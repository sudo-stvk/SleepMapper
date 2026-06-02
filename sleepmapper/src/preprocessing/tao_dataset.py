"""
SleepMapper — Tao et al. (Scientific Data 2025) Multimodal Dataset
DOI: https://doi.org/10.57760/sciencedb.19070

Handles:
  - smartphone_audio.mp3  (48kHz stereo → resample 16kHz mono)
  - spo2.csv              (1Hz, columns: idx, timestamp, spo2)
  - annotations.json      (AASM event format)

Returns per clip: (mel_tensor (1,128,T), spo2_tensor (50,), label float32)

Run as module:
    python -m src.preprocessing.tao_dataset
"""

import os
import json
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SAMPLE_RATE    = 16000       # resample everything to 16kHz (matches pipeline)
CLIP_DURATION  = 30          # seconds per clip (matches config.yaml)
HOP_SIZE       = 15          # sliding window hop: 50% overlap to get more clips from 50 patients
N_MELS         = 128         # mel bands (matches existing spectrogram.py)
HOP_LENGTH     = 512         # STFT hop (matches config.yaml)
N_FFT          = 2048        # FFT size (matches config.yaml)
SPO2_WINDOW    = 50          # 30s clip + 20s lookahead to capture physiological SpO2 lag
SPO2_HZ        = 1           # SpO2 sensor sampling rate: 1 Hz
SPO2_LEN       = SPO2_WINDOW * SPO2_HZ  # = 50 samples

# SpO2 clinical normalisation range (sensor dropout readings outside this are clamped)
SPO2_MIN = 70.0
SPO2_MAX = 100.0

# AASM event types that map to label=1
# Long names (Tao format) and short names (existing PSG format) both handled in _load_annotations
APNEA_TYPES = {"obstructive apnea", "central apnea", "mixed apnea", "hypopnea"}

# ──────────────────────────────────────────────────────────────────────────────


def _hms_to_seconds(hms_str):
    """
    Convert hh:mm:ss.ms string to total seconds (float).
    Handles the format used in the real SpO2 CSV: '21:31:17.000'
    """
    parts = hms_str.strip().split(":")
    hours   = float(parts[0])
    minutes = float(parts[1])
    seconds = float(parts[2])
    return hours * 3600 + minutes * 60 + seconds


def _parse_spo2(spo2_path):
    """
    Load SpO2 CSV and return:
        (spo2_arr, spo2_start_offset_sec)

    Real CSV format (from the PSG device, patient 21 example):
        relative position (hh:mm:ss.ms), absolute position (hh:mm:ss.ms), OSat ("%")
        00:00:00.000, 21:31:17.000, 93
        00:00:01.000, 21:31:18.000, 93
        ...

    spo2_arr              : 1Hz numpy array of SpO2 values (already in raw % units)
                            Index 0 = first second of SpO2 recording.
    spo2_start_offset_sec : absolute wall-clock seconds from midnight when the
                            SpO2 sensor started. The caller uses this to compute
                            how many seconds into the SpO2 array a given audio
                            clip falls, by subtracting record_start.

    Example for patient 21:
        record_start = 73847.0   (audio start: 20:30:47)
        spo2 start   = 77477.0   (SpO2 start:  21:31:17)
        offset       = 77477 - 73847 = 3630 s  (SpO2 began 60.5 min after audio)
        → a clip at audio-relative second 4000 maps to spo2_arr[4000 - 3630] = spo2_arr[370]

    Forward-fill then back-fill because sensor dropout is common.
    """
    df = pd.read_csv(spo2_path)

    # Normalise column names: strip whitespace and lowercase
    orig_cols = list(df.columns)
    df.columns = [c.strip().lower() for c in orig_cols]

    # ── Find SpO2 value column ──
    # Real device exports: 'osat ("%")'
    # Tao dataset may use: 'spo2', 'spo2_%', 'value'
    spo2_col = None
    for candidate in ['osat ("%")', 'osat(%)', 'osat', 'spo2', 'spo2_%', 'value']:
        if candidate in df.columns:
            spo2_col = candidate
            break

    if spo2_col is None:
        raise ValueError(
            f"Cannot find SpO2 column in {spo2_path}.\n"
            f"Found columns: {df.columns.tolist()}\n"
            f"Expected one of: osat, spo2, spo2_%, value"
        )

    # pd.to_numeric with errors='coerce' converts any non-numeric value
    # (e.g. '-' which the PSG device writes during sensor dropout) to NaN.
    # The subsequent ffill/bfill then fills those gaps with the last valid reading.
    values = pd.to_numeric(df[spo2_col], errors='coerce')
    values = values.ffill().bfill()
    if values.isna().any():
        # If the entire recording is NaN (sensor completely absent), fill with 95%
        values = values.fillna(95.0)
        print(f"[WARNING] All SpO2 values are NaN in {spo2_path}. Using default 95%.")
    spo2_arr = values.to_numpy(dtype=np.float32)

    # ── Compute the absolute start time of the SpO2 recording ──
    # Look for the 'absolute position' column to get wall-clock time of row 0.
    # This is critical for aligning with the audio recording's record_start.
    abs_col = None
    for candidate in ["absolute position (hh:mm:ss.ms)", "absolute position", "abs_time", "timestamp"]:
        if candidate in df.columns:
            abs_col = candidate
            break

    if abs_col is not None:
        # Parse the first row's absolute time → seconds from midnight
        first_abs = df[abs_col].iloc[0].strip()
        spo2_start_offset_sec = _hms_to_seconds(first_abs)
    else:
        # No absolute time column — assume SpO2 and audio started simultaneously
        # (will be correct if record_start matches, wrong otherwise — warn the user)
        print(f"[WARNING] No absolute time column found in {spo2_path}. "
              f"Assuming SpO2 started at the same time as the audio recording. "
              f"Verify this manually.")
        spo2_start_offset_sec = 0.0

    return spo2_arr, spo2_start_offset_sec


def _extract_spo2_window(spo2_arr, spo2_start_offset_sec, clip_start_sec, record_start_sec):
    """
    Extract a SPO2_LEN (50-sample) window aligned with an audio clip.

    SpO2 desaturation lags apnea onset by 15–30 seconds physiologically,
    so we use a window that covers [clip_start, clip_start + SPO2_WINDOW)
    = 30s clip + 20s lookahead = 50 samples total.

    Alignment math (real example — patient 21):
        record_start          = 73847 s  (audio recording started at 20:30:47)
        spo2_start_offset_sec = 77477 s  (SpO2 sensor started at 21:31:17)
        clip_start_sec        = 4000 s   (clip is 4000s after audio start)

        → absolute time of clip = 73847 + 4000 = 77847 s
        → SpO2 index of clip    = 77847 - 77477 = 370
        → we read spo2_arr[370 : 370 + 50]

    If clip falls before SpO2 sensor started (index < 0), we return
    a neutral healthy SpO2 window (95% normalised) rather than crashing.

    Args:
        spo2_arr             : 1Hz SpO2 array, index 0 = first second of SpO2 recording
        spo2_start_offset_sec: absolute seconds-from-midnight when SpO2 sensor started
        clip_start_sec       : clip start in seconds relative to record_start (audio start)
        record_start_sec     : absolute seconds-from-midnight when audio recording started

    Returns:
        np.ndarray of shape (50,) in [0.0, 1.0], float32
    """
    # Convert clip's audio-relative time → absolute time → SpO2 array index
    clip_abs_time = record_start_sec + clip_start_sec
    start_idx     = int(clip_abs_time - spo2_start_offset_sec)
    end_idx       = start_idx + SPO2_LEN

    # Clip falls before SpO2 sensor started — return neutral healthy signal
    # 95% SpO2 normalised: (95 - 70) / (100 - 70) = 0.833
    if start_idx < 0:
        return np.full(SPO2_LEN, 0.833, dtype=np.float32)

    # Pad end if clip's lookahead extends past end of SpO2 recording
    if end_idx > len(spo2_arr):
        pad_len  = end_idx - len(spo2_arr)
        last_val = spo2_arr[-1] if len(spo2_arr) > 0 else 95.0
        spo2_arr = np.concatenate([spo2_arr, np.full(pad_len, last_val)])

    window = spo2_arr[start_idx:end_idx].copy()

    # Clamp to clinical range [70, 100] then normalise to [0, 1]
    window = np.clip(window, SPO2_MIN, SPO2_MAX)
    window = (window - SPO2_MIN) / (SPO2_MAX - SPO2_MIN)

    return window.astype(np.float32)


def _load_annotations(ann_path):
    """
    Load annotation JSON — supports BOTH formats in the codebase:

    Format A (Tao et al. 2025 — new dataset):
        { "record_start": 79200,
          "awake_intervals": [{"start": 100, "end": 350}],  ← dict-based
          "events": [{"event_type": "Obstructive Apnea",
                      "event_start": 79450, "event_duration": 12.5}] }

    Format B (existing PSG data — already in the pipeline):
        { "record_start": 73847.0,
          "awake_intervals": [[77478.0, 77815.0], ...],  ← list-based, ABSOLUTE times
          "events": [{"event_type": "hypo",
                      "evnet_start": 77815.0,  ← NOTE the typo: 'evnet_start'
                      "event_duration": 16.0}] }

    All intervals returned are in seconds RELATIVE to record_start.
    """
    with open(ann_path, "r") as f:
        ann = json.load(f)

    record_start = float(ann.get("record_start", 0.0))

    # ── Parse apnea events (handles both 'event_start' and 'evnet_start' typo) ──
    apnea_intervals = []

    # New PSG format: unified 'events' list
    for ev in ann.get("events", []):
        etype = ev.get("event_type", "").strip().lower()
        # Map Tao long names and existing short names to the same set
        is_apnea = etype in APNEA_TYPES or etype in {"osa", "csa", "msa", "hypo"}
        if not is_apnea:
            continue

        # Handle both key spellings — 'event_start' (correct) and 'evnet_start' (typo in PSG data)
        if "event_start" in ev:
            ev_abs_start = float(ev["event_start"])
        elif "evnet_start" in ev:
            ev_abs_start = float(ev["evnet_start"])   # preserve existing typo
        else:
            continue

        ev_dur         = float(ev["event_duration"])
        ev_start_rel   = ev_abs_start - record_start   # relative to record_start
        apnea_intervals.append((ev_start_rel, ev_start_rel + ev_dur))

    # Old OSDB format: separate lists per apnea type (kept for backward compat)
    for apnea_key in ["osa", "csa", "msa", "hypo"]:
        for ev in ann.get(apnea_key, []):
            if "event_start" in ev:
                s = float(ev["event_start"]) - record_start
            elif "evnet_start" in ev:
                s = float(ev["evnet_start"]) - record_start
            else:
                continue
            d = float(ev["event_duration"])
            apnea_intervals.append((s, s + d))

    # ── Parse awake intervals ──
    # Format A: list of dicts {"start": x, "end": y}  — times relative to record_start
    # Format B: list of [abs_start, abs_end]           — ABSOLUTE times, need to subtract record_start
    awake_intervals = []
    for awk in ann.get("awake_intervals", []):
        if isinstance(awk, dict):
            # Tao format: times are already relative to record_start
            s = float(awk["start"])
            e = float(awk["end"])
        elif isinstance(awk, list) and len(awk) == 2:
            # Existing PSG format: absolute times — subtract record_start
            s = float(awk[0]) - record_start
            e = float(awk[1]) - record_start
        else:
            continue
        awake_intervals.append((s, e))

    return record_start, apnea_intervals, awake_intervals


def _audio_to_mel(audio):
    """
    Convert raw waveform (16kHz) to log-mel spectrogram.
    Matches the parameters in spectrogram.py / config.yaml exactly.

    Returns: (128, T) float32 numpy array (NOT log10 — using power_to_db)
    """
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
    )
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)


def _spec_augment(mel, F=15, T=50, num_masks=2):
    """SpecAugment — same parameters as dataset.py (training only)."""
    mel = mel.copy()
    n_mels, n_frames = mel.shape
    for _ in range(num_masks):
        f  = np.random.randint(0, F)
        f0 = np.random.randint(0, max(n_mels - f, 1))
        mel[f0:f0 + f, :] = 0

        t  = np.random.randint(0, min(T, n_frames))
        t0 = np.random.randint(0, max(n_frames - t, 1))
        mel[:, t0:t0 + t] = 0
    return mel


def build_tao_splits(data_root, random_state=42):
    """
    Patient-level train / val / test split for the Tao et al. dataset.
    50 patients → 35 train / 8 val / 7 test.
    NEVER splits at the clip level — see Section 4 locked decisions.

    Args:
        data_root   : Path or str pointing to the Tao dataset root directory.
                      Each subdirectory must be a patient folder containing
                      smartphone_audio.mp3, spo2.csv, and annotations.json.
        random_state: int, for reproducibility (must be 42 per locked decisions).

    Returns:
        (train_dirs, val_dirs, test_dirs) — lists of Path objects
    """
    data_root = Path(data_root)
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if len(patient_dirs) == 0:
        raise ValueError(f"No patient directories found under {data_root}")

    # First split off test (7 patients, ~14%), then val from remainder
    train_val, test_dirs = train_test_split(
        patient_dirs,
        test_size=7,
        random_state=random_state,
    )
    train_dirs, val_dirs = train_test_split(
        train_val,
        test_size=8,
        random_state=random_state,
    )

    print(f"[Tao split] Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)} patients")
    return train_dirs, val_dirs, test_dirs

def build_psg_splits(data_root, random_state=42):
    """
    Patient-level train / val / test split for the existing PSG dataset.
    20 patients (01-22, gaps at 11 and 17) → 14 train / 3 val / 3 test.
    NEVER splits at clip level.

    File naming convention per patient folder:
        {id}_phone.wav           ← full-night audio at 16kHz
        {id}_annotation.json     ← apnea event labels
        {id}_SpO2.csv            ← SpO2 readings (placed here in this step)

    Args:
        data_root   : path to data/raw/psg/
        random_state: int, must be 42 for reproducibility

    Returns:
        (train_dirs, val_dirs, test_dirs) — lists of Path objects
    """
    data_root = Path(data_root)
    patient_dirs = sorted([d for d in data_root.iterdir() if d.is_dir()])

    if len(patient_dirs) == 0:
        raise ValueError(f"No patient directories found under {data_root}")

    # Filter to patients that have ALL three required files
    valid_dirs = []
    for d in patient_dirs:
        pid = d.name
        has_audio = (d / f"{pid}_phone.wav").exists()
        has_ann   = (d / f"{pid}_annotation.json").exists()
        has_spo2  = (d / f"{pid}_SpO2.csv").exists()
        if has_audio and has_ann and has_spo2:
            valid_dirs.append(d)
        else:
            missing = []
            if not has_audio: missing.append(f"{pid}_phone.wav")
            if not has_ann:   missing.append(f"{pid}_annotation.json")
            if not has_spo2:  missing.append(f"{pid}_SpO2.csv")
            print(f"[WARNING] Skipping patient {pid} — missing: {', '.join(missing)}")

    print(f"[PSG] {len(valid_dirs)} valid patients found (of {len(patient_dirs)} folders)")

    if len(valid_dirs) < 6:
        raise ValueError(f"Need at least 6 patients for a 3-way split. Found only {len(valid_dirs)}.")

    # 20 patients → 3 test, 3 val, 14 train
    train_val, test_dirs = train_test_split(
        valid_dirs,
        test_size=3,
        random_state=random_state,
    )
    train_dirs, val_dirs = train_test_split(
        train_val,
        test_size=3,
        random_state=random_state,
    )

    print(f"[PSG split] Train: {len(train_dirs)} | Val: {len(val_dirs)} | Test: {len(test_dirs)} patients")
    print(f"  Train patients: {sorted([d.name for d in train_dirs])}")
    print(f"  Val   patients: {sorted([d.name for d in val_dirs])}")
    print(f"  Test  patients: {sorted([d.name for d in test_dirs])}")
    return train_dirs, val_dirs, test_dirs


class TaoMultimodalDataset(Dataset):
    """
    Multimodal dataset — works with BOTH data sources:

    PSG format (data/raw/psg/):
        {id}_phone.wav, {id}_annotation.json, {id}_SpO2.csv

    Tao et al. format:
        smartphone_audio.mp3, annotations.json, spo2.csv

    File naming is auto-detected per patient folder.
    Returns per clip: (mel_tensor (1,128,T), spo2_tensor (50,), label float32)
    """
    def __init__(self, patient_dirs, split="train", augment=False, cache_audio=False, cache_dir=None):
        """
        Args:
            patient_dirs: list of Path objects from build_psg_splits or build_tao_splits
            split       : 'train', 'val', or 'test' — used for logging only
            augment     : apply SpecAugment (training only)
            cache_audio : cache full night audio in memory (only use when num_workers=0)
        """
        self.augment = augment
        self.cache_audio = cache_audio
        self.clips   = []
        self.last_audio_path = None
        self.last_audio_data = None

        # Disk cache: pre-computed mel spectrograms (eliminates librosa I/O after first pass)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/cache/multimodal")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        n_normal  = 0
        n_apnea   = 0
        n_skipped = 0

        for patient_dir in patient_dirs:
            patient_dir = Path(patient_dir)
            pid = patient_dir.name

            # Auto-detect naming: PSG uses {id}_phone.wav, Tao uses smartphone_audio.mp3
            audio_path = patient_dir / f"{pid}_phone.wav"
            if not audio_path.exists():
                audio_path = patient_dir / "smartphone_audio.mp3"
            if not audio_path.exists():
                audio_path = patient_dir / "voice_recorder_1.wav"
            if not audio_path.exists():
                print(f"[WARNING] No audio found in {patient_dir}, skipping.")
                continue

            # PSG: {id}_annotation.json, Tao: annotations.json
            ann_path = patient_dir / f"{pid}_annotation.json"
            if not ann_path.exists():
                ann_path = patient_dir / "annotations.json"
            if not ann_path.exists():
                print(f"[WARNING] No annotation JSON in {patient_dir}, skipping.")
                continue

            # PSG: {id}_SpO2.csv, Tao: spo2.csv
            spo2_path = patient_dir / f"{pid}_SpO2.csv"
            if not spo2_path.exists():
                spo2_path = patient_dir / "spo2.csv"
            if not spo2_path.exists():
                print(f"[WARNING] No SpO2 CSV in {patient_dir}, skipping.")
                continue

            try:
                record_start, apnea_intervals, awake_intervals = _load_annotations(ann_path)
                # _parse_spo2 now returns (values_array, spo2_start_offset_sec)
                # spo2_start_offset_sec is the absolute wall-clock second (from midnight)
                # when the SpO2 sensor first started recording.
                spo2_arr, spo2_start_offset_sec = _parse_spo2(spo2_path)
            except Exception as e:
                print(f"[WARNING] Failed to parse {patient_dir.name}: {e}")
                continue

            # Get audio duration without loading the full file
            try:
                duration_sec = librosa.get_duration(path=str(audio_path))
            except Exception as e:
                print(f"[WARNING] Cannot get duration for {audio_path}: {e}")
                continue

            # Slide a 30s window with 15s hop (50% overlap) across the recording
            total_clips = int((duration_sec - CLIP_DURATION) / HOP_SIZE) + 1

            for i in range(total_clips):
                start_sec = i * HOP_SIZE
                end_sec   = start_sec + CLIP_DURATION

                if end_sec > duration_sec:
                    break

                # Skip awake windows — awake_intervals are relative to record_start
                is_awake = False
                for awk_s, awk_e in awake_intervals:
                    if max(start_sec, awk_s) < min(end_sec, awk_e):
                        is_awake = True
                        break
                if is_awake:
                    n_skipped += 1
                    continue

                # Determine label: apnea if any event overlaps this clip
                is_apnea = False
                for apn_s, apn_e in apnea_intervals:
                    if max(start_sec, apn_s) < min(end_sec, apn_e):
                        is_apnea = True
                        break

                label = 1.0 if is_apnea else 0.0

                self.clips.append({
                    "audio_path":           str(audio_path),
                    "spo2_arr":             spo2_arr,
                    "spo2_start_offset_sec": spo2_start_offset_sec,  # wall-clock start of SpO2
                    "record_start":         record_start,             # wall-clock start of audio
                    "start_sec":            start_sec,                # clip offset from audio start
                    "label":                label,
                })

                if is_apnea:
                    n_apnea += 1
                else:
                    n_normal += 1

        print(f"[{split} Tao] Normal: {n_normal} | Apnea: {n_apnea} | Skipped (awake): {n_skipped}")

    def __len__(self):
        return len(self.clips)

    @property
    def labels(self):
        """Expose labels list for WeightedRandomSampler compatibility (matches existing datasets)."""
        return [int(c["label"]) for c in self.clips]

    def __getitem__(self, idx):
        clip = self.clips[idx]

        # ── Fast path: load from disk cache ──
        audio_stem = Path(clip["audio_path"]).stem
        cache_file = self.cache_dir / f"{audio_stem}_{int(clip['start_sec']):06d}.pt"
        if cache_file.exists():
            try:
                cached = torch.load(cache_file, weights_only=True)
                mel = cached['mel'].float().numpy()       # (128, T) float16 → float32
                spo2_window = cached['spo2'].numpy()      # (50,) float32
                if self.augment:
                    mel = _spec_augment(mel)
                mel_tensor  = torch.from_numpy(mel).float().unsqueeze(0)
                spo2_tensor = cached['spo2'].float()
                label_tensor = torch.tensor(clip["label"], dtype=torch.float32)
                return mel_tensor, spo2_tensor, label_tensor
            except Exception:
                pass  # fall through to slow path if cache is corrupted

        try:
            audio_path = clip["audio_path"]
            if self.cache_audio:
                # Cache the full audio to avoid open/decode/close overhead for every clip
                if not hasattr(self, 'audio_cache'):
                    self.audio_cache = {}
                    
                if audio_path not in self.audio_cache:
                    self.audio_cache[audio_path], _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
                    
                start_sample = int(clip["start_sec"] * SAMPLE_RATE)
                end_sample = start_sample + int(CLIP_DURATION * SAMPLE_RATE)
                audio = self.audio_cache[audio_path][start_sample:end_sample]
            else:
                # Load only the 30s segment during training/validation to avoid memory blowup
                audio, _ = librosa.load(
                    audio_path,
                    sr=SAMPLE_RATE,
                    mono=True,
                    offset=clip["start_sec"],
                    duration=float(CLIP_DURATION),
                )

            # Pad if the last clip is shorter than 30s (edge case near end of recording)
            target_len = SAMPLE_RATE * CLIP_DURATION
            if len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)), mode="constant")
            else:
                audio = audio[:target_len]

            # Compute log-mel spectrogram — (128, T)
            mel = _audio_to_mel(audio)

            # Extract SpO2 window: 30s clip + 20s physiological lookahead = 50 samples
            spo2_window = _extract_spo2_window(
                clip["spo2_arr"],
                clip["spo2_start_offset_sec"],
                clip["start_sec"],
                clip["record_start"],
            )

            # Save to disk cache for subsequent epochs (float16 mel → ~235 KB/clip)
            try:
                torch.save({
                    'mel': torch.from_numpy(mel).half(),
                    'spo2': torch.from_numpy(spo2_window),
                }, cache_file)
            except Exception:
                pass  # don't crash training if cache write fails

            if self.augment:
                mel = _spec_augment(mel)

            mel_tensor  = torch.from_numpy(mel).float().unsqueeze(0)  # (1, 128, T)
            spo2_tensor = torch.from_numpy(spo2_window).float()        # (50,)
            label_tensor = torch.tensor(clip["label"], dtype=torch.float32)

            return mel_tensor, spo2_tensor, label_tensor

        except Exception as e:
            print(f"Error loading clip {idx} ({clip['audio_path']} @ {clip['start_sec']}s): {e}")
            # Return zero tensors with label=-1 so the training loop can skip this batch
            # Calculate dummy T dynamically to match collation shape
            dummy_T = 1 + (SAMPLE_RATE * CLIP_DURATION) // HOP_LENGTH
            mel_dummy  = torch.zeros((1, N_MELS, dummy_T), dtype=torch.float32)
            spo2_dummy = torch.zeros(SPO2_LEN, dtype=torch.float32)
            return mel_dummy, spo2_dummy, torch.tensor(-1.0, dtype=torch.float32)

    def precompute_cache(self):
        """
        Pre-compute and cache all mel spectrograms to disk.
        Loads each audio file ONCE, extracts all clips, saves mels as float16.
        Run before training for maximum speed:
            python -m src.training.train_multimodal data/raw/psg --dataset psg --precompute
        """
        from collections import defaultdict
        from tqdm import tqdm

        # Group clips by audio file → load each WAV only once
        clips_by_audio = defaultdict(list)
        for idx, clip in enumerate(self.clips):
            audio_stem = Path(clip["audio_path"]).stem
            cf = self.cache_dir / f"{audio_stem}_{int(clip['start_sec']):06d}.pt"
            if not cf.exists():
                clips_by_audio[clip["audio_path"]].append((idx, clip, cf))

        total = sum(len(v) for v in clips_by_audio.values())
        if total == 0:
            print("  All clips already cached — nothing to do.")
            return

        print(f"  Caching {total} clips from {len(clips_by_audio)} audio files...")

        for audio_path, clip_list in clips_by_audio.items():
            print(f"  Loading {Path(audio_path).name} ({len(clip_list)} clips)...")
            try:
                full_audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            except Exception as e:
                print(f"  ERROR: {e}")
                continue

            for idx, clip, cf in tqdm(clip_list, desc=f"    {Path(audio_path).stem}", leave=False):
                start_sample = int(clip["start_sec"] * SAMPLE_RATE)
                end_sample   = start_sample + SAMPLE_RATE * CLIP_DURATION
                audio_clip   = full_audio[start_sample:end_sample]

                target_len = SAMPLE_RATE * CLIP_DURATION
                if len(audio_clip) < target_len:
                    audio_clip = np.pad(audio_clip, (0, target_len - len(audio_clip)))
                else:
                    audio_clip = audio_clip[:target_len]

                mel = _audio_to_mel(audio_clip)
                spo2_window = _extract_spo2_window(
                    clip["spo2_arr"], clip["spo2_start_offset_sec"],
                    clip["start_sec"], clip["record_start"],
                )

                torch.save({
                    'mel': torch.from_numpy(mel).half(),
                    'spo2': torch.from_numpy(spo2_window),
                }, cf)

            del full_audio  # free RAM before loading next patient

        print(f"  Cache complete -> {self.cache_dir}")


# ─── Quick smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing.tao_dataset <path_to_tao_data_root>")
        sys.exit(1)

    data_root = sys.argv[1]
    train_dirs, val_dirs, test_dirs = build_tao_splits(data_root, random_state=42)

    ds = TaoMultimodalDataset(train_dirs, split="train", augment=True)
    print(f"Total train clips: {len(ds)}")

    if len(ds) > 0:
        mel, spo2, label = ds[0]
        print(f"mel shape : {mel.shape}")    # (1, 128, T)
        print(f"spo2 shape: {spo2.shape}")   # (50,)
        print(f"label     : {label}")
