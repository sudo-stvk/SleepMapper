import os
import multiprocessing

# Set thread environment variables before importing numpy/librosa
os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

import json
import yaml
import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from functools import partial
from tqdm import tqdm
import psutil
import time

def str2seconds(time_str: str, record_start_sec: float = 0.0) -> float:
    """
    Converts hh:mm:ss string to relative timestamp in seconds.
    """
    try:
        parts = time_str.split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid time format: {time_str}")
            
        hours, minutes, seconds = map(float, parts)
        
        if hours < 12:
            hours += 24
            
        total_seconds = hours * 3600 + minutes * 60 + seconds
        return total_seconds - record_start_sec
    except Exception as e:
        raise ValueError(f"Error converting time {time_str}: {e}")

def parse_time(val, record_start_sec):
    if isinstance(val, (int, float)):
        return float(val) - record_start_sec
    return str2seconds(str(val), record_start_sec)

def process_chunk(chunk_info, annotations_dict, config):
    """
    Processes a specific chunk of a patient's audio.
    """
    patient_id = chunk_info['patient_id']
    audio_path = chunk_info['audio_path']
    start_sample = chunk_info['start_sample']
    stop_sample = chunk_info['stop_sample']
    window_indices = chunk_info['window_indices']
    
    sample_rate = config['sample_rate']
    clip_duration = config['clip_duration']
    out_clips_dir = Path(config['out_clips_dir'])
    
    annotation = annotations_dict.get(patient_id)
    if not annotation:
        return None

    record_start_val = annotation.get('record_start', 0.0)
    try:
        if isinstance(record_start_val, (int, float)):
            record_start_sec = float(record_start_val)
        else:
            record_start_sec = str2seconds(str(record_start_val), 0.0)
    except Exception:
        record_start_sec = 0.0

    awake_intervals = []
    for awk in annotation.get('awake_intervals', []):
        if len(awk) == 2:
            try:
                s_awk = parse_time(awk[0], record_start_sec)
                e_awk = parse_time(awk[1], record_start_sec)
                awake_intervals.append((s_awk, e_awk))
            except Exception:
                pass

    apnea_intervals = []
    
    # Combine events from both formats into a single list to process
    events_to_process = []
    
    # New psg format: unified 'events' list
    for event in annotation.get('events', []):
        apnea_type = event.get('event_type', '').lower()
        if apnea_type in ['osa', 'csa', 'msa', 'hypo']:
            events_to_process.append((event, apnea_type))
            
    # Old osdb format: separate lists per type
    for apnea_type in ['osa', 'csa', 'msa', 'hypo']:
        for event in annotation.get(apnea_type, []):
            events_to_process.append((event, apnea_type))
            
    for event, apnea_type in events_to_process:
        try:
            if 'start' in event and 'end' in event:
                s_apn = parse_time(event['start'], record_start_sec)
                e_apn = parse_time(event['end'], record_start_sec)
            elif 'evnet_start' in event and 'event_duration' in event:
                s_apn = parse_time(event['evnet_start'], record_start_sec)
                e_apn = s_apn + float(event['event_duration'])
            elif 'event_start' in event and 'event_duration' in event:
                s_apn = parse_time(event['event_start'], record_start_sec)
                e_apn = s_apn + float(event['event_duration'])
            else:
                continue
            apnea_intervals.append((s_apn, e_apn, apnea_type))
        except Exception:
            continue

    # Load only the required chunk
    try:
        # Use soundfile for fast seeking and reading
        y, sr = sf.read(audio_path, start=start_sample, stop=stop_sample, dtype='float32')
        if sr != sample_rate:
            # Note: soundfile doesn't resample on the fly, 
            # if resampling is needed, librosa would be used, 
            # but here we assume data is already 16kHz or we handle it.
            # For max speed, we expect the raw data to match.
            pass 
    except Exception:
        return None

    labels_data = []
    stats = {'total_windows': 0, 'apnea_clips': 0, 'normal_clips': 0, 'windows_skipped_awake': 0}

    # Process windows in this chunk
    # window_indices are relative to the whole file
    for w_idx in window_indices:
        w_start_rel = (w_idx * clip_duration) - (start_sample / sample_rate)
        w_end_rel = w_start_rel + clip_duration
        
        w_start_abs = w_idx * clip_duration
        w_end_abs = w_start_abs + clip_duration
        
        # Determine if window is awake
        is_awake = False
        for awk_start, awk_end in awake_intervals:
            if max(w_start_abs, awk_start) < min(w_end_abs, awk_end):
                is_awake = True
                break
        
        if is_awake:
            stats['windows_skipped_awake'] += 1
            continue
            
        # Determine apnea type
        is_apnea = False
        primary_apnea_type = 'normal'
        for apn_start, apn_end, a_type in apnea_intervals:
            if max(w_start_abs, apn_start) < min(w_end_abs, apn_end):
                is_apnea = True
                primary_apnea_type = a_type
                break
        
        label = 1 if is_apnea else 0
        clip_filename = f"{patient_id}_window_{w_idx:04d}.wav"
        clip_path = out_clips_dir / clip_filename
        
        # Extract window from chunk using numpy
        start_idx = int(w_start_rel * sample_rate)
        end_idx = start_idx + int(clip_duration * sample_rate)
        
        # Safety check for indices
        if start_idx < 0 or end_idx > len(y):
            continue
            
        clip_y = y[start_idx:end_idx]
        
        # Save using soundfile (much faster than librosa)
        max_retries = 3
        write_success = False
        for attempt in range(max_retries):
            try:
                sf.write(clip_path, clip_y, sample_rate)
                write_success = True
                break
            except Exception as e:
                time.sleep(0.5)
                
        if not write_success:
            continue
        
        labels_data.append({
            'filename': clip_filename,
            'label': label,
            'patient_id': patient_id,
            'window_index': w_idx,
            'apnea_type': primary_apnea_type
        })
        
        stats['total_windows'] += 1
        if label == 1:
            stats['apnea_clips'] += 1
        else:
            stats['normal_clips'] += 1

    return {
        'labels_data': labels_data,
        'stats': stats,
        'patient_id': patient_id
    }

def main():
    # Find project root (2 levels up from src/preprocessing/)
    project_root = Path(__file__).parents[2]
    
    # Load configuration
    config_path = project_root / 'configs' / 'config.yaml'
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yaml: {e}")
        return

    sample_rate = config.get('sample_rate', 16000)
    clip_duration = config.get('clip_duration', 30)
    
    raw_psg_dir = project_root / 'data' / 'raw' / 'psg'
    # Use S: drive to avoid running out of space on C:
    out_clips_dir = Path('S:/SleepMapper_Data/clips')
    out_clips_dir.mkdir(parents=True, exist_ok=True)
    
    config_ext = {
        'sample_rate': sample_rate,
        'clip_duration': clip_duration,
        'out_clips_dir': str(out_clips_dir)
    }

    if not raw_psg_dir.exists():
        print(f"Error: Raw data directory {raw_psg_dir} does not exist.")
        return

    patient_folders = [f for f in raw_psg_dir.iterdir() if f.is_dir()]
    patient_folders.sort()

    # 1. Pre-load all annotations
    print("Pre-loading annotations...")
    annotations_dict = {}
    for folder in patient_folders:
        patient_id = folder.name
        json_path = folder / f"{patient_id}_annotation.json"
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    annotations_dict[patient_id] = json.load(f)
            except Exception as e:
                print(f"Error loading annotation for {patient_id}: {e}")

    # 2. Create chunked tasks
    print("Analyzing audio files and creating tasks...")
    tasks = []
    windows_per_chunk = 50  # Process 50 windows (25 mins) per task to balance overhead vs parallelism
    
    for folder in patient_folders:
        patient_id = folder.name
        audio_path = folder / f"{patient_id}_phone.wav"
        
        if not audio_path.exists() or patient_id not in annotations_dict:
            continue
            
        try:
            # Use soundfile.info to get metadata without loading
            info = sf.info(str(audio_path))
            duration_sec = info.duration
            total_windows = int(duration_sec // clip_duration)
            
            for i in range(0, total_windows, windows_per_chunk):
                end_win = min(i + windows_per_chunk, total_windows)
                win_indices = list(range(i, end_win))
                
                start_sample = int(i * clip_duration * sample_rate)
                # Read slightly more to ensure we have the full last window
                stop_sample = int(end_win * clip_duration * sample_rate)
                
                tasks.append({
                    'patient_id': patient_id,
                    'audio_path': str(audio_path),
                    'start_sample': start_sample,
                    'stop_sample': stop_sample,
                    'window_indices': win_indices
                })
        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}")

    num_workers = os.cpu_count()
    print(f"Starting processing with {num_workers} workers and {len(tasks)} tasks...")
    
    all_labels_data = []
    total_windows_gen = 0
    windows_skipped_awake = 0
    apnea_clips = 0
    normal_clips = 0
    
    # 3. Multiprocessing with Pool and tqdm
    # Using partial to pass shared data
    worker_fn = partial(process_chunk, annotations_dict=annotations_dict, config=config_ext)
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Using imap_unordered for better performance as chunks are independent
        pbar = tqdm(total=len(tasks), desc="Processing chunks")
        
        for res in pool.imap_unordered(worker_fn, tasks):
            if res:
                all_labels_data.extend(res['labels_data'])
                total_windows_gen += res['stats']['total_windows']
                windows_skipped_awake += res['stats']['windows_skipped_awake']
                apnea_clips += res['stats']['apnea_clips']
                normal_clips += res['stats']['normal_clips']
            
            # Update progress bar with stats
            cpu_usage = psutil.cpu_percent()
            pbar.set_postfix({
                'CPU': f"{cpu_usage}%",
                'Windows': total_windows_gen,
                'Apnea': apnea_clips
            })
            pbar.update(1)
            
        pbar.close()

    # 4. Save results
    if all_labels_data:
        df = pd.DataFrame(all_labels_data)
        df.to_csv(out_clips_dir / 'labels.csv', index=False)
        print(f"\nSaved {len(all_labels_data)} labels to {out_clips_dir / 'labels.csv'}")
        
    print("\n=== Processing Summary ===")
    print(f"Total windows generated: {total_windows_gen}")
    print(f"Windows skipped (awake): {windows_skipped_awake}")
    print(f"Apnea clips (label=1):   {apnea_clips}")
    print(f"Normal clips (label=0):  {normal_clips}")
    
    if normal_clips > 0:
        ratio = apnea_clips / normal_clips
        print(f"Class imbalance ratio:   {ratio:.2f} (Apnea:Normal)")

if __name__ == '__main__':
    # Fix for Windows multiprocessing
    multiprocessing.freeze_support()
    main()