# SleepMapper

SleepMapper is an on-device sleep apnea detection project that utilizes smartphone audio recordings to identify potential sleep apnea events.

## Project Structure

- `data/`: Contains raw, processed, and split datasets.
- `src/`: Core logic for preprocessing, modeling, and training.
  - `preprocessing/`: Audio feature extraction and cleaning.
  - `models/`: ML model architectures (e.g., Transformers, CNNs).
  - `training/`: Training loops and evaluation scripts.
  - `utils/`: Common utility functions.
- `configs/`: Configuration files for experiments.
- `outputs/`: Training checkpoints and logs.
- `notebooks/`: Jupyter notebooks for EDA and experimentation.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure parameters in `configs/config.yaml`.

## Usage

(Details to be added as implementation progresses)
