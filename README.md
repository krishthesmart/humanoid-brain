# YouTube Robot Training Data Collector

A complete Python system to collect YouTube training data for robot task models, with built-in storage constraints and metadata tracking.

## Features

- **Storage Management**: Automatically stops at 50GB (configurable).
- **Video Filtering**: Filters by resolution (480p), duration (5-20 min), and views (>10k).
- **Manifest Tracking**: Keeps a detailed CSV manifest of all downloaded content.
- **Task-Based**: Supports specific household tasks out of the box.
- **Resume & Duplicate Detection**: Skips already downloaded videos.
- **Cleanup Mode**: Option to delete video files while preserving metadata.
- **Bonus**: Automatic frame extraction for video previews.

## Installation

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Check current status
```bash
python collect_data.py --status
```

### 2. Download videos for a specific task
```bash
python collect_data.py --task laundry --videos 5
```

### 3. Download for all tasks
```bash
python collect_data.py --task all
```

### 4. Cleanup storage (deletes videos, keeps manifest)
```bash
python collect_data.py --cleanup
```

## Fast Task Classifier Training (Recommended on CPU)

For per-task classification accuracy (`cleaning`, `cooking`, `dishwashing`, `laundry`, `organizing`) without heavy VLM fine-tuning:

```bash
python train_task_classifier.py --epochs 12 --batch-size 32
```

This trains a MobileNet classifier and prints:
- Validation accuracy per epoch
- Final test overall accuracy
- Final per-task accuracy

Checkpoint output:
- `task_classifier.pt`

## Configuration

Adjust settings in `config.yaml`:
- `storage.max_gb`: Maximum storage limit.
- `video.max_resolution`: Target resolution (default 480).
- `tasks`: Add search terms or change targets for each task.

## File Structure

```text
training_data/
├── laundry/
├── dishwashing/
├── cooking/
├── cleaning/
├── organizing/
├── manifest.csv
└── storage_report.txt
```

## Technical Details

- **Downloader**: `yt-dlp`
- **Metadata**: `pandas`
- **Progress**: `tqdm`
- **Config**: `PyYAML`
- **OS**: Cross-platform ready (uses `pathlib`)
