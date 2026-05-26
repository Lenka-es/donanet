# User Guide

## Workflow Overview

```
Raw images + labels
        │
        ▼
 prepare-dataset          ← splits into train / val / test
        │
        ▼
     train                ← fine-tunes YOLO, writes weights/
        │
        ▼
      test                ← evaluates or runs inference on test/
```

---

## prepare-dataset

Split a directory of raw images (and their YOLO `.txt` label files) into the
three partitions used by training.

```bash
python donanet.py prepare-dataset \
    --source /path/to/raw \
    --train 0.7 \
    --val   0.2 \
    --test  0.1
```

| Option | Default | Description |
|---|---|---|
| `--source` | *(required)* | Directory containing images (and matching labels) |
| `--train` | `0.7` | Fraction for training |
| `--val` | `0.2` | Fraction for validation |
| `--test` | `0.1` | Fraction for testing |
| `--seed` | `42` | Random seed for reproducibility |
| `--move` | `False` | Move files instead of copying |

The command looks for image files (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)
in `--source` and expects a label file with the same stem in the same directory
(e.g. `fox_001.jpg` → `fox_001.txt`).

---

## train

Fine-tune a YOLO model on `dataset/train` and `dataset/val`.

```bash
python donanet.py train \
    --model  yolov8n.pt \
    --epochs 100 \
    --name   wildlife_v1
```

| Option | Default | Description |
|---|---|---|
| `--model` | `yolov8n.pt` | Base model checkpoint (Ultralytics hub or local path) |
| `--epochs` | `100` | Number of training epochs |
| `--imgsz` | `640` | Input image size (pixels) |
| `--batch` | `16` | Batch size (-1 = auto) |
| `--name` | `run` | Run name — weights saved to `weights/<name>/` |
| `--device` | *(auto)* | Device: `cpu`, `0`, `0,1`, … |
| `--resume` | `False` | Resume from the last checkpoint |
| `--patience` | `50` | Early-stopping patience (epochs without improvement) |

After training finishes, weights are copied to:

```
weights/<name>/best.pt
weights/<name>/last.pt
```

---

## test

Evaluate the model on `dataset/test` **or** run inference on arbitrary images.

```bash
# Evaluate on the built-in test partition
python donanet.py test --weights weights/wildlife_v1/best.pt

# Inference on a custom image directory
python donanet.py test \
    --weights weights/wildlife_v1/best.pt \
    --source  /path/to/images \
    --conf    0.25 \
    --save-images
```

| Option | Default | Description |
|---|---|---|
| `--weights` | *(required)* | Path to `.pt` weight file |
| `--source` | `dataset/test/images` | Images or video to run on |
| `--conf` | `0.25` | Confidence threshold |
| `--iou` | `0.45` | IoU threshold for NMS |
| `--imgsz` | `640` | Input image size |
| `--device` | *(auto)* | Device |
| `--save-images` | `False` | Save annotated output images |
| `--save-txt` | `False` | Save predictions as YOLO `.txt` files |

---

## list-datasets

Print a summary table of all three partitions:

```bash
python donanet.py list-datasets
```

Example output:

```
┌───────────┬────────┬────────┐
│ Partition │ Images │ Labels │
├───────────┼────────┼────────┤
│ train     │    350 │    350 │
│ val       │    100 │    100 │
│ test      │     50 │     50 │
└───────────┴────────┴────────┘
```

---

## info

Show available weights and a dataset summary in one go:

```bash
python donanet.py info
```
