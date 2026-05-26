# Installation Guide

## Prerequisites

- Python 3.11 or later (3.12 / 3.13 work too)
- [uv](https://docs.astral.sh/uv/) *(recommended)* — or plain `pip`
- A CUDA-capable GPU is **strongly recommended** for training; CPU-only inference works
- `libGL` system library (required by OpenCV, which Ultralytics depends on)

```bash
# Fedora / RHEL
sudo dnf install mesa-libGL

# Debian / Ubuntu
sudo apt-get install libgl1
```

---

## 1. Clone and Enter the Project

```bash
git clone <your-repo-url>
cd donanet
```

---

## 2. Install Python Dependencies

### With uv (recommended)

```bash
uv sync
```

### With pip

```bash
pip install -e .
```

---

## 3. Verify the CLI

```bash
python donanet.py --help
```

Expected output:

```
 Usage: donanet.py [OPTIONS] COMMAND [ARGS]...

 DonaNet — WildINTEL YOLO training & inference CLI

╭─ Commands ──────────────────────────────────────────────────────────────────╮
│ prepare-dataset  Split raw images + labels into train / val / test.         │
│ train            Train a YOLO model on the dataset.                         │
│ test             Run inference or evaluation on the test partition.          │
│ list-datasets    Show dataset partitions and image counts.                  │
│ info             Display available weights and dataset summary.              │
╰─────────────────────────────────────────────────────────────────────────────╯
```

---

## 4. GPU Setup (optional but recommended)

Ultralytics automatically uses the first available CUDA device.
To force CPU:

```bash
python donanet.py train --device cpu ...
```

To use a specific GPU:

```bash
python donanet.py train --device 0 ...
```

---

## 5. Directory Layout After Installation

```
donanet/
├── donanet.py          ← CLI entry point
├── pyproject.toml
├── dataset/
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── weights/            ← best.pt / last.pt stored here after training
├── runs/               ← Ultralytics training artefacts
└── docs/
```
