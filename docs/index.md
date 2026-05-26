# DonaNet Documentation

Welcome to the DonaNet documentation.

DonaNet provides a CLI (`donanet.py`) that drives Ultralytics YOLO for the full
training → evaluation → inference lifecycle on wildlife imagery.

![WildINTEL](img/wildIntel_logo.webp)

---

## Documentation Map

- [Installation Guide](installation-guide.md)
- [User Guide](user-guide.md)
- [Administrator Guide](admin-guide.md)

---

## Core Concepts

- **Dataset partitions** live under `dataset/train`, `dataset/val`, `dataset/test`.
- Each partition has `images/` and `labels/` sub-folders.
- Labels follow the **YOLO format**: `<class_id> <cx> <cy> <w> <h>` (normalised).
- Trained weights are saved under `weights/<run-name>/`.
- Training artefacts (metrics, plots) are saved under `runs/<run-name>/`.
