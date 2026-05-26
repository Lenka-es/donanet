# DonaNet Documentation

Welcome to the DonaNet documentation.

**DonaNet** is a neural network designed to detect and classify the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (Spain).

This project provides the **pre-trained weights** of a [YOLO](https://docs.ultralytics.com/) network
specifically adapted to Doñana, as well as the **`donanet.py`** CLI, which allows you to retrain
the network with new image datasets.

![WildINTEL](img/wildIntel_logo.webp)

---

## Documentation Map

- [DonaNet Model](donanet-model.md)
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
