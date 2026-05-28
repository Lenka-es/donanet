# <img src="img/wildIntel_logo.webp" alt="Wildintel Logo" height="60">  DonaNet


![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-GPLv3-blue.svg)
[![WildINTEL](https://img.shields.io/badge/WildINTEL-v1.0-blue)](https://wildintel.eu/)
[![Ultralytics YOLO](https://img.shields.io/badge/Ultralytics-YOLO-00ADEF)](https://docs.ultralytics.com/)
[![Typer](https://img.shields.io/badge/Typer-CLI-009688)](https://typer.tiangolo.com/)
[![uv](https://img.shields.io/badge/uv-package_manager-5C4EE5)](https://github.com/astral-sh/uv)

<hr>

## Neural network for detecting and classifying mammals in Doñana National Park

**DonaNet** is a neural network designed to detect and classify the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (Spain)**,  located in the Mediterranean biogeographical region, within the WildINTEL project, funded by Biodiversa+**. 

This repository contains the **weights file** produced by a training process of a
[YOLO](https://docs.ultralytics.com/) network specifically adapted to Doñana National Park.
The dataset used during this training process is available at XXXXX.

The repository also includes the **`donanet.py`** application, which allows retraining this network
with new image datasets.

## 📚 Documentation

Full documentation — installation guide, user guide and administrator guide — is available under
the `docs/:

- Installation guide: `docs/installation-guide.md`
- User guide: `docs/user-guide.md`
- Administrator guide: `docs/admin-guide.md`

You can also access the online documentation at https://wildintelproject.github.io/donanet

## 🚀 Features

- Interactive CLI powered by [Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/)
- Automatic dataset splitting into `train / val / test` partitions with configurable ratios
- Fine-tune any Ultralytics YOLO model and save weights under `weights/`
- Run inference or evaluation on the test partition with a chosen weight file
- Inspect dataset statistics and available weights at a glance
- Designed for seamless integration with the WildINTEL image storage

## 📋 Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) (for dependency management and packaging)
- CUDA-capable GPU recommended for training (CPU fallback supported)

## 🧭 Overview

`donanet.py` exposes a set of Typer sub-commands that drive Ultralytics YOLO:

| Command            | Description                                            |
|--------------------|--------------------------------------------------------|
| `prepare-dataset`  | Split images + labels into `train / val / test`        |
| `train`            | Fine-tune a YOLO model and write weights to `weights/` |
| `test`             | Run inference or evaluation on the test partition      |
| `list-datasets`    | Show dataset partitions and their image counts         |
| `info`             | Show available weights and dataset summary             |

### Dataset layout

```
dataset/
├── train/
│   ├── images/   ← training images (.jpg / .png)
│   └── labels/   ← YOLO-format .txt annotations
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

Each label file follows the YOLO format — one row per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalised to `[0, 1]` relative to the image dimensions.

---

## Quick start

### 0. Get the code

```bash
git clone <your-repo-url>
cd donanet
```

### 1. Install dependencies

```bash
uv sync          # or: pip install -e .
```

### 2. Prepare your dataset

Place raw images inside a source directory, then split them:

```bash
python donanet.py prepare-dataset --source /path/to/raw --train 0.7 --val 0.2 --test 0.1
```

### 3. Train

```bash
python donanet.py train --model yolov8n.pt --epochs 50 --name my_run
```

Weights are saved to `weights/my_run/best.pt` and `weights/my_run/last.pt`.

### 4. Test

```bash
python donanet.py test --weights weights/my_run/best.pt --conf 0.25
```

### 5. Inspect

```bash
python donanet.py info
python donanet.py list-datasets
```

---

## Main Commands

```bash
python donanet.py prepare-dataset --source /data/raw --train 0.7 --val 0.2 --test 0.1
python donanet.py train --model yolov8n.pt --epochs 100 --name wildlife_v1
python donanet.py train --model yolov8s.pt --epochs 100 --resume --name wildlife_v1
python donanet.py test --weights weights/wildlife_v1/best.pt
python donanet.py test --weights weights/wildlife_v1/best.pt --save-images
python donanet.py list-datasets
python donanet.py info
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the GNU General Public License v3.0 or later - see the [LICENSE](LICENSE) file for details.

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

## 🏛️ Funding

This work is part of the [WildINTEL project](https://wildintel.eu/), funded by the [Biodiversa+](https://www.biodiversa.eu/) Joint Research Call 2022-2023 "Improved
transnational monitoring of biodiversity and ecosystem change for science and society (BiodivMon)". Biodiversa+ is the
European co-funded biodiversity partnership supporting excellent research on biodiversity with an impact for policy and
society. Biodiversa+ is part of the European Biodiversity Strategy for 2030 that aims to put Europe's biodiversity on a
path to recovery by 2030 and is co-funded by the European Commission.

**WildINTEL has been co-funded by the [European Commission](https://commission.europa.eu/) (GA No. 101052342) and the following funding organisations: [Agencia Estatal de Investigación](https://www.aei.gob.es/) (Spain, PCI2023-145963-2, PCI2024-153489), [National Science Centre](https://www.ncn.gov.pl/?language=en) (Poland, UMO-2023/05/Y/NZ8/00104), the [Research Council of Norway](https://www.forskningsradet.no/en/) (Norway, NFR350962) and the [German Research Foundation](https://www.dfg.de/en/) (Germany).**
