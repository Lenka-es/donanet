# DonaNet Documentation

![WildINTEL](img/wildIntel_logo.webp){ style="display: block; margin: 0 auto;" }

Welcome to the DonaNet documentation.

**DonaNet** is a neural network designed to detect and classify the mammals that inhabit
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/) (**SW**  Spain). **located in the Mediterranean biogeographical region**. **It has been developed within the framework of the WildINTEL project, funded by Biodiversa+ under the 2022-2023 BiodivMon joint call. It has been co-funded by the European Commission (GA No. 101052342) and the following funding organisations: Agencia Estatal de Investigación (Spain, PCI2023-145963-2, PCI2024-153489), National Science Centre (Poland, UMO-2023/05/Y/NZ8/00104), the Research Council of Norway (Norway, NFR350962) and the German Research Foundation (Germany).** 

This project provides the **pre-trained weights** of a [YOLO](https://docs.ultralytics.com/) network
specifically adapted to Doñana, as well as the **`donanet.py`** CLI, which allows you to retrain
the network with new image datasets.

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
