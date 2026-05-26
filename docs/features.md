# Features

## Pre-trained YOLO network for Doñana

DonaNet ships with a **ready-to-use YOLO model** specifically trained on camera-trap imagery from
[Doñana National Park](https://www.miteco.gob.es/es/red-parques-nacionales/nuestros-parques/donana/).
No training required — download the weights and start detecting mammals immediately.

- Detects and classifies the mammal species present in Doñana National Park.
- Optimised for the lighting conditions, vegetation and camera angles typical of the park.
- Based on [Ultralytics YOLO](https://docs.ultralytics.com/), one of the most performant and
  widely adopted object-detection architectures.
- Weights are versioned and distributed alongside this repository so you always know exactly
  which model version produced a given result.

---

## Retraining on new datasets

The included **`donanet.py`** CLI makes it straightforward to fine-tune the network with your
own image datasets, adapting the model to new species, locations or camera setups.

- **Dataset preparation** — automatically splits a raw collection of images and YOLO-format labels
  into `train / val / test` partitions with configurable ratios.
- **Fine-tuning** — resume training from the provided weights or from any Ultralytics checkpoint,
  with full control over epochs, batch size and model variant.
- **Evaluation** — run inference or evaluation on the test partition and collect standard detection
  metrics (mAP, precision, recall).
- **Inspection** — list available weights and dataset statistics at a glance with `donanet info`
  and `donanet list-datasets`.

---

## Interactive CLI

All functionality is exposed through an interactive command-line interface built with
[Typer](https://typer.tiangolo.com/) and [Rich](https://rich.readthedocs.io/), providing
a clean experience with coloured output, progress bars and inline help.

```bash
# Run inference with the pre-trained weights
python donanet.py test --weights weights/donana/best.pt

# Fine-tune on a new dataset
python donanet.py train --model weights/donana/best.pt --epochs 50 --name my_run
```
