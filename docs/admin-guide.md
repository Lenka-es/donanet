# Administrator Guide

## Project Layout

```
donanet/
├── donanet.py              ← CLI entry point (Typer app)
├── pyproject.toml          ← project metadata & dependencies
├── mkdocs.yml              ← documentation site config
├── .gitignore
├── dataset/
│   ├── train/
│   │   ├── images/         ← training images
│   │   └── labels/         ← YOLO .txt annotations
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── weights/                ← saved weight files (best.pt, last.pt)
│   └── <run-name>/
├── runs/                   ← Ultralytics training artefacts
│   └── <run-name>/
│       ├── results.csv
│       ├── confusion_matrix.png
│       └── …
└── docs/
    ├── index.md
    ├── installation-guide.md
    ├── user-guide.md
    ├── admin-guide.md      ← this file
    ├── img/
    └── stylesheets/
```

---

## dataset.yaml

`donanet train` auto-generates a `dataset.yaml` at the project root before
handing off to Ultralytics.  Its format is:

```yaml
path: /absolute/path/to/donanet   # project root
train: dataset/train/images
val:   dataset/val/images
test:  dataset/test/images

nc: <number of classes>
names:
  0: <class_0_name>
  1: <class_1_name>
  …
```

You can edit this file manually before calling `train` if you need custom
class names or want to point at a different dataset location.

---

## Weights Storage

After a successful training run, DonaNet copies the best and last checkpoints
from the Ultralytics `runs/` directory into `weights/<run-name>/`:

| File | Contents |
|---|---|
| `best.pt` | Checkpoint with the lowest validation loss |
| `last.pt` | Checkpoint from the final epoch |

Both files are standard PyTorch state-dicts compatible with `ultralytics`.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `DONANET_DATASET_DIR` | `./dataset` | Override dataset root |
| `DONANET_WEIGHTS_DIR` | `./weights` | Override weights root |
| `DONANET_RUNS_DIR` | `./runs` | Override runs root |

---

## Adding New Commands

`donanet.py` uses a single `typer.Typer` application object named `app`.
Add new sub-commands with:

```python
@app.command()
def my_command(
    option: str = typer.Option("default", help="My option"),
):
    """Short description shown in --help."""
    ...
```

---

## Building Documentation

```bash
uv run mkdocs build     # static site → site/
uv run mkdocs serve     # live-reload dev server at http://127.0.0.1:8000
```
