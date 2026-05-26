#!/usr/bin/env python3
"""DonaNet — WildINTEL YOLO training & inference CLI."""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.resolve()
DATASET_DIR = Path(ROOT, "dataset")
WEIGHTS_DIR = Path(ROOT, "weights")
RUNS_DIR = Path(ROOT, "runs")
DATASET_YAML = ROOT / "dataset.yaml"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

PARTITIONS = ("train", "val", "test")

console = Console()

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="donanet",
    help="DonaNet — WildINTEL YOLO training & inference CLI",
    add_completion=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dirs() -> None:
    """Create the standard dataset / weights / runs directories if missing."""
    for partition in PARTITIONS:
        for sub in ("images", "labels"):
            (DATASET_DIR / partition / sub).mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def _count_partition(partition: str) -> tuple[int, int]:
    """Return (image_count, label_count) for a partition."""
    img_dir = DATASET_DIR / partition / "images"
    lbl_dir = DATASET_DIR / partition / "labels"
    imgs = [f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS] if img_dir.exists() else []
    lbls = [f for f in lbl_dir.iterdir() if f.suffix == ".txt"] if lbl_dir.exists() else []
    return len(imgs), len(lbls)


def _write_dataset_yaml(class_names: list[str]) -> None:
    """Write (or overwrite) dataset.yaml at the project root."""
    data = {
        "path": str(ROOT),
        "train": "dataset/train/images",
        "val": "dataset/val/images",
        "test": "dataset/test/images",
        "nc": len(class_names),
        "names": {i: name for i, name in enumerate(class_names)},
    }
    with DATASET_YAML.open("w") as fh:
        yaml.dump(data, fh, default_flow_style=False, allow_unicode=True)
    rprint(f"[green]✔[/green] dataset.yaml written → [bold]{DATASET_YAML}[/bold]")


def _copy_weights(run_dir: Path, run_name: str) -> None:
    """Copy best.pt / last.pt from a Ultralytics run dir into weights/<name>/."""
    dest = WEIGHTS_DIR / run_name
    dest.mkdir(parents=True, exist_ok=True)
    copied = False
    for fname in ("best.pt", "last.pt"):
        src = run_dir / "weights" / fname
        if src.exists():
            shutil.copy2(src, dest / fname)
            rprint(f"[green]✔[/green] {fname} → [bold]{dest / fname}[/bold]")
            copied = True
    if not copied:
        rprint(f"[yellow]⚠[/yellow] No weight files found in {run_dir / 'weights'}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command()
def prepare_dataset(
    source: Annotated[
        Path,
        typer.Option("--source", "-s", help="Directory with raw images (and label .txt files beside them).", show_default=False),
    ],
    train_ratio: Annotated[
        float,
        typer.Option("--train", "-tr", help="Fraction of images for training.", min=0.0, max=1.0),
    ] = 0.7,
    val_ratio: Annotated[
        float,
        typer.Option("--val", "-v", help="Fraction of images for validation.", min=0.0, max=1.0),
    ] = 0.2,
    test_ratio: Annotated[
        float,
        typer.Option("--test", "-te", help="Fraction of images for testing.", min=0.0, max=1.0),
    ] = 0.1,
    seed: Annotated[int, typer.Option("--seed", help="Random seed for reproducibility.")] = 42,
    move: Annotated[bool, typer.Option("--move", help="Move files instead of copying.")] = False,
) -> None:
    """Split raw images + labels into train / val / test partitions."""

    # --- validate ratios ---
    total = round(train_ratio + val_ratio + test_ratio, 6)
    if abs(total - 1.0) > 1e-4:
        rprint(f"[red]✗[/red] Ratios must sum to 1.0 — got {total:.4f}")
        raise typer.Exit(code=1)

    if not source.exists() or not source.is_dir():
        rprint(f"[red]✗[/red] Source directory not found: {source}")
        raise typer.Exit(code=1)

    images = sorted(f for f in source.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        rprint(f"[yellow]⚠[/yellow] No images found in {source}")
        raise typer.Exit(code=0)

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # the rest go to test (avoids rounding drift)
    splits: dict[str, list[Path]] = {
        "train": images[:n_train],
        "val": images[n_train : n_train + n_val],
        "test": images[n_train + n_val :],
    }

    _ensure_dirs()
    op = shutil.move if move else shutil.copy2
    op_label = "Moved" if move else "Copied"

    rprint(Panel.fit(f"[bold]Splitting {n} images[/bold]  seed={seed}", title="prepare-dataset"))

    for partition, files in splits.items():
        img_dst = DATASET_DIR / partition / "images"
        lbl_dst = DATASET_DIR / partition / "labels"
        count_img = count_lbl = 0
        for img_path in files:
            shutil.copy2(img_path, img_dst / img_path.name) if not move else shutil.move(str(img_path), img_dst / img_path.name)
            count_img += 1
            label_src = img_path.with_suffix(".txt")
            if label_src.exists():
                shutil.copy2(label_src, lbl_dst / label_src.name) if not move else shutil.move(str(label_src), lbl_dst / label_src.name)
                count_lbl += 1
        rprint(
            f"  [cyan]{partition:5s}[/cyan]  {op_label} {count_img} images"
            + (f", {count_lbl} labels" if count_lbl else " (no labels found)")
        )

    rprint("[green]✔[/green] Dataset split complete.")


@app.command()
def train(
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Base YOLO model (e.g. yolov8n.pt, yolov8s.pt) or path to a .pt file."),
    ] = "yolov8n.pt",
    epochs: Annotated[int, typer.Option("--epochs", "-e", help="Number of training epochs.", min=1)] = 100,
    imgsz: Annotated[int, typer.Option("--imgsz", help="Input image size (pixels).", min=32)] = 640,
    batch: Annotated[int, typer.Option("--batch", "-b", help="Batch size. -1 = auto.")] = 16,
    name: Annotated[str, typer.Option("--name", "-n", help="Run name. Weights stored in weights/<name>/")] = "run",
    device: Annotated[
        Optional[str],
        typer.Option("--device", "-d", help="Device: cpu, 0, 0,1, … (default: auto)"),
    ] = None,
    patience: Annotated[int, typer.Option("--patience", help="Early-stopping patience (epochs).", min=1)] = 50,
    resume: Annotated[bool, typer.Option("--resume", help="Resume from the last checkpoint.")] = False,
    class_names: Annotated[
        Optional[str],
        typer.Option("--classes", help='Comma-separated class names, e.g. "fox,deer,boar". Required if dataset.yaml is missing.'),
    ] = None,
) -> None:
    """Train a YOLO model on the dataset and save weights to weights/<name>/."""

    try:
        from ultralytics import YOLO
    except ImportError:
        rprint("[red]✗[/red] ultralytics is not installed. Run: pip install ultralytics")
        raise typer.Exit(code=1)

    # --- ensure dataset.yaml exists ---
    if not DATASET_YAML.exists():
        if class_names is None:
            rprint(
                "[yellow]⚠[/yellow] dataset.yaml not found.\n"
                "       Pass [bold]--classes[/bold] to generate it automatically.\n"
                "       Example: [italic]--classes fox,deer,boar[/italic]"
            )
            raise typer.Exit(code=1)
        _write_dataset_yaml([c.strip() for c in class_names.split(",")])

    # --- sanity check: are there training images? ---
    n_train, _ = _count_partition("train")
    if n_train == 0:
        rprint("[yellow]⚠[/yellow] No images found in dataset/train/images. Run [bold]prepare-dataset[/bold] first.")
        raise typer.Exit(code=1)

    rprint(
        Panel.fit(
            f"[bold]model:[/bold] {model}  [bold]epochs:[/bold] {epochs}  "
            f"[bold]batch:[/bold] {batch}  [bold]imgsz:[/bold] {imgsz}  "
            f"[bold]name:[/bold] {name}",
            title="[green]train[/green]",
        )
    )

    yolo = YOLO(model)

    extra: dict = {}
    if device is not None:
        extra["device"] = device

    results = yolo.train(
        data=str(DATASET_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=str(RUNS_DIR),
        patience=patience,
        resume=resume,
        **extra,
    )

    # --- copy weights ---
    run_dir = RUNS_DIR / name
    if run_dir.exists():
        _copy_weights(run_dir, name)
    else:
        # Ultralytics may append a numeric suffix on collisions
        candidates = sorted(RUNS_DIR.glob(f"{name}*/"))
        if candidates:
            _copy_weights(candidates[-1], name)

    rprint(f"[green]✔[/green] Training complete. Results in [bold]{RUNS_DIR / name}[/bold]")


@app.command()
def test(
    weights: Annotated[
        Path,
        typer.Option("--weights", "-w", help="Path to the .pt weight file.", show_default=False),
    ],
    source: Annotated[
        Optional[Path],
        typer.Option("--source", "-s", help="Images / directory to run inference on (default: dataset/test/images)."),
    ] = None,
    conf: Annotated[float, typer.Option("--conf", help="Confidence threshold.", min=0.0, max=1.0)] = 0.25,
    iou: Annotated[float, typer.Option("--iou", help="IoU threshold for NMS.", min=0.0, max=1.0)] = 0.45,
    imgsz: Annotated[int, typer.Option("--imgsz", help="Input image size.", min=32)] = 640,
    device: Annotated[Optional[str], typer.Option("--device", "-d", help="Device: cpu, 0, …")] = None,
    save_images: Annotated[bool, typer.Option("--save-images", help="Save annotated output images.")] = False,
    save_txt: Annotated[bool, typer.Option("--save-txt", help="Save predictions as YOLO .txt files.")] = False,
) -> None:
    """Run inference or evaluation on the test partition (or a custom source)."""

    try:
        from ultralytics import YOLO
    except ImportError:
        rprint("[red]✗[/red] ultralytics is not installed. Run: pip install ultralytics")
        raise typer.Exit(code=1)

    if not weights.exists():
        rprint(f"[red]✗[/red] Weights file not found: {weights}")
        raise typer.Exit(code=1)

    if source is None:
        source = DATASET_DIR / "test" / "images"

    if not source.exists():
        rprint(f"[red]✗[/red] Source not found: {source}")
        raise typer.Exit(code=1)

    rprint(
        Panel.fit(
            f"[bold]weights:[/bold] {weights}\n"
            f"[bold]source:[/bold]  {source}\n"
            f"[bold]conf:[/bold]    {conf}   [bold]iou:[/bold] {iou}",
            title="[cyan]test[/cyan]",
        )
    )

    yolo = YOLO(str(weights))

    extra: dict = {}
    if device is not None:
        extra["device"] = device

    yolo.predict(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        save=save_images,
        save_txt=save_txt,
        project=str(RUNS_DIR),
        name="predict",
        **extra,
    )

    rprint(f"[green]✔[/green] Inference complete.  Results in [bold]{RUNS_DIR / 'predict'}[/bold]")


@app.command()
def list_datasets() -> None:
    """Show dataset partitions and their image / label counts."""

    table = Table(title="Dataset partitions", show_header=True, header_style="bold cyan")
    table.add_column("Partition", style="cyan", width=12)
    table.add_column("Images", justify="right")
    table.add_column("Labels", justify="right")
    table.add_column("Paired", justify="right")

    for partition in PARTITIONS:
        n_img, n_lbl = _count_partition(partition)
        paired = min(n_img, n_lbl)
        style = "green" if n_img > 0 else "dim"
        table.add_row(partition, str(n_img), str(n_lbl), str(paired), style=style)

    console.print(table)


@app.command()
def info() -> None:
    """Display available weights and a dataset summary."""

    rprint(Panel.fit("[bold white]DonaNet[/bold white] — WildINTEL YOLO CLI", subtitle=f"root: {ROOT}"))

    # --- dataset ---
    rprint("\n[bold cyan]Dataset[/bold cyan]")
    list_datasets()

    # --- dataset.yaml ---
    if DATASET_YAML.exists():
        with DATASET_YAML.open() as fh:
            data = yaml.safe_load(fh)
        nc = data.get("nc", "?")
        names = data.get("names", {})
        rprint(f"\n[bold cyan]dataset.yaml[/bold cyan]  classes={nc}")
        for idx, cls_name in (names.items() if isinstance(names, dict) else enumerate(names)):
            rprint(f"  [dim]{idx}[/dim] {cls_name}")
    else:
        rprint("\n[yellow]⚠[/yellow] dataset.yaml not found (run [bold]train --classes ...[/bold] to generate it)")

    # --- weights ---
    rprint("\n[bold cyan]Weights[/bold cyan]")
    if WEIGHTS_DIR.exists():
        weight_files = sorted(WEIGHTS_DIR.rglob("*.pt"))
        if weight_files:
            wt = Table(show_header=True, header_style="bold")
            wt.add_column("File", style="green")
            wt.add_column("Size", justify="right")
            for wf in weight_files:
                size_mb = wf.stat().st_size / (1024 * 1024)
                wt.add_row(str(wf.relative_to(ROOT)), f"{size_mb:.1f} MB")
            console.print(wt)
        else:
            rprint("  [dim]No weights found. Run [bold]train[/bold] first.[/dim]")
    else:
        rprint("  [dim]weights/ directory does not exist.[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _ensure_dirs()
    app()
