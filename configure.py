#!/usr/bin/env python3
"""
DonaNet — Docker Stack Configurator

Interactive configuration tool for running DonaNet (YOLO wildlife detector)
inside a Docker container. Generates docker-compose and environment files
based on user input.

Usage:
    python configure.py [OPTIONS]
    ./configure.py [OPTIONS]

Requirements (install via setup.sh):
    uv sync --group configure
"""

import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader

TEMPLATES = Path("templates")

env = Environment(loader=FileSystemLoader(TEMPLATES))


def render(name: str, dest: Path, ctx: dict) -> None:
    tpl = env.get_template(name)
    with open(dest, "w") as f:
        f.write(tpl.render(**ctx))


try:
    import typer
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
except ImportError:
    print("Required packages not found. Run: ./setup.sh")
    sys.exit(1)

from core import system

BANNER = """
╔══════════════════════════════════════════════════════════════════╗

  ██████╗  ██████╗ ███╗   ██╗ █████╗ ███╗   ██╗███████╗████████╗
  ██╔══██╗██╔═══██╗████╗  ██║██╔══██╗████╗  ██║██╔════╝╚══██╔══╝
  ██║  ██║██║   ██║██╔██╗ ██║███████║██╔██╗ ██║█████╗     ██║
  ██║  ██║██║   ██║██║╚██╗██║██╔══██║██║╚██╗██║██╔══╝     ██║
  ██████╔╝╚██████╔╝██║ ╚████║██║  ██║██║ ╚████║███████╗   ██║
  ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝   ╚═╝

    DONANET — Docker Configuration Wizard
    Copyright (c) WildINTEL Project 2026.

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    YOLO neural network for wildlife detection in Doñana N.P.
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Project: https://wildintel.eu/
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

╚══════════════════════════════════════════════════════════════════╝
"""

app = typer.Typer(
    name="configure",
    help="DonaNet Docker stack configuration wizard.",
    rich_markup_mode="rich",
)

console = Console()

SCRIPT_DIR = Path(__file__).parent.resolve()
PROFILES_DIR = SCRIPT_DIR / "profiles"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_memory_mb(value: str) -> int:
    """Parse memory string into integer MB: '4G', '4096M', '4096MB', '4096'."""
    s = value.strip().upper().replace(" ", "")
    if s.endswith("GB"):
        return int(s[:-2]) * 1024
    if s.endswith("MB"):
        return int(s[:-2])
    if s.endswith("G"):
        return int(s[:-1]) * 1024
    if s.endswith("M"):
        return int(s[:-1])
    return int(s)


def load_env_file(env_path: Path) -> dict[str, str]:
    """Load a .env file and return key/value pairs."""
    env_vars: dict[str, str] = {}
    if not env_path.exists():
        return env_vars
    with open(env_path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                env_vars[key.strip()] = value.strip()
    return env_vars


def print_banner() -> None:
    console.print(Panel(BANNER, style="blue"))


def print_section(title: str) -> None:
    console.print()
    console.print(f"[green]{'━' * 67}[/green]")
    console.print(f"[green]  {title}[/green]")
    console.print(f"[green]{'━' * 67}[/green]")


def get_existing_profiles() -> list[str]:
    if not PROFILES_DIR.exists():
        return []
    return sorted(
        [d.name for d in PROFILES_DIR.iterdir() if d.is_dir() and (d / ".env").exists()]
    )


def validate_profile_name(name: str) -> bool:
    return bool(re.match(r"^[a-zA-Z0-9_-]+$", name))


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

class Configuration:
    """Configuration data container for the DonaNet Docker stack."""

    ENV_TO_ATTR_MAP: dict[str, str] = {
        "TZ":           "tz",
        "DATASET_PATH": "dataset_path",
        "WEIGHTS_PATH": "weights_path",
        "RUNS_PATH":    "runs_path",
        "MEM_LIMIT":    "mem_limit",
        "CPU_LIMIT":    "cpu_limit",
    }

    def __init__(self) -> None:
        self.profile_name: str = "production"

        # Paths
        self.dataset_path: str = str(SCRIPT_DIR / "dataset")
        self.weights_path: str = str(SCRIPT_DIR / "weights")
        self.runs_path: str    = str(SCRIPT_DIR / "runs")

        # Runtime
        self.tz: str = system.get_system_timezone()

        # GPU
        self.gpu: str = "n"

        # Resources
        self.mem_limit: str = "8192M"
        self.cpu_limit: str = "4.0"

    def get_profile_dir(self) -> Path:
        return PROFILES_DIR / self.profile_name


def load_config_from_env(env_path: Path) -> "Configuration":
    """Load configuration from an existing .env file."""
    env_vars = load_env_file(env_path)
    config = Configuration()
    for env_name, attr_name in Configuration.ENV_TO_ATTR_MAP.items():
        if env_name in env_vars:
            setattr(config, attr_name, env_vars[env_name])
    if "GPU" in env_vars:
        config.gpu = env_vars["GPU"]
    return config


# ---------------------------------------------------------------------------
# Interactive steps
# ---------------------------------------------------------------------------

def configure_profile_name(config: Configuration) -> None:
    print_section("Profile Configuration")

    console.print()
    console.print("[blue]i[/blue] Profiles allow multiple independent deployments")
    console.print("    (e.g. cpu-dev, gpu-production). Each profile lives under")
    console.print("    ./profiles/<name>/ and is fully self-contained.")
    console.print()

    existing = get_existing_profiles()
    if existing:
        console.print("Existing profiles:")
        for p in existing:
            console.print(f"  * {p}")
        console.print()

    while True:
        name = Prompt.ask("Profile name", default=config.profile_name)
        if not validate_profile_name(name):
            console.print("[red]x[/red] Use only letters, numbers, hyphens and underscores.")
            continue
        profile_dir = PROFILES_DIR / name
        if profile_dir.exists() and not (profile_dir / ".env").exists() is False:
            overwrite = Confirm.ask(f"Profile '{name}' already exists. Overwrite?", default=False)
            if not overwrite:
                continue
        config.profile_name = name
        break

    console.print(f"[green]v[/green] Profile: {config.profile_name}")
    console.print(f"    Directory: ./profiles/{config.profile_name}/")


def configure_storage(config: Configuration) -> None:
    print_section("Storage Configuration")

    console.print()
    console.print("[blue]i[/blue] These host paths are mounted inside the container.")
    console.print("    Use absolute paths or paths relative to this directory.")
    console.print()

    config.dataset_path = Prompt.ask(
        "Dataset path on host  (mounted at /app/dataset)",
        default=config.dataset_path,
    )
    config.weights_path = Prompt.ask(
        "Weights path on host  (mounted at /app/weights)",
        default=config.weights_path,
    )
    config.runs_path = Prompt.ask(
        "Runs path on host     (mounted at /app/runs)",
        default=config.runs_path,
    )

    for label, path in [
        ("dataset", config.dataset_path),
        ("weights", config.weights_path),
        ("runs",    config.runs_path),
    ]:
        p = Path(path)
        if not p.exists():
            console.print(f"[yellow]![/yellow] {label} path does not exist yet — it will be created")
            p.mkdir(parents=True, exist_ok=True)
        console.print(f"[green]v[/green] {label}: {path}")


def configure_gpu(config: Configuration) -> None:
    print_section("GPU Configuration")

    console.print()
    console.print("[blue]i[/blue] GPU acceleration is strongly recommended for training.")
    console.print("    Requires NVIDIA Container Toolkit (nvidia-ctk) installed on the host.")
    console.print("    CPU-only mode is supported for inference and dataset preparation.")
    console.print()

    default_gpu = config.gpu.lower() == "y"
    gpu = Confirm.ask("Enable NVIDIA GPU support?", default=default_gpu)
    config.gpu = "y" if gpu else "n"

    if config.gpu == "y":
        console.print("[green]v[/green] GPU overlay enabled — docker-compose.gpu.yml will be generated.")
        console.print(
            "[blue]i[/blue] Verify NVIDIA driver: [cyan]nvidia-smi[/cyan]  "
            "and toolkit: [cyan]nvidia-ctk --version[/cyan]"
        )
    else:
        console.print("[blue]i[/blue] GPU disabled — running in CPU-only mode.")


def configure_resources(config: Configuration) -> None:
    print_section("Resource Limits")

    total_mb  = system.get_system_total_memory_mb()
    total_cpu = system.get_system_total_cpus()

    console.print()
    console.print(
        f"[blue]i[/blue] System: [cyan]{total_mb} MB[/cyan] RAM, "
        f"[cyan]{total_cpu}[/cyan] logical CPU cores."
    )
    console.print("[blue]i[/blue] Training is memory-intensive — allocate at least 8 GB for YOLO.")
    console.print()

    default_mem = f"{max(4096, int(total_mb * 0.75))}M"
    mem_input = Prompt.ask("Memory limit", default=default_mem)
    config.mem_limit = f"{parse_memory_mb(mem_input)}M"

    default_cpu = str(round(max(2.0, total_cpu * 0.75), 2))
    config.cpu_limit = Prompt.ask("CPU limit (cores)", default=default_cpu)

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Service")
    table.add_column("Memory", justify="right")
    table.add_column("CPU", justify="right")
    table.add_row("donanet", config.mem_limit, config.cpu_limit)
    console.print(table)


def configure_timezone(config: Configuration) -> None:
    config.tz = Prompt.ask("Timezone", default=config.tz)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_env_file(config: Configuration) -> Path:
    print_section("Generating environment file")

    profile_dir = config.get_profile_dir()
    profile_dir.mkdir(parents=True, exist_ok=True)

    env_file = profile_dir / ".env"
    if env_file.exists():
        backup = env_file.with_suffix(
            f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        env_file.rename(backup)
        console.print(f"[blue]i[/blue] Previous .env backed up → {backup.name}")

    render("env_template.j2", env_file, {"config": config})
    console.print(f"[green]v[/green] {env_file.relative_to(SCRIPT_DIR)}")
    return env_file


def generate_docker_compose(config: Configuration) -> None:
    print_section("Generating docker-compose files")

    profile_dir = config.get_profile_dir()
    ctx = {
        "config":      config,
        "profile_dir": str(profile_dir),
        "project_dir": str(SCRIPT_DIR),
    }

    dest = profile_dir / "docker-compose.yml"
    render("docker-compose.yml.j2", dest, ctx)
    console.print(f"[green]v[/green] {dest.relative_to(SCRIPT_DIR)}")

    if config.gpu == "y":
        dest = profile_dir / "docker-compose.gpu.yml"
        render("docker-compose.gpu.yml.j2", dest, ctx)
        console.print(f"[green]v[/green] {dest.relative_to(SCRIPT_DIR)}  (GPU overlay)")


def generate_helper_scripts(config: Configuration) -> None:
    print_section("Generating helper scripts")

    profile_dir = config.get_profile_dir()
    project_name = f"donanet-{config.profile_name}"

    compose_files = [f"-f {profile_dir}/docker-compose.yml"]
    if config.gpu == "y":
        compose_files.append(f"-f {profile_dir}/docker-compose.gpu.yml")

    compose_cmd = (
        f"docker compose --project-name {project_name} "
        f"{' '.join(compose_files)} "
        f"--env-file {profile_dir}/.env"
    )

    ctx = {
        "profile_dir":  str(profile_dir),
        "profile_name": config.profile_name,
        "compose_cmd":  compose_cmd,
        "banner":       BANNER.strip("\n"),
        "config":       config,
    }

    scripts = ["build.sh", "train.sh", "test.sh", "prepare.sh", "info.sh", "exec.sh", "ps.sh"]

    for script in scripts:
        dest = profile_dir / script
        render(f"{script}.j2", dest, ctx)
        dest.chmod(0o755)

    console.print(
        f"[green]v[/green] Scripts in ./profiles/{config.profile_name}/: "
        + ", ".join(scripts)
    )
    console.print(f"[blue]i[/blue] Compose command: {compose_cmd}")


def print_summary(config: Configuration) -> None:
    print_section("Configuration Summary")

    table = Table(show_header=False, box=None)
    table.add_column("Setting", style="cyan")
    table.add_column("Value",   style="white")

    table.add_row("Profile Name",      config.profile_name)
    table.add_row("Profile Directory", f"./profiles/{config.profile_name}/")
    table.add_row("", "")
    table.add_row("Timezone",     config.tz)
    table.add_row("GPU",          "enabled" if config.gpu == "y" else "disabled (CPU only)")
    table.add_row("", "")
    table.add_row("Dataset path", config.dataset_path)
    table.add_row("Weights path", config.weights_path)
    table.add_row("Runs path",    config.runs_path)
    table.add_row("", "")
    table.add_row("Memory limit", config.mem_limit)
    table.add_row("CPU limit",    config.cpu_limit)

    console.print(table)

    profile_dir = f"./profiles/{config.profile_name}"
    console.print()
    console.print("[blue]Build the Docker image:[/blue]")
    console.print(f"  {profile_dir}/build.sh")
    console.print()
    console.print("[blue]Run training:[/blue]")
    console.print(f"  {profile_dir}/train.sh --model yolov8n.pt --epochs 100 --name my_run")
    console.print()
    console.print("[blue]Run inference:[/blue]")
    console.print(f"  {profile_dir}/test.sh --weights weights/my_run/best.pt")
    console.print()
    console.print("[blue]Prepare dataset:[/blue]")
    console.print(f"  {profile_dir}/prepare.sh --source /path/to/raw --train 0.7 --val 0.2 --test 0.1")
    console.print()
    console.print("[green]v[/green] Configuration complete!")


# ---------------------------------------------------------------------------
# Typer commands
# ---------------------------------------------------------------------------

@app.command(rich_help_panel="Main Commands")
def interactive(
    from_profile: Optional[str] = typer.Option(
        None,
        "--from-profile",
        help="Name of existing profile to use as base configuration.",
    ),
):
    """Run the interactive configuration wizard.

    Examples:

        # Start fresh
        python configure.py interactive

        # Edit an existing profile
        python configure.py interactive --from-profile production
    """
    print_banner()

    config = Configuration()

    if from_profile:
        base_env = PROFILES_DIR / from_profile / ".env"
        if not base_env.exists():
            console.print(f"[red]x[/red] Profile '{from_profile}' not found or has no .env file")
            raise typer.Exit(1)
        console.print(f"[blue]i[/blue] Loading base configuration from: {base_env}")
        config = load_config_from_env(base_env)
        console.print("[green]v[/green] Base configuration loaded")
        console.print()

    configure_profile_name(config)

    if not from_profile:
        auto_env = config.get_profile_dir() / ".env"
        if auto_env.exists():
            saved_name = config.profile_name
            config = load_config_from_env(auto_env)
            config.profile_name = saved_name
            console.print(
                f"[blue]i[/blue] Loaded existing configuration from "
                f"profiles/{saved_name}/.env"
            )
            console.print()

    configure_timezone(config)
    configure_storage(config)
    configure_gpu(config)
    configure_resources(config)
    generate_env_file(config)
    generate_docker_compose(config)
    generate_helper_scripts(config)
    print_summary(config)


@app.command(rich_help_panel="Main Commands")
def quick(
    profile: str = typer.Option("production", "--profile", "-p", help="Profile name."),
    gpu: bool    = typer.Option(False,         "--gpu",           help="Enable GPU support."),
):
    """Quick non-interactive configuration using sensible defaults.

    Example:

        python configure.py quick --profile production --gpu
    """
    print_banner()

    if not validate_profile_name(profile):
        console.print(f"[red]x[/red] Invalid profile name: {profile}")
        raise typer.Exit(1)

    config = Configuration()
    config.profile_name = profile
    config.gpu = "y" if gpu else "n"

    total_mb  = system.get_system_total_memory_mb()
    total_cpu = system.get_system_total_cpus()
    config.mem_limit = f"{max(4096, int(total_mb * 0.75))}M"
    config.cpu_limit = str(round(max(2.0, total_cpu * 0.75), 2))

    generate_env_file(config)
    generate_docker_compose(config)
    generate_helper_scripts(config)
    print_summary(config)


@app.command(name="list-profiles", rich_help_panel="Utility Commands")
def list_profiles() -> None:
    """List all existing configuration profiles."""
    print_banner()
    print_section("Available Profiles")

    profiles = get_existing_profiles()

    if not profiles:
        console.print()
        console.print("[yellow]No profiles found.[/yellow]")
        console.print()
        console.print("Create a new profile with:")
        console.print("  python configure.py interactive")
        return

    console.print()
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Profile Name")
    table.add_column("GPU")
    table.add_column("Memory")
    table.add_column("Directory")

    for p in profiles:
        ev = load_env_file(PROFILES_DIR / p / ".env")
        table.add_row(
            p,
            "yes" if ev.get("GPU", "n") == "y" else "no",
            ev.get("MEM_LIMIT", "?"),
            f"./profiles/{p}/",
        )

    console.print(table)


@app.command(name="delete-profile", rich_help_panel="Utility Commands")
def delete_profile(
    name:  str  = typer.Argument(...,                  help="Profile to delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete an existing configuration profile."""
    if not validate_profile_name(name):
        console.print(f"[red]x[/red] Invalid profile name: {name}")
        raise typer.Exit(1)

    profile_dir = PROFILES_DIR / name
    if not profile_dir.exists():
        console.print(f"[red]x[/red] Profile '{name}' does not exist")
        raise typer.Exit(1)

    if not force:
        console.print(f"[yellow]![/yellow] This will delete profile '{name}' and all its files:")
        console.print(f"    {profile_dir}")
        console.print()
        if not Confirm.ask("Are you sure?", default=False):
            console.print("[blue]i[/blue] Cancelled")
            return

    import shutil
    shutil.rmtree(profile_dir)
    console.print(f"[green]v[/green] Profile '{name}' deleted")


@app.command(rich_help_panel="Utility Commands")
def check() -> None:
    """Verify Docker and system prerequisites."""
    import shutil
    print_banner()
    print_section("System Check")

    mem_mb  = system.get_system_total_memory_mb()
    cpus    = system.get_system_total_cpus()
    docker  = shutil.which("docker") is not None

    console.print(f"  RAM:    {mem_mb} MB")
    console.print(f"  CPUs:   {cpus}")
    console.print(f"  Docker: {'found' if docker else 'NOT FOUND'}")

    if not docker:
        console.print("[red]x[/red] Docker is required.")
        raise typer.Exit(1)

    try:
        subprocess.run(
            ["docker", "compose", "version"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        console.print("  Docker Compose v2: found")
    except Exception:
        console.print("[red]x[/red] Docker Compose v2 is required.")
        raise typer.Exit(1)

    nvidia = shutil.which("nvidia-smi") is not None
    console.print(f"  nvidia-smi: {'found' if nvidia else 'not found (GPU unavailable)'}")

    console.print("[green]v[/green] All required checks passed.")


if __name__ == "__main__":
    app()
