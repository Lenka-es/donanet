import subprocess
import psutil


def get_system_timezone() -> str:
    """Return the host timezone string (e.g. 'Europe/Madrid'), falling back to 'UTC'."""
    try:
        tz = subprocess.check_output(
            ["timedatectl", "show", "--property=Timezone", "--value"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        if tz:
            return tz
    except Exception:
        pass
    try:
        from pathlib import Path
        localtime = Path("/etc/localtime").resolve()
        zoneinfo = Path("/usr/share/zoneinfo")
        return str(localtime.relative_to(zoneinfo))
    except Exception:
        pass
    return "UTC"


def get_system_total_memory_mb() -> int:
    """Total installed RAM in MB."""
    return psutil.virtual_memory().total // 1024 // 1024


def get_system_total_cpus() -> int:
    """Number of logical CPU cores."""
    return psutil.cpu_count(logical=True) or 4
