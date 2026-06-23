from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Dict, Optional

import numpy as np


THREAD_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_PROC_BIND",
    "OMP_PLACES",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def _cpu_model() -> str:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as cpuinfo:
                for line in cpuinfo:
                    if line.lower().startswith(("model name", "hardware")):
                        return line.split(":", 1)[1].strip()
        except (OSError, IndexError):
            pass
    elif system == "Darwin":
        for key in ("machdep.cpu.brand_string", "hw.model"):
            try:
                value = subprocess.check_output(
                    ["sysctl", "-n", key], text=True, timeout=1
                ).strip()
                if value:
                    return value
            except (OSError, subprocess.SubprocessError):
                pass
    elif system == "Windows":
        value = os.environ.get("PROCESSOR_IDENTIFIER")
        if value:
            return value

    for value in (platform.processor(), platform.uname().processor):
        if value:
            return value.strip()

    return platform.machine() or "unknown"


def _physical_cpu_count() -> Optional[int]:
    try:
        import psutil

        return psutil.cpu_count(logical=False)
    except (ImportError, RuntimeError):
        return None


def _available_cpu_count() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def _total_memory_gib() -> Optional[float]:
    try:
        import psutil

        return psutil.virtual_memory().total / 1024**3
    except (ImportError, RuntimeError):
        pass

    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    except (AttributeError, OSError, ValueError):
        return None


def _package_version() -> str:
    try:
        return importlib.metadata.version("deal")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _native_runtime_info() -> Dict[str, object]:
    try:
        from .sgp import _C_deal_sgp
    except ImportError as error:
        return {
            "native_extension": f"unavailable ({error})",
            "openmp_enabled": "unknown",
        }

    info: Dict[str, object] = {
        "native_extension": getattr(_C_deal_sgp, "__file__", "unknown")
    }
    runtime_info = getattr(_C_deal_sgp, "parallel_runtime_info", None)
    if runtime_info is None:
        info["openmp_enabled"] = "unknown (extension needs rebuilding)"
    else:
        info.update(runtime_info())
    return info


def collect_runtime_info() -> Dict[str, object]:
    memory_gib = _total_memory_gib()
    info: Dict[str, object] = {
        "host": platform.node() or "unknown",
        "platform": platform.platform(),
        "architecture": platform.machine() or "unknown",
        "cpu_model": _cpu_model(),
        "logical_cpus": os.cpu_count() or 1,
        "physical_cpus": _physical_cpu_count(),
        "available_cpus": _available_cpu_count(),
        "memory_gib": memory_gib,
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "deal_version": _package_version(),
        "numpy_version": np.__version__,
        "thread_environment": {
            variable: os.environ.get(variable, "<unset>")
            for variable in THREAD_ENVIRONMENT_VARIABLES
        },
    }
    info.update(_native_runtime_info())
    return info


def format_runtime_info(info: Optional[Dict[str, object]] = None) -> str:
    info = collect_runtime_info() if info is None else info
    physical_cpus = info.get("physical_cpus")
    memory_gib = info.get("memory_gib")
    openmp_enabled = info.get("openmp_enabled", "unknown")

    cpu_counts = (
        f"logical={info['logical_cpus']}, "
        f"physical={physical_cpus if physical_cpus is not None else 'unknown'}, "
        f"available={info['available_cpus']}"
    )
    memory = f"{memory_gib:.1f} GiB" if memory_gib is not None else "unknown"

    if openmp_enabled is True:
        openmp = (
            f"enabled, version={info['openmp_version']}, "
            f"max_threads={info['openmp_max_threads']}, "
            f"runtime_cpus={info['openmp_num_procs']}, "
            f"dynamic={info['openmp_dynamic']}"
        )
    elif openmp_enabled is False:
        openmp = "disabled (native extension runs serially)"
    else:
        openmp = str(openmp_enabled)

    thread_environment = info["thread_environment"]
    thread_settings = ", ".join(
        f"{variable}={thread_environment[variable]}"
        for variable in THREAD_ENVIRONMENT_VARIABLES
    )

    return "\n".join(
        (
            "[INFO] Runtime environment:",
            f"- Host: {info['host']}",
            f"- Platform: {info['platform']} ({info['architecture']})",
            f"- CPU: {info['cpu_model']} ({cpu_counts})",
            f"- Memory: {memory}",
            (
                f"- Software: DEAL {info['deal_version']}, "
                f"Python {info['python_version']}, NumPy {info['numpy_version']}"
            ),
            f"- Python executable: {info['python_executable']}",
            f"- Native extension: {info['native_extension']}",
            f"- OpenMP: {openmp}",
            f"- Thread settings: {thread_settings}",
        )
    )
