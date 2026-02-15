import os
import shutil
from typing import Dict, Any, Tuple

import torch

try:
    import psutil  # type: ignore
except Exception:
    psutil = None


def _to_gb(v: float) -> float:
    return float(v) / (1024.0 ** 3)


def _read_meminfo_kb() -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for ln in f:
                if ":" not in ln:
                    continue
                k, val = ln.split(":", 1)
                parts = val.strip().split()
                if not parts:
                    continue
                try:
                    out[k] = int(parts[0])
                except Exception:
                    continue
    except Exception:
        pass
    return out


def _cpu_percent_fallback() -> float:
    # Fallback estimate from 1-minute load avg when psutil is unavailable.
    try:
        one_min = os.getloadavg()[0]
        cores = os.cpu_count() or 1
        return max(0.0, min(100.0, (one_min / float(cores)) * 100.0))
    except Exception:
        return 0.0


def get_cpu_stats() -> Dict[str, Any]:
    if psutil is not None:
        try:
            return {
                "percent": float(psutil.cpu_percent(interval=0.0)),
                "cores_physical": int(psutil.cpu_count(logical=False) or 0),
                "cores_logical": int(psutil.cpu_count(logical=True) or 0),
            }
        except Exception:
            pass

    return {
        "percent": _cpu_percent_fallback(),
        "cores_physical": int(os.cpu_count() or 0),
        "cores_logical": int(os.cpu_count() or 0),
    }


def get_ram_stats() -> Dict[str, Any]:
    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            used = float(vm.total - vm.available)
            return {
                "total_gb": _to_gb(vm.total),
                "used_gb": _to_gb(used),
                "available_gb": _to_gb(vm.available),
                "percent": float(vm.percent),
            }
        except Exception:
            pass

    mi = _read_meminfo_kb()
    total_kb = int(mi.get("MemTotal", 0))
    avail_kb = int(mi.get("MemAvailable", mi.get("MemFree", 0)))
    used_kb = max(0, total_kb - avail_kb)
    percent = (used_kb / total_kb * 100.0) if total_kb > 0 else 0.0
    return {
        "total_gb": _to_gb(total_kb * 1024.0),
        "used_gb": _to_gb(used_kb * 1024.0),
        "available_gb": _to_gb(avail_kb * 1024.0),
        "percent": float(percent),
    }


def get_storage_stats(path: str = "/") -> Dict[str, Any]:
    try:
        du = shutil.disk_usage(path)
        used = float(du.used)
        total = float(du.total)
        percent = (used / total * 100.0) if total > 0 else 0.0
        return {
            "path": path,
            "total_gb": _to_gb(total),
            "used_gb": _to_gb(used),
            "free_gb": _to_gb(float(du.free)),
            "percent": float(percent),
        }
    except Exception:
        return {
            "path": path,
            "total_gb": 0.0,
            "used_gb": 0.0,
            "free_gb": 0.0,
            "percent": 0.0,
        }


def get_process_stats() -> Dict[str, Any]:
    pid = int(os.getpid())
    if psutil is not None:
        try:
            p = psutil.Process(pid)
            rss = float(p.memory_info().rss)
            return {
                "pid": pid,
                "rss_gb": _to_gb(rss),
                "threads": int(p.num_threads()),
            }
        except Exception:
            pass
    return {"pid": pid, "rss_gb": 0.0, "threads": 0}


def get_vram_stats() -> Tuple[float, float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0, 0.0

    try:
        allocated = float(torch.cuda.memory_allocated())
        total = float(torch.cuda.get_device_properties(0).total_memory)
        percent = (allocated / total) * 100.0 if total > 0 else 0.0
        return _to_gb(allocated), _to_gb(total), float(percent)
    except Exception:
        return 0.0, 0.0, 0.0


def get_system_stats(storage_path: str = "/") -> Dict[str, Any]:
    cpu = get_cpu_stats()
    ram = get_ram_stats()
    storage = get_storage_stats(storage_path)
    vram_used, vram_total, vram_pct = get_vram_stats()
    process = get_process_stats()
    return {
        "cpu": cpu,
        "ram": ram,
        "storage": storage,
        "vram": {
            "used_gb": float(vram_used),
            "total_gb": float(vram_total),
            "percent": float(vram_pct),
        },
        "process": process,
    }
