import subprocess

import torch


def get_gpu_stats(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {}

    gpu_index = device.index
    if gpu_index is None:
        gpu_index = torch.cuda.current_device()

    command = [
        "nvidia-smi",
        f"--id={gpu_index}",
        "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}

    values = [value.strip() for value in result.stdout.strip().split(",")]
    if len(values) != 5:
        return {}

    util, memory_used, memory_total, power, temp = values
    return {
        "gpu/utilization_percent": float(util),
        "gpu/memory_used_mb": float(memory_used),
        "gpu/memory_total_mb": float(memory_total),
        "gpu/power_watts": float(power),
        "gpu/temperature_c": float(temp),
    }
