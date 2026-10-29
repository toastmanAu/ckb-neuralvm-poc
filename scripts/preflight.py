#!/usr/bin/env python3
"""Read-only, no-download GPU/runtime preflight. Does not prove model compatibility."""
import importlib.util
import json
import platform
import shutil
import sys

def main():
    report = {"python": sys.version.split()[0], "platform": platform.platform()}
    report["commands"] = {x: bool(shutil.which(x)) for x in
        ("uv", "hf", "riscv64-unknown-elf-gcc", "riscv64-unknown-elf-readelf", "rocminfo")}
    report["torch_installed"] = bool(importlib.util.find_spec("torch"))
    report["triton_installed"] = bool(importlib.util.find_spec("triton"))
    if report["torch_installed"]:
        import torch
        report["torch_version"] = torch.__version__
        report["torch_hip"] = torch.version.hip
        report["torch_cuda_visible"] = torch.cuda.is_available()  # works for ROCm/HIP too
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            report["gpu_name"] = props.name
            report["gpu_vram_gib"] = round(props.total_memory / 1024**3, 2)
            try:
                a = torch.tensor([1, 2, 3], device="cuda"); report["gpu_smoke_test"] = int(a.sum().item()) == 6
            except Exception as exc:
                report["gpu_error"] = repr(exc)
    print(json.dumps(report, indent=2))
    print("Preflight only: does NOT establish Qwenputer/GDN or CUDA-graph compatibility.")
if __name__ == "__main__": main()
