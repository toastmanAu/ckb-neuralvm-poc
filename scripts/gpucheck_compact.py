#!/usr/bin/env python3
"""Upstream gpucheck (strict bf16, bit-exact vs the IR evaluator) on the *compacted* chip model,
i.e. the module set RvChat's fast path runs (fast.compact(qwen=False) -> SparseLinear).

    PYTHONPATH=<qwenputer>/experimental/riscv:<this repo>/scripts \
        python gpucheck_compact.py [--progs a,b] [--decode] [--no-compact] [--no-adapter]

Exits 1 on any mismatch. Run in <qwenputer>/experimental/riscv (programs load from tests/data).
"""
import argparse
import sys
import time

import torch

from chipformer.q35 import fast
from chipformer.rv import core, elf, hf, ref
from chipformer.rv.measure import DATA
from chipformer.rv.programs import ALL, program

import rocm_exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--progs", default="signext,call,memcpy,rv32ui-lb,c-crc32")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--no-compact", action="store_true", help="dense nn.Linear (the slow path)")
    ap.add_argument("--no-adapter", action="store_true", help="reproduce the ROCm divergence")
    a = ap.parse_args()
    print(f"device {torch.cuda.get_device_name()}, compact={not a.no_compact}, "
          f"adapter={not a.no_adapter}", flush=True)
    c = core.build()
    bad = 0
    for name in a.progs.split(","):
        prog = program(name) if name in ALL else elf.program(DATA / f"{name}.elf")
        core.snapshot(c, ref.Machine.from_program(prog))
        toks = ref.Machine.from_program(prog).run(10**6)
        model, lay, ids = hf.build_model(c, toks, torch.bfloat16)
        model = model.to("cuda")
        if not a.no_compact:
            fast.compact(model, qwen=False, n_chip=c.n_layers())
        wrapped = 0 if a.no_adapter else rocm_exact.install([model.model])
        for mode in ["prefill"] + (["decode"] if a.decode else []):
            t = time.time()
            try:
                n = hf.check(c, model, lay, ids, toks, mode)
                print(f"{name:12s} {mode:7s} exact on {n} positions, {wrapped} wrapped "
                      f"({time.time() - t:.1f} s)", flush=True)
            except AssertionError as e:
                bad += 1
                print(f"{name:12s} {mode:7s} MISMATCH, {wrapped} wrapped: {str(e)[:200]}",
                      flush=True)
        del model
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
