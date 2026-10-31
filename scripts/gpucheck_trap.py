#!/usr/bin/env python3
"""Upstream's strict bf16 circuit check (hf.check: every IR dim bit-exact at every position) on
traces that take the ecall trap, for the PATCHED circuit (patches/qwenputer-ecall-trap.patch).

Each case: a v0.3 fixture's context injected into a bridge-free self-test ELF
(scripts/build_selftest.sh), run on ref.Machine from the patched tree; the trace must halt
cleanly with the fixture's expected output and contain J(TRAP_WORD) tokens (the trap taken).
First the circuit's own predictions (core.predictions over ir.evaluate) must equal the
reference trace at every position: the circuit computes what ref.Machine does. hf.check alone
cannot show that, since it compares the weights with the same IR (an IR bug passes it). Then the
chip-only HF model of core.build() runs the trace, dense and compacted (fast path), with the
ROCm adapter, bit-exact against the IR. Exits 1 on any mismatch. Run in the PATCHED tree:

    cd /data/qwenputer/qwenputer-trap/experimental/riscv && \\
        PYTHONPATH=.:<this repo>/scripts python gpucheck_trap.py [--cases a,b] [--decode]
"""
import argparse
import json
import pathlib
import sys
import time

import torch

from chipformer.q35 import fast, ir
from chipformer.rv import core, elf, hf, ref
from chipformer.rv import spec as S

import ctx_v2
import rocm_exact

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT = "sys_two_in_two_out,sys_long_lock_last_byte,sys_witness_missing,sys_probe_long"


def trace(case: dict) -> tuple[list[int], object]:
    name = "selftest-V3_T.elf" if case["payload"] == "V3;T" else "selftest-V3.elf"
    path = ctx_v2.case_elf(ROOT / "build" / name, case, ROOT / "build/ctx")
    prog = elf.program(path)
    m = ref.Machine.from_program(prog)
    toks = m.run(10**6)
    out = bytes(m.out).decode()
    if m.fault is not None or not m.halted or out != case["expected"]:
        raise SystemExit(f"{case['id']}: ref.Machine fault={m.fault} halted={m.halted} out={out!r}")
    return toks, prog


def main() -> None:
    if not hasattr(S, "TRAP_CALLS"):
        raise SystemExit("run from the patched tree (spec.TRAP_CALLS missing)")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=DEFAULT)
    ap.add_argument("--decode", action="store_true")
    a = ap.parse_args()
    fixtures = {v["id"]: v for v in json.loads((ROOT / "fixtures/syscall_vectors.json").read_text())}
    print(f"device {torch.cuda.get_device_name()}", flush=True)
    c = core.build()
    trap = S.tok("J", S.TRAP_WORD)
    bad = 0
    for cid in a.cases.split(","):
        toks, prog = trace(fixtures[cid])
        n_trap = toks.count(trap)
        if not n_trap:
            raise SystemExit(f"{cid}: no J(TRAP_WORD) in the trace: the trap is not exercised")
        core.snapshot(c, ref.Machine.from_program(prog))
        pred = core.predictions(c, ir.evaluate(c, toks, check=False))
        wrong = [i for i in range(len(toks) - 1) if pred[i] != toks[i + 1]]
        if wrong:
            i = wrong[0]
            bad += 1
            print(f"{cid:28s} IR predictions: {len(wrong)} wrong of {len(toks) - 1}; first at {i}: "
                  f"{S.name(pred[i])} for {S.name(toks[i + 1])}", flush=True)
            continue
        print(f"{cid:28s} IR predictions match the reference on {len(toks) - 1} positions", flush=True)
        for compact in (False, True):
            model, lay, ids = hf.build_model(c, toks, torch.bfloat16)
            model = model.to("cuda")
            if compact:
                fast.compact(model, qwen=False, n_chip=c.n_layers())
            wrapped = rocm_exact.install([model.model])
            for mode in ["prefill"] + (["decode"] if a.decode else []):
                t = time.time()
                label = f"{cid:28s} {'compact' if compact else 'dense':7s} {mode:7s}"
                try:
                    n = hf.check(c, model, lay, ids, toks, mode)
                    print(f"{label} exact on {n} positions ({n_trap} trap jumps), {wrapped} wrapped "
                          f"({time.time() - t:.1f} s)", flush=True)
                except AssertionError as e:
                    bad += 1
                    print(f"{label} MISMATCH: {str(e)[:200]}", flush=True)
            del model
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
