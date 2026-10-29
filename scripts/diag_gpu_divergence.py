#!/usr/bin/env python3
"""Diagnostic only: find the first HF submodule whose bf16 output differs between CPU and GPU.

The compiled RV32 core is bit-exact in bf16 on CPU (upstream gpucheck) but not on ROCm. This runs
the same small chip model (upstream hf.build_model, snapshot injected via hf.inject) on both
devices with forward hooks on every leaf module, and reports modules in execution order with the
max |cpu - gpu| of their outputs. Run in the upstream tree with PYTHONPATH set.
"""
import argparse

import torch

from chipformer.rv import core, elf, hf, ref
from chipformer.rv.measure import DATA
from chipformer.rv.programs import ALL, program


def trace(model, c, ids, toks, dev):
    """Prefill with the snapshot injected; returns [(module name, output tensor on cpu)]."""
    model = model.to(dev)
    rec = []
    hooks = []
    for name, mod in model.named_modules():
        def hook(m, inp, out, name=name):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o):
                rec.append((name, o.detach().float().cpu()))
        hooks.append(mod.register_forward_hook(hook))
    x = torch.tensor([[ids[t] for t in toks]], device=dev)
    with torch.no_grad():
        out = model(input_ids=torch.tensor([[ids[None]]], device=dev), use_cache=True)
        hf.inject(c, out.past_key_values)
        rec.clear()  # only the trace after the injected snapshot
        model(input_ids=x, past_key_values=out.past_key_values, use_cache=True)
    for h in hooks:
        h.remove()
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prog", default="signext")
    ap.add_argument("--show", type=int, default=12, help="differing modules to print")
    a = ap.parse_args()
    prog = program(a.prog) if a.prog in ALL else elf.program(DATA / f"{a.prog}.elf")
    c = core.build()
    core.snapshot(c, ref.Machine.from_program(prog))
    toks = ref.Machine.from_program(prog).run(10**6)
    model, lay, ids = hf.build_model(c, toks, torch.bfloat16)
    cpu = trace(model, c, ids, toks, "cpu")
    gpu = trace(model, c, ids, toks, "cuda")
    assert [n for n, _ in cpu] == [n for n, _ in gpu], "module call order differs"
    shown = 0
    for (name, x), (_, y) in zip(cpu, gpu):
        if x.shape != y.shape:
            print(f"{name}: shape {tuple(x.shape)} vs {tuple(y.shape)}")
            continue
        d = (x - y).abs()
        if d.max() > 0:
            pos = int(d.reshape(d.shape[0] * d.shape[1], -1).amax(-1).argmax()) if d.dim() >= 3 else -1
            print(f"{name:55s} max|diff|={float(d.max()):.3e} n_diff={int((d > 0).sum())} "
                  f"first_pos={pos}")
            shown += 1
            if shown >= a.show:
                break
    if not shown:
        print("CPU and GPU agree bit-exactly on every hooked module output")


if __name__ == "__main__":
    main()
