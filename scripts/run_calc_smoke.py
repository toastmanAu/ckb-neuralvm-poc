#!/usr/bin/env python3
"""Upstream chipformer.rv.chat main() (calc baseline), with the ROCm exact-GEMM adapter on the
chip layers when running on HIP. Run in the upstream tree with PYTHONPATH including this dir."""
import argparse

from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv.chat import RvChat, replay

import rocm_exact


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--firmware", default="tests/data/rv/calc-bridge.elf")
    ap.add_argument("--ask", default="Use calc: what is 123456789 * 987654321?")
    ap.add_argument("--max-new", type=int, default=600)
    ap.add_argument("--slow", action="store_true")
    ap.add_argument("--no-adapter", action="store_true", help="reproduce the ROCm divergence")
    a = ap.parse_args()
    chat = RvChat(a.model, fast=not a.slow)
    if rocm_exact.needed() and not a.no_adapter:
        n = rocm_exact.install(rocm_exact.chip_modules(chat))
        print(f"[rocm_exact] fp32-accumulate on {n} Linear modules in {chat.n_chip} chip layers")
    m = ref.Machine.from_program(elf.program(a.firmware))
    BR.park(m)
    ids = chat.prompt_ids([{"role": "user", "content": a.ask}])
    m2 = ref.Machine.from_program(elf.program(a.firmware))
    BR.park(m2)
    lock = BR.Session(chat.T, m2)
    lock.history(ids)
    r = chat.generate(ids, m, a.max_new, check_head=True, verbose=True, lock=lock)
    print(chat.text_of(r["tokens"]))
    print(f"[{r['text_tokens']} text + {r['chip_tokens']} chip tokens in {r['seconds']:.1f}s, "
          f"head mismatches {r['head_mismatch']}, lock error {r['lock_error']}]")
    print(replay(chat, ids, r["tokens"], a.firmware))


if __name__ == "__main__":
    main()
