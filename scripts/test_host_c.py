#!/usr/bin/env python3
"""Differential test of the guest C (built natively with -DTOY_HOST) against the Python oracles.

NOT the RV32 firmware and NOT neural execution: this checks the contract and context-API C
sources on x86-64 before they run on ref.Machine and the model layers.

  1. every v0.2 fixture (fixtures/ctx_vectors.json) and v0.1 fixture through `host_toy run`
  2. --n random contexts x API calls through `host_toy api`, compared with reference_v2.Ctx,
     weighted toward the edges (offsets 0/len/len+1/UINT64_MAX, indices 0..3/UINT64_MAX,
     unknown sources, missing witness, invalid contexts)
"""
import argparse
import json
import pathlib
import random
import subprocess
import sys

import ctx_v2
import reference_rules
import reference_v2 as R

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRCS = ["host_main.c", "contract.c", "ctx.c", "emit.c", "parse.c"]


def build(cc: str) -> pathlib.Path:
    out = ROOT / "build/host_toy"
    out.parent.mkdir(exist_ok=True)
    subprocess.run([cc, "-DTOY_HOST", "-O2", "-Wall", "-Wextra", "-Werror", "-fsanitize=address,undefined",
                    "-o", str(out)] + [str(ROOT / "guest" / s) for s in SRCS], check=True)
    return out


def fixtures(exe) -> list[str]:
    bad = []
    for v in json.loads((ROOT / "fixtures/ctx_vectors.json").read_text()):
        raw = bytes(ctx_v2.CTX_SIZE) if v["ctx"] is None else ctx_v2.encode(v["ctx"])
        got = subprocess.run([exe, "run", raw.hex(), v["payload"]], capture_output=True, text=True, check=True).stdout
        if got != v["expected"]:
            bad.append(f"v2 {v['id']}: {got!r} != {v['expected']!r}")
    any_ctx = bytes(ctx_v2.CTX_SIZE).hex()
    for v in json.loads((ROOT / "fixtures/test_vectors.json").read_text()):
        # bridge.c strips newlines from the captured payload; the v0.1 path ignores the context
        got = subprocess.run([exe, "run", any_ctx, v["payload"]], capture_output=True, text=True, check=True).stdout
        if got != v["expected"] or got != reference_rules.verify(v["payload"]):
            bad.append(f"v1 {v['id']}: {got!r} != {v['expected']!r}")
    return bad


def random_ctx(rng: random.Random) -> dict:
    caps = lambda: [f"{rng.choice([0, 1, rng.getrandbits(64), R.U64, R.U64 - 1]):016X}"
                    for _ in range(rng.randint(0, 2))]
    nbytes = lambda: rng.choice([0, 1, 3, 4, 5, 31, 32, rng.randint(0, 32)])
    ctx = {"inputs": caps(), "outputs": caps(), "args": rng.randbytes(nbytes()).hex(),
           "witness": None if rng.random() < 0.15 else rng.randbytes(nbytes()).hex()}
    r = rng.random()  # raw header overrides: invalid values, and valid ones disagreeing with the lists
    if r < 0.05:
        ctx["magic"] = "CKX0"
    elif r < 0.08:
        ctx["n_inputs"] = rng.randint(3, 255)
    elif r < 0.11:
        ctx["n_outputs"] = rng.randint(3, 255)
    elif r < 0.14:
        ctx["args_len"] = rng.choice([33, 34, 254, 255, rng.randint(33, 255)])
    elif r < 0.17:
        ctx["witness_len"] = rng.choice([33, 34, 254, rng.randint(33, 254)])
    elif r < 0.20:
        ctx[rng.choice(["n_inputs", "n_outputs"])] = rng.randint(0, 2)
    elif r < 0.23:
        ctx[rng.choice(["args_len", "witness_len"])] = rng.randint(0, 32)
    return ctx


def api(exe, n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    lines, want = [], []
    for _ in range(n):
        ctx = random_ctx(rng)
        c = R.Ctx(ctx)
        lines.append("C " + ctx_v2.encode(ctx).hex())
        size = max(len(c.args), len(c.wit or b""))
        for _ in range(12):
            if rng.random() < 0.6:
                src = rng.choice([R.ARGS, R.WITNESS, R.ARGS, R.WITNESS, 0, 3, R.U64])
                buf = rng.choice([0, 1, 4, 5, 32, 40, rng.randint(0, 40)])
                off = rng.choice([0, 1, 3, 4, size - 1 if size else 0, size, size + 1, 200, R.U64,
                                  rng.getrandbits(64)])
                rc, ln, got = c.partial_load(buf, off, src)
                lines.append(f"L {src} {buf} {off}")
                want.append(f"{rc} {ln} {got.hex().upper() or '-'}")
            else:
                idx = rng.choice([0, 1, 2, 3, R.U64, rng.getrandbits(64)])
                src = rng.choice([R.INPUT, R.OUTPUT, R.INPUT, R.OUTPUT, 0, 3, R.U64])
                rc, cap = c.capacity(idx, src)
                lines.append(f"K {idx} {src}")
                want.append(f"{rc} {'-' if cap is None else cap}")
    got = subprocess.run([exe, "api"], input="\n".join(lines) + "\n", capture_output=True, text=True,
                         check=True).stdout.splitlines()
    if len(got) != len(want):
        return [f"api: {len(got)} results for {len(want)} calls"]
    calls = [x for x in lines if x[0] != "C"]
    return [f"api {calls[i]}: C {got[i]!r} != Python {want[i]!r}" for i in range(len(want)) if got[i] != want[i]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cc", default="cc")
    p.add_argument("--n", type=int, default=2000, help="random contexts (12 API calls each)")
    p.add_argument("--seed", type=int, default=20261030)
    a = p.parse_args()
    exe = build(a.cc)
    bad = fixtures(exe)
    print(f"fixtures: {len(bad)} mismatches")
    bad_api = api(exe, a.n, a.seed)
    print(f"api: {a.n * 12} random calls, {len(bad_api)} mismatches")
    for line in (bad + bad_api)[:20]:
        print("  " + line)
    sys.exit(1 if bad or bad_api else 0)


if __name__ == "__main__":
    main()
