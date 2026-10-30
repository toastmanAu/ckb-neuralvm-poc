#!/usr/bin/env python3
"""Checks deterministic test vectors (v0.1 payloads and v0.2 contexts) against the independent
Python oracles. No model, GPU or RV32 build needed."""
import json
import pathlib
import sys

import reference_v2
from reference_rules import verify

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    sets = [("test_vectors.json", lambda t: verify(t["payload"])),
            ("ctx_vectors.json", reference_v2.verify)]
    errors, total = [], 0
    for name, oracle in sets:
        for t in json.loads((ROOT / "fixtures" / name).read_text()):
            total += 1
            actual = oracle(t)
            if actual != t["expected"]:
                errors.append(f"{t['id']}: {actual!r} != {t['expected']!r}")
            print(("PASS" if actual == t["expected"] else "FAIL"), t["id"], actual.strip())
    if errors:
        print("\n".join(errors), file=sys.stderr); raise SystemExit(1)
    print(f"{total}/{total} independent Python fixture checks passed.")


if __name__ == "__main__": main()
