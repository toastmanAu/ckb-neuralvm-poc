#!/usr/bin/env python3
"""Checks deterministic test vectors against a completely independent Python oracle."""
import json
import pathlib
import sys
from reference_rules import verify

ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    vectors = json.loads((ROOT / "fixtures" / "test_vectors.json").read_text())
    errors = []
    for t in vectors:
        actual = verify(t["payload"])
        if actual != t["expected"]:
            errors.append(f"{t['id']}: {actual!r} != {t['expected']!r}")
        print(("PASS" if actual == t["expected"] else "FAIL"), t["id"], actual.strip())
    if errors:
        print("\n".join(errors), file=sys.stderr); raise SystemExit(1)
    print(f"{len(vectors)}/{len(vectors)} independent Python fixture checks passed.")
if __name__ == "__main__": main()
