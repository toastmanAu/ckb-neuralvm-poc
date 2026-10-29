#!/usr/bin/env python3
"""
Run the compiled RV32 firmware using Qwenputer's *software reference machine*.
NOT neural execution: a separate pass must run model layers on GPU.
Run in upstream 'uv' environment (see HANDOFF.md).
"""
import argparse
import json
import pathlib
from chipformer.rv import bridge as BR, elf, ref

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run_one(tok, elf_path, payload, max_records):
    machine = ref.Machine.from_program(elf.program(elf_path))
    BR.park(machine)
    session = BR.Session(tok, machine)
    call = ("<tool_call>\n<function=ckb_verify>\n<parameter=payload>\n" + payload +
            "\n</parameter>\n</function>\n</tool_call>")
    session.qwen(tok.encode(call), max_records=max_records)
    if len(session.runs) != 1:
        raise RuntimeError(f"expected one run, found {len(session.runs)}; check tokenizer/tool-call syntax")
    output = tok.decode_bytes(session.runs[-1]).decode("utf-8", "replace")
    start_tag, end_tag = "<tool_response>\n", "</tool_response>"
    if start_tag not in output or end_tag not in output:
        raise RuntimeError(f"response tags missing: {output!r}")
    actual = output.split(start_tag, 1)[1].split(end_tag, 1)[0]
    return actual, output

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--elf", default=str(ROOT / "build/ckb_bridge.elf"))
    p.add_argument("--tokenizer", required=True, help="Path to the downloaded model's tokenizer.json")
    p.add_argument("--max-records", type=int, default=200000)
    a = p.parse_args()
    text = BR.Text(a.tokenizer)
    vectors = json.loads((ROOT / "fixtures/test_vectors.json").read_text())
    errors = []
    for v in vectors:
        try:
            actual, _ = run_one(text, a.elf, v["payload"], a.max_records)
            ok = actual == v["expected"]
        except Exception as exc:
            ok, actual = False, f"error {exc!r}"
        print("PASS" if ok else "FAIL", v["id"], repr(actual))
        if not ok: errors.append(v["id"])
    if errors: raise SystemExit("Reference emulator failures: " + ", ".join(errors))
    print(f"All {len(vectors)} firmware/reference checks passed (software-only, not neural).")
if __name__ == "__main__": main()
