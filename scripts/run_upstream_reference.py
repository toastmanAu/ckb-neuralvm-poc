#!/usr/bin/env python3
"""
Run the compiled RV32 firmware using Qwenputer's *software reference machine*.
NOT neural execution: a separate pass must run model layers on GPU.
Run in upstream 'uv' environment (see HANDOFF.md).
"""
import argparse
import json
import pathlib
from chipformer.rv import bridge as BR, elf, ref, spec as S

import ctx_v2
import reference_rules
import reference_v2

ROOT = pathlib.Path(__file__).resolve().parents[1]

def run_one(tok, elf_path, payload, max_records):
    machine = ref.Machine.from_program(elf.program(elf_path))
    BR.park(machine)
    session = BR.Session(tok, machine)
    call = ("<tool_call>\n<function=ckb_verify>\n<parameter=payload>\n" + payload +
            "\n</parameter>\n</function>\n</tool_call>")
    session.qwen(tok.encode(call), max_records=max_records)
    if machine.fault is not None:
        raise RuntimeError(f"fault {S.FAULTS[machine.fault[0]]} at {machine.fault[1]:#x}")
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
    v1 = json.loads((ROOT / "fixtures/test_vectors.json").read_text())
    v2 = json.loads((ROOT / "fixtures/ctx_vectors.json").read_text())
    v3 = json.loads((ROOT / "fixtures/syscall_vectors.json").read_text())
    ctx_dir = ROOT / "build/ctx"
    errors, code_shas = [], set()
    for tag, vectors in (("v1", v1), ("v2", v2), ("v3", v3)):
        for v in vectors:
            # each v0.2 fixture runs its own ELF: the base build with the fixture's context injected
            path = ctx_v2.case_elf(a.elf, v, ctx_dir) if tag != "v1" else pathlib.Path(a.elf)
            if tag != "v1":
                code_shas.add(ctx_v2.code_sha256(path))
            oracle = reference_v2.verify(v) if tag != "v1" else reference_rules.verify(v["payload"])
            # stock upstream (no trap patch): a v0.3 case must fault on its first syscall; only an
            # invalid context, rejected before any syscall, gets the normal verdict
            want_fault = tag == "v3" and not hasattr(S, "TRAP_CALLS") and oracle != "FAIL;R=CTX\n"
            try:
                actual, _ = run_one(text, path, v["payload"], a.max_records)
                ok = not want_fault and actual == v["expected"] == oracle
            except Exception as exc:
                actual = f"error {exc!r}"
                ok = want_fault and str(exc).startswith("fault ecall")
            print("PASS" if ok else "FAIL", tag, v["id"], repr(actual) + (" (expected: stock circuit)" if want_fault else ""))
            if not ok: errors.append(f"{tag}:{v['id']}")
    if len(code_shas) != 1:
        errors.append(f"context injection changed code bytes: {sorted(code_shas)}")
    if errors: raise SystemExit("Reference emulator failures: " + ", ".join(errors))
    circuit = "patched" if hasattr(S, "TRAP_CALLS") else "stock: v0.3 syscalls fault as expected"
    print(f"[{circuit}] All {len(v1)} v0.1 + {len(v2)} v0.2 + {len(v3)} v0.3 firmware/reference checks passed "
          f"(software-only, not neural); code sha256 (context zeroed) {code_shas.pop()}")
if __name__ == "__main__": main()
