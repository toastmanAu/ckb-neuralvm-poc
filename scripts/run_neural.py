#!/usr/bin/env python3
"""Run one fixture through Qwenputer's model layers (the neural RV32 CPU) and verify it.

Requires a real tool call in the model output; otherwise outcome is NOT_EXERCISED. Verification:
live lockstep against upstream ref.Machine (lock_error must be None) and RvChat.replay (IR
evaluator + reference machine over the whole stream). The Python oracle in reference_rules.py is
only compared against, never used to produce the answer.

Run in the upstream tree: PYTHONPATH=<qwenputer>/experimental/riscv:<this repo>/scripts
"""
import argparse
import datetime
import hashlib
import json
import pathlib
import time

import torch

from chipformer.rv import bridge as BR, elf, ref
from chipformer.rv.chat import RvChat, replay

import rocm_exact
from reference_rules import verify as oracle

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_REV = "2d9068508f2b680badb7abcded96fdd107e9ba5d"
CKB_TOOL = {"type": "function", "function": {
    "name": "ckb_verify",
    "description": "Run a toy CKB-style capacity/lock-tag verifier. It is NOT a valid CKB transaction verification. Use payload exactly as provided without changes.",
    "parameters": {"type": "object", "properties": {
        "payload": {"type": "string", "description": "Exact V1;I=...;O=...;L=...;W=... payload"}},
        "required": ["payload"]}}}
START, END = "<tool_response>\n", "</tool_response>"


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def tool_output(text: str) -> str | None:
    """The firmware's response bytes between the (circuit-printed) tool-response tags."""
    if START not in text or END not in text:
        return None
    return text.split(START, 1)[1].split(END, 1)[0]


def sent_payload(text: str) -> str | None:
    """The payload Qwen actually wrote into the tool call (what the CPU was given)."""
    head, tail = "<parameter=payload>\n", "\n</parameter>"
    if head not in text or tail not in text.split(head, 1)[1]:
        return None
    return text.split(head, 1)[1].split(tail, 1)[0]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--elf", default=str(ROOT / "build/ckb_bridge.elf"))
    p.add_argument("--case", default="valid_fee")
    p.add_argument("--slow", action="store_true")
    p.add_argument("--executor", action="store_true", help="NVIDIA-oriented fast graph path; try only after baseline")
    p.add_argument("--no-adapter", action="store_true", help="skip the ROCm exact-GEMM adapter (reproduces divergence)")
    p.add_argument("--max-new", type=int, default=600)
    p.add_argument("--max-chip", type=int, default=60000)
    p.add_argument("--no-check", action="store_true", help="Debug only: skip IR replay")
    p.add_argument("--manifest", default=str(ROOT / "reports/runs.jsonl"))
    a = p.parse_args()
    vectors = json.loads((ROOT / "fixtures/test_vectors.json").read_text())
    v = next((x for x in vectors if x["id"] == a.case), None)
    if v is None:
        raise SystemExit(f"Unknown case {a.case}")
    assert oracle(v["payload"]) == v["expected"], "fixture disagrees with the independent oracle"

    chat = RvChat(a.model, fast=not a.slow, executor=a.executor)
    adapter = rocm_exact.needed() and not a.no_adapter
    if adapter:
        rocm_exact.install(chat.lm.layers[: chat.n_chip])
    m = ref.Machine.from_program(elf.program(a.elf))
    BR.park(m)
    prompt = "Call ckb_verify using precisely this payload, with no modifications: " + v["payload"]
    ids = chat.prompt_ids([{"role": "user", "content": prompt}], tools=(CKB_TOOL,))
    lock_m = ref.Machine.from_program(elf.program(a.elf))
    BR.park(lock_m)
    lock = BR.Session(chat.T, lock_m)
    lock.history(ids)
    t0 = time.time()
    r = chat.generate(ids, m, max_new=a.max_new, max_chip=a.max_chip,
                      check_head=True, verbose=True, lock=lock)
    elapsed = time.time() - t0
    generated = chat.text_of(r["tokens"])
    actual = tool_output(generated)
    called = "<function=ckb_verify>" in generated
    exercised = called and actual is not None and r["chip_tokens"] > 0
    replay_result = replay(chat, ids, r["tokens"], a.elf) if exercised and not a.no_check else None
    replay_ok = None if replay_result is None else bool(replay_result.get("ok"))
    sent = sent_payload(generated)
    payload_exact = sent == v["payload"]
    match = exercised and actual == v["expected"]
    trace_ok = r["lock_error"] is None and replay_ok is True
    # A matching output from a mis-transcribed payload is not a pass for this fixture.
    status = ("TRACE_MISMATCH" if r["lock_error"] is not None
              else "NOT_EXERCISED" if not exercised else "UNVERIFIED" if a.no_check
              else "INPUT_MISMATCH" if trace_ok and not payload_exact
              else "PASS" if match and trace_ok else "FAIL")
    mode = "neural_executor" if a.executor else "neural_hf" if a.slow else "neural_hf_fast"
    props = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    out = {
        "mode": mode, "status": status, "case_id": v["id"],
        "device": props.name if props else "cpu",
        "backend": f"hip {torch.version.hip}" if torch.version.hip else f"cuda {torch.version.cuda}",
        "torch": torch.__version__, "rocm_exact_adapter": adapter,
        "model_rev": MODEL_REV, "firmware_sha256": sha256(a.elf), "elf": a.elf,
        "expected": v["expected"], "actual": actual,
        "reference_match": match, "chip_tokens": r["chip_tokens"], "text_tokens": r["text_tokens"],
        "elapsed_s": round(elapsed, 2), "lock_error": r["lock_error"], "replay_ok": replay_ok,
        "replay": replay_result, "tool_called": called,
        "sent_payload": sent, "payload_exact": payload_exact,
        "head_mismatch": r["head_mismatch"], "generated": generated,
        "timestamp": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    print(json.dumps(out, indent=2))
    with open(a.manifest, "a") as f:
        f.write(json.dumps(out) + "\n")
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
