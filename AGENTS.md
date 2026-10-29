# Coding agent instructions — CKB-NeuralVM PoC

You are the implementation agent responsible for finishing an **evidence-backed neural execution PoC**. Read `README.md`, `HANDOFF.md`, `docs/RESEARCH.md` and the supplied source files before editing anything.

## Single objective

Run a small CKB-*inspired* validation program compiled as RV32IM+Zba ELF through Qwenputer's embedded neural processor, verify output with conventional reference execution, and capture reproducible evidence. We are not building a production CKB script engine, RPC service, on-chain product or security validator.

## Do not cut corners

- DO NOT substitute Python arithmetic or a software interpreter and call it neural inference.
- DO NOT use a normal Qwen model's textual guess as the validation verdict.
- DO NOT describe the toy witness byte equality as signature verification.
- DO NOT claim existing CKB VM RV64 script binaries run on this RV32 processor.
- DO NOT implement GPU quantization, alter compiled circuit weights or change arithmetic dtype without proving bitwise equivalence.
- DO NOT use real keys, wallet seed phrases, live funds, node credentials or chain broadcasts.
- DO NOT make PoC depend on external cloud or on-chain data. Local fixture inputs only.
- DO NOT introduce on-chain consensus claims, proof-of-execution claims, or security guarantees.
- DO NOT remove upstream lockstep/replay checks merely to achieve apparent output success.

## Implementation priorities

1. Make the 12 host oracle tests pass; these already pass at handoff, maintain them.
2. Pin qwenputer commit/model revision and get upstream reference `calc` working on the target machine.
3. Build `guest/ckb_bridge.c` with upstream linker/runtime map, or inspect the included Clang/LLD-built ELF32 and rerun its compiler recipe; fix compile/link issues without replacing neural execution with host execution.
4. Use upstream `chipformer.rv.elf.program`, `ref.Machine`, `bridge.park`, `bridge.Session` to run fixtures. Capture actual firmware response bytes and compare with fixture expectations.
5. Run `scripts/run_neural.py` and ensure an actual `ckb_verify` tool call triggers chip execution. If it doesn't, inspect `RvChat.generate` and introduce a **verified forced-text-token path that still passes tokens through the neural chip layers**; never fake a tool response.
6. Produce comparison logs for at least 5 model-layer fixtures, with `lock_error=None`, positive `chip_tokens`, `replay.ok=True`, and exact expected firmware response.
7. Run on 7900 XTX ROCm if technically supported; try safe slow HF path before CUDA Graph fast executor. Keep NVIDIA fallback optional and clearly labelled.
8. Write `reports/RESULTS.md` describing what was actually run, device/backend, failures, workarounds, and remaining architectural distance to real CKB-VM.

## First-run commands

- `python3 scripts/check_reference.py`
- `python3 scripts/preflight.py`
- Follow the numbered procedure in `HANDOFF.md`.

## Deliverables

- Working bare-metal RV32 ELF build recipe; source, linker/runtime references, reproducible build SHA.
- 12 reference-machine fixtures passing (or diagnosed blockers).
- 5+ true neural-layer cases passing with independent traces (or a rigorously documented `ROCM-BLOCKED` status).
- One small CLI to run one fixture and JSON output with fields `mode`, `device`, `model_rev`, `firmware_sha256`, `case_id`, `expected`, `actual`, `chip_tokens`, `elapsed_s`, `reference_match`, `replay_ok`, `status`.
- Results document and minimal rerun instructions.
- No untested claims.

## Decision gates

- **Gate 0:** upstream dependency/runtime usable; if not, record exact exception and retry with supported versions.
- **Gate 1:** software reference is exact; if not, debug ELF/load/bridge before GPU.
- **Gate 2:** neural layers actually execute; if model does not choose tool, add controlled token driving with lockstep.
- **Gate 3:** 5+ neural cases exact and replayed; only then mark PoC achieved.

## Definition of 'neural execution'

A documented program ELF is injected into the snapshot representing the extra neural CPU's ROM/RAM/registers; the model's compiled transformer layers generate its instruction-trace tokens and the specified result; the Python reference runner is only used as an oracle and lockstep check. No conventional host interpreter performs the computation on the model's behalf.

If blocked, preserve a working software-reference baseline and a precise plan for the missing piece. Prefer a reproducible partial success to fabricated completion.
