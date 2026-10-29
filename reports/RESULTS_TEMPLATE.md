# CKB-NeuralVM PoC experiment results

**Status:** NOT RUN ON TARGET HARDWARE. Fill in with observed results only.

## Environment

- Date:
- Host OS / kernel:
- CPU / RAM:
- GPU / VRAM:
- ROCm / HIP:
- Torch / Transformers / Triton:
- qwenputer git SHA:
- model ID / immutable revision:
- RISC-V GCC version:
- ELF SHA256:

## Gate record

| Gate | Status | Evidence / exact command / location |
|---|---|---|
| Python fixture oracle | PASS in handoff environment (12/12) | `python3 scripts/check_reference.py` |
| Upstream calc baseline | NOT RUN | |
| ELF compiled | PASS Clang17+LLD in handoff environment; not model-executed | `build/ckb_bridge_clang.elf` / `scripts/check_elf_map.py` |
| `ref.Machine` firmware fixtures | NOT RUN | |
| Actual neural CPU verified | NOT RUN | |
| 5+ model-layer cases match | NOT RUN | |

## Fixture results

Record `case_id`, mode (`python_oracle` / `qwenputer_ref` / `neural_hf` / `neural_executor`), expected and actual stdout bytes, chip tokens, elapsed s, lockstep result, replay status, stderr, and device for each case.

## Blockers

Write precise error, logs, provenance, smallest repro, and next action. Use status labels from `HANDOFF.md`.

## Conclusions

Which exact claim is supported by evidence? Which remains untested? Do **not** claim CKB consensus or cryptographic validation.
