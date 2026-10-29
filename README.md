# CKB-NeuralVM — Research PoC handoff (2026-10-29)

**Purpose:** Demonstrate that a CKB-inspired validation program, compiled to RISC-V, executes **inside Qwenputer's actual neural CPU layers** and returns a result matching independent conventional computation.

**State at handoff:** Research, sample firmware, build helpers, a Clang-17 cross-compiled **static ELF32 artifact**, an ELF memory-map check, a Python oracle (12/12 passing), and experimental runners are supplied. **The Qwenputer reference machine, full model, neural GPU execution and ROCm compatibility have not been tested here.** A successfully linked ELF is not evidence it actually executed.

**This is NOT a CKB-VM implementation.** The first firmware is deliberately a 32-bit toy transaction check (1 input capacity, 1 output capacity and a matching 8-bit toy tag), with a Qwenputer-specific bridge. Never call it a valid CKB transaction check, cryptographic signature check or a CKB consensus proof.

## Entry points

- **`AGENTS.md`** — exact instructions to paste/hand to a coding agent.
- **`HANDOFF.md`** — architecture, build stages, exit criteria, CLI commands, pitfalls, AMD fallback.
- **`docs/RESEARCH.md`** — upstream sources, precise VM differences and longer-term CKB-VM roadmap.
- **`guest/ckb_bridge.c`** — initial standalone RV32 bare-metal firmware (draft; needs upstream build testing).
- **`fixtures/test_vectors.json`** — 12 host-oracle-tested fixtures.
- **`scripts/build_firmware.sh`** — GNU cross-compile against Qwenputer's own `rom.ld` and `crt0.S`.
- **`scripts/build_firmware_clang.sh`** — self-contained Clang+LLD cross-build fallback, verified to create ELF32 here.
- **`build/ckb_bridge_clang.elf`** — included sample ELF; static format/address map verified, not executed.
- **`scripts/check_reference.py`** — completely independent Python implementation (runs without model).
- **`scripts/run_upstream_reference.py`** — software `ref.Machine` firmware execution (requires upstream clone, RISC-V compiler, tokenizer).
- **`scripts/run_neural.py`** — experimental real model-layer runner (requires actual model and compatible GPU runtime).
- **`scripts/preflight.py`** — GPU/toolchain capability report.

## Offline test available right now

```bash
python3 scripts/check_reference.py
python3 scripts/check_elf_map.py build/ckb_bridge_clang.elf
```

## First build sequence on target Linux machine

```bash
# From a directory outside this package:
git clone https://github.com/divagr18/qwenputer.git
export QWENPUTER_DIR="$PWD/qwenputer"
cd "$QWENPUTER_DIR/experimental/riscv"
uv sync --extra hf
uv run pytest tests -q
# In another terminal, from this package root:
export QWENPUTER_DIR=/absolute/path/to/qwenputer
bash scripts/build_firmware.sh  # OR: bash scripts/build_firmware_clang.sh
# See HANDOFF.md for tokenizer and reference/real neural tests.
```

Do not install or download arbitrary torch builds without checking your ROCm version and compatibility matrix. Do not blindly run generic Hugging Face `pipeline()` or vLLM examples: their basic text-generation path does not establish that the neural CPU executes.

## Definition of success

1. Upstream Qwenputer `calc` demo runs and is checked on a supported GPU/backend (or AMD caveat recorded).
2. Compiled firmware executes through upstream `ref.Machine` and passes the 12 fixtures.
3. **At least five selected fixtures** execute through **actual Qwenputer model layers** with nonzero circuit tokens; expected response is present, replay confirms zero discrepancies and the reference lockstep reports no error.
4. Logs include upstream commit SHA, model revision, device details, backend, toolchain, ELF SHA256, exact cases, output, circuit token count and execution time.

If 3 is blocked, mark outcome `CPU-REFERENCE-ONLY` or `ROCM-BLOCKED` rather than announcing a successful neural PoC.
