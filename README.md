# CKB-NeuralVM — research PoC

A toy CKB-*inspired* validation program, compiled to RV32IM, executed **inside the compiled neural
CPU layers** of [Qwenputer](https://github.com/divagr18/qwenputer) (Qwen3.5-4B-Qwenputer) and checked
against an independent Python oracle and upstream's software reference machine.

**Status (2026-10-30): Gate 3 reached on AMD ROCm (RX 7900 XTX), on both the slow HF path and the
fast compacted path (~45 s per case instead of ~3.5 min).** 11 of 12 fixtures
ran exactly on the neural chip layers, with live lockstep against `ref.Machine` and a full IR replay
both clean. The 12th (`leading_noise`) failed because Qwen dropped a leading space when writing the
tool call; the CPU executed what it was given correctly. Full write-up, evidence and rerun commands:
**[`reports/RESULTS.md`](reports/RESULTS.md)**.

**This is NOT a CKB-VM implementation.** The firmware is deliberately a 32-bit toy check (1 input
capacity, 1 output capacity and a matching 8-bit toy tag) with a Qwenputer-specific I/O bridge. It
is not a valid CKB transaction check, not signature verification, not RV64, not real CKB syscalls,
and makes no consensus, proof-of-execution or security claim.

## Results at a glance

| | |
|---|---|
| Python oracle | 12/12 |
| Upstream `ref.Machine` (GNU and Clang ELFs) | 12/12 each |
| Upstream `calc` demo on ROCm | fails without adapter (lockstep divergence at chip token 15); passes with it |
| Neural fixtures, slow path (`--slow`) | 11 PASS, 1 INPUT_MISMATCH; ~3.1–3.9k chip tokens and ~3.5 min each |
| Neural fixtures, fast path (default) | identical results and chip token counts; 41–52 s each |

**ROCm finding.** The circuit is exact only if GEMMs accumulate in fp32. Upstream enforces that with
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`, which has **no effect on
ROCm**, so bf16 GEMMs flip bits the circuit depends on. `scripts/rocm_exact.py` makes only the chip
layers' `nn.Linear`s (and the fast path's compacted `SparseLinear`s) compute in fp32 and round once to bf16. Weights, the circuit and Qwen's text
layers are unchanged. Bitwise equivalence was shown with upstream's strict gpucheck (0/10 → 10/10
exact, dense and compacted) and with lockstep + replay on every neural run.

**Not covered yet:** the Triton/CUDAGraph executor, a neural run of the Clang ELF, and the
forced-token feed that would rescue `leading_noise`.

## Repository layout

- **`AGENTS.md`** — rules for a coding agent working on this PoC (what must not be claimed or changed).
- **`HANDOFF.md`** — architecture, build stages, gates, pitfalls, AMD fallback.
- **`docs/RESEARCH.md`** — upstream sources, VM differences and the longer-term CKB-VM roadmap.
- **`guest/`** — bare-metal RV32 firmware (`ckb_bridge.c`) plus linker script and startup code for the Clang build.
- **`fixtures/test_vectors.json`** — 12 fixtures with expected firmware output.
- **`scripts/check_reference.py`**, **`reference_rules.py`** — independent Python oracle (no model needed).
- **`scripts/build_firmware.sh`** — GNU cross-compile against Qwenputer's own `rom.ld`/`crt0.S`
  (`RISCV_MARCH=rv32im` for GCC < 12, which lacks Zba).
- **`scripts/build_firmware_clang.sh`** — self-contained Clang+LLD build (`rv32im_zba`).
- **`scripts/check_elf_map.py`** — checks the ELF's segments fit Qwenputer's address map.
- **`scripts/run_upstream_reference.py`** — runs the ELF on upstream's software `ref.Machine`.
- **`scripts/run_calc_smoke.py`** — upstream `calc` baseline, with the ROCm adapter.
- **`scripts/run_neural.py`** — runs one fixture through the model layers; prints a JSON record and appends it to `reports/runs.jsonl`.
- **`scripts/rocm_exact.py`** — the ROCm fp32-accumulate adapter.
- **`scripts/gpucheck_compact.py`** — upstream strict bit-exact check on the compacted (fast-path) chip model.
- **`scripts/diag_gpu_divergence.py`** — CPU-vs-GPU per-module diff used to find the ROCm divergence.
- **`scripts/preflight.py`** — GPU/toolchain capability report.
- **`reports/`** — `RESULTS.md`, per-run manifests (`runs.jsonl`) and raw logs.

Built ELFs are not committed (`build/` is ignored). Rebuild them; the SHA-256s of the ELFs used are
recorded in `reports/RESULTS.md`.

## Quick start

No model or GPU needed:

```bash
python3 scripts/check_reference.py
```

Full reproduction (upstream clone, model download, GPU):

```bash
git clone https://github.com/divagr18/qwenputer.git
export QWENPUTER_DIR="$PWD/qwenputer"
RISCV_MARCH=rv32im bash scripts/build_firmware.sh     # or: bash scripts/build_firmware_clang.sh
python3 scripts/check_elf_map.py build/ckb_bridge.elf
```

Then follow **Rerun** in `reports/RESULTS.md` for the reference, `calc` baseline and neural runs.

On **ROCm**, do not run upstream's `uv sync --extra hf`: its lockfile pins the PyPI (CUDA) torch build.
Install the same torch version from `https://download.pytorch.org/whl/rocm<version>` for your ROCm
release, and put `$QWENPUTER_DIR/experimental/riscv` on `PYTHONPATH`. Generic Hugging Face
`pipeline()` or vLLM text generation does not exercise the neural CPU and proves nothing here.

## Definition of success

1. Upstream Qwenputer `calc` demo runs and is checked on the GPU/backend (or the AMD caveat recorded).
2. Compiled firmware executes through upstream `ref.Machine` and passes the 12 fixtures.
3. **At least five fixtures** execute through **actual Qwenputer model layers** with nonzero chip
   tokens; the expected response is present, replay confirms zero discrepancies and the reference
   lockstep reports no error.
4. Logs include upstream commit SHA, model revision, device, backend, toolchain, ELF SHA-256, cases,
   output, chip token count and execution time.

If 3 is blocked, the outcome is `CPU-REFERENCE-ONLY` or `ROCM-BLOCKED`, not a neural PoC.

## License

MIT, see [`LICENSE`](LICENSE). `guest/start.S` is adapted from Qwenputer's MIT-licensed `crt0.S`;
its notice is reproduced in `LICENSE`. The scripts drive an unmodified upstream clone rather than
vendoring it.
