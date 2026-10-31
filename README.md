# CKB-NeuralVM — research PoC

A toy CKB-*inspired* validation program, compiled to RV32IM, executed **inside the compiled neural
CPU layers** of [Qwenputer](https://github.com/divagr18/qwenputer) (Qwen3.5-4B-Qwenputer) and checked
against an independent Python oracle and upstream's software reference machine.

**Status (2026-10-30): Gate 3 reached on AMD ROCm (RX 7900 XTX), on both the slow HF path and the
fast compacted path (~45 s per case instead of ~3.5 min), with both the GNU and Clang ELFs.** All
12 fixtures run exactly on the neural chip layers, with live lockstep against `ref.Machine` and a
full IR replay both clean: 11 unaided, and `leading_noise` (where Qwen drops a leading space when
writing the tool call) via the HANDOFF §8 controlled token feed.

**v0.2 (HANDOFF §10):** the firmware now also reads a small **host-injected transaction
context** (2 inputs, 2 outputs, script args, witness) through a toy loader API whose partial-load
rules and return codes follow ckb-script's `store_data`. All 22 context fixtures run exactly on the
neural layers, covering overflow, missing witness, out-of-range indices and partial reads.

**v0.3 (HANDOFF §10, modified circuit):** the firmware reads the context through real `ecall`s
with CKB syscall numbers 2052/2071/2074. Upstream's circuit faults on those, so
`patches/qwenputer-ecall-trap.patch` makes the circuit jump to a firmware handler instead, and
the checkpoint is rebuilt locally with upstream's own merge. Its unpatched control build matches
the published chip layers byte for byte; the patched one differs from that control only in chip-layer
MLPs (from the published one also in 8 Qwen attention keys fitted at merge time; see RESULTS). All 22
syscall fixtures run exactly on it; the published checkpoint faults on them, as expected. This is
a **modified local checkpoint**, labelled as such in every run. Full write-up, evidence and rerun
commands: **[`reports/RESULTS.md`](reports/RESULTS.md)**.

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
| Neural fixtures, Clang ELF (fast) | 11 PASS, 1 INPUT_MISMATCH; ~10% fewer chip tokens |
| `leading_noise` with `--force-call` | PASS (`FAIL;R=FMT`) on both ELFs; forced `valid_fee` control unchanged |
| Triton executor, verify mode | matches real-weights residual on every token (0 mismatches) |
| v0.2 context fixtures: oracle / native C / `ref.Machine` | 22/22 each; native C also 24,000 random API calls, 0 mismatches |
| v0.3 syscall fixtures, neural (fast, patched checkpoint) | 22/22 PASS on GNU; published checkpoint faults on them (negative control) |
| v0.2 context fixtures, neural (fast) | 22/22 PASS on GNU, 7/7 run on Clang PASS; 1.1k–7.2k chip tokens, 16–95 s |

**ROCm finding.** The circuit is exact only if GEMMs accumulate in fp32. Upstream enforces that with
`torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False`, which has **no effect on
ROCm**, so bf16 GEMMs flip bits the circuit depends on. `scripts/rocm_exact.py` makes only the chip
layers' `nn.Linear`s (and the fast path's compacted `SparseLinear`s) compute in fp32 and round once to bf16. Weights, the circuit and Qwen's text
layers are unchanged. Bitwise equivalence was shown with upstream's strict gpucheck (0/10 → 10/10
exact, dense and compacted) and with lockstep + replay on every neural run.

**Executor caveat:** upstream's Triton executor runs the circuit IR, not the checkpoint weights, so
an executor-only run is not neural inference and is labelled `circuit_executor_not_neural`. Only
its verify mode (weights produce every token, executor cross-checked) is reported.

## Repository layout

- **`AGENTS.md`** — rules for a coding agent working on this PoC (what must not be claimed or changed).
- **`HANDOFF.md`** — architecture, build stages, gates, pitfalls, AMD fallback.
- **`docs/RESEARCH.md`** — upstream sources, VM differences and the longer-term CKB-VM roadmap.
- **`guest/`** — bare-metal RV32 firmware: `bridge.c` (Qwenputer I/O bridge), `parse.c` (v0.1 payload),
  `ctx.c`/`ctx.h` (toy context API, a shim and not CKB syscalls), `trap.c`/`syscall.h` (v0.3 syscall
  handler), `contract.c`, `emit.c`; `selftest.c` (bridge-free build); `host_main.c`
  (native test harness); linker script and startup code for the Clang build.
- **`fixtures/test_vectors.json`** — 12 v0.1 payload fixtures with expected firmware output.
- **`fixtures/ctx_vectors.json`** — 22 v0.2 context fixtures (payload `V2` or `V2;T`).
- **`fixtures/syscall_vectors.json`** — 22 v0.3 fixtures (payload `V3` or `V3;T`; need the patched circuit).
- **`patches/qwenputer-ecall-trap.patch`** — the circuit + reference-machine change for v0.3 (applied to a separate clone).
- **`scripts/gpucheck_trap.py`**, **`scripts/build_selftest.sh`** — patched-circuit check: IR vs reference trace, weights vs IR.
- **`scripts/compare_checkpoints.py`**, **`reports/checkpoints/`** — per-tensor checkpoint comparison and hash manifests.
- **`scripts/check_reference.py`**, **`reference_rules.py`**, **`reference_v2.py`** — independent Python oracles (no model needed).
- **`scripts/ctx_v2.py`** — encodes a fixture's context and injects it into a copy of the ELF.
- **`scripts/test_host_c.py`** — native C build of the guest sources vs the oracle (fixtures + random API calls).
- **`scripts/test_capture_guard.py`** — checks the 80-row tool-call capture guard.
- **`scripts/build_firmware.sh`** — GNU cross-compile against Qwenputer's own `rom.ld`/`crt0.S`
  (`RISCV_MARCH=rv32im` for GCC < 12, which lacks Zba).
- **`scripts/build_firmware_clang.sh`** — self-contained Clang+LLD build (`rv32im_zba`).
- **`scripts/check_elf_map.py`** — checks the ELF's segments fit Qwenputer's address map.
- **`scripts/run_upstream_reference.py`** — runs the ELF on upstream's software `ref.Machine`.
- **`scripts/run_calc_smoke.py`** — upstream `calc` baseline, with the ROCm adapter.
- **`scripts/run_neural.py`** — runs one fixture through the model layers; prints a JSON record and appends it to `reports/runs.jsonl`.
- **`scripts/rocm_exact.py`** — the ROCm fp32-accumulate adapter.
- **`scripts/forced_feed.py`** — HANDOFF §8 controlled tool-call feed (`run_neural.py --force-call`).
- **`scripts/gpucheck_compact.py`** — upstream strict bit-exact check on the compacted (fast-path) chip model.
- **`scripts/diag_gpu_divergence.py`** — CPU-vs-GPU per-module diff used to find the ROCm divergence.
- **`scripts/preflight.py`** — GPU/toolchain capability report.
- **`reports/`** — `RESULTS.md`, per-run manifests (`runs.jsonl`) and raw logs.

Built ELFs are not committed (`build/` is ignored). Rebuild them; the SHA-256s of the ELFs used are
recorded in `reports/RESULTS.md`. The v0.1 results' ELFs build from tag `v0.1-gate3`.

## Quick start

No model or GPU needed:

```bash
python3 scripts/check_reference.py
python3 scripts/test_host_c.py      # needs a native C compiler with ASan/UBSan
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

MIT, see [`LICENSE`](LICENSE). `guest/start.S` is adapted from Qwenputer's MIT-licensed `crt0.S`,
and `patches/qwenputer-ecall-trap.patch` modifies (and quotes) Qwenputer source; its notice is
reproduced in `LICENSE`. The scripts drive an unmodified upstream clone rather than
vendoring it.
