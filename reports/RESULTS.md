# CKB-NeuralVM PoC — experiment results

**Status: PoC achieved (Gate 3) on AMD ROCm, on both the slow HF path and the fast (compacted +
CUDA-graph) path, with one documented arithmetic adapter.**
11 of 12 fixtures executed exactly on Qwenputer's neural RV32 CPU with lockstep and IR replay clean;
the 12th (`leading_noise`) failed at Qwen's *text transcription* of the payload, not in execution.

What this shows: a toy CKB-*inspired* capacity/tag check, compiled to RV32IM, runs inside the
compiled (non-learned) chip layers of Qwen3.5-4B-Qwenputer on an RX 7900 XTX, and its output bytes
are exactly the independent Python oracle's. It is **not** CKB-VM, not RV64, not real CKB syscalls,
not signature verification, and not a consensus or security claim.

## Environment

| | |
|---|---|
| Date | 2026-10-29 |
| Host | Ubuntu 22.04.5 LTS, kernel 6.8.0-106-generic |
| CPU / RAM | Intel i7-14700KF / 62 GiB |
| GPU | AMD Radeon RX 7900 XTX (gfx1100), 24 GiB; ~3.6 GiB used by other processes (ComfyUI, desktop) during runs |
| ROCm / HIP | ROCm 7.2.2; torch HIP 7.2.53211 |
| Python env | `/data/qwenputer/venv-rocm`, Python 3.12.13 |
| Torch / Transformers / Triton | 2.14.1+rocm7.2 / 5.19.0 / triton-rocm 3.8.0; tokenizers 0.23.3 |
| Kernels | HF torch fallbacks (no `flash-linear-attention`, no `causal_conv1d`) |
| qwenputer | `ca13310e2f9f788d1dd3feeea8213c1b0c23913c` (unmodified clone) |
| Model | `divagr1925/Qwen3.5-4B-Qwenputer` @ `2d9068508f2b680badb7abcded96fdd107e9ba5d` (BF16) |
| RISC-V GCC | riscv64-unknown-elf-gcc 10.2.0 (no Zba support → built `-march=rv32im`) |
| Clang | Ubuntu clang 17.0.6 + LLD 21.0.0 (swiftly's `ld.lld`, first on PATH) |
| ELF (GNU, used for neural runs) | `build/ckb_bridge.elf` sha256 `05dff6da06e4b9b268d71fc474b5c7327af475e1cbebbd255d6291a9190c5631` |
| ELF (Clang, rv32im_zba) | `build/ckb_bridge_clang.elf` sha256 `d71d1d6323b00509f45fe440fdf051c88d86e2a469e2f265ad1a275ab96c056c` |

Torch note: upstream `uv.lock` pins `torch==2.14.1` from **PyPI**, which is a CUDA build. The same
version was installed from `https://download.pytorch.org/whl/rocm7.2` instead; `uv sync --extra hf`
was deliberately not used. `chipformer` is used via `PYTHONPATH`, not installed.

## Gate record

| Gate | Status | Evidence |
|---|---|---|
| Python fixture oracle | **PASS 12/12** | `python3 scripts/check_reference.py` |
| Upstream test suite | **231 passed, 1 failed, 22 skipped** | `reports/logs/pytest-full.log`. Failure: `test_rv_fuzz.py::test_fuzz_free_running_cpu` → `KeyError: 'Q'` in `fastexec.FreeRun` (stale token family in upstream's free-run executor; not used by `RvChat`). Skips: tokenizer not in HF cache (6 files), `RV_SLOW` long traces |
| ELF compiled (GNU, upstream `rom.ld`/`crt0.S`) | **PASS** | `RISCV_MARCH=rv32im bash scripts/build_firmware.sh`; opcode audit: RV32I only, no C/CSR |
| ELF compiled (Clang) | **PASS** | `RISCV_CLANG=clang-17 bash scripts/build_firmware_clang.sh` |
| `ref.Machine` firmware fixtures (Gate 1) | **PASS 12/12 for both ELFs** | `scripts/run_upstream_reference.py` |
| Upstream calc baseline on ROCm, no adapter | **TRACE-MISMATCH** | `reports/logs/calc-slow.log`: correct tool call, 15 chip tokens, then lock error `F0(24012)` vs `F0(32716)`; replay `ok=False` |
| Upstream calc baseline on ROCm, with adapter (Gate 0) | **PASS** | `reports/logs/calc-slow-adapter.log`: `121932631112635269` (correct), 2430 chip tokens, lock error None, replay ok, 0 wrong of 2460 owned |
| Actual neural CPU (Gate 2) | **PASS** | model chose `ckb_verify` unaided in 12/12 runs; no forced-token path needed |
| 5+ model-layer cases exact (Gate 3) | **PASS 11/12** (all 5 required) | `reports/runs.jsonl`, `reports/logs/neural-*.log` |
| Same, fast path (2026-10-30) | **PASS 11/12**, ~4.6× faster | `reports/runs.jsonl` (`neural_hf_fast`), `reports/logs/neural-*-fast.log` |

## ROCm blocker and fix (root cause, not workaround)

**Symptom.** On the slow HF path the neural CPU starts correctly and diverges after 15–21 chip tokens.

**Isolation** (one variable at a time, upstream `chipformer.rv.gpucheck`, strict bf16 = bit-exact vs the IR):

1. bf16 on **CPU**: exact on all programs/modes → circuit and HF modules are correct.
2. bf16 on ROCm, hipBLASLt: mismatch everywhere (residues like −2⁻²⁴ where the IR has exactly 0).
3. bf16 on ROCm, rocBLAS (`TORCH_BLAS_PREFER_HIPBLASLT=0`): identical mismatch → not the BLAS library choice.
4. Per-module CPU-vs-GPU hook diff (`scripts/diag_gpu_divergence.py`): first divergence is
   `layers.0.linear_attn.in_proj_qkv`, a bf16 `Linear` whose **input is bit-identical** on both devices.
5. Minimal repro on that GEMM (K=2816): CPU bf16 == fp32-accumulate-then-round-once; ROCm bf16 differs
   by up to 2.0 regardless of `allow_bf16_reduced_precision_reduction`; ROCm fp32 GEMM rounded to
   bf16 == CPU bit-for-bit.

**Root cause.** The circuit is exact only if GEMMs accumulate in fp32 (upstream
`q35.fast.exact_matmuls`). On CUDA that is enforced by a cuBLAS flag. **On ROCm that flag has no
effect**, so bf16 GEMMs round partial sums and flip the circuit's threshold dims.

**Fix.** `scripts/rocm_exact.py` wraps the `nn.Linear`s of the chip layers only (88 modules in 11
layers): upcast the same bf16 inputs/weights exactly to fp32, GEMM, round once to bf16. No weights,
stored dtypes or circuit are changed; Qwen's text layers are untouched.

**Bitwise-equivalence proof.** Upstream strict bf16 gpucheck with the adapter: exact on
signext/call/memcpy/rv32ui-lb/c-crc32, prefill and decode (10/10; up to 4166 positions), versus
0/10 without. Then full-model lockstep + IR replay on every neural run below.

**Negative control.** `valid_fee` with `--no-adapter`: lock error at chip token 21, same
`F0(24012)`≠`F0(32716)` signature. (Its manifest line says `NOT_EXERCISED`; the runner has since been
changed to label lockstep divergence `TRACE_MISMATCH`.)

## Neural fixture results (mode `neural_hf`, `--slow`, adapter on, GNU ELF)

| case | status | output (firmware putchar) | payload transcribed exactly | chip tokens | elapsed s | lock_error | replay |
|---|---|---|---|---|---|---|---|
| valid_fee | PASS | `PASS;FEE=000000003B9ACA00` | yes | 3881 | 215.2 | None | ok |
| valid_exact | PASS | `PASS;FEE=0000000000000000` | yes | 3839 | 216.0 | None | ok |
| lock_mismatch | PASS | `FAIL;R=LOCK` | yes | 3558 | 202.4 | None | ok |
| capacity_fail | PASS | `FAIL;R=CAP` | yes | 3510 | 201.4 | None | ok |
| max_u64 | PASS | `PASS;FEE=FFFFFFFFFFFFFFFF` | yes | 3662 | 203.8 | None | ok |
| zero_fee_zero_cap | PASS | `PASS;FEE=0000000000000000` | yes | 3839 | 215.7 | None | ok |
| bad_hex | PASS | `FAIL;R=FMT` | yes | 3094 | 186.4 | None | ok |
| truncated | PASS | `FAIL;R=FMT` | yes | 3306 | 204.5 | None | ok |
| trailing | PASS | `FAIL;R=FMT` | yes | 3708 | 228.1 | None | ok |
| lowercase | PASS | `FAIL;R=FMT` | yes | 3094 | 197.4 | None | ok |
| max_fee | PASS | `PASS;FEE=FFFFFFFFFFFFFFFE` | yes | 3686 | 207.1 | None | ok |
| leading_noise | **INPUT_MISMATCH** | `PASS;FEE=0000000000000000` | **no** — leading space dropped | 3839 | 215.2 | None | ok |

`leading_noise`: Qwen wrote `V1;I=…` without the fixture's leading space. The CPU executed exactly
what it was given (lockstep and replay clean; the oracle on the transcribed payload also says
`PASS;FEE=0…0`). This is a text-layer transcription failure. The manifest line says `FAIL`; the
runner now checks the transcribed payload and labels this `INPUT_MISMATCH`.

Replays: 0 wrong predictions and 0 owner mismatches on every run (~3.5–4.4k positions each). Each
run is one tool call (`runs: 1`). Throughput on this path is ~18 chip tokens/s.

## Fast path (mode `neural_hf_fast`, 2026-10-30)

The default `RvChat(fast=True)` path builds a chip-only view whose Linears `fast.compact(qwen=False)`
replaces with upstream `SparseLinear` (nonzero-block GEMMs) and runs one CUDA graph per token via
`torch.compile(mode="reduce-overhead")`. Compaction happens inside `RvChat.__init__`, so the first
adapter wrapped none of these modules: the fast path ran unprotected bf16 GEMMs on ROCm.

**Fix.** `rocm_exact.install` now also wraps `SparseLinear` (same fp32-accumulate, round-once rule,
applied to its block and dense-prefill products), and `rocm_exact.chip_modules(chat)` includes the
fast path's chip view `chat.cm`.

**Bitwise proof first** (`scripts/gpucheck_compact.py`: upstream's strict bf16 `hf.check` on the
small chip model after `fast.compact(qwen=False)`, mirroring `_build_fast`):

- old adapter (wraps only the 7 remaining `nn.Linear`s): signext/call, prefill and decode, **0/4 exact**;
- new adapter (95 modules): signext/call/memcpy/rv32ui-lb/c-crc32, prefill and decode, **10/10 exact**
  (up to 4166 positions).

**Full model** (same GNU ELF, adapter on, default fast path; `torch.compile` + HIP graphs worked unchanged):

| case | status | output | chip tokens | fast s | slow s |
|---|---|---|---|---|---|
| valid_fee | PASS | `PASS;FEE=000000003B9ACA00` | 3881 | 45.4 | 215.2 |
| valid_exact | PASS | `PASS;FEE=0000000000000000` | 3839 | 43.0 | 216.0 |
| lock_mismatch | PASS | `FAIL;R=LOCK` | 3558 | 41.2 | 202.4 |
| capacity_fail | PASS | `FAIL;R=CAP` | 3510 | 46.7 | 201.4 |
| max_u64 | PASS | `PASS;FEE=FFFFFFFFFFFFFFFF` | 3662 | 41.1 | 203.8 |
| zero_fee_zero_cap | PASS | `PASS;FEE=0000000000000000` | 3839 | 46.6 | 215.7 |
| bad_hex | PASS | `FAIL;R=FMT` | 3094 | 43.8 | 186.4 |
| truncated | PASS | `FAIL;R=FMT` | 3306 | 50.8 | 204.5 |
| trailing | PASS | `FAIL;R=FMT` | 3708 | 52.0 | 228.1 |
| lowercase | PASS | `FAIL;R=FMT` | 3094 | 50.7 | 197.4 |
| max_fee | PASS | `PASS;FEE=FFFFFFFFFFFFFFFE` | 3686 | 45.5 | 207.1 |
| leading_noise | INPUT_MISMATCH | `PASS;FEE=0000000000000000` (leading space dropped again) | 3839 | 46.6 | 215.2 |

Every run: lock error None, replay ok, 0 wrong predictions. Chip token counts equal the slow
path's case for case. Elapsed includes the first-call graph capture.

**Negative control.** `valid_fee --no-adapter` on the fast path: `TRACE_MISMATCH` at chip token 21,
the same `F0(24012)` ≠ `F0(32716)` signature as the slow path.

## Clang ELF on the neural layers (fast path, 2026-10-30)

`build/ckb_bridge_clang.elf` (clang-17 + LLD, `rv32im_zba`, sha256 `d71d1d63…6c056c`): **11/12 PASS**
with exactly the same outputs as the GNU ELF, lock error None, replay ok and 0 wrong predictions
on every run; `leading_noise` is INPUT_MISMATCH for the same transcription reason. Chip token counts
are ~10% lower (e.g. `valid_fee` 3441 vs 3881; range 2817–3441) from Zba and Clang's codegen.
Logs: `reports/logs/neural-*-clang-fast.log`.

## Controlled text-token feed (HANDOFF §8, 2026-10-30)

`scripts/forced_feed.py` (`run_neural.py --force-call`) subclasses `RvChat` without modifying
upstream. Every forward pass is unchanged, so each forced token runs through the chip layers and
Qwen's layers after the firmware snapshot is injected. Only the *returned logits* are replaced by
a one-hot choice of the next token of the exact tool call, and only while the circuit does not own
the position. The model's own first token must open the call (it did: `<tool_call>`); after the
last forced token (`</tool_call>`) the circuit owns GO and its output exactly as unforced. The run
fails as `FORCE_FAILED` unless the generated stream begins with exactly the forced tokens and the
circuit never claimed a forced slot. Lockstep and replay stay on.

| case | ELF | status | output | forced tokens | chip tokens | lock / replay |
|---|---|---|---|---|---|---|
| valid_fee (positive control) | GNU | PASS | `PASS;FEE=000000003B9ACA00` | 74 | 3881 (= unforced) | None / ok, 0 wrong |
| leading_noise | GNU | **PASS** | `FAIL;R=FMT` | 75 | 2855 | None / ok, 0 wrong |
| leading_noise | Clang | **PASS** | `FAIL;R=FMT` | 75 | 2645 | None / ok, 0 wrong |

With the exact payload (leading space included) reaching the CPU, the firmware rejects it as the
oracle does. **All 12 fixtures now pass on the neural layers**, 11 unaided and `leading_noise` via
the controlled feed. Logs: `reports/logs/neural-*-forced-*.log`.

## Triton executor (2026-10-30)

Upstream's executor (`tritonexec.Executor2`) is built from the circuit **IR** (`core.build`), not
from the checkpoint's chip-layer weights: it is a fast interpreter of the circuit. Per AGENTS.md
an executor-only run is therefore *not* neural inference, and `run_neural.py --executor` labels
its mode `circuit_executor_not_neural`. Not reported as a result here.

What was tested is upstream's `executor="verify"` (`--executor-verify`): the real-weights fast chip
view produces every token, and the executor runs alongside with its residual compared on every
token. `valid_fee`: **PASS**, `verify_mismatch` 0 over 3881 chip tokens, lock None, replay ok, 56.6 s.
So the Triton kernels and `torch.cuda.CUDAGraph` work on HIP and agree with the weights bit-for-bit
on this run. Log: `reports/logs/neural-valid_fee-executor-verify.log`.

## Not done / open

- Executor-only timing was not measured (it is not neural inference); only verify mode was run.
- The forced feed and executor verify were run on the cases above, not the full 12 × 2 matrix.
- Upstream `fastexec` `KeyError: 'Q'` not investigated (outside the `RvChat` path).
- Margin: the longest tool call (`trailing`) uses 79 of upstream's 80-row capture window; longer
  payloads silently overwrite the last row (`rows[min(cap, CAP_MAX)]`).

## Distance to real CKB-VM (unchanged by this result)

RV32IM(+Zba) vs CKB's RV64IMC + B extensions; no ELF64 loader, 4 MiB memory model, cycle model or
versioned syscalls (2052/2071/2074 and partial-loading semantics); no Molecule, script groups,
hashing or signatures. The payload's `L`/`W` byte equality is a toy tag check, not authentication.
See `HANDOFF.md` §10–11 for the staged roadmap.

## Rerun

```bash
export QW=/home/phill/qwencupter/qwenputer/experimental/riscv POC=/home/phill/qwencupter/ckb-neuralvm-poc
PY=/data/qwenputer/venv-rocm/bin/python M=/data/qwenputer/qwenputer-rv
python3 $POC/scripts/check_reference.py
RISCV_MARCH=rv32im QWENPUTER_DIR=/home/phill/qwencupter/qwenputer bash $POC/scripts/build_firmware.sh
cd $QW && export PYTHONPATH=$QW:$POC/scripts
$PY $POC/scripts/run_upstream_reference.py --elf $POC/build/ckb_bridge.elf --tokenizer $M/tokenizer.json
$PY $POC/scripts/run_calc_smoke.py $M --slow                       # Gate 0 baseline
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee --slow
$PY $POC/scripts/gpucheck_compact.py --decode                    # fast-path bitwise proof
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case leading_noise --force-call
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee --executor-verify
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee   # fast path
```

`run_neural.py` prints one JSON object (`mode`, `device`, `model_rev`, `firmware_sha256`,
`case_id`, `expected`, `actual`, `chip_tokens`, `elapsed_s`, `reference_match`, `replay_ok`,
`status`, plus lock/replay/payload detail) and appends it to `reports/runs.jsonl`.
