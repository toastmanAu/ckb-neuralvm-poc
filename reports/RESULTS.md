# CKB-NeuralVM PoC — experiment results

**Status: PoC achieved (Gate 3) on AMD ROCm, on both the slow HF path and the fast (compacted +
CUDA-graph) path, with one documented arithmetic adapter. v0.2 (HANDOFF §10, host-injected
transaction context) also runs exactly on the neural layers: 22/22 context fixtures.**
v0.1: 11 of 12 fixtures executed exactly on Qwenputer's neural RV32 CPU with lockstep and IR
replay clean; the 12th (`leading_noise`) failed at Qwen's *text transcription* of the payload,
not in execution, and passes with the HANDOFF §8 controlled feed. v0.2: see its section below.

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
| v0.1 ELF (GNU, used for neural runs; tag `v0.1-gate3`) | `build/ckb_bridge.elf` sha256 `05dff6da06e4b9b268d71fc474b5c7327af475e1cbebbd255d6291a9190c5631` |
| v0.1 ELF (Clang, rv32im_zba; tag `v0.1-gate3`) | `build/ckb_bridge_clang.elf` sha256 `d71d1d6323b00509f45fe440fdf051c88d86e2a469e2f265ad1a275ab96c056c` |

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

## v0.2: host-injected transaction context (HANDOFF §10, 2026-10-30)

**Result: all 22 context fixtures PASS on the neural layers (GNU ELF, fast path), and the 7 of
them run on the Clang ELF PASS too; every run with lockstep `lock_error` None, replay ok and the exact payload.**
The 12 v0.1 fixtures still pass on the refactored firmware (11 unaided, `leading_noise` via the
forced feed, as before).

**Firmware split** (`guest/`): `bridge.c` (Qwenputer ecalls and the text-bridge capture),
`parse.c` (the v0.1 payload), `ctx.c`/`ctx.h` (the virtual context and its API), `contract.c`
(contracts and dispatch), `emit.c` (output). The payload selects the contract: `V2` runs the toy
lock over the context, `V2;T` prints an API probe transcript, anything else is the v0.1 check.

**The context is host-injected, not sent through the tool call.** It is a 128-byte read-only
object at the DROM symbol `ckb_ctx` (layout in `guest/ctx.h`). `scripts/ctx_v2.py` writes a
fixture's context into a copy of the ELF, so `ref.Machine`, the live lockstep and
`RvChat.replay` all load the same bytes and upstream code stays unmodified. This mirrors CKB,
where the transaction comes from the host, not from the script's caller; it also means the tool
call carries only `V2` (26–28 of 80 capture rows) instead of a growing transcription.
The injected ELFs differ from their base build only in those 128 bytes: `code_sha256` (the ELF
with `ckb_ctx` zeroed) is the same for all 22 fixtures of a build, checked by
`run_upstream_reference.py` and recorded per run.

| | GNU (`-march=rv32im`) | Clang (`rv32im_zba`) |
|---|---|---|
| base ELF sha256 | `2a19a0b6a4f379d5611efc1f97bdc6ff3df66a65d8c3d074fb0ed7b79f2116c5` | `4cec66eaa22e6a45c8b25ece04f74b016feb30c04571a769d317a3b3d3c22306` |
| code sha256 (context zeroed) | `2110a697a6afa86ea5e43f8082fde923d7b7e64b2b246ae96676f4772b5d5030` | `e986c5e34edec88dd678569466ab26fcd61f6283e24c7e6a2636bb8a004bc4ee` |

**API (a shim, not CKB syscalls).** `toy_load_script_args`, `toy_load_witness`,
`toy_load_cell_capacity(index, source)` over 2 inputs and 2 outputs, and
`toy_partial_load(dst, *len, offset, source)`, which the two byte loaders are built on. Return
codes (`0` success, `1` INDEX_OUT_OF_BOUND, `2` ITEM_MISSING) and the partial-load rule are taken
from ckb-script 1.1.1 (`syscalls/mod.rs`, `syscalls/utils.rs::store_data`): the offset clamps to
the item length, `*len` returns the full size remaining after the offset, and
min(`*len` in, remaining) bytes are copied. Toy departures: an unknown source returns
INDEX_OUT_OF_BOUND (CKB aborts the VM); there is one witness and no witness index, and a missing
witness returns ITEM_MISSING (ckb-script returns INDEX_OUT_OF_BOUND for an absent witness index);
an invalid context makes every loader return ITEM_MISSING; no ecall numbers, Molecule, script
groups or hashes.

**Contract `V2`** (toy, not authentication): valid context (`CTX`), witness present (`NOWIT`),
args == witness byte for byte, compared in 4-byte partial loads (`LOCK`), at least one input
(`NOIN`), no u64 overflow in either sum (`OVF`), inputs ≥ outputs (`CAP`); else
`PASS;FEE=<inputs − outputs>`. Inputs and outputs are iterated until INDEX_OUT_OF_BOUND, as a CKB
script does.

**Differential testing, three layers before the model:**

1. *Python oracle* (`scripts/reference_v2.py`): written from the rules above and the fixture's
   structured fields, not from the encoded bytes, so it shares no code with the C. The contract
   fixtures' expected outputs were written by hand and the oracle is checked against them; the
   three probe transcripts are oracle-generated.
2. *Native C* (`scripts/test_host_c.py`): the same guest sources built for x86-64 with
   ASan/UBSan, run on all 34 fixtures and on 24,000 random API calls weighted toward the edges
   (offsets 0/len/len+1/UINT64_MAX, indices 0–3/UINT64_MAX, unknown sources, missing witness,
   invalid contexts, and raw header bytes that disagree with the listed items), compared with
   the oracle: 0 mismatches. This is not an independent
   implementation (same source), it is the C checked off-target. Mutation check: removing the
   offset clamp, making the index bound `>`, deleting the overflow check, or deleting either
   context length guard (`args_len`/`witness_len` ≤ 32, which keep loads inside the 128 bytes)
   each turns it red.
3. *Upstream `ref.Machine`* (`run_upstream_reference.py`): 12 v0.1 + 22 v0.2 fixtures, both ELFs,
   all pass.

**Neural runs** (`neural_hf_fast`, adapter on; chip tokens / wall time; logs
`reports/logs/neural-v2-<case>[-clang]-fast.log`):

| case | purpose | expected | GNU | Clang |
|---|---|---|---|---|
| `two_in_two_out` | 2 inputs, 2 outputs, fee 5 CKB | `PASS;FEE=000000001DCD6500` | 3308 / 38 s | 2524 / 31 s |
| `one_in_one_out` | v0.1 valid_fee as a context | `PASS;FEE=000000003B9ACA00` | 2883 / 33 s | — |
| `ctx_lock_mismatch` | toy tag differs | `FAIL;R=LOCK` | 1880 / 23 s | — |
| `witness_missing` | loader returns ITEM_MISSING | `FAIL;R=NOWIT` | 1391 / 18 s | 1220 / 18 s |
| `lock_len_mismatch` | args longer than witness | `FAIL;R=LOCK` | 1606 / 19 s | — |
| `long_lock_match` | 32-byte args == witness, 8 partial loads | `PASS;FEE=0000000000000004` | 5378 / 56 s | — |
| `long_lock_last_byte` | differs only in byte 31 (last chunk) | `FAIL;R=LOCK` | 4431 / 47 s | 4101 / 46 s |
| `input_sum_overflow` | inputs sum past UINT64_MAX | `FAIL;R=OVF` | 2281 / 27 s | 1797 / 23 s |
| `output_sum_overflow` | outputs sum past UINT64_MAX | `FAIL;R=OVF` | 2584 / 31 s | — |
| `max_sum_exact` | sums reach UINT64_MAX exactly, no overflow | `PASS;FEE=0000000000000000` | 3038 / 37 s | — |
| `ctx_capacity_fail` | outputs exceed inputs by 1 CKB | `FAIL;R=CAP` | 2931 / 34 s | — |
| `no_inputs` | first input load is INDEX_OUT_OF_BOUND | `FAIL;R=NOIN` | 1989 / 24 s | 1668 / 25 s |
| `no_outputs` | no outputs: whole input is fee | `PASS;FEE=000000001DCD6500` | 2658 / 33 s | — |
| `empty_lock` | zero-length args and witness (toy semantics) | `PASS;FEE=0000000000000001` | 2547 / 31 s | — |
| `bad_magic` | context magic wrong | `FAIL;R=CTX` | 1265 / 16 s | 1131 / 17 s |
| `bad_count` | n_inputs 3 > toy maximum 2 | `FAIL;R=CTX` | 1269 / 17 s | — |
| `bad_args_len` | args_len header 255 > 32: would read past the 128-byte context | `FAIL;R=CTX` | 1276 / 20 s | — |
| `bad_witness_len` | witness_len header 33 > 32 (and not 0xFF) | `FAIL;R=CTX` | 1280 / 18 s | — |
| `unpatched` | base ELF, no context injected | `FAIL;R=CTX` | 1253 / 18 s | — |
| `probe_long` | API probe: partial reads, clamped offsets, OOB indices | `T;0,20,;0,20,00112233;…` | 7217 / 95 s | 6332 / 88 s |
| `probe_missing` | API probe: empty args, missing witness, empty outputs | `T;0,00,;0,00,;0,00,;0,…` | 4996 / 59 s | — |
| `probe_bad_ctx` | API probe on an invalid context: every load ITEM_MISSING | `T;2,00,;2,04,;2,04,;2,…` | 3911 / 50 s | — |

The probe transcripts are the API's return codes, `*len` values and copied bytes for a fixed call
list (`contract.c::probe_v2` = `reference_v2.PROBES`), so the partial-read, clamped-offset,
out-of-range-index and unknown-source rules themselves ran on the neural CPU and matched the
oracle byte for byte, not only the contracts built on them. Their expected strings are
oracle-generated, so this shows the neural CPU, native C and oracle agree; the oracle's rule is
checked against ckb-script only by reading its source, not by running it.

**Firmware build note.** GCC 10 placed a `const volatile` array in writable `.sdata` (RAM 0x100)
while Clang put it in `.rodata`; the context is now a plain `const` array read through a volatile
pointer, which both compilers place in DROM (0x20070), with every read still a real load.

**Capture guard.** `run_neural.py` now refuses (status `BRIDGE_BLOCKED`, written to the manifest)
any call whose captured tokens exceed `CAP_MAX` = 80 or include a token longer than 16 bytes,
before loading the model. `scripts/test_capture_guard.py`: v0.1 fixtures use 58–79 rows, v0.2
26–28; a synthetic 80-row call is accepted and held intact in the bridge's text buffer, and the
81-row one is refused and is in fact corrupted in the buffer; a call containing a real 32-byte
vocabulary token (`-` × 32) is refused and that row is in fact cut to 16 bytes. Deleting either
branch of the guard turns the test red.

## Not done / open

- Executor-only timing was not measured (it is not neural inference); only verify mode was run.
- The forced feed and executor verify were run on the cases above, not the full 12 × 2 matrix.
- Upstream `fastexec` `KeyError: 'Q'` not investigated (outside the `RvChat` path).
- Margin: the longest tool call (`trailing`) uses 79 of upstream's 80-row capture window; longer
  payloads silently overwrite the last row (`rows[min(cap, CAP_MAX)]`). Now guarded (v0.2
  section): such calls are refused before the run instead of silently corrupted.
- v0.2 on Clang: 7 of the 22 context fixtures were run on the neural layers (all 22 on
  `ref.Machine`). The forced feed and executor verify were not repeated on v0.2.
- A neural ecall decoder for selected CKB syscall numbers (2052/2071/2074), the HANDOFF §10 step
  after this one, is not started.

## Distance to real CKB-VM (unchanged by these results)

RV32IM(+Zba) vs CKB's RV64IMC + B extensions; no ELF64 loader, 4 MiB memory model, cycle model or
versioned syscalls (2052/2071/2074 and partial-loading semantics); no Molecule, script groups,
hashing or signatures. The payload's `L`/`W` byte equality is a toy tag check, not authentication.
See `HANDOFF.md` §10–11 for the staged roadmap.

## Rerun

```bash
export QW=/home/phill/qwencupter/qwenputer/experimental/riscv POC=/home/phill/qwencupter/ckb-neuralvm-poc
PY=/data/qwenputer/venv-rocm/bin/python M=/data/qwenputer/qwenputer-rv
git -C $POC checkout v0.1-gate3   # only to reproduce the v0.1 ELF hashes above; skip for v0.2
python3 $POC/scripts/check_reference.py                          # 12 v0.1 + 22 v0.2 oracle checks
python3 $POC/scripts/test_host_c.py                             # native C vs oracle (v0.2)
RISCV_MARCH=rv32im QWENPUTER_DIR=/home/phill/qwencupter/qwenputer bash $POC/scripts/build_firmware.sh
cd $QW && export PYTHONPATH=$QW:$POC/scripts
$PY $POC/scripts/run_upstream_reference.py --elf $POC/build/ckb_bridge.elf --tokenizer $M/tokenizer.json
$PY $POC/scripts/run_calc_smoke.py $M --slow                       # Gate 0 baseline
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee --slow
$PY $POC/scripts/gpucheck_compact.py --decode                    # fast-path bitwise proof
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case leading_noise --force-call
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee --executor-verify
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case valid_fee   # fast path
$PY $POC/scripts/test_capture_guard.py --tokenizer $M/tokenizer.json --elf $POC/build/ckb_bridge.elf
$PY $POC/scripts/run_neural.py --model $M --elf $POC/build/ckb_bridge.elf --case two_in_two_out  # v0.2
```

`--case` takes any id from either fixture file (ids are unique across both); a v0.2 id makes the
runner inject that fixture's context into a copy of `--elf` under `build/ctx/` and run that.

`run_neural.py` prints one JSON object (`mode`, `device`, `model_rev`, `firmware_sha256`,
`case_id`, `expected`, `actual`, `chip_tokens`, `elapsed_s`, `reference_match`, `replay_ok`,
`status`, plus lock/replay/payload detail) and appends it to `reports/runs.jsonl`.
