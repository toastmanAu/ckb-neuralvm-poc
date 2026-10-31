# CKB-NeuralVM PoC — experiment results

**Status: PoC achieved (Gate 3) on AMD ROCm, on both the slow HF path and the fast (compacted +
CUDA-graph) path, with one documented arithmetic adapter. v0.2 (HANDOFF §10, host-injected
transaction context) also runs exactly on the neural layers: 22/22 context fixtures. v0.3: with a
locally patched and rebuilt circuit that decodes CKB syscall numbers 2052/2071/2074, 22/22 syscall
fixtures run exactly (a modified checkpoint, not the published one).**
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

## v0.3: CKB syscall numbers decoded by a patched circuit (HANDOFF §10, 2026-10-30)

**Result: with a locally rebuilt checkpoint whose circuit decodes ecall 2052/2071/2074, all 22
syscall fixtures PASS on the neural layers (fast path) with both the GNU and the Clang ELF, lockstep `None`, replay ok,
exact payload; the v0.2 and v0.1 checks pass on it too. The published checkpoint faults on the
same calls, as upstream's own reference machine predicts.** This is a **modified circuit** and a
**new local checkpoint**, not the published Qwen3.5-4B-Qwenputer; every run on it is labelled
`+trap_patched_circuit` and records the patch SHA instead of the published revision.

**The circuit change** (`patches/qwenputer-ecall-trap.patch`, applied to a separate clone at
`ca13310`; the upstream checkout is untouched). The stock circuit hard-faults on every ecall
except putchar (1) and exit (93) (`core.py`, `fault.7`). The patch adds 2052/2071/2074 to the
decoded a7 values, removes them from `fault.7`, and makes each a `J` record to a fixed ROM word
(`TRAP_WORD` = 0x10, address 0x40040): a jump with no link register and no CSR. The reference
machine (`ref.py`) gets the same rule, so lockstep and replay check it. Circuit size: 11 layers
either way; 2918 → 2921 dims, 42114 → 42126 FFN units, 1265 → 1268 slots; hidden 3968 unchanged.

**The firmware side.** `guest/start.S` puts `ckb_trap_entry` at 0x40040 (`.org 0x40`; both
build scripts now use this repo's `start.S` and `rom.ld` and fail if the symbol moves). Calls go
through `ckb_syscall` (a7 = number, then `ecall`); the trap leaves `ra` pointing at the caller,
so the handler (`guest/trap.c`) tail-returns there with the result in a0. Toy semantics, in
`guest/syscall.h`: 2052 returns the script args bytes (not a Molecule Script), 2071 the cell's
capacity as 8 bytes LE (not a CellOutput), 2074 the single witness at index 0; partial loading
and return codes as v0.2; an absent witness index, including a missing witness, is
INDEX_OUT_OF_BOUND as in ckb-script. RV32 pointers and offsets, sources 1/2 only. Contract `V3`
is the v0.2 lock with every read a syscall (same verdicts by construction, checked by the
fixtures); `V3;T` prints a syscall probe transcript.

**Checks, in the order they were run:**

1. *Oracle / native C*: `reference_v2.syscall` and the native build (`trap.c` called directly)
   agree on all 22 v0.3 fixtures (56/56 fixtures overall), and on the random calls, about 30% of
   which now go through `ckb_trap_dispatch` with 32-bit arguments as on RV32. Mutation check:
   dropping the missing-witness → INDEX_OUT_OF_BOUND remap or the witness index check turns it red.
2. *Reference machines*: on the **patched** `ref.Machine` all 56 fixtures pass on both ELFs. On
   **stock** upstream, the 17 v0.3 cases that reach a syscall die with fault `ecall` at 0x4004c
   (the `ecall` in `ckb_syscall`); the 5 invalid-context cases pass because they never make one.
3. *Circuit semantics* (`scripts/gpucheck_trap.py`, bridge-free self-test ELFs from
   `scripts/build_selftest.sh`): the patched IR's own predictions equal the reference trace at
   every position, and the HF weights equal the IR bit for bit, dense and compacted, prefill and
   token-by-token decode (4 cases, 552–5305 positions, 40 trap jumps; log
   `reports/logs/gpucheck-trap.log`).
4. *Checkpoint provenance* (`scripts/compare_checkpoints.py`, manifests in
   `reports/checkpoints/`): upstream's `chipformer.rv.merge` was run on base `Qwen/Qwen3.5-4B` @
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, first **unpatched** as a positive control:
   572 of 580 tensors are byte-identical to the published checkpoint, including all 11 chip
   layers, the embeddings and configs. So the positive control is **not** fully identical: the
   other 8 are the full-attention `k_proj` of Qwen's layers 14, 18, …, 42, each differing only in
   column 2776 (`mu_dim` in `chipformer.json`, the mute dim), in 280–418 of 4,063,232 elements,
   by ≤ 4.9e-4 (`reports/checkpoints/merge-trap-vs-published.detail.json`, from
   `compare_checkpoints.py --detail`). That column is written only by `q35.mute`'s anti-sink keys,
   which `fit_anti_sink` computes by running base Qwen on calibration prompts on whatever GPU the
   merge runs on (here ROCm, in bf16). Differing GPU numerics are the likely cause, consistent
   with the v0.1 finding, but **not isolated**: a different library version would look the same.
   These keys act on Qwen's attention to chip positions, not on the circuit. The **patched**
   merge then differs from our stock merge in exactly 16 tensors, all MLP weights of chip layers
   4–10; every Qwen layer, the embeddings and the anti-sink keys are byte-identical to it. Relative
   to the published checkpoint the patched one therefore differs in 24 tensors (those 16 + the 8).
5. *Neural* (`run_neural.py` from the patched tree, model `/data/qwenputer/qwenputer-rv-trap`;
   GNU logs `reports/logs/neural-v3-<case>-trap-fast.log`, Clang `…-trap-clang-fast.log`). The
   Clang ELF (`rv32im_zba`) needs 13.8% fewer chip tokens over the 22 cases (68,026 vs 78,927).

| v0.3 case | expected | GNU chip tokens / wall | Clang chip tokens / wall |
|---|---|---|---|
| `sys_two_in_two_out` | `PASS;FEE=000000001DCD6500` | 5468 / 57 s | 4284 / 46 s |
| `sys_one_in_one_out` | `PASS;FEE=000000003B9ACA00` | 4250 / 45 s | 3472 / 38 s |
| `sys_ctx_lock_mismatch` | `FAIL;R=LOCK` | 2276 / 27 s | 1995 / 24 s |
| `sys_witness_missing` | `FAIL;R=NOWIT` | 1553 / 19 s | 1363 / 17 s |
| `sys_lock_len_mismatch` | `FAIL;R=LOCK` | 1887 / 24 s | 1653 / 21 s |
| `sys_long_lock_match` | `PASS;FEE=0000000000000004` | 7481 / 75 s | 6819 / 69 s |
| `sys_long_lock_last_byte` | `FAIL;R=LOCK` | 5604 / 58 s | 5402 / 56 s |
| `sys_input_sum_overflow` | `FAIL;R=OVF` | 3475 / 38 s | 2807 / 31 s |
| `sys_output_sum_overflow` | `FAIL;R=OVF` | 4263 / 45 s | 3366 / 36 s |
| `sys_max_sum_exact` | `PASS;FEE=0000000000000000` | 4771 / 49 s | 3832 / 41 s |
| `sys_ctx_capacity_fail` | `FAIL;R=CAP` | 5083 / 52 s | 3936 / 42 s |
| `sys_no_inputs` | `FAIL;R=NOIN` | 2476 / 28 s | 2164 / 24 s |
| `sys_no_outputs` | `PASS;FEE=000000001DCD6500` | 3623 / 39 s | 3062 / 34 s |
| `sys_empty_lock` | `PASS;FEE=0000000000000001` | 3767 / 41 s | 3073 / 34 s |
| `sys_bad_magic` | `FAIL;R=CTX` | 1324 / 17 s | 1181 / 16 s |
| `sys_bad_count` | `FAIL;R=CTX` | 1328 / 17 s | 1186 / 16 s |
| `sys_bad_args_len` | `FAIL;R=CTX` | 1335 / 17 s | 1193 / 15 s |
| `sys_bad_witness_len` | `FAIL;R=CTX` | 1339 / 17 s | 1194 / 16 s |
| `sys_unpatched` | `FAIL;R=CTX` | 1312 / 17 s | 1170 / 15 s |
| `sys_probe_long` | `S;0,20,;0,02,3210;0,04…` | 6596 / 69 s | 6039 / 65 s |
| `sys_probe_missing` | `S;0,00,;0,00,;1,04,;1,…` | 5876 / 62 s | 5298 / 58 s |
| `sys_probe_bad_ctx` | `S;2,00,;2,04,;2,04,;2,…` | 3840 / 65 s | 3537 / 64 s |

Also on the patched checkpoint: v0.2 `two_in_two_out` PASS (3326 chip tokens) and v0.1
`valid_fee` PASS (4192). Only these two non-syscall cases were re-run neurally on it; the other
v0.1/v0.2 fixtures pass on the patched reference machine but were not re-run on the patched
weights, and every `gpucheck_trap` trace takes the trap.

**Negative control on the published checkpoint:** `sys_two_in_two_out` from the stock tree is
`FAIL` with lockstep `None` and replay ok, i.e. the published weights predict the stock fault
token for token. After the fault Qwen itself wrote a plausible but invented tool response ("The
payload "V3" is not a valid CKB transaction format…") and closed the tag: another reason the
verdict is only ever the firmware's bytes.

**A bug the checks caught.** The first patched checkpoint failed every v0.3 run with
TRACE_MISMATCH at the first trap: the model predicted `J(0)` where the reference has `J(16)`.
Upstream's `case()` drops an `int` payload (only family ids are ints), so the constant trap word
never reached the payload bits. `gpucheck_trap.py` had passed because `hf.check` compares the
weights with the same IR; it now first compares the IR's predictions with the reference trace,
which reproduced the failure exactly (11 wrong of 4437, all at trap positions) before the fix
(the payload as constant bits) and passes after it. That failing check's output was not saved;
the 6 failed neural runs remain in `reports/runs.jsonl`, with `chipformer_json_sha256`
`04076f15…`. The merge logs are `reports/logs/merge-{stock,trap}.log`.

| artifact | sha256 |
|---|---|
| `patches/qwenputer-ecall-trap.patch` | `a35edece3f813b9f4428d43c1900ffcc630313f30de892c4e3ff5fcc3c12adcf` |
| patched checkpoint, per-tensor manifest `reports/checkpoints/merge-trap.hashes.json` | `3bd3045eaa79368e8783f363e348fe4be939ff82beeb204d386420e9d949f0a2` |
| patched checkpoint `chipformer.json` (recorded per run as `chipformer_json_sha256`) | `76a9432fed8119e66e4aee35983aa8228d3fc32e5729f5aa812200c96433ccd2` |
| published checkpoint `chipformer.json` | `e62bc08b357b900ad84d007034ce4738985a7db1908994a4c85cf2a6657c576d` |

The per-tensor manifests pin the weights; runs record only `chipformer_json_sha256`, which is
layout metadata (it told the first, buggy patched checkpoint, `04076f15…`, apart from the final
one) and not a hash of the weights. `model_index_sha256` is the same for every rv1 merge and
identifies nothing. `run_neural.py` now refuses to run (`CIRCUIT_MISMATCH`) unless the code
tree's circuit has the same size statistics as the checkpoint's `chipformer.json` (checked on all
four tree/checkpoint pairings); circuits of identical size would not be told apart.
| v0.3 firmware, GNU `build/ckb_bridge.elf` | `213d4f154b8fe33eee8047dc53154144e78f04d0c514ba79678015bff77348a4` |
| v0.3 firmware, Clang `build/ckb_bridge_clang.elf` | `4c40c9a92e81c049e1766d91102edbc4a6a66a65f86d910ba53454c4e7415a53` |
| self-test `build/selftest-V3.elf` / `selftest-V3_T.elf` | `e7f74755…0ba3c4` / `430f2974…6ab9fa` |

The v0.2 section's ELF hashes are those of commit `e8f7d6c`; the v0.3 firmware adds the trap
entry and syscall code, and the v0.2 and v0.1 fixtures were re-checked on it on both reference
machines (stock and patched) but re-run neurally only for the two cases above.

### Text fidelity of the patched checkpoint (2026-10-31)

**Result: on everything measured, the patched checkpoint's Qwen text is identical to the
published checkpoint's, and both reproduce upstream's published text-fidelity numbers on ROCm.**

*Upstream's check* (`python -m chipformer.rv.textcheck BASE MERGED`, unmodified, each model run
from its own tree; JSON in `reports/textcheck/`, logs `reports/logs/textcheck-*.log`, ~1.5 min
each). It compares the merged model's next-token distributions over Qwen's vocabulary with base
Qwen3.5-4B (`851bf6e`) on 16 plain-text prompts, against the noise floor between base Qwen's own
sdpa and eager attention kernels:

| | mean KL vs base | noise floor | greedy text identical to base | top-1 agreement | tokenizer |
|---|---|---|---|---|---|
| upstream's report (`docs/rv32.md`, their hardware) | 6.1e-4 | 5.1e-4 | 10/16 | – | 4/4 |
| published checkpoint, this machine | 6.088e-4 | 4.918e-4 | 11/16 | 98.76% | 4/4 |
| patched checkpoint, this machine | 6.088e-4 | 4.918e-4 | 11/16 | 98.76% | 4/4 |

Published and patched agree exactly on every per-prompt KL, top-1 rate and greedy outcome. That
is expected rather than a strong result: these prompts contain no chip tokens, and the two
checkpoints differ only in chip-layer MLP weights (and, against the published one, the mute
column of 8 `k_proj`s), so this check shows the patch has no effect on plain text but cannot
see past that.

*Text around the computer.* To cover text written with chip positions in context, the two
non-syscall cases already run on the patched checkpoint were run on the published one with the
same v0.3 ELF (`213d4f15…`, stock tree; logs `reports/logs/neural-v3fw-*-published-fast.log`).
v0.1 `valid_fee` and v0.2 `two_in_two_out` both PASS on both, and the complete generated text,
the tool call, the 4192 / 3326 chip tokens and Qwen's reply after `</tool_response>`, is
identical between the two checkpoints. (In `valid_fee` that reply says "The fee is
000000003B9A00" for the firmware's `000000003B9ACA00`, identically on both models: a Qwen
transcription slip, and one more reason the verdict is only ever the firmware's bytes.)

## Not done / open

- Executor-only timing was not measured (it is not neural inference); only verify mode was run.
- The forced feed and executor verify were run on the cases above, not the full 12 × 2 matrix.
- Upstream `fastexec` `KeyError: 'Q'` not investigated (outside the `RvChat` path).
- Margin: the longest tool call (`trailing`) uses 79 of upstream's 80-row capture window; longer
  payloads silently overwrite the last row (`rows[min(cap, CAP_MAX)]`). Now guarded (v0.2
  section): such calls are refused before the run instead of silently corrupted.
- v0.2 on Clang: 7 of the 22 context fixtures were run on the neural layers (all 22 on
  `ref.Machine`). The forced feed and executor verify were not repeated on v0.2.
- v0.3 on Clang: all 22 syscall fixtures were run neurally on the patched checkpoint and pass
  (2026-10-31). The forced feed and executor verify were not repeated on v0.3.
- Text fidelity of the patched checkpoint was measured on upstream's 16 plain-text prompts and
  on two tool-call cases (see "Text fidelity" above), not on a benchmark such as MMLU, and not
  on text following a v0.3 syscall run (the published model cannot produce one to compare).
- The trap has no return-address register: syscalls must be made through `ckb_syscall`. A real
  trap (saving the PC, returning with `mret`) would need a register write in the circuit.

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

v0.3 (patched circuit; needs the base model, ~37 GB RAM for the merge, and a separate clone):

```bash
git clone https://github.com/divagr18/qwenputer.git qwenputer-trap && cd qwenputer-trap
git checkout ca13310e2f9f788d1dd3feeea8213c1b0c23913c && git apply $POC/patches/qwenputer-ecall-trap.patch
hf download Qwen/Qwen3.5-4B --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --local-dir $BASE
cd experimental/riscv && export PYTHONPATH=$PWD:$POC/scripts
$PY -m chipformer.rv.merge $BASE --out $TRAPM                     # ~1 min, peak ~37 GB RAM
RISCV_MARCH=rv32im bash $POC/scripts/build_selftest.sh
$PY $POC/scripts/gpucheck_trap.py                                 # IR vs reference, weights vs IR
$PY $POC/scripts/compare_checkpoints.py $TRAPM $POC/reports/checkpoints/merge-stock.hashes.json
$PY $POC/scripts/run_upstream_reference.py --elf $POC/build/ckb_bridge.elf --tokenizer $M/tokenizer.json
# (from the stock tree, the same command asserts that each v0.3 syscall faults with cause ecall)
$PY $POC/scripts/run_neural.py --model $TRAPM --elf $POC/build/ckb_bridge.elf --case sys_two_in_two_out
$PY $POC/scripts/run_neural.py --model $TRAPM --elf $POC/build/ckb_bridge_clang.elf --case sys_two_in_two_out
$PY -m chipformer.rv.textcheck $BASE $TRAPM --out $POC/reports/textcheck/trap.json   # text fidelity
```

`--case` takes any id from any fixture file (ids are unique across all three); a v0.2 id makes the
runner inject that fixture's context into a copy of `--elf` under `build/ctx/` and run that.

`run_neural.py` prints one JSON object (`mode`, `device`, `model_rev`, `firmware_sha256`,
`case_id`, `expected`, `actual`, `chip_tokens`, `elapsed_s`, `reference_match`, `replay_ok`,
`status`, plus lock/replay/payload detail) and appends it to `reports/runs.jsonl`.
