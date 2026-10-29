# Engineering handoff — CKB-NeuralVM PoC v0.1

**Date:** 2026-10-29  
**Hardware target:** Ubuntu/Linux + AMD Radeon RX 7900 XTX (24 GiB, ROCm, gfx1100), desktop i7-14700/64 GiB RAM.  
**Primary upstream:** [divagr18/qwenputer](https://github.com/divagr18/qwenputer)  
**Checkpoint:** [divagr1925/Qwen3.5-4B-Qwenputer](https://huggingface.co/divagr1925/Qwen3.5-4B-Qwenputer)  
**Future reference:** [nervosnetwork/ckb-vm](https://github.com/nervosnetwork/ckb-vm)

## 0. Objective and scientific claim

Test whether a CKB-*inspired* validator written in C and compiled to RISC-V runs with **instruction execution occurring inside compiled, non-learned Qwenputer transformer circuitry**, not in a host subprocess. Compare with a Python independent specification and upstream software RISC-V reference. Require lockstep/replay confirmation on the generated model-layer trace.

**Not within v0.1 scope:** RV64, compressed instructions, real CKB ELF/consensus semantics, actual CKB syscalls, Molecule, cryptographic signatures, network RPC, any real transaction validation or on-chain operation.

Why this scope: Qwenputer is RV32IM+Zba with custom putchar (ecall a7=1) and park/exit (a7=93). Production CKB uses RV64 plus C, bitmanip extensions and versioned CKB syscalls. Reusing Qwenputer's existing neural CPU is feasible for the first experiment; loading genuine CKB binaries is a separate architecture effort.

## 1. Evidence-grounded upstream implementation

- Upstream code: `experimental/riscv/chipformer/rv/{chat.py,bridge.py,firmware.py,core.py,ref.py,elf.py,spec.py}`.
- Existing firmware: `experimental/riscv/chipformer/rv/env/calc/calc.c`, with its BRIDGE section near the end.
- Existing launcher: `uv run python -m chipformer.rv.chat qwenputer-rv --firmware tests/data/rv/calc-bridge.elf --ask 'Use calc: what is 123456789 * 987654321?'`.
- Firmware loaded into state from an ELF image; `RvChat` runs text tokens and chip tokens with different handling. P3 tool bridge captures up to 80 model text tokens with up to 16 bytes per captured token row, resumes the CPU on closing tool-call token, and echoes `putchar` output.
- **Address map**: RAM `0x00000000..0x00007FFF`, text bridge at `0x8000` (special buffer), DROM at `0x20000` (128 KiB), executable ROM at `0x40000` (256 KiB). Follow upstream `rom.ld`.
- **Tool call convention**: `<tool_call>\n<function=ckb_verify>\n<parameter=payload>\n...\n</parameter>\n</function>\n</tool_call>`; the upstream bridge captures token bytes, runs after close. Upstream calc firmware recognises value after the *second `>`* and before next `<`.
- **CPU ecall semantics**: a7=1 outputs one byte/special token; a7=93 yields HALT/park. Do not emit real CKB syscall ecall numbers on the unmodified neural CPU.

## 2. Source-tree structure

```
ckb-neuralvm-poc/
  README.md
  AGENTS.md                # main coding agent assignment
  HANDOFF.md               # this document
  guest/ckb_bridge.c       # initial bare-metal C firmware
  guest/start.S, rom.ld   # standalone Clang/LLD fallback map
  fixtures/test_vectors.json
  scripts/build_firmware.sh
  scripts/build_firmware_clang.sh
  scripts/check_elf_map.py
  scripts/preflight.py
  scripts/reference_rules.py
  scripts/check_reference.py
  scripts/run_upstream_reference.py
  scripts/run_neural.py
  docs/RESEARCH.md
  reports/                  # empty until experiments
  build/ckb_bridge_clang.elf # shipped static ELF32 artifact, UNEXECUTED
```

Keep upstream qwenputer checkout separate and unmodified until needed; submit any adaptations as a small patch file or a tiny local adapter. Pin exact commit and model checkpoint hash in your results.

## 3. Hardware / toolchain preflight (Gate 0)

Required: Linux, Python >=3.11, `uv`, `hf`, a working PyTorch ROCm installation, `riscv64-unknown-elf-gcc` + `riscv64-unknown-elf-readelf`, and enough free space for model files and temporary build data.

```bash
python3 scripts/preflight.py
rocminfo | grep -m 3 gfx1100 || true
python3 - <<'PY'
import torch
print(torch.__version__, torch.version.hip, torch.cuda.is_available())
if torch.cuda.is_available(): print(torch.cuda.get_device_properties(0))
PY
```

Do not replace an already-working ROCm Python environment blindly. Upstream `pyproject.toml` requests Torch >=2.14.1 and Transformers >=5.19.0 (as checked 2026-10-29); confirm a matching ROCm wheel is available. The card states BF16 checkpoint ~18.7–18.8 GB; actual peak allocated memory may exceed 24 GB. Upstream fast executor explicitly uses `torch.cuda.CUDAGraph`; HIP exposes many `torch.cuda` APIs but compatibility is **not verified** for this repository. Start with no executor and `--slow` if the fast path fails. If not feasible on gfx1100, preserve reference runs and optionally use an NVIDIA CUDA system for Gate 2; do not silently call a CPU-only reference run neural.

## 4. Upstream smoke (Gate 0 continued)

```bash
git clone https://github.com/divagr18/qwenputer.git
export QWENPUTER_DIR="$PWD/qwenputer"
cd "$QWENPUTER_DIR/experimental/riscv"
uv sync --extra hf
uv run pytest tests -q
hf download divagr1925/Qwen3.5-4B-Qwenputer --local-dir qwenputer-rv
uv run python -m chipformer.rv.chat qwenputer-rv \
  --firmware tests/data/rv/calc-bridge.elf \
  --ask 'Use calc: what is 123456789 * 987654321?' --check
```

Record a successful actual in-model `calc` execution (chip tokens >0 and clean replay); otherwise classify the failure by model version, GPU backend, kernels, OOM or tool invocation.

Generic HF `pipeline()` cannot demonstrate the full CPU bridge by itself. `RvChat` loads the snapshot and makes chip tokens invisible to the Qwen text layers. The model checkpoint has no guarantee of good generic quantization support.

## 5. Firmware wire contract (v0.1)

Payload exact ASCII, strictly uppercase hex:

```
V1;I=00000002540BE400;O=0000000218711A00;L=2A;W=2A
```

- `I`: one input capacity, 16 hex digits (uint64 shannons).
- `O`: one output capacity, 16 hex digits (uint64 shannons).
- `L`: 8-bit toy lock identifier, not a real script argument or signature.
- `W`: 8-bit toy witness value, not a cryptographic signature.
- Rules, in order: malformed format => `FAIL;R=FMT`; unequal `L/W` => `FAIL;R=LOCK`; `I<O` => `FAIL;R=CAP`; otherwise `PASS;FEE=<16 uppercase hex digits>`.
- The example is 100 CKB input, 90 CKB output, 10 CKB difference (1 CKB = 100,000,000 shannons). This is merely a capacity-conservation arithmetic fixture; *not* complete CKB validation.
- Firmware prints the surrounding Qwenputer tool-response tags via special `putchar` values 256..259 and HALT parks between requests.

Important: The C bridge copied from the upstream mechanism is Qwenputer-specific. The conventional Python independent oracle is `scripts/reference_rules.py`; don't use it to compute a result on behalf of neural inference.

### Why no real CKB syscalls yet?

CKB uses ecall a7=2052 (`load_script`), 2071 (`load_cell`), 2074 (`load_witness`) and a 64-bit length-pointer/partial loading convention (RFC0009). None are implemented in the present neural CPU. A guest-side function can emulate the *intent* of loading fixture data, but may not be mislabelled as a CKB syscall implementation. Follow-on work may add machine-level ecall decoding and memory-write semantics.

## 6. Cross-build guest firmware (Gate 1a)

From package root:

```bash
export QWENPUTER_DIR=/absolute/path/to/qwenputer
bash scripts/build_firmware.sh
file build/ckb_bridge.elf
riscv64-unknown-elf-readelf -h -lW build/ckb_bridge.elf
sha256sum build/ckb_bridge.elf
# Alternative (Clang + LLD) already compiled here:
bash scripts/build_firmware_clang.sh
python3 scripts/check_elf_map.py build/ckb_bridge_clang.elf
```

The GNU build uses upstream `rom.ld`, `crt0.S`, `-march=rv32im_zba -mabi=ilp32`, bare-metal static linking and libgcc. The source intentionally avoids libc and dynamic allocation. **GNU script is supplied as an unexecuted build scaffold:** compiler/linker flags or `crt0`/`exit` symbol details may need adjusting to actual installed cross compiler. **An alternative `scripts/build_firmware_clang.sh` and locally generated `build/ckb_bridge_clang.elf` were successfully cross-compiled with Clang 17 + LLD and verified by static ELF program headers; they were NOT run on Qwenputer.** Check emitted ELF is ELF32 little-endian RISC-V, uses only supported ISA, and all PT_LOAD sections fit the defined ROM/DROM/RAM layout. Inspect program disassembly and runtime behavior before declaring success.

If unsupported `ecall`, misalignment or out-of-range fault occurs, inspect target ISA, test-map/linker and the bridge before changing the circuit. The Qwenputer ELF loader (`chipformer.rv.elf.program`) is a strict loader of the custom memory map and rejects incompatible segments.

## 7. Run Qwenputer's *software* reference machine (Gate 1b)

```bash
python3 scripts/check_reference.py
cd "$QWENPUTER_DIR/experimental/riscv"
uv run python /absolute/path/to/ckb-neuralvm-poc/scripts/run_upstream_reference.py \
 --elf /absolute/path/to/ckb-neuralvm-poc/build/ckb_bridge.elf \
 --tokenizer /absolute/path/to/ckb-neuralvm-poc-or-qwenputer-rv/tokenizer.json
```

`run_upstream_reference.py` uses `elf.program`, `ref.Machine`, `BR.park`, and `BR.Session` and explicitly marks results software-only. It deliberately replays a synthetic tool-call using the exact upstream bridge protocol, avoiding unreliable natural-language tool selection at this stage. Use `--elf /.../build/ckb_bridge_clang.elf` to try the included Clang build if the GNU build is unavailable. Correct any tokenizer special-token encoding or bridge parsing discrepancies the upstream reference exposes. Require 12/12 exact guest firmware outputs compared with the oracle. Capture the instruction count and trace if possible.

## 8. Execute on actual model layers (Gate 2)

First, run reference-passing fixtures one at a time, beginning with a minimal valid case and one failure case:

```bash
cd "$QWENPUTER_DIR/experimental/riscv"
uv run python /absolute/path/to/ckb-neuralvm-poc/scripts/run_neural.py \
 --model "$PWD/qwenputer-rv" \
 --elf /absolute/path/to/ckb-neuralvm-poc/build/ckb_bridge.elf \
 --case valid_fee --slow
```

The initial runner is **experimental**, intentionally checked and explicitly fails if the model does not invoke `ckb_verify`. It supplies a tool schema through `RvChat.prompt_ids`, then measures `chip_tokens` and checks `RvChat.replay`. It must be adapted after actual upstream tests, because `RvChat`'s original system prompting is tuned to `calc` and may not spontaneously invoke the new tool despite its schema.

### If Qwen fails to call `ckb_verify`

- Inspect actual generated `<tool_call>` format, function name, `<parameter=payload>` capture, max 80-token window and closing `</tool_call>`.
- Try a more explicit tool instruction while keeping payload and firmware unchanged.
- Then implement a **controlled text-token feed** inside a fork of the `RvChat` decoding loop. Feed an exact tool-call sequence after the firmware snapshot is injected, running every forced text token through the chip layers' `step` as well as updating Qwen's cache. On `</tool_call>`, allow the *neural CPU layers* to own/emit GO and subsequent circuit trace tokens. Do not prefill a tool call and then inject snapshot, because that resets the bridge capture state.
- Compare with the bridge `Session` reference token-by-token. Use `replay` for an additional IR-based check. Document whether response bytes were written by the circuit's special putchar path.
- If this cannot be made to work, label project `BRIDGE-BLOCKED`, not complete.

### ROCm fallback ordering

1. Confirm `torch.cuda.is_available()` works with HIP and BF16 checkpoint loads in VRAM; avoid other GPU users.
2. Test `RvChat(..., fast=False, executor=False)` (full HF chip layers, slow but conceptually direct).
3. Test default `fast=True, executor=False` if kernels allow.
4. Only then try `executor=True` CUDA-graph/Triton path. Differences must be traced to exact numerical behavior; never turn checks off and claim success.
5. When blocked, preserve host-reference results, exact traceback, device info and kernel versions; consider access to a compatible NVIDIA card for a separate neural run.

## 9. Differential/acceptance tests (Gate 3)

Required subsets to run on model layers: `valid_fee`, `valid_exact`, `lock_mismatch`, `capacity_fail`, `max_u64` (5). If costs permit, run all 12.

| Test | Independent oracle | Firmware through software ref | Firmware through model layers |
|---|---|---|---|
| Input encoding | Spec regex | Captured ASCII payload | Token row capture |
| Program used | Python rules | Same ELF | Same ELF, neural CPU |
| Output check | Fixture `expected` | Exact output bytes | Exact output bytes |
| Trace verification | N/A | `ref.Machine` instruction trace | `lock_error=None`, `replay.ok=True`, chip tokens >0 |
| Statement proved | Expected toy policy | RISC-V firmware implements policy | Neural RISC-V executes firmware |

For the five neural cases, record:

```json
{
  "mode": "neural_hf",
  "status": "PASS",
  "case_id": "valid_fee",
  "gpu": "RX 7900 XTX",
  "hip_version": "...",
  "qwenputer_commit": "...",
  "model_revision": "...",
  "elf_sha256": "...",
  "expected": "PASS;FEE=000000003B9ACA00",
  "actual": "PASS;FEE=000000003B9ACA00",
  "chip_tokens": 1234,
  "elapsed_s": 0.0,
  "lock_error": null,
  "replay_ok": true
}
```

The numeric `chip_tokens`/time above are illustrative schema placeholders, not measurements. Agent must populate observed values. A positive toy result means *only* neural CPU execution of that program under that configuration.

## 10. First extension: more CKB-like execution context

Once the neural proof works, refactor firmware to separate `bridge` / `fixture parser` / `virtual context` / `contract` / `emit`. Implement a small **host-injected context object** and guest-side API:

- `toy_load_script_args` (one-byte value initially)
- `toy_load_witness` (one-byte value initially)
- `toy_load_cell_capacity(index, source)` with 2 inputs and 2 outputs
- `toy_partial_load(dst, uint64_t *len, offset, source)` implementing RFC0009's length-return and offset rules, **but call it a shim, not a real VM syscall**.

Differentially test with independent Python and C reference; include out-of-range indices, partial reads, missing items and `UINT64_MAX` sum overflow. Only after that consider a generated neural ecall decoder accepting selected 2052/2071/2074 numbers.

## 11. Long-term RV64 / real CKB-VM research gates — intentionally out of initial scope

1. RV64I integer datapath, sign extension, 64-bit memory/register handling and end-to-end tests.
2. M, C, Zba/Zbb/Zbc/Zbs extension parity under the chosen VM version; compare against pinned CKB-VM, not informal general RISC-V documentation.
3. ELF64 loader, CKB address-space/memory protection, 4 MiB working memory and limits.
4. Native CKB syscall semantics and Molecule serialization, script groups and transaction context.
5. Versioned cycle model and syscalls (`exec`, `spawn`, and others as applicable).
6. Real mainnet-relevant lock/type scripts in a hermetic offline test harness; differential testing across thousands of vectors, faults and out-of-bounds cases.
7. Cross-device reproducibility and deterministic error/trace equivalence. Without all of these, **never** call it a CKB consensus engine.

Do not choose a post-quantum CKB lock as the first example: SPHINCS+/ML-DSA is expensive and would obscure whether the base mechanism works.

## 12. Blockers and reporting

Blocker categories: `DEP-INCOMPATIBLE`, `COMPILER-BLOCKED`, `REFERENCE-BLOCKED`, `MODEL-OOM`, `ROCM-KERNEL`, `BRIDGE-BLOCKED`, `TRACE-MISMATCH`, `OUTPUT-MISMATCH`.

For every blocker, save exact command, full error, versions, SHA, minimal reproducer, reasoned next step. There is no requirement to achieve a fraudulent 'success' by bypassing the difficult component. Provide `reports/RESULTS.md` and append a run manifest (JSON) for each trial.

## 13. External sources / pin list

See `docs/RESEARCH.md`. Resolve and record actual git SHA and model revision before running. These are moving repositories; do not assume a generated sketch has already been compiled or tested on the target GPU.
