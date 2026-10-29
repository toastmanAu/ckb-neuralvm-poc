# Research notes and source-of-truth links

**Research date:** 2026-10-29. The linked repositories and published docs were inspected, but no live GPU build or Git checkout was possible within the handoff creation environment. Commit IDs must be pinned on the builder's machine.

## Verified Qwenputer architecture

1. [Qwenputer repo README](https://github.com/divagr18/qwenputer) — Qwen3.5-4B plus 11 compiled CPU layers. RV32IM + Zba + decimal bignum; compute weights generated directly, no training. Gated DeltaNet state as exact-addressable RAM. Checkpoint about 18.7–18.8 GB BF16. `RvChat` boots firmware into a state snapshot.
2. [RV32 design notes](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/docs/rv32.md) — ~32 KB RAM at 0, DROM at 0x20000, ROM at 0x40000. Qwen text bridge, special chip trace-token families, HF vs Triton executable trace conformance. Fast 3,200 chip tokens/s on L4 reported by author; not an AMD benchmark.
3. [Upstream firmware bridge](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/env/calc/calc.c) — `#ifdef BRIDGE` code; capture rows from 0x8000 after text tool invocation; the example firmware initially parks on exit then resumes on GO; prints special token values 256..259.
4. [Bridge implementation](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/bridge.py) — `Text`, `Session`, `park`, 16-byte per-token rows, 80 captured tokens. The source is used as the authoritative guidance for our synthetic test-call protocol.
5. [RvChat implementation](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/chat.py) — custom loader/driver, `prompt_ids`, `generate`, `replay`, original `CALC_TOOL`, `--slow`, `--check`; fast path uses GPU graphs; `generate()` accepts `on_token` but not a direct forced tool-call interface (as checked). Extend minimally if needed.
6. [ELF loader](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/elf.py) — static 32-bit little-endian ELF PT_LOAD loader restricted to firmware map.
7. [Link script](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/env/rom.ld), [startup](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/env/crt0.S), [test cross-build recipe](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/chipformer/rv/env/build_tests.sh) — practical starting points for custom firmware.
8. [Qwenputer experimental README](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/README.md) — RV32UI/UM/Zba tests, program trace replay, QEMU comparisons; supports loading C firmware in model without retraining.
9. [Upstream pyproject](https://raw.githubusercontent.com/divagr18/qwenputer/main/experimental/riscv/pyproject.toml) — Python >=3.11, Torch >=2.14.1, Transformers >=5.19.0, `uv sync --extra hf` (check ROCm wheel availability).

### Major architectural warning

The HF model card's generic Transformers / SGLang / vLLM snippets do not establish the custom state snapshot and transparent chip-token handling necessary to exercise the CPU. Use upstream `RvChat`, not a generic chat endpoint, for the actual PoC. There is no evidence the checkpoint's compiled logic survives quantization or arbitrary numerical transformations.

## Verified CKB VM differences

1. [CKB-VM repository](https://github.com/nervosnetwork/ckb-vm) — production RISC-V scripting VM; Rust implementation has both 32/64 register modes, but **mainnet executes RV64**. Development Rust mode and production ASM mode can differ; do not imply a successful RV32 simulation is mainnet equivalence.
2. [CKB VM RFC0003](https://github.com/nervosnetwork/rfcs/blob/master/rfcs/0003-ckb-vm/0003-ckb-vm.md) — early spec: RV64IMC, ELF64, maximum ~4 MiB runtime memory, no native floating-point extension.
3. [VM Syscalls RFC0009](https://raw.githubusercontent.com/nervosnetwork/rfcs/master/rfcs/0009-vm-syscalls/0009-vm-syscalls.md) — a7 numbers and ABI: six args a0–a5, result a0, partial read `(addr, uint64_t *len, offset)`. Representative IDs: exit 93, load script 2052, load cell 2071, load witness 2074, load cell by field 2081. **None of these except exit 93 are wired into stock Qwenputer.**
4. [CKB C stdlib](https://github.com/nervosnetwork/ckb-c-stdlib) — real C header syscall declarations and Molecule support. Future phase reference only. Do not paste its RV64 inline syscall wrappers directly into stock RV32 Qwenputer; that will fault.
5. [VM Syscalls 2 RFC0034](https://github.com/nervosnetwork/rfcs/blob/master/rfcs/0034-vm-syscalls-2/0034-vm-syscalls-2.md) — VM version/cycle/exec, version-specific availability.
6. [VM Syscalls 3 RFC0050](https://github.com/nervosnetwork/rfcs/blob/master/rfcs/0050-vm-syscalls-3/0050-vm-syscalls-3.md) — spawn and other later execution semantics; not in first PoC.
7. [CKB-VM Rust library](https://raw.githubusercontent.com/nervosnetwork/ckb-vm/develop/src/lib.rs) — library currently exposes `run<R, M>` and `run_with_memory`, plus machine creation constructs and `ISA_IMC`, `ISA_B`, `ISA_MOP` flags. Pinned version APIs must be consulted before writing a conformance harness.

**A key distinction:** The current v0.1 toy checks overall capacities and a byte-equality tag. Actual CKB transaction validation includes script-group semantics, transaction context, serialized structures, cryptography/locks, cell type rules, fees, occupied-capacity requirements and consensus policy. The PoC proves only the neural-RISC-V execution pathway, not any of those semantics.

## AMD runtime verification

- [AMD GPU specifications](https://rocmdocs.amd.com/en/develop/reference/gpu-specs.html): RX 7900 XTX `gfx1100`, 24 GB VRAM.
- [AMD ROCm Radeon support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-6.4.0/reference/system-requirements.html): gfx1100 card supported on selected Linux ROCm stacks. Match installed OS/runtime version.
- [AMD ROCm Triton instructions](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-triton.html): Triton can run on supported AMD GPUs, but this does **not** certify upstream `RvChat`'s CUDA graph/fast executor on ROCm.

### Why the Radeon experiment is a research risk

Upstream expects Torch >=2.14.1 while the October 2026 ROCm ecosystem is evolving. If a matching HIP wheel cannot be installed or exact neural computations fail, the agent must not silently switch to arbitrary pip CUDA builds, generic GGUF quantization, float16 instead of BF16, or host-simulated execution and call that a neural success. Record the blockers, then test using a supported NVIDIA machine if available, keeping target-specific results separate.

## Recommended reading order for the agent

1. Qwenputer README + design.
2. `calc.c` BRIDGE section, `bridge.py`, `chat.py`, `elf.py`, `rom.ld`.
3. `guest/ckb_bridge.c`, fixtures, `HANDOFF.md`.
4. CKB-VM repo, RFC0003/0009, only after primary neural circuit is demonstrably running.

## Known open questions to resolve in code, not assume

- Does this exact upstream commit allow a generic `RvChat` tool schema to replace `calc` without also altering the in-model tool formatting, and does Qwen reliably select it?
- Does `BR.Text.encode` preserve the synthetic special token delimiters exactly in `run_upstream_reference.py`? Verify against `BR.Session` traces.
- Does minimal libc-free RV32 firmware link with the upstream `crt0.S` on the installed cross toolchain? Inspect ELF segments and `exit` symbol.
- Are exact GDN state writes and CUDA-graph optimizations valid on ROCm gfx1100 with chosen Torch/Transformers revisions?
- What are actual peak VRAM, elapsed time and instruction/token counts for guest firmware? No measurements have been made here.
- Do we want a second PoC building actual CKB syscall hardware logic, or initially keep a guest-side deterministic shim for the next iteration? Decide after this version is demonstrated.

## Handoff build artifact provenance

A standalone Clang 17 + lld 17 cross-build was produced locally using the `guest/start.S` and `guest/rom.ld` compatibility files, with `--target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32`. The resulting **ELF32** file is `build/ckb_bridge_clang.elf`, SHA256 `01bbce2c381e48e1a5c0abd567be69b12d1864d033e982313e33d336e93a060c`, with PT_LOAD sections at `0x40000` (R-X, 2796 bytes), `0x20000` (R, 17 bytes), and `0x100` (RW, 160-byte bss). Static syntax and ELF layout were checked. **The circuit, tokenizer bridge, software RV32 emulator and real Qwenputer model were not run during artifact creation**, so binary semantics, model-layer compatibility and correctness are unverified. The agent should run all dynamic gates before claiming success.
