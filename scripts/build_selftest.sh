#!/usr/bin/env bash
# Bridge-free self-test ELFs (guest/selftest.c) for scripts/gpucheck_trap.py; GNU toolchain as in
# build_firmware.sh. One ELF per payload: build/selftest-V3.elf, build/selftest-V3_T.elf.
set -euo pipefail
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"
NM="${RISCV_NM:-riscv64-unknown-elf-nm}"
MARCH="${RISCV_MARCH:-rv32im_zba}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/build"
for payload in V3 "V3;T"; do
  out="$ROOT/build/selftest-${payload//;/_}.elf"
  "$GCC" -march="$MARCH" -mabi=ilp32 -O2 -ffreestanding -fno-builtin \
    -fno-stack-protector -fno-pic -fno-pie -nostdlib -nostartfiles -static -Wl,--no-relax \
    -DSELFTEST_PAYLOAD="\"$payload\"" -T"$ROOT/guest/rom.ld" -o "$out" \
    "$ROOT/guest/start.S" "$ROOT/guest/selftest.c" "$ROOT/guest/contract.c" "$ROOT/guest/ctx.c" \
    "$ROOT/guest/emit.c" "$ROOT/guest/parse.c" "$ROOT/guest/trap.c" -lgcc
  "$NM" "$out" | grep -q '^00040040 T ckb_trap_entry$' || { echo "ckb_trap_entry not at 0x40040 in $out" >&2; exit 3; }
  echo "$out $(sha256sum "$out" | cut -c1-64)"
done
