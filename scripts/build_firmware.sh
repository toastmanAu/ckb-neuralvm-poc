#!/usr/bin/env bash
set -euo pipefail
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"
READELF="${RISCV_READELF:-riscv64-unknown-elf-readelf}"
MARCH="${RISCV_MARCH:-rv32im_zba}"  # GCC < 12 lacks Zba: RISCV_MARCH=rv32im (a strict subset)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$ROOT/build" "$ROOT/reports"
command -v "$GCC" >/dev/null || { echo "Missing $GCC; install a RISC-V bare-metal GCC cross compiler" >&2; exit 2; }
"$GCC" -march="$MARCH" -mabi=ilp32 -O2 -ffreestanding -fno-builtin \
  -fno-stack-protector -fno-pic -fno-pie -nostdlib -nostartfiles -static \
  -Wl,--no-relax -Wl,-Map,"$ROOT/build/ckb_bridge.map" \
  -T"$ROOT/guest/rom.ld" -o "$ROOT/build/ckb_bridge.elf" \
  "$ROOT/guest/start.S" "$ROOT/guest/bridge.c" "$ROOT/guest/contract.c" "$ROOT/guest/ctx.c" "$ROOT/guest/emit.c" "$ROOT/guest/parse.c" "$ROOT/guest/trap.c" -lgcc
NM="${RISCV_NM:-riscv64-unknown-elf-nm}"
# guest/start.S, linked first, must put the trap entry at ROM word 0x10 (spec.TRAP_WORD in the patch)
"$NM" "$ROOT/build/ckb_bridge.elf" | grep -q '^00040040 T ckb_trap_entry$' || { echo "ckb_trap_entry not at 0x40040" >&2; exit 3; }
"$READELF" -h -lW "$ROOT/build/ckb_bridge.elf" | tee "$ROOT/reports/elf-layout.txt"
"$GCC" -dumpmachine
printf 'Built %s\n' "$ROOT/build/ckb_bridge.elf"
