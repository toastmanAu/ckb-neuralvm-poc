#!/usr/bin/env bash
# Proven in the handoff authoring environment with clang 17.0.0 + lld; not runtime-tested.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CLANG="${RISCV_CLANG:-clang}"
command -v "$CLANG" >/dev/null || { echo "Missing $CLANG" >&2; exit 2; }
mkdir -p "$ROOT/build"
"$CLANG" --target=riscv32-unknown-elf -march=rv32im_zba -mabi=ilp32 \
  -O2 -ffreestanding -fno-builtin -fno-stack-protector -nostdlib \
  -fuse-ld=lld -Wl,--no-relax -Wl,-T,"$ROOT/guest/rom.ld" \
  -o "$ROOT/build/ckb_bridge_clang.elf" \
  "$ROOT/guest/start.S" "$ROOT/guest/bridge.c" "$ROOT/guest/contract.c" "$ROOT/guest/ctx.c" "$ROOT/guest/emit.c" "$ROOT/guest/parse.c" "$ROOT/guest/trap.c"
nm "$ROOT/build/ckb_bridge_clang.elf" | grep -q '^00040040 T ckb_trap_entry$' || { echo "ckb_trap_entry not at 0x40040" >&2; exit 3; }
file "$ROOT/build/ckb_bridge_clang.elf"
readelf -h -lW "$ROOT/build/ckb_bridge_clang.elf"
