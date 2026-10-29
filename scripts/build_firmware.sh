#!/usr/bin/env bash
set -euo pipefail
: "${QWENPUTER_DIR:?Set QWENPUTER_DIR to the upstream qwenputer repository root}"
GCC="${RISCV_GCC:-riscv64-unknown-elf-gcc}"
READELF="${RISCV_READELF:-riscv64-unknown-elf-readelf}"
MARCH="${RISCV_MARCH:-rv32im_zba}"  # GCC < 12 lacks Zba: RISCV_MARCH=rv32im (a strict subset)
UPSTREAM="${QWENPUTER_DIR}/experimental/riscv/chipformer/rv/env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
mkdir -p "$ROOT/build" "$ROOT/reports"
command -v "$GCC" >/dev/null || { echo "Missing $GCC; install a RISC-V bare-metal GCC cross compiler" >&2; exit 2; }
for f in "$UPSTREAM/rom.ld" "$UPSTREAM/crt0.S"; do test -f "$f" || { echo "Missing upstream file $f" >&2; exit 2; }; done
"$GCC" -march="$MARCH" -mabi=ilp32 -O2 -ffreestanding -fno-builtin \
  -fno-stack-protector -fno-pic -fno-pie -nostdlib -nostartfiles -static \
  -Wl,--no-relax -Wl,-Map,"$ROOT/build/ckb_bridge.map" \
  -T"$UPSTREAM/rom.ld" -o "$ROOT/build/ckb_bridge.elf" \
  "$UPSTREAM/crt0.S" "$ROOT/guest/ckb_bridge.c" -lgcc
"$READELF" -h -lW "$ROOT/build/ckb_bridge.elf" | tee "$ROOT/reports/elf-layout.txt"
"$GCC" -dumpmachine
printf 'Built %s\n' "$ROOT/build/ckb_bridge.elf"
echo 'NEXT: run the upstream chipformer.rv.elf loader to verify the generated ELF maps into Qwenputer ROM/DROM/RAM.'
