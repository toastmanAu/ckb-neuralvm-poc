#!/usr/bin/env python3
"""Static ELF32 RISC-V PT_LOAD address-range gate; does not execute any instructions."""
import argparse
import pathlib
import struct

# RAM starts at upstream spec.RAM_BASE=0: GNU ld maps the ELF header at 0, as upstream calc-bridge.elf does.
REGIONS = {'RAM': (0x00000000, 0x00008000), 'DROM': (0x00020000, 0x00040000), 'ROM': (0x00040000, 0x00080000)}

def check(path):
    b = pathlib.Path(path).read_bytes()
    if b[:6] != b'\x7fELF\x01\x01': raise ValueError('expected little-endian ELF32')
    if struct.unpack_from('<H', b, 18)[0] != 243: raise ValueError('expected machine EM_RISCV=243')
    entry, phoff = struct.unpack_from('<II', b, 0x18)
    phentsize, phnum = struct.unpack_from('<HH', b, 0x2A)
    if not (REGIONS['ROM'][0] <= entry < REGIONS['ROM'][1]): raise ValueError('entry not in ROM')
    count = 0
    for i in range(phnum):
        pt, off, va, pa, filesz, memsz, flags, align = struct.unpack_from('<8I', b, phoff + i * phentsize)
        if pt != 1: continue
        count += 1
        if filesz > memsz or off + filesz > len(b): raise ValueError(f'bad PT_LOAD file size segment {i}')
        match = [name for name, (lo, hi) in REGIONS.items() if lo <= va and va + memsz <= hi]
        if len(match) != 1: raise ValueError(f'PT_LOAD out of bounds at {va:#x} size {memsz:#x}')
        print(f'segment {i}: {match[0]:4} VA={va:#08x} bytes={memsz} flags={flags:#x}')
    if count < 2: raise ValueError('expected loadable segments for code/data')
    print(f'ELF map PASS: {count} segments, entry={entry:#x}; ABI/ISA/runtime correctness NOT tested')

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('elf'); a = ap.parse_args(); check(a.elf)
if __name__ == '__main__': main()
