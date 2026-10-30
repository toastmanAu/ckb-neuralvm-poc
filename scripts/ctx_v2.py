"""Host side of the v0.2 toy context: encode a fixture's context and inject it into the ELF.

The firmware reads a 128-byte context from the DROM symbol ``ckb_ctx`` (guest/ctx.h has the
layout). Injection overwrites that symbol's bytes in a copy of the ELF, so every consumer that
loads the ELF (ref.Machine, the neural run's lockstep, RvChat.replay) sees the same context with
upstream code unmodified. The firmware code bytes are unchanged; ``code_sha256`` checks that.
"""
import hashlib
import pathlib
import struct

CTX_SIZE, MAX_CELLS, MAX_BYTES, WITNESS_MISSING = 128, 2, 32, 0xFF
SYMBOL = b"ckb_ctx"


def encode(ctx: dict) -> bytes:
    """ctx: {"inputs": [hex u64], "outputs": [hex u64], "args": hex, "witness": hex | None},
    plus optional raw overrides "magic" (4 ASCII chars), "n_inputs", "n_outputs", "args_len",
    "witness_len" (header bytes written as given) for malformed-context fixtures."""
    ins = [int(x, 16) for x in ctx["inputs"]]
    outs = [int(x, 16) for x in ctx["outputs"]]
    args = bytes.fromhex(ctx["args"])
    wit = None if ctx["witness"] is None else bytes.fromhex(ctx["witness"])
    if len(ins) > MAX_CELLS or len(outs) > MAX_CELLS or len(args) > MAX_BYTES or \
            (wit is not None and len(wit) > MAX_BYTES):
        raise ValueError("context exceeds the toy layout")
    b = bytearray(CTX_SIZE)
    magic = ctx.get("magic", "CKX1").encode()
    if len(magic) != 4:
        raise ValueError("magic must be 4 bytes")
    b[0:4] = magic
    b[4] = ctx.get("n_inputs", len(ins))
    b[5] = ctx.get("n_outputs", len(outs))
    b[6] = ctx.get("args_len", len(args))
    b[7] = ctx.get("witness_len", WITNESS_MISSING if wit is None else len(wit))
    for i, v in enumerate(ins):
        struct.pack_into("<Q", b, 8 + 8 * i, v)
    for i, v in enumerate(outs):
        struct.pack_into("<Q", b, 24 + 8 * i, v)
    b[40:40 + len(args)] = args
    if wit:
        b[72:72 + len(wit)] = wit
    return bytes(b)


def _symbol_file_offset(elf: bytes) -> int:
    """File offset of ckb_ctx's bytes: .symtab lookup, then the PROGBITS section holding it."""
    assert elf[:4] == b"\x7fELF" and elf[4] == 1 and elf[5] == 1, "ELF32 little-endian only"
    shoff, = struct.unpack_from("<I", elf, 0x20)
    shentsize, shnum = struct.unpack_from("<HH", elf, 0x2E)
    secs = [struct.unpack_from("<10I", elf, shoff + i * shentsize) for i in range(shnum)]
    for name, typ, _, addr, off, size, link, _, _, entsize in secs:
        if typ != 2:  # SHT_SYMTAB
            continue
        strtab = secs[link]
        for k in range(size // entsize):
            st_name, st_value, st_size, _, _, _ = struct.unpack_from("<IIIBBH", elf, off + k * entsize)
            s0 = strtab[4] + st_name
            if elf[s0:elf.index(b"\0", s0)] != SYMBOL:
                continue
            if st_size != CTX_SIZE:
                raise ValueError(f"ckb_ctx is {st_size} bytes, want {CTX_SIZE}")
            for _, t, _, a, o, sz, *_ in secs:
                if t == 1 and a <= st_value and st_value + CTX_SIZE <= a + sz:  # SHT_PROGBITS
                    return o + st_value - a
            raise ValueError("ckb_ctx is not in a PROGBITS section")
    raise ValueError("no ckb_ctx symbol (not a v0.2 firmware?)")


def inject(base_elf, ctx_bytes: bytes, out_elf) -> pathlib.Path:
    assert len(ctx_bytes) == CTX_SIZE
    elf = bytearray(pathlib.Path(base_elf).read_bytes())
    off = _symbol_file_offset(bytes(elf))
    elf[off:off + CTX_SIZE] = ctx_bytes
    out = pathlib.Path(out_elf)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(bytes(elf))
    return out


def code_sha256(elf_path) -> str:
    """SHA-256 of the ELF with ckb_ctx zeroed: equal for every context injected into one build."""
    elf = bytearray(pathlib.Path(elf_path).read_bytes())
    off = _symbol_file_offset(bytes(elf))
    elf[off:off + CTX_SIZE] = bytes(CTX_SIZE)
    return hashlib.sha256(elf).hexdigest()


def case_elf(base_elf, case: dict, out_dir) -> pathlib.Path:
    """The fixture's ELF: the base build with its context injected (ctx None: a plain copy)."""
    base = pathlib.Path(base_elf)
    if case["ctx"] is None:
        out = pathlib.Path(out_dir) / f"{base.stem}-{case['id']}.elf"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(base.read_bytes())
        return out
    return inject(base, encode(case["ctx"]), pathlib.Path(out_dir) / f"{base.stem}-{case['id']}.elf")
