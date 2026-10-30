"""Python-only oracle for the v0.2 toy context API and contract. NOT a CKB consensus engine and
NOT neural execution. Written from the rules in guest/ctx.h (ckb-script 1.1.1 store_data) and
the fixture's structured context, not from the encoded bytes, so it shares no code with the C.
"""
SUCCESS, INDEX_OUT_OF_BOUND, ITEM_MISSING = 0, 1, 2
INPUT, OUTPUT = 1, 2  # capacity sources
ARGS, WITNESS = 1, 2  # partial-load items
U64 = (1 << 64) - 1


class Ctx:
    def __init__(self, ctx: dict | None):
        if ctx is None:  # nothing injected: the firmware's default bytes are not a context
            ctx = {"magic": "NONE", "inputs": [], "outputs": [], "args": "", "witness": None}
        ins = [int(x, 16) for x in ctx["inputs"]]
        outs = [int(x, 16) for x in ctx["outputs"]]
        args = bytes.fromhex(ctx["args"])
        wit = None if ctx["witness"] is None else bytes.fromhex(ctx["witness"])
        n_in, n_out = ctx.get("n_inputs", len(ins)), ctx.get("n_outputs", len(outs))
        args_len = ctx.get("args_len", len(args))
        wit_len = ctx.get("witness_len", 0xFF if wit is None else len(wit))
        self.valid = (ctx.get("magic", "CKX1") == "CKX1" and n_in <= 2 and n_out <= 2
                      and args_len <= 32 and (wit_len <= 32 or wit_len == 0xFF))
        # a count or length header overrides the listed items, as the firmware reads it: absent
        # slots read as zero bytes
        self.ins = (ins + [0, 0])[:n_in]
        self.outs = (outs + [0, 0])[:n_out]
        self.args = args.ljust(32, b"\0")[:args_len]
        self.wit = None if wit_len == 0xFF else (wit or b"").ljust(32, b"\0")[:wit_len]

    def partial_load(self, buf_len: int, offset: int, source: int):
        """(rc, *len out, copied bytes); *len is left as given when rc != SUCCESS."""
        if not self.valid:
            return ITEM_MISSING, buf_len, b""
        if source == ARGS:
            data = self.args
        elif source == WITNESS:
            if self.wit is None:
                return ITEM_MISSING, buf_len, b""
            data = self.wit
        else:
            return INDEX_OUT_OF_BOUND, buf_len, b""
        off = min(offset, len(data))
        rest = data[off:]
        return SUCCESS, len(rest), rest[:buf_len]

    def capacity(self, index: int, source: int):
        if not self.valid:
            return ITEM_MISSING, None
        cells = self.ins if source == INPUT else self.outs if source == OUTPUT else None
        if cells is None or index >= len(cells):
            return INDEX_OUT_OF_BOUND, None
        return SUCCESS, cells[index]


def contract(c: Ctx) -> str:
    if not c.valid:
        return "FAIL;R=CTX\n"
    if c.wit is None:
        return "FAIL;R=NOWIT\n"
    if c.args != c.wit:
        return "FAIL;R=LOCK\n"
    if sum(c.ins) > U64:
        return "FAIL;R=OVF\n"
    if not c.ins:
        return "FAIL;R=NOIN\n"
    if sum(c.outs) > U64:
        return "FAIL;R=OVF\n"
    if sum(c.ins) < sum(c.outs):
        return "FAIL;R=CAP\n"
    return f"PASS;FEE={sum(c.ins) - sum(c.outs):016X}\n"


# mirrors guest/contract.c probe_v2
PROBES = [("L", ARGS, 0, 0), ("L", ARGS, 4, 0), ("L", ARGS, 4, 30), ("L", ARGS, 4, U64),
          ("L", WITNESS, 0, 0), ("L", WITNESS, 4, 1), ("L", 3, 4, 0),
          ("K", 0, INPUT), ("K", 1, INPUT), ("K", 2, INPUT), ("K", U64, INPUT),
          ("K", 0, OUTPUT), ("K", 0, 3)]


def probe(c: Ctx) -> str:
    out = "T"
    for p in PROBES:
        if p[0] == "L":
            rc, n, got = c.partial_load(p[2], p[3], p[1])
            out += f";{rc:X},{n:02X}," + got.hex().upper()
        else:
            rc, cap = c.capacity(p[1], p[2])
            out += f";{rc:X}" + ("" if cap is None else f",{cap:016X}")
    return out + "\n"


def verify(case: dict) -> str:
    c = Ctx(case["ctx"])
    if case["payload"] == "V2":
        return contract(c)
    if case["payload"] == "V2;T":
        return probe(c)
    raise ValueError(f"not a v0.2 payload: {case['payload']!r}")
