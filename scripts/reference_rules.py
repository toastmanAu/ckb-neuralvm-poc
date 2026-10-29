"""Python-only fixture oracle. NOT a CKB consensus engine and NOT neural execution."""
import re

PAT = re.compile(r"^V1;I=([0-9A-F]{16});O=([0-9A-F]{16});L=([0-9A-F]{2});W=([0-9A-F]{2})$")

def verify(payload: str) -> str:
    m = PAT.fullmatch(payload)
    if not m:
        return "FAIL;R=FMT\n"
    i, o, lock, witness = (int(x, 16) for x in m.groups())
    if lock != witness:
        return "FAIL;R=LOCK\n"
    if i < o:
        return "FAIL;R=CAP\n"
    return f"PASS;FEE={i-o:016X}\n"
