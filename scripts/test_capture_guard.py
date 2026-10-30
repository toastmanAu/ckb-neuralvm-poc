#!/usr/bin/env python3
"""The 80-row capture guard (run_neural.capture_check) against real and synthetic payloads.

Prints each fixture's captured-token count, then finds the longest v0.1-shaped payload that fits
and checks that (a) the guard accepts it and refuses one token more, and (b) in upstream's
bridge (BR.Session) the text buffer the firmware reads holds the accepted call intact but not
the refused one (the reason for the guard). Then the same for the other branch: a payload
holding one real vocabulary token longer than 16 bytes is refused and is cut in the buffer.
Software only, no model weights.

    cd <qwenputer>/experimental/riscv && PYTHONPATH=.:<this repo>/scripts \\
        python test_capture_guard.py --tokenizer <model>/tokenizer.json [--elf build/ckb_bridge.elf]
"""
import argparse
import json
import pathlib
import sys

from chipformer.rv import bridge as BR, elf, ref

from run_neural import CALL_HEAD, CALL_TAIL, capture_check

ROOT = pathlib.Path(__file__).resolve().parents[1]


def n_captured(T, payload: str) -> int:
    ids = T.encode(CALL_HEAD + payload + CALL_TAIL)
    return ids.index(T.TCC) - ids.index(T.TCO) - 1


def buffer_intact(T, elf_path, payload: str) -> bool:
    """Do the firmware-visible rows 1..80 hold exactly the captured tokens' bytes?"""
    m = ref.Machine.from_program(elf.program(elf_path))
    BR.park(m)
    ids = T.encode(CALL_HEAD + payload + CALL_TAIL)
    body = ids[ids.index(T.TCO) + 1: ids.index(T.TCC)]
    sess = BR.Session(T, m)
    sess.qwen(ids[: ids.index(T.TCC)], run=False)  # capture only, no GO
    rows = [bytes(r).rstrip(b"\0") for r in m.text[1: len(body) + 1]]
    return rows == [T.decode_bytes([q]) for q in body]


def long_token_payload(T) -> str:
    """A short payload whose tokenization contains a token longer than one 16-byte row."""
    for q in range(T.n):
        b = T.bytes[q]
        if len(b) > 16 and b.isascii() and b.decode().isprintable() and b"<" not in b:
            payload = "V1;" + b.decode()
            ids = T.encode(CALL_HEAD + payload + CALL_TAIL)
            if q in ids:
                return payload
    raise SystemExit("no >16-byte ASCII token found in the vocabulary")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--elf", default=str(ROOT / "build/ckb_bridge.elf"))
    a = p.parse_args()
    T = BR.Text(a.tokenizer)
    for name in ("test_vectors.json", "ctx_vectors.json"):
        counts = {v["id"]: n_captured(T, v["payload"]) for v in json.loads((ROOT / "fixtures" / name).read_text())}
        print(f"{name}: max {max(counts.values())} rows ({max(counts, key=counts.get)}), "
              f"min {min(counts.values())}")
    # grow a V1 payload with trailing digits (one token each in Qwen's tokenizer) until it overflows
    pad = ""
    while n_captured(T, "V1;I=0000000000000001;O=0000000000000001;L=01;W=01" + pad + "0") <= 80:
        pad += "0"
    fits = "V1;I=0000000000000001;O=0000000000000001;L=01;W=01" + pad
    over = fits + "0"
    bad = []
    if capture_check(T, fits) is not None:
        bad.append(f"guard refused an 80-row payload: {capture_check(T, fits)}")
    if capture_check(T, over) is None:
        bad.append("guard accepted an 81-row payload")
    ok_fits, ok_over = buffer_intact(T, a.elf, fits), buffer_intact(T, a.elf, over)
    print(f"{n_captured(T, fits)} rows: guard {capture_check(T, fits)}; buffer intact {ok_fits}")
    print(f"{n_captured(T, over)} rows: guard {capture_check(T, over)!r}; buffer intact {ok_over}")
    if not ok_fits:
        bad.append("an 80-row call is not held intact")
    if ok_over:
        bad.append("an 81-row call was held intact: the overflow claim is wrong")
    longp = long_token_payload(T)
    ok_long = buffer_intact(T, a.elf, longp)
    print(f"long-token payload {longp!r} ({n_captured(T, longp)} rows): guard {capture_check(T, longp)!r}; "
          f"buffer intact {ok_long}")
    if capture_check(T, longp) is None or "longer than" not in capture_check(T, longp):
        bad.append("guard accepted a call with a >16-byte token")
    if ok_long:
        bad.append("a >16-byte token was held intact: the truncation claim is wrong")
    for b in bad:
        print("FAIL " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
