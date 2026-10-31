#!/usr/bin/env python3
"""Compare two merged checkpoints tensor by tensor (SHA-256 of each tensor's raw bytes), plus
their JSON configs and tokenizer, streaming one tensor at a time.

    python compare_checkpoints.py A B [--json out.json]
    python compare_checkpoints.py --hashes DIR out.json   # a manifest, usable as A or B later

A or B may be a checkpoint directory or such a manifest (so a deleted checkpoint can still be
compared against).

Used twice: (1) positive control, our stock merge vs the published Qwen3.5-4B-Qwenputer (must be
identical for the merge recipe to be trusted); (2) our patched merge vs the published one (the
differences must be confined to the chip layers and the chip-token rows the circuit change
touches). Prints every differing tensor with a short summary.
"""
import argparse
import hashlib
import json
import pathlib

import torch
from safetensors import safe_open


def tensors(d: pathlib.Path) -> dict[str, str]:
    """name -> shard file."""
    idx = d / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    return {k: "model.safetensors" for k in safe_open(str(d / "model.safetensors"), "pt").keys()}


def digest(d: pathlib.Path, shard: str, name: str) -> list:
    with safe_open(str(d / shard), "pt") as f:
        t = f.get_tensor(name)
    raw = t.contiguous().view(-1).view(torch.uint8).numpy()
    return [hashlib.sha256(raw).hexdigest(), list(t.shape), str(t.dtype)]


def manifest(d: pathlib.Path) -> dict:
    """{"files": {name: parsed JSON}, "tensors": {name: [sha256, shape, dtype]}}"""
    if d.is_file():
        m = json.loads(d.read_text())
        return compact(m["files"], m["tensors"])
    files = {f: json.loads((d / f).read_text()) for f in
             ("config.json", "chipformer.json", "tokenizer.json", "generation_config.json") if (d / f).exists()}
    return compact(files, {n: digest(d, sh, n) for n, sh in sorted(tensors(d).items())})


def compact(files: dict, tensors_: dict) -> dict:
    """The tokenizer as a hash of its canonical JSON (it is ~10 MB); other files kept whole."""
    files = dict(files)
    if isinstance(files.get("tokenizer.json"), dict) and "sha256_canonical" not in files["tokenizer.json"]:
        canon = json.dumps(files["tokenizer.json"], sort_keys=True).encode()
        files["tokenizer.json"] = {"sha256_canonical": hashlib.sha256(canon).hexdigest()}
    return {"files": files, "tensors": tensors_}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json")
    ap.add_argument("--hashes", action="store_true", help="write a's manifest to b and exit")
    ap.add_argument("--detail", action="store_true",
                    help="both directories: for each differing 2-D tensor, which columns/rows differ and by how much")
    p = ap.parse_args()
    if p.hashes:
        pathlib.Path(p.b).write_text(json.dumps(manifest(pathlib.Path(p.a))))
        return
    A, B = pathlib.Path(p.a), pathlib.Path(p.b)
    ma, mb = manifest(A), manifest(B)
    report = {"a": str(A), "b": str(B), "files": {}, "only_a": [], "only_b": [], "differ": [], "same": 0}
    for f in ("config.json", "chipformer.json", "tokenizer.json", "generation_config.json"):
        ja, jb = ma["files"].get(f), mb["files"].get(f)
        if ja is None or jb is None:
            report["files"][f] = f"missing in {'a' if ja is None else 'b'}"
        else:
            report["files"][f] = "same" if ja == jb else sorted(
                k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    ta, tb = ma["tensors"], mb["tensors"]
    report["only_a"] = sorted(set(ta) - set(tb))
    report["only_b"] = sorted(set(tb) - set(ta))
    for name in sorted(set(ta) & set(tb)):
        da, db = ta[name], tb[name]
        if da == db:
            report["same"] += 1
        else:
            report["differ"].append({"tensor": name, "a": da[1:], "b": db[1:]})
    if p.detail and A.is_dir() and B.is_dir():
        wa, wb = tensors(A), tensors(B)
        for d in report["differ"]:
            n = d["tensor"]
            with safe_open(str(A / wa[n]), "pt") as fa, safe_open(str(B / wb[n]), "pt") as fb:
                x, y = fa.get_tensor(n).float(), fb.get_tensor(n).float()
            if x.shape != y.shape or x.dim() != 2:
                continue
            ne = x != y
            diff = (x - y).abs()[ne]
            d["detail"] = {"elements": int(ne.sum()), "of": x.numel(),
                           "columns": ne.any(0).nonzero().flatten().tolist()[:20],
                           "n_columns": int(ne.any(0).sum()), "n_rows": int(ne.any(1).sum()),
                           "max_abs_diff": float(diff.max()), "max_abs_a": float(x[ne].abs().max())}
            print("  DETAIL", n, json.dumps(d["detail"]))
    print(json.dumps({k: v for k, v in report.items() if k != "differ"}, indent=1))
    print(f"{report['same']} identical tensors, {len(report['differ'])} differ")
    for d in report["differ"][:60]:
        print("  DIFF", d["tensor"], d["a"], d["b"])
    if p.json:
        pathlib.Path(p.json).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
