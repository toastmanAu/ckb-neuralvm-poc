"""ROCm adapter: make bf16 nn.Linear in the chip layers accumulate in fp32 and round once.

Why: the compiled circuit is exact only if every GEMM accumulates in fp32 (upstream
chipformer.q35.fast.exact_matmuls). On CUDA that is enforced with
``torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False``. On ROCm
(torch 2.14.1+rocm7.2, RX 7900 XTX, hipBLASLt and rocBLAS alike) that flag has no effect: a bf16
GEMM with K = 2816 differs from the fp32-accumulated result by up to one bf16 ulp (2.0 at
magnitude ~512), which breaks the circuit's threshold dims. Measured: CPU bf16 GEMM ==
fp32-accumulate-then-round, and ROCm fp32 GEMM rounded to bf16 == CPU bit-for-bit.

What it changes: only the arithmetic of the wrapped Linear calls (inputs and weights are the same
bf16 values, upcast exactly; the product is rounded to bf16 once). Weights, dtypes of stored
tensors and the circuit are untouched. Bitwise equivalence is proven by upstream gpucheck's strict
bf16 mode and the chat lockstep/replay checks, not assumed.
"""
import torch
import torch.nn.functional as F


def _fp32_linear_forward(self: torch.nn.Linear, x: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        return F.linear(x, self.weight, self.bias)
    b = None if self.bias is None else self.bias.float()
    return F.linear(x.float(), self.weight.float(), b).to(torch.bfloat16)


def _lin32(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if x.dtype != torch.bfloat16 or x.device.type != "cuda":
        return F.linear(x, w)
    return F.linear(x.float(), w.float()).to(torch.bfloat16)


def _fp32_sparse_forward(self, x: torch.Tensor) -> torch.Tensor:
    """upstream chipformer.q35.fast.SparseLinear.forward (ca13310) with each F.linear via _lin32."""
    if self.dense is not None and x.numel() > self.dense_above * x.shape[-1]:
        return _lin32(x, self.dense)
    if self.r1 == self.r0 or self.c1 == self.c0:
        return x.new_zeros(*x.shape[:-1], self.out_features)
    y = _lin32(x[..., self.c0:self.c1], self.weight)
    if self.r0 or self.r1 != self.out_features:
        y = F.pad(y, (self.r0, self.out_features - self.r1))
    return y


def install(modules) -> int:
    """Wrap every nn.Linear and fast.SparseLinear (the fast path's compacted chip Linears) under
    ``modules`` (an iterable of modules); returns how many. Call after fast.compact."""
    from chipformer.q35.fast import SparseLinear

    n = 0
    for root in modules:
        for m in root.modules():
            if getattr(m, "_rocm_exact", False):
                continue
            if isinstance(m, torch.nn.Linear):
                m.forward = _fp32_linear_forward.__get__(m, torch.nn.Linear)
            elif isinstance(m, SparseLinear):
                m.forward = _fp32_sparse_forward.__get__(m, SparseLinear)
            else:
                continue
            m._rocm_exact = True
            n += 1
    return n


def patch_hf_build_model() -> None:
    """For upstream gpucheck/hf.check: every layer of hf.build_model's model is a chip layer."""
    from chipformer.rv import hf

    orig = hf.build_model

    def build_model(*args, **kwargs):
        model, lay, ids = orig(*args, **kwargs)
        install([model.model])
        return model, lay, ids

    hf.build_model = build_model


def needed() -> bool:
    return torch.version.hip is not None


def chip_modules(chat) -> list:
    """RvChat's chip layers, plus the fast path's chip view (chat.cm: same layers + a zero layer)."""
    mods = list(chat.lm.layers[: chat.n_chip])
    if getattr(chat, "cm", None) is not None:
        mods.append(chat.cm)
    return mods
