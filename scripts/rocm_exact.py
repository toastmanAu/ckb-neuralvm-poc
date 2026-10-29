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


def install(modules) -> int:
    """Wrap every nn.Linear under ``modules`` (an iterable of modules); returns how many."""
    n = 0
    for root in modules:
        for m in root.modules():
            if isinstance(m, torch.nn.Linear) and not getattr(m, "_rocm_exact", False):
                m.forward = _fp32_linear_forward.__get__(m, torch.nn.Linear)
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
