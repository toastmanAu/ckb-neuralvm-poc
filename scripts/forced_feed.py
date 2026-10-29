"""HANDOFF §8 controlled text-token feed for RvChat (upstream left unmodified).

Upstream's decoding loop picks each text token as ``circuit_token(hc)`` if the circuit owns the
position, else ``argmax(logits)``. ForcedRvChat leaves every forward pass as it is (each forced
token still runs through the chip layers and Qwen's layers, updating both caches, after the
firmware snapshot is injected) and only replaces the *returned logits* with a one-hot choice of
the next forced token, and only while the circuit does not own the position. After the last forced
token (``</tool_call>``) the circuit owns GO and everything it prints, exactly as unforced.

The first generated token comes from the prompt's prefill logits, which this does not touch: the
caller checks it equals the first forced token (Qwen opens with ``<tool_call>`` unaided).
"""
import torch

from chipformer.rv.chat import RvChat


class ForcedRvChat(RvChat):
    def set_force(self, ids: list[int]) -> None:
        """Force ``ids[1:]`` (``ids[0]`` must be the model's own first token)."""
        self.force_ids = list(ids)
        self._queue = list(ids[1:])
        self.forced = 0
        self.force_conflict = None  # first position where the circuit claimed a forced slot

    def _force(self, hc, logits):
        if not getattr(self, "_queue", None):
            return logits
        if self.circuit_token(hc) is not None:
            if self.force_conflict is None:
                self.force_conflict = self.forced
            return logits
        t = self._queue.pop(0)
        self.forced += 1
        out = torch.full_like(logits, float("-inf"))
        out[..., t] = 0
        return out

    def _qwen_step(self, hc, cache, pos):  # fast and executor paths
        return self._force(hc, super()._qwen_step(hc, cache, pos))

    def _layers(self, ids, cache, pos):  # slow path; pos is None for chip tokens
        hc, logits = super()._layers(ids, cache, pos)
        return hc, (logits if pos is None else self._force(hc, logits))

    def force_done(self) -> bool:
        return not self._queue and self.force_conflict is None
