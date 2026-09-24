"""The model boundary: turn (state, rules) into violation probabilities."""

from __future__ import annotations

import os
from typing import Protocol, Sequence

from .ruleset import VIOLATION_KEY, ModelPin, Rule

# Files `laya.load` needs from a checkpoint repo. The repo also holds sibling
# checkpoints we do not want to download.
_CHECKPOINT_FILES = ["rl_agent_config.json", "model.safetensors", "tokenizer/*", "encoder/*"]

# Room left for special tokens and the per-line token-count approximation in
# `diff.chunk_diff`.
_STATE_MARGIN = 24


class Judge(Protocol):
    state_budget: int

    def count_tokens(self, text: str) -> int: ...

    def violation_probabilities(self, state: str, rules: Sequence[Rule]) -> dict[str, float]: ...


class LayaJudge:
    """Laya checkpoint pinned to an exact Hugging Face revision."""

    def __init__(self, pin: ModelPin, device: str = "cpu") -> None:
        # transformers probes TensorFlow on import; the Laya card says that can
        # deadlock model construction.
        os.environ.setdefault("USE_TF", "0")
        import laya
        from huggingface_hub import snapshot_download

        model_dir = snapshot_download(
            pin.repo, revision=pin.revision, allow_patterns=_CHECKPOINT_FILES
        )
        self._agent = laya.load(model_dir, device=device)
        cfg = self._agent.cfg
        self.state_budget = cfg.get("max_len", 512) - cfg.get("head_max_len", 192) - _STATE_MARGIN

    def count_tokens(self, text: str) -> int:
        return len(self._agent.tok(text, add_special_tokens=False)["input_ids"])

    def violation_probabilities(self, state: str, rules: Sequence[Rule]) -> dict[str, float]:
        if not rules:
            return {}
        questions = {rule.id: rule.question_spec() for rule in rules}
        answers = self._agent.predict(state, questions)["answers"]
        return {rule.id: float(answers[rule.id]["probabilities"][VIOLATION_KEY]) for rule in rules}
