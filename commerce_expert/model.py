"""Local MLX inference with one shared, gold-free prompt for every variant.

MLX imports are deliberately lazy: data generation, tests and the demo's recorded
runs remain usable on Linux without Apple Metal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """You are a commerce analytics SQL specialist. Use SQLite and the supplied schema and business policy. Return exactly one JSON object, no markdown or explanation.
If the question is answerable: {"action":"query","sql":"SELECT ..."}.
If an essential metric or period is ambiguous and the policy does not resolve it: {"action":"clarify","question":"A short specific clarifying question"}.
Generate a single read-only query. Do not invent columns. Use explicit output aliases, deterministic ordering and the dates supplied."""


def render_messages(case: dict[str, Any]) -> list[dict[str, str]]:
    """Construct inputs exclusively from merchant-visible context, never gold labels."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Schema:\n{case['schema']}\n\n"
                f"As of (UTC): {case['as_of_date']}\n\n"
                f"Business policy:\n{case['policy']}\n\n"
                f"Merchant question: {case['question']}"
            ),
        },
    ]


def render_prompt(case: dict[str, Any]) -> str:
    """Readable prompt for inspection; MLX uses tokenizer's native chat template."""
    return "\n\n".join(f"{m['role'].upper()}:\n{m['content']}" for m in render_messages(case))


@dataclass(frozen=True)
class ModelConfig:
    model_path: str | Path
    adapter_path: str | Path | None = None
    max_tokens: int = 512
    seed: int = 42


class MLXModel:
    """Load exactly one model and measure uncached, greedy, single-request inference."""

    def __init__(self, config: ModelConfig):
        import mlx.core as mx
        from mlx_lm import load

        self.config = config
        self.mx = mx
        kwargs = {}
        if config.adapter_path is not None:
            kwargs["adapter_path"] = str(config.adapter_path)
        start = time.perf_counter()
        self.model, self.tokenizer = load(str(config.model_path), **kwargs)
        mx.eval(self.model.parameters())
        mx.synchronize()
        self.load_ms = (time.perf_counter() - start) * 1000

    def generate(self, case: dict[str, Any]) -> dict[str, Any]:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        mx = self.mx
        mx.random.seed(self.config.seed)
        # No cross-request prefix/KV cache, so all variants pay the same prefill cost.
        mx.clear_cache()
        mx.reset_peak_memory()
        prompt = self.tokenizer.apply_chat_template(
            render_messages(case),
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_tokens = len(self.tokenizer.encode(prompt))
        chunks = []
        completion_tokens = 0
        first_token_ms = None
        start = time.perf_counter()
        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=self.config.max_tokens,
            sampler=make_sampler(temp=0.0),
            prefill_step_size=256,
        ):
            if first_token_ms is None:
                mx.synchronize()
                first_token_ms = (time.perf_counter() - start) * 1000
            chunks.append(response.text)
            completion_tokens = response.generation_tokens
        mx.synchronize()
        return {
            "text": "".join(chunks),
            "generation_ms": (time.perf_counter() - start) * 1000,
            "first_token_ms": first_token_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "peak_memory_bytes": mx.get_peak_memory(),
            "active_memory_bytes": mx.get_active_memory(),
            "model_load_ms": self.load_ms,
        }
