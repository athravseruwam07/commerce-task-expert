# Local training and quantization

This project uses **Qwen/Qwen2.5-Coder-1.5B-Instruct** (Apache 2.0) as a practical small-model starting point, rather than claiming to use the newest or strongest model. It is converted to FP16 in MLX. All training and inference run on the local Apple GPU; downloading the public starting weights is the only network requirement. No hosted generation API or teacher API is required.

## Reproduce

From the repository root on Apple silicon with Python 3.11 or newer:

```sh
uv sync --extra ml --extra dev
uv run commerce generate
uv run python scripts/train_local.py --iters 300
uv run python scripts/quantize_local.py
```

The data-generation command and full benchmark commands are documented in the repository README. `train_local.py` converts the base checkpoint if needed, reads **only train and validation cases**, constructs the same chat prompt used at inference, verifies every complete prompt-plus-answer fits the configured sequence length, and trains with prompt masking. It refuses to truncate targets. The test set is never converted into training examples.

The script saves a LoRA adapter under `models/adapters` and fuses it with the original FP16 base into `models/fine_tuned`. `quantize_local.py` then quantizes the **fused fine-tuned checkpoint** to MLX 4-bit affine weights with group size 64 in `models/quantized`. This is weight-only quantization, not 4-bit activation arithmetic.

## Controlled variants

| Variant | Weights | Adaptation |
|---|---|---|
| Original | FP16 | None |
| Fine-tuned | FP16 | LoRA merged into the base |
| Quantized fine-tuned | Mixed: eligible matrices 4-bit, remaining parameters FP16 | Same merged LoRA checkpoint |

Training is **LoRA on a floating-point base**, not QLoRA. Keeping this distinction allows the fine-tuned FP16 versus quantized comparison to isolate post-training quantization. The initial small configuration trains rank-8 adapters on the final four transformer layers with batch size 1, gradient checkpointing, seed 42 and completion-only loss. `configs/lora.yaml` is the source of truth.

## Measurement

`MLXModel.generate` uses the native chat template, greedy decoding, an identical output cap, and no prefix/KV reuse across requests. All variants run in the same MLX implementation. Reported generation time includes prompt processing and output generation; load time is separately recorded. First token time and token counts are available. MLX peak allocated memory is measured after resetting the peak immediately before a request, with model weights already resident. It is not the machine's total memory or process RSS; the benchmark reports its definition. Metal is synchronized before timestamps are finalized.

The training record is `reports/training.json`; console traces are in ignored `reports/raw/training.log`. The record includes data hashes, iteration count, seed, package versions and elapsed time. Quantization metadata and output file sizes are in `reports/quantization.json`. Downloaded weights, adapters and synthetic SQLite databases are intentionally ignored by Git. Preserve the public source model's attribution and license if distributing derived weights.

## Memory limits

The target is a 16 GB M4 Mac. Fit is checked by a short real run before the longer experiment; a theoretical weight-size calculation does not establish training feasibility. Close memory-heavy applications while running local experiments. Do not run multiple variants simultaneously. If a larger dataset needs longer sequences, adjust `--max-seq-length` explicitly; if memory becomes insufficient, lower the number of adapted layers or sequence length only after checking that examples still fit. Changes must be recorded in the final experiment metadata.

Quantization can reduce memory without improving latency. Accuracy and speed gains are hypotheses to measure, not guaranteed outcomes. Model selection and prompt revisions use validation cases only; test failures are for reporting and subsequent versioned experiments.

## Primary references

- [MLX LM](https://github.com/ml-explore/mlx-lm)
- [Official MLX LoRA guide](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- [Qwen2.5-Coder-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct)
