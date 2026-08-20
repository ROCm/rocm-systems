# Causal-LM Smoke Test Results

Date: 2026-08-20

These results use the fixed five-prompt workload in this directory:

- greedy generation with `max_new_tokens=1`
- float32 model weights
- eager attention
- CPU and real-GPU outputs compared with `rtol=1e-3`, `atol=1e-3`
- simulated `gfx1250` launched with rocjitsu using `configs/gfx1250.json`

## Summary

CPU, real MI350X, and simulated `gfx1250` runs passed for all four models
across the fixed five-prompt workload. All CPU-vs-GPU and CPU-vs-simulator
comparisons matched token IDs and logits within tolerance.

| Model key | Model | CPU | Real GPU | CPU vs GPU | Simulated `gfx1250` |
| --- | --- | --- | --- | --- | --- |
| `tinyllama_1b` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | pass, 5 prompts, 3.922s | pass, 5 prompts, 7.073s | pass, max abs `2.0503997802734375e-05` | pass, 5 prompts, 1670.609s, max abs `4.863739013671875e-05` |
| `qwen3_0_6b` | `Qwen/Qwen3-0.6B` | pass, 5 prompts, 4.270s | pass, 5 prompts, 7.835s | pass, max abs `5.0067901611328125e-05` | pass, 5 prompts, 988.118s, max abs `8.869171142578125e-05` |
| `llama32_1b` | `meta-llama/Llama-3.2-1B-Instruct` | pass, 5 prompts, 6.433s | pass, 5 prompts, 7.725s | pass, max abs `4.00543212890625e-05` | pass, 5 prompts, 1947.919s, max abs `6.651878356933594e-05` |
| `qwen2p5_1p5b` | `Qwen/Qwen2.5-1.5B-Instruct` | pass, 5 prompts, 6.284s | pass, 5 prompts, 7.226s | pass, max abs `4.673004150390625e-05` | pass, 5 prompts, 2437.115s, max abs `0.00011539459228515625` |

## Generated Tokens

The listed token IDs are the single generated token from the CPU run. The real
GPU and simulated `gfx1250` generated the same token ID for each checked prompt.

| Model key | Prompt 0 | Prompt 1 | Prompt 2 | Prompt 3 | Prompt 4 |
| --- | --- | --- | --- | --- | --- |
| `tinyllama_1b` | `[432]` | `[367]` | `[13]` | `[13]` | `[29871]` |
| `qwen3_0_6b` | `[34208]` | `[387]` | `[279]` | `[220]` | `[220]` |
| `llama32_1b` | `[35308]` | `[387]` | `[11892]` | `[720]` | `[11]` |
| `qwen2p5_1p5b` | `[34208]` | `[387]` | `[11631]` | `[220]` | `[279]` |

## Tests Run

- `python3 -m py_compile emulation/rocjitsu/model_validation/causal_lm/*.py`
- `python3 emulation/rocjitsu/model_validation/causal_lm/run_validation.py --help`
- `python3 emulation/rocjitsu/model_validation/causal_lm/compare.py --help`
- `python3 emulation/rocjitsu/model_validation/causal_lm/generate.py --help`
- `git diff --check -- emulation/rocjitsu/model_validation/causal_lm emulation/rocjitsu/model_validation/tinyllama`
- synthetic comparator checks for exact match, token mismatch, NaN logits, shape mismatch, and case-count mismatch
- CPU generation smoke with `trl-internal-testing/tiny-random-LlamaForCausalLM`
- full harness run for `tinyllama_1b`
- full harness run for `qwen3_0_6b`
- full harness run for `llama32_1b`
- full harness run for `qwen2p5_1p5b`

## Interpretation

These results show rocjitsu can complete and numerically match this small
causal-LM smoke workload on simulated `gfx1250`. This is a functional
correctness signal for one-token generation on these models, not a broad
readiness claim: simulator runtime is still very slow compared with CPU and the
real GPU, and this test does not cover longer generation, batching, quantized
weights, KV-cache growth, or serving-style launch/runtime behavior.
