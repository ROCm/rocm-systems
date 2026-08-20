# TinyLlama Reproducer

This directory contains a small TinyLlama 1.1B model check for CPU, a real ROCm
GPU, and rocjitsu simulated `gfx1250` execution.

Model URL: <https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0>

The expected passing simulator result requires the rocjitsu `v_cls_i32` fix for
the all-zero/all-sign-bit case.

## Setup

Run from this directory:

```bash
cd /path/to/rocm-systems/emulation/rocjitsu/model_validation/tinyllama

# Point these at your local ROCm PyTorch and rocjitsu build.
export ROCM_PYTHON=/path/to/rocm-python
export ROCM_LIB_DIR=/path/to/rocm-runtime-libs
export ROCJITSU_BUILD=/path/to/rocjitsu-build
export ROCJITSU_CONFIG=${ROCJITSU_CONFIG:-../../configs/gfx1250.json}

# Directory-local defaults are fine for these.
export HF_DEPS=${HF_DEPS:-"$PWD/.deps/hf"}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
export MODEL_DIR=${MODEL_DIR:-"$PWD/model_cache/TinyLlama_TinyLlama-1.1B-Chat-v1.0"}
export XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-"$PWD/.run"}
export ROCJITSU_RUNTIME_DIR=${ROCJITSU_RUNTIME_DIR:-"$PWD/.run/rocjitsu-tinyllama"}
mkdir -p "$HF_DEPS" "$HF_HOME" "$XDG_RUNTIME_DIR" "$ROCJITSU_RUNTIME_DIR" results

$ROCM_PYTHON -m pip install \
  --target "$HF_DEPS" \
  transformers huggingface_hub hf_xet tokenizers sentencepiece safetensors numpy

cmake --build "$ROCJITSU_BUILD" --target rocjitsu_bin -j "$(nproc)"
```

## Download Model

```bash
PYTHONPATH="$HF_DEPS" HF_HOME="$HF_HOME" "$ROCM_PYTHON" - <<'PY'
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    local_dir=os.environ["MODEL_DIR"],
    allow_patterns=["*.json", "*.model", "*.safetensors", "*.txt"],
    ignore_patterns=["*.bin", "*.pth", "*.pt", "optimizer.pt", "training_args.bin"],
)
PY
```

## Run Default Prompt

The default prompt is `The quick brown fox`; `max_new_tokens=1`, greedy
generation, float32 weights, and eager attention are handled by
`tinyllama_generate.py`.

```bash
PYTHONPATH="$HF_DEPS" \
LD_LIBRARY_PATH="$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROCM_PYTHON" ./tinyllama_generate.py \
  --device cpu \
  --model-path "$MODEL_DIR" \
  --output-json results/tinyllama_cpu.json

PYTHONPATH="$HF_DEPS" \
LD_LIBRARY_PATH="$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
HSA_ENABLE_SDMA=1 \
  "$ROCM_PYTHON" ./tinyllama_generate.py \
  --device cuda \
  --model-path "$MODEL_DIR" \
  --output-json results/tinyllama_real_gpu.json

timeout 900s env \
PYTHONPATH="$HF_DEPS" \
LD_LIBRARY_PATH="$ROCJITSU_BUILD:$ROCJITSU_BUILD/lib/rocjitsu/src/rocjitsu/hooks:$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
XDG_RUNTIME_DIR="$XDG_RUNTIME_DIR" \
ROCJITSU_RUNTIME_DIR="$ROCJITSU_RUNTIME_DIR" \
HSA_ENABLE_SDMA=1 \
HSA_HOTSWAP_DISABLE=1 \
  "$ROCJITSU_BUILD/tools/rocjitsu/rocjitsu" \
  --daemon \
  --config "$ROCJITSU_CONFIG" \
  -- "$ROCM_PYTHON" ./tinyllama_generate.py \
  --device cuda \
  --model-path "$MODEL_DIR" \
  --output-json results/tinyllama_sim_gfx1250.json
```

Expected default-prompt output after the `v_cls_i32` fix:

```text
CPU new token:      [432]
real GPU new token: [432]
sim new token:      [432]
decoded text:       <s> The quick brown fox j
```

Before the fix, simulated `gfx1250` produced token `[13]` for this prompt.

## Compare

```bash
PYTHONPATH="$HF_DEPS" "$ROCM_PYTHON" ./tinyllama_compare.py \
  results/tinyllama_cpu.json \
  results/tinyllama_real_gpu.json \
  results/tinyllama_sim_gfx1250.json
```

The fixed simulator should report:

```text
sequence_ids_match: true
new_token_ids_match: true
allclose: true
```

## More Prompts

Pass `--prompt` to `tinyllama_generate.py` and use different output filenames.
Good smoke-test prompts:

```text
A good GPU simulator should
In one sentence, explain matrix multiplication:
Write a short haiku about rain.
The answer to life is
ROCm kernels are useful because
```

For extra prompts, compare token IDs and logits between CPU, real GPU, and
simulated `gfx1250`; do not treat the generated prose as meaningful.

## Notes

- Keep `HSA_HOTSWAP_DISABLE=1` in the simulator command so the run uses the
  intended worktree build instead of an external hotswap library.
- The fixed bug affected int64 position IDs in the LLaMA RoPE path:
  `position_ids[:, None, :].float()`.
