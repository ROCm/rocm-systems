# TinyLlama Reproducer

TinyLlama is now one model entry in the causal-LM smoke-test harness.
Use `../causal_lm/run_validation.py` as the canonical reproducer.

## Setup

```bash
cd /path/to/rocm-systems/emulation/rocjitsu/model_validation/causal_lm

export ROCM_PYTHON=/path/to/rocm-python
export ROCM_LIB_DIR=/path/to/rocm-runtime-libs

export ROCJITSU_ROOT=${ROCJITSU_ROOT:-"$(git rev-parse --show-toplevel)/emulation/rocjitsu"}
export ROCJITSU_BUILD=${ROCJITSU_BUILD:-"$ROCJITSU_ROOT/build-causal-lm"}
export ROCJITSU_CONFIG=${ROCJITSU_CONFIG:-"$ROCJITSU_ROOT/configs/gfx1250.json"}

export HF_DEPS=${HF_DEPS:-"$PWD/.deps/hf"}
export HF_HOME=${HF_HOME:-"$PWD/.cache/huggingface"}
mkdir -p "$HF_DEPS" "$HF_HOME"

"$ROCM_PYTHON" -m pip install \
  --target "$HF_DEPS" \
  transformers huggingface_hub hf_xet tokenizers sentencepiece safetensors numpy

cmake -S "$ROCJITSU_ROOT" -B "$ROCJITSU_BUILD" -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROCJITSU_BUILD" \
  --target rocjitsu_bin rocjitsu_shared rocjitsu_hooks \
  -j "$(nproc)"
```

## Run

```bash
PYTHONPATH="$HF_DEPS" \
LD_LIBRARY_PATH="$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROCM_PYTHON" ./run_validation.py \
  tinyllama_1b
```

The intended passing behavior is that all fixed prompts produce matching CPU,
real-GPU, and simulated `gfx1250` token IDs and logits. See
`../causal_lm/RESULTS.md` for the checked results.
