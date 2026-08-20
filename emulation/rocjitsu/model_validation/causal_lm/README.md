# Causal-LM Smoke Test

This directory is a small harness for checking whether rocjitsu can run Hugging
Face causal-LM models on simulated `gfx1250` and match CPU/real-GPU output for
fixed one-token generation prompts.

The fixed workload is five prompts:

- `The quick brown fox`
- `A good GPU simulator should`
- `In one sentence, explain matrix multiplication:`
- `ROCm kernel launch status:`
- `The answer is 42 because`

Each prompt uses:

- generation: greedy, `max_new_tokens=1`
- dtype: float32
- attention: eager
- validation: compare real GPU and simulated `gfx1250` against CPU token IDs and logits

## Setup

Run from this directory:

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

List the model keys in `models.json`, then run one:

```bash
PYTHONPATH="$HF_DEPS" \
LD_LIBRARY_PATH="$ROCM_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  "$ROCM_PYTHON" ./run_validation.py tinyllama_1b
```

The initial model keys are:

- `tinyllama_1b`
- `qwen3_0_6b`
- `qwen2p5_1p5b`
- `llama32_1b`

Each run downloads the model if needed, then runs CPU, real GPU, simulated
`gfx1250`, and comparison. Results are written under `results/<model-key>/`.
Current checked results are summarized in `RESULTS.md`.

## Outputs

- `<model-key>_cpu.json`
- `<model-key>_real_gpu.json`
- `<model-key>_sim_gfx1250.json`
- `<model-key>_summary.json`
- per-step stdout, stderr, and timing JSON files

## Notes

- `HSA_ENABLE_SDMA=1` is set for real GPU and simulator runs.
- `HSA_HOTSWAP_DISABLE=1` is set for simulator runs so the selected rocjitsu
  launcher controls execution.
- Simulator runtime directories default to short paths under `$TMPDIR` or the
  system temporary directory to avoid daemon socket path-length issues.
