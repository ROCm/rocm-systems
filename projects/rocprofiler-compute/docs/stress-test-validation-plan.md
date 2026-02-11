# rocprofiler-compute Stress Test: Framework & Model Validation Plan

**Goal:** Stress test rocprofiler-compute across ML frameworks and select a practical set of models for validation on AMD MI GPUs (MI100, MI200, MI300, MI350).

---

## 1. Frameworks (deduplicated and categorized)

| Framework | Role | Typical use | ROCm support |
|-----------|------|-------------|--------------|
| **PyTorch** | Training + inference | Native ROCm; most model ecosystems | ✅ Primary |
| **JAX** | Training + inference | Often via Flax; ROCm via jax-rocm | ✅ |
| **Megatron** | Training | Large-scale LLM training (NVIDIA-origin, AMD ports) | ⚠️ Check port status |
| **Triton Inference Server** | Inference serving | Multi-framework backend; ROCm support varies | ⚠️ Check backend |
| **vLLM** | Inference | High-throughput LLM serving; PyTorch/CUDA/ROCm | ✅ |
| **SGLang** | Inference | RadixAttention, LLM serving; ROCm support | ✅ |
| **Hugging Face TGI** | Inference | Text Generation Inference; ROCm support | ✅ |

**Recommendation:** Treat **PyTorch** and **JAX** as the two core “framework backends”; use **vLLM**, **SGLang**, and **TGI** as the main **inference serving** stacks to stress test. Add **Megatron** and **Triton** if your team has existing setups or needs coverage for them.

---

## 2. Models: availability and constraints

| Model | Type | Size / variant | Open weights? | Framework fit | Notes |
|-------|------|----------------|---------------|---------------|--------|
| **DeepSeek-V3** | LLM | 671B (MoE) | ✅ | vLLM, SGLang, TGI, PyTorch | Heavy; good stress test |
| **DeepSeek-V3.2** | LLM | Multiple | ✅ | Same as V3 | Newer; validate both if resources allow |
| **DeepSeek-R1** | Reasoning | 671B (MoE) | ✅ | vLLM, SGLang, TGI | Reasoning-specific patterns |
| **Grok 2 8T** | LLM | 8T params | ❌ Proprietary | N/A for open validation | Drop for open stress test |
| **Qwen3** | LLM | 8B–72B+ | ✅ | vLLM, SGLang, TGI, PyTorch | Good small–medium coverage |
| **Qwen3 Coder 480B** | Code LLM | 480B | ✅ | vLLM, SGLang, TGI | Large; strong compute stress |
| **Qwen3 VL 235B** | Vision–Language | 235B | ✅ | Depends on VL support in serving | VL path; different kernels |
| **GPT-OSS** | LLM | Varies | ✅ (name suggests open) | Depends on implementation | Clarify exact model/repo |
| **Kimi K2 Thinking** | Reasoning | Large | ⚠️ Check license | If available: vLLM/SGLang | Reasoning; validate if accessible |
| **GLM-4.5** | LLM | Multiple | ✅ (e.g. GLM-4) | vLLM, TGI, PyTorch | Good diversity |
| **Proprietary** | — | — | ❌ | Internal only | Keep 1–2 for internal validation only |

---

## 2.1 Model details: parameter size and disk space

All disk figures are estimates. Actual usage depends on precision (FP16/BF16/FP8/INT8), quantization, and how long you profile.

**Storage rules of thumb**

- **Weights (FP16/BF16):** 2 bytes/param → **≈ 2 GB per 1B parameters** (e.g. 70B ≈ 140 GB).
- **Weights (FP8):** 1 byte/param → **≈ 1 GB per 1B parameters** (when available).
- **Training full checkpoint (weights + optimizer):** **≈ 3× weight size** per checkpoint (Adam: weights + momentum + variance).
- **rocprofiler-compute output:** Depends on run length and metrics. Typical ranges: **~0.1–2 GB** (short inference), **~5–50 GB** (long inference), **~50–500 GB** (short training), **~200 GB–2 TB** (long/multi-step training with many kernels).

**Abbreviations:** *T = total params (B)*, *A = active params per token (B)*, *MoE = mixture-of-experts.*

| Model | Parameters (T / A) | Weights only (FP16/BF16) | Weights only (FP8) | Disk: inference profiling (approx.) | Disk: training profiling (approx.) |
|-------|--------------------|--------------------------|--------------------|--------------------------------------|------------------------------------|
| **DeepSeek-V3** | 671B / 37B (MoE) | ~1.34 TB | ~671 GB | **1.4–1.5 TB** | **4–6 TB** |
| **DeepSeek-V3.2** | 671B / 37B (MoE) | ~1.34 TB | ~671 GB | **1.4–1.5 TB** | **4–6 TB** |
| **DeepSeek-R1** | 671B / 37B (MoE) | ~1.34 TB | ~671 GB | **1.4–1.5 TB** | **4–6 TB** |
| **Grok-2** (open ~270B) | 270B / 115B (MoE) | ~540 GB | ~270 GB | **560–640 GB** | **1.7–2.5 TB** |
| **Grok 2 8T** (if 8T params) | 8T (proprietary) | ~16 TB | ~8 TB | **16+ TB** | **50+ TB** (out of scope for open validation) |
| **Qwen3-8B** | 8B (dense) | ~16 GB | ~8 GB | **20–50 GB** | **80–200 GB** |
| **Qwen3-14B** | 14B (dense) | ~28 GB | ~14 GB | **35–80 GB** | **120–300 GB** |
| **Qwen3-32B** | 32.8B (dense) | ~66 GB | ~33 GB | **70–170 GB** | **270–700 GB** |
| **Qwen3-30B-A3B** | 30B / 3B (MoE) | ~60 GB | ~30 GB | **65–160 GB** | **250–650 GB** |
| **Qwen3-235B-A22B** | 235B / 22B (MoE) | ~470 GB | ~235 GB | **490–570 GB** | **1.5–2.5 TB** |
| **Qwen3 Coder 480B** | 480B / 35B (MoE) | ~960 GB | ~480 GB | **1.0–1.1 TB** | **3–5 TB** |
| **Qwen3 VL 235B** | 235B / 22B (MoE) | ~470 GB | ~235 GB | **490–570 GB** | **1.5–2.5 TB** |
| **GPT-OSS** | (define exact model) | (depends) | (depends) | (depends) | (depends) |
| **Kimi K2** | 1T / 32B (MoE) | ~2 TB | ~1 TB | **2.0–2.2 TB** | **6–10 TB** |
| **GLM-4.5** | 355B / 32B (MoE) | ~710 GB | ~355 GB | **730–810 GB** | **2.2–3.5 TB** |
| **GLM-4.5-Air** | 106B / 12B (MoE) | ~212 GB | ~106 GB | **220–310 GB** | **0.7–1.2 TB** |
| **Proprietary** | (your model) | (your size) | (if FP8) | (weights + 20–100 GB) | (4–7× weights + 100 GB–2 TB) |

**Inference profiling disk** = model weights (on disk) + rocprofiler-compute output (tens of GB for short runs, up to ~100–200 GB for long or kernel-heavy runs). KV cache is usually in VRAM and not written to disk.

**Training profiling disk** = model weights + full checkpoints (typically 3× weight size per checkpoint; 1–2 checkpoints assumed) + rocprofiler-compute output (large for multi-step training: 50 GB–2 TB depending on steps and metrics). So total ≈ **4×–7× weight size + profiling output**.

Use **FP8** columns when the model is served or trained in FP8 (e.g. Qwen3-Coder-480B-Instruct-FP8) to reduce weight storage.

**Smaller Qwen3 variants (for minimal disk / CI):** Qwen3-0.6B (~1.2 GB), Qwen3-1.7B (~3.4 GB), Qwen3-4B (~8 GB). Inference profiling then fits in **&lt; 20 GB**; training in **&lt; 50 GB**.

**Quick formulas**

- **Weights (GB)** ≈ params_B × 2 (FP16/BF16) or params_B × 1 (FP8).
- **Inference profiling (GB)** ≈ weights + 20–100 (short run) or + 50–200 (long run).
- **Training profiling (GB)** ≈ 4×–7× weights + 100–500 (short) or 200–2000 (long run).

---

## 3. Suggested model set for validation

Select a **small core set** for daily/CI stress tests and an **extended set** for deeper validation.

### 3.1 Core validation set (must-have)

- **Qwen3 (e.g. 8B or 32B)**  
  - Fast, widely supported, good for “smoke” and regression.
- **DeepSeek-V3 or DeepSeek-V3.2 (one of)**  
  - Large MoE; stresses multi-GPU, memory, and kernel diversity.
- **DeepSeek-R1 (if same infra as V3)**  
  - Covers reasoning-style workloads; add if no extra infra cost.
- **One code model**  
  - **Qwen3 Coder** (e.g. 32B if 480B is too heavy) or similar — stresses different attention/code patterns.

### 3.2 Extended validation set (when resources allow)

- **Qwen3 Coder 480B** — max compute/memory stress.
- **Qwen3 VL 235B** — if your stack supports VL and you care about vision kernels.
- **GLM-4 / GLM-4.5** — architecture diversity.
- **Kimi K2 Thinking** — only if weights and license are available and you need reasoning coverage.

### 3.3 Explicitly exclude from open validation

- **Grok 2 8T** — proprietary; no open weights.
- **Proprietary** — keep as a separate, internal-only category (1–2 models if needed).

---

## 4. Framework × model matrix (recommended)

Use this to assign models to frameworks so that each framework is stressed and no model is over-tested.

| Framework | Core models | Extended (optional) |
|-----------|-------------|----------------------|
| **PyTorch** (native) | Qwen3 8B/32B, GLM-4 | DeepSeek-V3, Qwen3 Coder |
| **JAX** (e.g. Flax) | Qwen3 small, GLM-4 (if available) | — |
| **vLLM** | Qwen3 32B, DeepSeek-V3, DeepSeek-R1 | Qwen3 Coder 480B, Qwen3 VL 235B |
| **SGLang** | Qwen3 32B, DeepSeek-V3 | DeepSeek-R1, Qwen3 Coder |
| **TGI** | Qwen3 8B/32B, GLM-4 | DeepSeek-V3 |
| **Megatron** | — | One large model if port exists (e.g. 70B class) |
| **Triton** | — | Same as backend (e.g. vLLM or PyTorch) |

---

## 5. What to stress test with rocprofiler-compute

- **Profile**  
  - `rocprof-compute profile -n <workload_name> -- <launch_cmd>`  
  - Confirm no hangs, no missing kernels, and that output dirs and CSVs are produced.
- **Analyze**  
  - Run `rocprof-compute analyze` on the produced workloads; check roofline, PMC, and any block-specific reports.
- **Scenarios**  
  - Single-GPU and multi-GPU (where applicable).  
  - Short runs (few steps / few tokens) for CI; longer runs for weekly stress.  
  - Different batch sizes and sequence lengths for inference.

---

## 6. Decisions to make as a team

1. **Megatron / Triton**  
   - Include only if you have (or will add) ROCm-capable setups; otherwise mark “out of scope” for this round.
2. **DeepSeek V3 vs V3.2**  
   - Pick one for core set to avoid duplication unless you need both.
3. **Qwen3 Coder 480B**  
   - Include if hardware (e.g. 8× MI300X) and time allow; otherwise use a smaller Coder variant in core set.
4. **Qwen3 VL 235B**  
   - Include only if VL is in scope and your serving stack supports it on ROCm.
5. **GPT-OSS**  
   - Replace with a concrete model name and repo once defined; then add to core or extended set.
6. **Proprietary**  
   - Define 1–2 internal models and run them only in internal CI, not in open/public validation.

---

## 7. Next steps

1. Confirm framework list (drop or add Megatron/Triton).  
2. Lock core model set (e.g. Qwen3 8B/32B, DeepSeek-V3 or V3.2, DeepSeek-R1, one code model, GLM-4).  
3. Assign each core model to at least one framework in the matrix above.  
4. Add a small script or CI job that runs `rocprof-compute profile` + `analyze` for each (framework, model) pair and checks exit code and output presence.  
5. Schedule extended set (e.g. 480B, VL, Kimi K2) for weekly or manual runs.

Once you confirm frameworks and the core model set, we can add a minimal `tests/stress/` (or similar) layout and example commands under `rocprofiler-compute`.
