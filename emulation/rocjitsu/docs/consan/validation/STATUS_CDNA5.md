# ConSan validation on CDNA5 (gfx1250)

This table starts the September 25, 2026 revalidation of external workloads in
RocJITsu emulation. Empty cells mean **not yet revalidated**; historical results
are retained in git history. Rows use the existing gfx1250 workload manifest:
IREE/Sharktank, five hip-moi fixtures, the CDNA5 HipKittens BF16 kernel,
Tensile configurations, and PyTorch/Triton operations including tensor descriptors
and cluster synchronization. The unavailable gfx1250 Jakub fixture and broad
survey summaries are excluded, as is the FP16 sparse sweep whose selected
inputs fail solution applicability checks. See the
[workload audit and exact commands](CDNA5_WORKLOAD_AUDIT_20260925.md).
hip-moi ports can be added when implemented; corpus and Torch selections use
existing target support.

Qualification follows [RDNA4](STATUS_RDNA4.md) and [CDNA4](STATUS_CDNA4.md):
🟩 matching clean correctness, complete applicable coverage, healthy emulator
checks, and at least **6/8** fault detections; 🟨 pending, below bar, or outside
scope; 🟥 correctness or instrumentation failure. Fractions count detections
among admitted and reached fault trials. Default cells name the lowest passing
preset at or above `default`; SuperCollider cells name the tested controls.
Emulator results do not qualify physical hardware or measure its performance.
Follow [VALIDATION.md](VALIDATION.md), beginning with allowlist discovery;
record the prerequisite gap if matching native rocprofv3 profiling is unavailable.
For the global-access scope limitation relevant to `pytorch-scatter-reduce`, see
[SuperCollider for global memory](../SUPERCOLLIDER_GLOBAL_MEMORY.md).

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) |  |  |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) |  |  |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) |  |  |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) |  |  |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) |  |  |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) |  |  |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) |  |  |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) |  |  |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) |  |  |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) |  |  |
| Test corpus | P0 | HipKittens CDNA5 naive BF16 (`hipkittens-bf16fp32-cdna5-naive`) |  |  |
| Tensile | P0 | `002_sk_mxf8gemm_explicit` (`tensile-sk-mxf8gemm-explicit`) |  |  |
| Tensile | P0 | `003_sk_mxf4gemm_explicit` (`tensile-sk-mxf4gemm-explicit`) |  |  |
| Tensile | P1 | `037_spmm_tdm_f16_transposes` (`tensile-spmm-tdm-f16-transposes`) |  |  |
| Tensile | P1 | `016_spmm_tdm_all` (`tensile-spmm-tdm-all`) |  |  |
| Tensile | P1 | `001_sk_mxf8f4gemm_tdm` (`tensile-sk-mxf8f4gemm-tdm`) |  |  |
| Tensile | P1 | `004_sk_mxf8gemm_tdm` (`tensile-sk-mxf8gemm-tdm`) |  |  |
| Tensile | P1 | `007_sk_mxf4gemm_tdm` (`tensile-sk-mxf4gemm-tdm`) |  |  |
| Tensile | P1 | bounded Stream-K smoke (`tensile-sk-sgemm-runtime-smoke`) |  |  |
| Tensile | P2 | `000_sk_sgemm_quick` (`tensile-sk-sgemm-quick`) |  |  |
| Tensile | P2 | `005_sk_f8gemm_quick` (`tensile-sk-f8gemm-quick`) |  |  |
| Tensile | P2 | `006_sk_hgemm_quick` (`tensile-sk-hgemm-quick`) |  |  |
| Tensile | P3 | `015_spmm_f8_ml` (`tensile-spmm-f8-ml`) |  |  |
| PyTorch | P0 | tensor-descriptor add (`pytorch-tdm-descriptor-add`) |  |  |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) |  |  |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) |  |  |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) |  |  |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) |  |  |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) |  |  |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) |  |  |
| PyTorch | P1 | cluster synchronization (`pytorch-cluster-load-sync`) |  |  |
