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
green requires matching clean correctness, complete applicable coverage, healthy
emulator checks, and at least **6/8** fault detections. Fractions count detections
among admitted and reached fault trials. Default cells name the lowest passing
preset at or above `default`; SuperCollider cells name the tested controls.
Emulator results do not qualify physical hardware or measure its performance.
Follow [VALIDATION.md](VALIDATION.md), beginning with allowlist discovery.
This campaign uses rocprofv3 inside the gfx1250 emulator; see the
[campaign setup and evidence](CDNA5_REVALIDATION_20260925.md) for profiler ordering
and matching runtime requirements.
For the global-access scope limitation relevant to `pytorch-scatter-reduce`, see
[SuperCollider for global memory](../SUPERCOLLIDER_GLOBAL_MEMORY.md).

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟨 default: all 3 filtered clean runs pass; fault trials pending | 🟨 delay-zero: all 3 filtered clean runs pass; fault trials pending |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 delay-zero: filtered clean pass; fault 0/8; calibration pending |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟥 default: clean run reports 3 conflicts; CDNA5 publication journal missing | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Test corpus | P0 | HipKittens CDNA5 naive BF16 (`hipkittens-bf16fp32-cdna5-naive`) | 🟥 default: heap corruption at teardown after instrumented run | 🟥 delay-zero: heap corruption at teardown after instrumented run |
| Tensile | P0 | `002_sk_mxf8gemm_explicit` (`tensile-sk-mxf8gemm-explicit`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Tensile | P0 | `003_sk_mxf4gemm_explicit` (`tensile-sk-mxf4gemm-explicit`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Tensile | P1 | `037_spmm_tdm_f16_transposes` (`tensile-spmm-tdm-f16-transposes`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Tensile | P1 | `016_spmm_tdm_all` (`tensile-spmm-tdm-all`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| Tensile | P1 | `001_sk_mxf8f4gemm_tdm` (`tensile-sk-mxf8f4gemm-tdm`) | 🟧 full-shard profiler discovery interrupted; longer-deadline retry pending | 🟧 full-shard profiler discovery interrupted; longer-deadline retry pending |
| Tensile | P1 | `004_sk_mxf8gemm_tdm` (`tensile-sk-mxf8gemm-tdm`) |  |  |
| Tensile | P1 | `007_sk_mxf4gemm_tdm` (`tensile-sk-mxf4gemm-tdm`) |  |  |
| Tensile | P1 | bounded Stream-K smoke (`tensile-sk-sgemm-runtime-smoke`) |  |  |
| Tensile | P2 | `000_sk_sgemm_quick` (`tensile-sk-sgemm-quick`) | 🟧 full-shard profiler discovery interrupted; longer-deadline retry pending | 🟧 full-shard profiler discovery interrupted; longer-deadline retry pending |
| Tensile | P2 | `005_sk_f8gemm_quick` (`tensile-sk-f8gemm-quick`) |  |  |
| Tensile | P2 | `006_sk_hgemm_quick` (`tensile-sk-hgemm-quick`) |  |  |
| Tensile | P3 | `015_spmm_f8_ml` (`tensile-spmm-f8-ml`) |  |  |
| PyTorch | P0 | tensor-descriptor add (`pytorch-tdm-descriptor-add`) | 🟥 default: heap corruption at teardown after instrumented run | 🟥 delay-zero: heap corruption at teardown after instrumented run |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟨 default: filtered clean pass (180 s deadline); fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only workload outside LDS detector scope | 🟨 numerical pass; global accesses outside SuperCollider scope |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
| PyTorch | P1 | cluster synchronization (`pytorch-cluster-load-sync`) | 🟨 default: filtered clean pass; fault trials pending | 🟨 delay-zero: filtered clean pass; fault trials pending |
