# ConSan validation on CDNA5 (gfx1250)

This table records the September 25–26, 2026 revalidation of external workloads
in RocJITsu emulation. All rows have been assessed; orange cells retain measured
clean-run execution limits. Historical results are retained in git history. Rows use the existing gfx1250 workload manifest:
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
[campaign setup and evidence](CDNA5_REVALIDATION_20260925.md) for profiler ordering,
matching runtime requirements, and the hook builds used by each campaign.
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
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 high: clean pass at 1200 s; fault 8/8 | 🟨 sleep=15: clean pass at 1200 s; fault 0/8 |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 higher: filtered clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8 |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 higher: clean pass with complete coverage; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8 |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 higher: all 3 clean runs pass; prefill fault 8/8 | 🟨 sleep=15: all 3 clean runs pass; prefill fault 0/8 |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 sleep=15: filtered clean pass; fault 0/8 |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high: clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high: filtered clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 high: clean pass; fault 8/8 with complete publication evidence | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Test corpus | P0 | HipKittens CDNA5 naive BF16 (`hipkittens-bf16fp32-cdna5-naive`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Tensile | P0 | `002_sk_mxf8gemm_explicit` (`tensile-sk-mxf8gemm-explicit`) | 🟩 high: complete clean pass with tensor-DMA coverage; fault 8/8 | 🟨 sleep_wave=15: clean pass with complete tensor-DMA comparison coverage; detector 0/8 despite oracle failures 8/8 |
| Tensile | P0 | `003_sk_mxf4gemm_explicit` (`tensile-sk-mxf4gemm-explicit`) | 🟩 high: complete clean pass with tensor-DMA coverage; fault 8/8 | 🟨 sleep=15: clean pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | `037_spmm_tdm_f16_transposes` (`tensile-spmm-tdm-f16-transposes`) | 🟩 high: all 4 clean objects pass with complete tensor-DMA coverage; fault 6/8 | 🟨 sleep=15: all 4 clean objects pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | `016_spmm_tdm_all` (`tensile-spmm-tdm-all`) | 🟩 higher + 256 banks: all 4 clean shards pass with complete tensor-DMA coverage; fault 8/8 | 🟨 sleep=15: all 4 clean shards pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | `001_sk_mxf8f4gemm_tdm` (`tensile-sk-mxf8f4gemm-tdm`) | 🟩 high: all 3 clean shards pass at 1200 s with complete tensor-DMA coverage; fault 8/8 | 🟨 sleep=15: all 3 clean shards pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | `004_sk_mxf8gemm_tdm` (`tensile-sk-mxf8gemm-tdm`) | 🟩 high: all 6 clean shards pass with complete tensor-DMA coverage; fault 8/8 | 🟨 sleep=15: all 6 clean shards pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | `007_sk_mxf4gemm_tdm` (`tensile-sk-mxf4gemm-tdm`) | 🟩 high + 192 MiB report cap: all 6 clean shards pass with complete tensor-DMA coverage; fault 8/8 | 🟨 sleep=15: all 6 clean shards pass with complete tensor-DMA comparison coverage; fault 0/8 |
| Tensile | P1 | bounded Stream-K smoke (`tensile-sk-sgemm-runtime-smoke`) | 🟩 high: complete clean pass; fault 8/8 | 🟨 sleep_wave=15: timed and untimed clean passes; detector 0/8 despite oracle failures 8/8 |
| Tensile | P2 | `000_sk_sgemm_quick` (`tensile-sk-sgemm-quick`) | 🟧 high + 1 GiB cap: 3/6 full-clean shards pass; 511/512/513 time out at 14,400 s; untimed matching clean passes; fault 8/8 | 🟧 sleep_wave=15, untimed: matching clean passes; detector 0/8, oracle failures 8/8; full clean 5/6; 513 times out at 14,400 s |
| Tensile | P2 | `005_sk_f8gemm_quick` (`tensile-sk-f8gemm-quick`) | 🟩 high: all 9 clean shards pass; exact-artifact fault 8/8 | 🟨 sleep=15: all 9 clean shards pass with complete coverage; earlier-build fault 0/8 |
| Tensile | P2 | `006_sk_hgemm_quick` (`tensile-sk-hgemm-quick`) | 🟩 high: all 6 clean shards pass; exact-artifact fault 6/8 | 🟨 sleep=15: all 6 clean shards pass with complete coverage; earlier-build fault 0/8 |
| Tensile | P3 | `015_spmm_f8_ml` (`tensile-spmm-f8-ml`) | 🟩 higher: all 3 clean shards pass with complete coverage; fault 8/8 | 🟨 sleep_wave=15: all 3 clean shards pass with complete coverage; detector 0/8 despite oracle failures 8/8 |
| PyTorch | P0 | tensor-descriptor add (`pytorch-tdm-descriptor-add`) | 🟨 default: clean pass with complete tensor-load/store coverage; barrier drop does not create a cross-wave race | 🟨 sleep_wave=15: clean pass with complete tensor-load/store coverage; barrier drop does not create a cross-wave race |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 high: filtered clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 higher: clean pass with complete coverage; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 high: filtered clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only workload outside LDS detector scope | 🟨 numerical pass; global accesses outside SuperCollider scope |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 high + 256 banks: repaired lane retention; clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟩 high: filtered clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P1 | cluster synchronization (`pytorch-cluster-load-sync`) | 🟨 default: clean pass; same-lane LDS accesses, barrier drop does not create a race | 🟨 delay-zero: clean pass; same-lane LDS accesses, barrier drop does not create a race |
