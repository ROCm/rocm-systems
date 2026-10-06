# ConSan validation on CDNA4 (gfx950)

Requalified on **October 5, 2026**, on a physical MI350X using only TheRock
`10.2.0a20261005`. The measured RocJITsu revision is `c88681ad6a` from
`shared/rocjitsu/sanitizers`, preserved across the campaign interruption;
hook SHA256 is `903301e8b07926c8a60344b78fdce16c6c87fbb2006f675573c8a21ea9a80d19`.
The end-to-end environment uses Python `3.12.15`, matching PyTorch
`2.15.0a0+rocm10.2.0a20261005`, HIP `7.17.26392`, and IREE `3.12.0`.
All 26 manifest workloads have fresh native numerical controls and profiler
allowlists. Both instrumented modes passed clean qualification in all 25
applicable workloads; `scatter_reduce` remains outside detector scope.

Fault counts are detections, with a prospective bar of **6/8**. Every displayed
fault batch ran eight admitted/reached trials with healthy pre/post probes.
Green also requires matching clean correctness and complete applicable coverage
and evidence. Each October 5 Default search starts at `default` and advances through
`high`, `higher`, then `max`, stopping at the first green result. “Lowest passing”
means all lower settings were tested and failed qualification. October 5 preset searches
use automatic bank sizing. Updated priorities follow the current executable manifest. [Procedure](VALIDATION.md).

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is below bar or outside detector scope;
🟧 clean qualification blocked by prerequisites, coverage/evidence, or timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟩 high (lowest passing): clean pass; fault 6/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟩 higher (lowest passing): clean pass; fault 8/8; supporting decode/combined clean pass | 🟨 sleep=15: clean pass; fault 0/8; below bar; supporting decode/combined clean pass |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟩 high (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟩 sleep=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P4 | hip-moi MFMA attention (`wmma-attention`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟩 sleep=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 higher (lowest passing): repaired hip-moi atomic cache; clean pass; fault 8/8 (bar 6/8) | 🟨 sleep=15: clean pass; fault 0/8 (bar 6/8); below bar |
| Test corpus | P0 | HIP matmul 128 cubed (`hip-matmul-m128-n128-k128`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P0 | HipKittens BF16 (`hipkittens-bf16fp32-16x32`) | 🟨 max: clean pass; fault 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P1 | HipKittens FP8 (`hipkittens-fp8fp32-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P1 | HipKittens MXFP8 (`hipkittens-mxfp8-4wave`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P1 | HIP Stream-K simple (`hip-streamk-simple-m256-n256-k256`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P1 | HIP Stream-K two-tile (`hip-streamk-two-tile-m256-n256-k256`) | 🟩 high (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Test corpus | P2 | rocBLAS SGEMM square-64 (`rocblas-sgemm-square-64`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| Tensile | P0 | gfx950 LDS-positive BF16 GEMM (`tensile-gfx950-lds-positive`) | 🟩 max + 64 banks: repaired lane retention; clean pass; fault 8/8 (bar 6/8) | 🟩 delay-zero: clean pass; fault 8/8 (bar 6/8) |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 2/8; below bar |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 higher (lowest passing): clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 higher (lowest passing): clean pass; fault 7/8 | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟨 max: clean pass; fault 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8; below bar |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage | 🟨 Out of scope: numerical pass; traced global-atomic kernels have no applicable LDS/FLAT race coverage |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟨 max: clean pass; fault 0/8; below bar | 🟨 sleep=15: clean pass; fault 0/8; below bar |

All 23 applicable reviewed fault recipes were regenerated from current binary
identities and final ISA. The October 5 campaign completed 21 searches through top-k.
Each assessed configuration has
eight prospective trials and a fixed six-detection bar. Unsuccessful lower
presets and previous named-configuration runs remain in the campaign artifacts. TP2 decode and combined are supporting clean controls for the
TP2-family row; `scatter_reduce` has no qualifying fault recipe.

The Sharktank decode attention dispatch is now number 24. The old PyTorch
softmax barrier selector is absent; the norm/softmax row uses the reviewed
`NormTwoOps` reduction publication probe and therefore tests a different site
from the September softmax fault. Tensile uses a physical client wrapper and
hashed, byte-identical native replay inputs. Its regenerated initial LDS store
uses address VGPR `v2`, replacing the old binary's `v54`. No old selectors were reused for the October 5 fault runs. The old explicit HipKittens BF16 512-bank configuration is not
reused; this search uses the named presets with automatic bank sizing. [Earlier BF16 analysis](HIPKITTENS_CDNA4_ANALYSIS.md) remains historical.

[Combined results](</home/benjacob/consan-rerun-20261005/validation-summary.json>)
retain each clean result and fault summary, including admission, reach,
detection intervals, numeric manifestations, and health evidence.
[Ascending search and per-preset results](</home/benjacob/consan-rerun-20261005/ascending/search-summary.json>)
retain all attempted presets and the selected lowest passing setting.
[Completion audit](</home/benjacob/consan-rerun-20261005/validation-completion-audit.json>)
independently verifies every attempted fault batch against its raw trial results,
reviewed selectors, fixed threshold, admission/reach witnesses, and health probes.
[Reviewed recipes](</home/benjacob/consan-rerun-20261005/fault-spec.reviewed.json>)
and [final ISA review](</home/benjacob/consan-rerun-20261005/fault-review-contexts.json>)
retain the exact prospective selections.
[Preparation and source identities](</home/benjacob/consan-rerun-20261005/README.md>)
record the external-source patches, TheRock-only paths, package versions, and
native Tensile discovery/replay procedure. These source and toolchain changes
limit direct comparison with the September campaign.

The refreshed ConSan test suite also passed **3,862 tests**, including **445
physical gfx950 tests**, with no failures or skips; its
[results](</home/benjacob/rocm-systems/emulation/rocjitsu/build/consan-results.xml>)
are retained. The two optional host transform/retry benchmarks subsequently
[passed](</home/benjacob/consan-rerun-20261005/standalone-benchmarks.xml>) with an
eligible generated RDNA4 input; they are CPU tests, separate from the physical
gfx950 qualification above.
