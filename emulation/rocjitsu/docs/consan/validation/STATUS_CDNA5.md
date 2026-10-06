# ConSan validation on CDNA5 (gfx1250)

This table records the October 5, 2026 revalidation of external workloads in
RocJITsu emulation at revision `c88681ad6a4`, using a freshly rebuilt GCC emulator
and ConSan hook. It covers the existing 31 rows (33 validation IDs), retaining
the selected Tensile shards; unfinished runs are identified below. These results
qualify the recorded emulator
configuration, not physical hardware or its performance.

The campaign retains the previously recorded Default presets, bank/report
capacities, and SuperCollider delay settings. Lower Default presets were not
recalibrated. Green requires matching clean numerical correctness, complete
applicable coverage and runtime evidence, healthy emulator checks, and at least
**6/8** detections among admitted and reached fault trials. Fault fractions
exclude neither detector misses nor numerical failures. An oracle failure alone
is not a detector finding. TP2's fault fraction refers to prefill; all three TP2
modes must pass their clean checks.

Fresh rocprofv3 traces and exact kernel allowlists were generated inside the
emulator with the matching SDK for each workload. Agent traces identify
`gfx1250` (target version 120500). Tensile fault qualification additionally uses
matching clean controls for the exact fault command and retained replay
artifacts where required; these supplement the full-shard clean gate.
All **597** selected gfx1250 ConSan host/simulator tests passed, including
446 simulator tests. The workload campaign established 62 clean configuration
passes and audited 42 complete eight-trial fault batches (336 trials); 26 table
cells qualify green. Twelve additional completed trials belong to incomplete
cohorts and do not qualify those cells. Final emulator discovery and D128 pressure
smoke checks passed; emulator, hook, and fault-spec hashes were unchanged.
This campaign did not rerun ASan/UBSan.

Evidence and reproduction scripts are retained under
`/home/benoit/workspace/consan-validation/cdna5-reevaluation-20261005/`:
`campaign-contract.json`, `settings.json`, `doctor.json`, `build.log`,
`gfx1250-tests.log`, `discovery/`, `allowlists/`, `clean/`, `matching-clean/`,
`fault/`, `clean-retry/`, `discovery-audit.json`, `qualification-audit.json`,
`user-time-budget.json`, `budget-stop-complete.json`, `partial-cohorts.json`,
and `final-health.json`.
Sparse-FP8 Default initially passed two shards but reached only seven of eight
clients in the third before its 1,200 s deadline. The identical third shard
passed in 1,403 s with a 3,600 s limit; the first two passing shards are retained.
The initial timeout and supplemental result remain in the artifacts.
The earlier [workload selection audit](CDNA5_WORKLOAD_AUDIT_20260925.md) and
[campaign setup](CDNA5_REVALIDATION_20260925.md) explain the selected workloads,
profiler preload ordering, and runtime requirements; their measurements remain
historical. Follow [VALIDATION.md](VALIDATION.md) for new qualification runs.
For the global-access scope limitation in `pytorch-scatter-reduce`, see
[SuperCollider for global memory](../SUPERCOLLIDER_GLOBAL_MEMORY.md).

At the requested time budget, unfinished long-running workloads remain yellow.
This includes the full SGEMM clean qualification and incomplete fault cohorts;
partial trials are retained but do not establish a passing fault fraction.
These yellow cells indicate deferred evidence, not a newly observed correctness
failure. The bounded SGEMM smoke is qualified separately.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope (also user-budget deferrals in this campaign);
🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default | SuperCollider |
| --- | ---: | --- | --- | --- |
| Main E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault qualification incomplete at user time budget |
| Main E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 higher: clean pass; fault qualification incomplete at user time budget | 🟨 sleep=15: clean pass; fault qualification incomplete at user time budget |
| Main E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 higher: clean pass; fault qualification incomplete at user time budget | 🟨 sleep=15: clean pass; fault qualification incomplete at user time budget |
| Main E2E | P2 | Sharktank TP2 prefill/decode/combined (`tp2-family`, `tp2-decode`, `tp2-combined`) | 🟨 higher: all 3 clean runs pass; prefill; fault qualification incomplete at user time budget | 🟨 sleep=15: all 3 clean runs pass; prefill; fault qualification incomplete at user time budget |
| Main E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🟨 high: clean pass; fault qualification incomplete at user time budget | 🟨 sleep=15: clean pass; fault qualification incomplete at user time budget |
| Main E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high: clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Main E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high: clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| Main E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Main E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Test corpus | P0 | HipKittens CDNA5 naive BF16 (`hipkittens-bf16fp32-cdna5-naive`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| Tensile | P0 | `002_sk_mxf8gemm_explicit` (`tensile-sk-mxf8gemm-explicit`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; detector 0/8 despite oracle failures 8/8 |
| Tensile | P0 | `003_sk_mxf4gemm_explicit` (`tensile-sk-mxf4gemm-explicit`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8 |
| Tensile | P1 | `037_spmm_tdm_f16_transposes` (`tensile-spmm-tdm-f16-transposes`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep=15: clean pass; fault 0/8 |
| Tensile | P1 | `016_spmm_tdm_all` (`tensile-spmm-tdm-all`) | 🟩 higher + 256 banks: all 4 clean shards pass; fault 8/8 | 🟨 sleep=15: all 4 clean shards pass; fault 0/8 |
| Tensile | P1 | `001_sk_mxf8f4gemm_tdm` (`tensile-sk-mxf8f4gemm-tdm`) | 🟩 high: all 3 clean shards pass; fault 8/8 | 🟨 sleep=15: all 3 clean shards pass; fault 0/8 |
| Tensile | P1 | `004_sk_mxf8gemm_tdm` (`tensile-sk-mxf8gemm-tdm`) | 🟩 high: all 6 clean shards pass; fault 8/8 | 🟨 sleep=15: all 6 clean shards pass; fault 0/8 |
| Tensile | P1 | `007_sk_mxf4gemm_tdm` (`tensile-sk-mxf4gemm-tdm`) | 🟩 high + 192 MiB report cap: all 6 clean shards pass; fault 8/8 | 🟨 sleep=15: all 6 clean shards pass; fault 0/8 |
| Tensile | P1 | bounded Stream-K smoke (`tensile-sk-sgemm-runtime-smoke`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; detector 0/8 despite oracle failures 8/8 |
| Tensile | P2 | `000_sk_sgemm_quick` (`tensile-sk-sgemm-quick`) | 🟨 high + 1 GiB report cap: full clean run stopped at user time budget; qualification incomplete | 🟨 sleep_wave=15: full clean run deferred at user time budget; qualification incomplete |
| Tensile | P2 | `005_sk_f8gemm_quick` (`tensile-sk-f8gemm-quick`) | 🟩 high: all 9 clean shards pass; fault 8/8 | 🟨 sleep=15: all 9 clean shards pass; fault 0/8 |
| Tensile | P2 | `006_sk_hgemm_quick` (`tensile-sk-hgemm-quick`) | 🟩 high: all 6 clean shards pass; fault 7/8 | 🟨 sleep=15: all 6 clean shards pass; fault qualification incomplete at user time budget |
| Tensile | P3 | `015_spmm_f8_ml` (`tensile-spmm-f8-ml`) | 🟨 higher: all 3 clean shards pass; fault qualification incomplete at user time budget | 🟨 sleep_wave=15: all 3 clean shards pass; fault qualification incomplete at user time budget |
| PyTorch | P0 | tensor-descriptor add (`pytorch-tdm-descriptor-add`) | 🟨 default: clean pass; barrier drop does not create a cross-wave race | 🟨 sleep_wave=15: clean pass; barrier drop does not create a cross-wave race |
| PyTorch | P0 | `torch.mode` (`pytorch-torch-mode`) | 🟩 high: clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| PyTorch | P0 | `torch.topk` (`pytorch-torch-topk`) | 🟩 higher: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P1 | `torch.sort` (`pytorch-torch-sort`) | 🟩 high: clean pass; fault 8/8 | 🟩 sleep_wave=15: clean pass; fault 8/8 |
| PyTorch | P1 | `scatter_reduce` (`pytorch-scatter-reduce`) | 🟨 numerical pass; global-only workload outside LDS detector scope | 🟨 numerical pass; global-only workload outside SuperCollider scope |
| PyTorch | P1 | `torch.histc` (`pytorch-torch-histc`) | 🟩 high + 256 banks: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P2 | norm/softmax (`pytorch-norm-softmax`) | 🟩 high: clean pass; fault 8/8 | 🟨 sleep_wave=15: clean pass; fault 0/8 |
| PyTorch | P1 | cluster synchronization (`pytorch-cluster-load-sync`) | 🟨 default: clean pass; same-lane LDS accesses, barrier drop does not create a race | 🟨 delay-zero: clean pass; same-lane LDS accesses, barrier drop does not create a race |
