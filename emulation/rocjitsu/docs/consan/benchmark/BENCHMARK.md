# ConSan benchmarking

This guide defines ConSan performance measurement. Benchmarking is deliberately
separate from correctness qualification in
[`validation/`](../validation/VALIDATION.md): benchmark results cannot promote a
validation cell, and validation status does not carry latency or overhead
claims.

The active benchmark targets are [gfx950](STATUS_GFX950.md) and
[gfx1201](STATUS_GFX1201.md). The [gfx942](STATUS_GFX942.md),
[gfx1100](STATUS_GFX1100.md), and [gfx1250](STATUS_GFX1250.md) ledgers are
placeholders for future expansion, not part of the current required matrix.
Unlike correctness ledgers, benchmark ledgers are specific to one concrete
target because performance does not generalize across all products implementing
an architecture family.

## Contract

The benchmark suite must be automatic and reproducible with one command. Its
latency target is less than 30 minutes over the complete default audited
baseline-plus-ConSan matrix on a warm machine with one supported GPU and at
least 16 GiB of device memory. Initial baselines may exceed that target while
the corpus and transform costs are being characterized; the runner records the
actual elapsed time rather than rejecting otherwise valid results. The target
includes the paired audit-on/audit-off work needed to measure the site-audit
cost and setup performed on every ordinary run;
preinstallation and a populated model cache may be documented prerequisites,
but hidden manual preparation may not be.

Every workload must:

- execute an end-to-end, production-shaped operation with a bounded input;
- run unchanged in the native baseline and each supported ConSan mode;
- retain an independent output or numerical oracle in timed and untimed runs;
- record source, executable, code-object, framework, compiler, runtime, target,
  input, and configuration identities;
- report end-to-end latency or throughput in the workload's natural unit; and
- fit on a 16 GiB GPU without relying on multi-GPU sharding.

The suite should span independently generated ISA rather than multiply nearly
identical model sizes. Framework or generator provenance must be observed from
the executed code objects. A model name alone does not prove whether a GEMM came
from Torch, Triton, Gluon, rocBLAS, hipBLASLt, or Tensile.

## Coverage audit

Every benchmark run verifies site instrumentation by default. A result is
admissible only when every site selected by the ordinary policy was
instrumented, static and dynamic analysis completed, and the final coverage
verdict passed. The audit has negligible measured cost, so its latency is the
ordinary instrumented latency; the suite does not run a second audit-disabled
control for every cell.

The committed audited suite should ultimately remain below the 30-minute target
so a new binary can be qualified without an unbounded preliminary campaign.

Each process measures one synchronized end-to-end operation with its numerical
oracle. PyTorch loads GPU code objects lazily, so ConSan transformation can be
interleaved with that operation rather than forming one framework-visible
phase. The hook therefore maintains a monotonic process clock covering the
union of intervals spent preparing instrumented replacement code objects.
Native samples bracket the four-mode matrix to expose drift.

For each mode, target status files report two values:

- **startup (seconds)** is the hook-measured time spent preparing selected
  instrumented code objects before or during the workload. Concurrent
  transformations count their union in wall time, not their summed CPU time.
- **runtime (ratio)** is the synchronized instrumented workload time after
  subtracting in-workload ConSan transformation time, divided by the bracketed
  median native workload time.

Detailed phase measurements, instrumentation-clock snapshots, audit evidence,
and provenance remain in the machine-readable benchmark artifacts.

## External Aorta checkout

Workloads come from an external ROCm/Aorta checkout. No Aorta source or model
artifact is vendored into RocJITsu. The future runner will require:

```sh
export CONSAN_BENCHMARK_AORTA_DIR=/path/to/aorta
```

It must reject a missing or unidentifiable checkout and record the Aorta commit
and dirty-tree state. The survey below describes Aorta commit
`da894f76e4c14cec03f428e7aed2aa45dfbf6a65` and must be revisited when the
selected workloads or that revision change.

## Aorta workload survey

Aorta currently exposes two materially different model paths.

| Aorta path | Model or shape | Actual implementation source | 16 GiB / all-target suitability | Distinct value |
| --- | --- | --- | --- | --- |
| `workload: inference`, `offline_batch` | Synthetic dense decoder, long prompt and short output | PyTorch eager. Matrix operations use the libraries selected by the installed PyTorch/ROCm stack; Aorta does not pin or prove a particular Tensile solution. | Good. Shape and model dimensions are fully bounded and the path is target-neutral. | Prefill-heavy attention, softmax, normalization, embedding, and large GEMM shapes. |
| `workload: inference`, `offline_batch` | The same decoder with a short prompt and longer token loop | PyTorch eager. Its `kv_cache: true` behavior is simulated by length-one forwards; it is not a real paged-KV implementation. | Good, with a bounded token count. | Decode-shaped small-M GEMMs, repeated launches, reductions, and argmax. It must be named synthetic decode, not Qwen decode. |
| `workload: inference`, `num_experts > 1` | Synthetic top-1 MoE decoder | PyTorch eager with argmax, boolean masks, indexed gather/write, GLU experts, and backend GEMMs. | Good at small expert and hidden sizes. | A different routing/indexing and sparse-expert kernel mix from the dense path. |
| `workload: training` or `llm_determinism` | Synthetic dense or top-1 MoE repeated-block model | PyTorch eager, including backward and optionally AdamW or FSDP/RCCL paths. | Single-rank bounded shapes fit, but distributed variants are not portable to a one-GPU 16 GiB contract. | Backward/optimizer kernels if the inference-only core leaves a meaningful ISA gap. |
| `workload: tokenspeed_serve` | Qwen3 0.6B, 1.7B, 4B, and 8B | TokenSpeed serving. Its registered provider set includes Gluon, Triton, and Torch paths; the provider actually selected must be captured from each run. | Not an all-target candidate as written: the pinned container accepts only `gfx950` and `gfx1250` and supplies its own ROCm stack. The 8B recipe was measured only on a 309 GiB device and is not a safe 16 GiB commitment. | Real Qwen prefill/decode and production TokenSpeed kernel selection. The 0.6B through 4B variants are the plausible size candidates once portability and local-runtime integration exist. |
| `workload: tokenspeed_serve` | `openai/gpt-oss-20b` | TokenSpeed's dedicated MXFP4 MoE path and Gluon JIT. | Exclude from the initial suite. Its MXFP4 checkpoint is designed for 16 GiB systems, but Aorta has only measured this exact stack on a 309 GiB GPU, requires a roughly 40 GB complete repository snapshot on disk, and spends minutes in startup. | Valuable future large-MoE stress, but its exact Aorta-plus-ConSan memory use and total-matrix latency are not yet qualified. |

Two useful Aorta facilities are not end-to-end model candidates:

- `tokenspeed-kernel-gemm-smoke.yaml` isolates Gluon BF16, Torch BF16, Triton
  FP8 block-scale, and Torch FP8 block-scale GEMMs. It is useful for proving
  generator-specific ISA properties, but it is a kernel benchmark and is
  currently `gfx950`-specific.
- `tokenspeed-kernel-suites-smoke.yaml` executes attention, MoE, quantization,
  sampling, and transform tests. These provide correctness and code-object
  coverage, not performance metrics or end-to-end model execution.

Aorta has no existing portable Gluon end-to-end workload and no existing
end-to-end workload that pins execution to Tensile. Its local PyTorch workloads
may reach Tensile through rocBLAS or hipBLASLt, but that is a runtime routing
result to record, not a source-level guarantee. Consequently the current
portable core can begin with the three bounded PyTorch inference shapes above,
but it does not yet satisfy the desired generator diversity. Closing that gap
requires adding portable Aorta workload recipes or adapters; relabeling the
TokenSpeed probes as end-to-end workloads would not close it.

## Provisional corpus

The useful starting set is therefore:

1. a long-prompt, short-output dense `inference` cell for PyTorch prefill;
2. a short-context, single continuous-batch tick for synthetic PyTorch decode;
3. a small `num_experts > 1` `inference` cell for PyTorch top-1 MoE routing;
4. a Qwen3-0.6B prefill cell and a Qwen3-0.6B decode cell through
   `tokenspeed_serve` on gfx950, and on gfx1201 only if that external stack can
   be enabled without substantial porting; and
5. a small pinned-hipBLASLt/Tensile end-to-end cell and a Gluon end-to-end cell
   on each target where those paths work naturally.

Items 1--3 are immediately portable candidates, not yet frozen benchmark
shapes. Items 4--5 are required diversity work, not claims about what the
current Aorta checkout can execute on both targets. The suite is a
target-specific union, not a requirement that every workload run on every
target. Record a difficult external-framework path as `not selected on this
target` rather than making its port a prerequisite; in particular, TokenSpeed
may be omitted on gfx1201 while remaining in the gfx950 suite. The
exact-provider kernel smokes may supplement this corpus as ISA probes, but
cannot satisfy an end-to-end slot.

The final shapes must be chosen from audited runs. Prefer the smallest shape
that retains the intended kernel mix and amortizes launch noise; reject any
candidate whose baseline-plus-four-mode matrix jeopardizes the 30-minute suite
bound. Current TokenSpeed recipes load the model into their serving stack and
do not establish host-to-device weight streaming, so a model is not considered
16-GiB-safe merely because an external system could cycle its weights.

## Suite selection status

The benchmark corpus is not yet frozen. Candidate choice should maximize
executed-ISA diversity per minute, using an audited code-object inventory to
show which candidate adds new instruction, synchronization, addressing, and
generator families. Model-size variants that add no meaningful ISA family
should be dropped in favor of distinct prefill, decode, routing, reduction, or
library-provider paths.

No number from the former validation campaigns is admitted into the benchmark
status ledgers. Those runs used different corpora and timing contracts and are
not comparable to the suite defined here.
