# DBI/DBT Literature Survey for rocjitsu/RocFuzz

This note distills the local surveys of DynamoRIO, QEMU TCG, and Frida
Gum/Stalker into design lessons for rocjitsu and RocFuzz. The focus is not on
reusing those projects directly. The useful question is which abstractions,
metadata contracts, and pass boundaries would make rocjitsu a stronger DBI
substrate for device-side coverage.

QBDI and rev.ng were left out of this pass because they are more LLVM-centered
than the direction we want to study here. The three projects below show that a
small native instruction representation plus disciplined patch metadata can be
enough for production DBI without making LLVM IR the central abstraction.

## Executive Takeaway

- The common pattern is a native, target-specific instruction/block model, not
  a large compiler IR. DynamoRIO uses `instr_t`/`opnd_t`/`instrlist_t`, QEMU TCG
  uses typed `TCGOp` streams inside Translation Blocks, and Frida Stalker uses
  decoded instructions plus architecture-specific relocators and writers.
- Mature DBI systems separate phases: decode, classify, analyze, transform,
  allocate resources, relocate, emit, publish, invalidate, and report. RocFuzz
  currently proves these pieces can work, but too much of the orchestration is
  preload-local rather than a stable rocjitsu DBI API.
- Code mutation is treated as a managed artifact. DynamoRIO has fragments and
  linkstubs, QEMU has Translation Blocks with patch/invalidation metadata, and
  Frida has exec blocks backed by code/data/slow slabs. rocjitsu needs the
  AMDGPU code-object version of this: patched kernel variants, caves,
  trampolines, relocation records, descriptor/resource updates, and rollback.
- State preservation is an explicit contract in mature DBI systems. For AMDGCN
  that means every probe must declare EXEC/SCC/VCC/register/scratch/memory
  effects, and every relocation must prove that app-visible state is preserved.
- Instrumentation should expose one product policy while preserving internal
  diagnostics. RocFuzz's adaptive branch/edge policy fits that model, but the
  DBI layer needs better counters for why sites were selected, degraded, or
  skipped.

## Projects Reviewed

| Project | What it contributes | What not to copy directly |
| --- | --- | --- |
| DynamoRIO | Mutable native instruction lists, ordered client events, scratch-register reservation, fragment/linkstub metadata, restore-state discipline. | CPU trace formation, clean-call-heavy instrumentation, and TLS assumptions. |
| QEMU TCG | Translation Block identity, typed linear IR, decode context/state keying, marker-then-inject plugin pipeline, cache invalidation and direct-branch patch ledger. | Whole-emulator execution, host helper callbacks at instrumentation frequency, dynamic TB chaining as a runtime requirement. |
| Frida Gum/Stalker | Iterator/writer transform API, relocator as a first-class service, event sinks, observer channels, code/data/slow slab ownership, targeted invalidation. | Per-thread following, return-address rewriting, W^X page mechanics, and host callouts as the primary instrumentation primitive. |

## ROCm Fuzzer Alternatives

As of May 2026, we have not found a public ROCm/HIP equivalent of
[cuFuzz][cufuzz]: a fuzzer that instruments AMDGCN/HSACO device basic blocks at
runtime and feeds deterministic device branch coverage into AFL++ or another
coverage-guided engine. CUDA-side cuFuzz is still the closest architecture
reference, but ROCm does not provide an NVBit-equivalent runtime instrumentation
service. RocFuzz uses rocjitsu DBI/DBT utilities to fill that gap for AMDGPU
code objects.

[ROCprofiler-SDK PC sampling][rocprofiler-pc-sampling] is useful for profiling
and offline sanity checks, but it is not a practical primary AFL feedback
channel. It samples PCs instead of recording deterministic edges, can miss
short-lived branches, does not naturally encode `previous_bb ^ current_bb`, and
requires post-processing rather than updating the AFL bitmap at the end of every
input. It can help rank hot blocks or debug low edge counts, but it is not part
of the AFL feedback path in this prototype.

## Key Ideas Worth Borrowing

### 1. Native Instruction IR With a Mutable Block Surface

All three systems keep the representation close to the target machine. That is
the right precedent for AMDGCN. RocFuzz does not need an LLVM-style middle end
to insert coverage probes; it needs a durable decoded AMDGCN block object that
can be inspected, annotated, rewritten, and emitted.

The useful minimum surface is:

- original instruction identity: code-object offset, kernel symbol, block id,
  raw words, decoded opcode, operand metadata, and opaque-instruction status;
- block structure: entry, exits, fallthrough, direct branch targets, and
  stable coverage ids independent of the mutable block object;
- mutation operations: insert before/after, replace, mark app vs meta
  instructions, attach relocation records, and attach emitted-probe ownership;
- emission view: final instruction stream, cave/trampoline allocations, branch
  fixups, and descriptor/resource changes.

rocjitsu already has generated decoders, CFG recovery, liveness helpers,
patching code, and probe builders. The gap is the stable mutable DBI layer
between "decoded facts" and "raw ELF bytes were patched".

### 2. Ordered Pass Pipeline Instead of Ad Hoc Probe Logic

DynamoRIO's `drmgr`, QEMU's translator plus plugin injection pipeline, and
Frida's transformer/iterator all make phase ordering explicit. RocFuzz has the
same conceptual phases today, but they are mostly encoded in fuzzer runtime
helpers.

The rocjitsu-facing pipeline should look like:

```text
discover code object
  -> extract compatible kernel images
  -> decode kernel text
  -> recover CFG and stable block ids
  -> analyze liveness, EXEC/SCC/VCC sensitivity, resources, and relocation risk
  -> select candidate coverage sites
  -> choose previous-BB, fixed-branch, block-entry, or skip per site
  -> reserve scratch/register/device-state resources
  -> lower probes through AMDGCN writer helpers
  -> relocate overwritten app instructions
  -> commit a transaction against a shadow code object
  -> publish, cache, invalidate, and report
```

This would let RocFuzz be one DBI client rather than the owner of every
intermediate structure. It would also make it possible to add future tracing or
profiling clients without duplicating code-object surgery.

### 3. Per-Site Analysis Records

The mature systems avoid re-deriving facts during emission. DynamoRIO samples
analyze a block once and pass per-block data to insertion. QEMU stores
instruction-start and Translation Block metadata. Frida tracks decoded
instruction addresses, relocation state, and emitted block ownership.

RocFuzz should move toward one explicit `ProbeSitePlan`-style record per site:

- stable site id, kernel id, block id, source offset, and branch target ids;
- selected coverage semantic and fallback reason if degraded;
- liveness summary and chosen SGPR/VGPR/scratch resources;
- clobbers and saved/restored special state;
- relocation requirements for overwritten instructions;
- cave/trampoline placement and branch-distance checks;
- emitted counter slot or previous-BB state layout;
- expected map pressure and report/debug metadata.

This record is the natural bridge between CFG analysis, probe lowering, patch
reporting, and test assertions.

### 4. Relocator and Writer as Reusable DBI Services

Frida's architecture-specific relocators and writers are the cleanest model for
rocjitsu. The fuzzer should not be stitching raw AMDGCN words for every case.
It should ask a writer for operations like "increment fixed counter",
"previous-BB edge update", "save and restore EXEC", "materialize device-state
base", "emit branch island", or "restore SCC-sensitive predicate state".

The relocator should answer whether an app instruction sequence can be moved
or replayed safely. For AMDGCN, that includes at least:

- SOPP/direct-branch relocation and target reachability;
- PC-relative operands and literal encodings;
- multiword and opaque instructions such as VOPD;
- SCC/VCC/EXEC dependencies around compare and branch sequences;
- scalar/vector register liveness and pressure;
- scratch and private-segment metadata consequences;
- wave32/wave64 behavior where emitted code depends on lane topology.

This is a rocjitsu DBI service, not a RocFuzz-only service. Coverage is just the
first client that exercises it hard.

### 5. Managed Patch Artifacts, Not One-Off Byte Mutation

DynamoRIO fragments, QEMU Translation Blocks, and Frida exec blocks all carry
metadata about generated code, exits, invalidation, and ownership. The GPU
equivalent is a patched kernel variant cache.

For rocjitsu, every patched variant should have a manifest:

- source container identity, selected device image, gfxip, kernel symbol, and
  wave mode assumptions;
- original text ranges and final text/cave/trampoline ranges;
- descriptor and metadata changes, including SGPR/VGPR/private-segment growth;
- branch fixups, relocation records, and rollback bytes;
- coverage-state ABI requirements;
- loader path that published it and lifetime owner;
- invalidation keys for module unload, fatbin unregister, shadow-cache eviction,
  and policy changes;
- structured report events for every selected, degraded, skipped, and failed
  site.

The current preload already produces useful JSONL reports and caches some
shadow modules. The survey makes the architecture point clearer: this should be
a central rocjitsu mutation backend with transaction semantics.

### 6. Explicit State and Clobber Contracts

Mature DBI systems force inserted code to declare how app state is translated
or restored. On CPUs this means registers, flags, PC translation, and fault
state. On AMDGCN the state surface is larger and more unusual:

- EXEC and lane activity;
- SCC, VCC, condition-code-producing instruction pairs, and branch predicates;
- scalar and vector temporaries;
- scratch/private-segment use and descriptor updates;
- device memory ordering for counter and previous-BB updates;
- per-workitem vs per-wave vs per-launch state;
- current coverage policy when lanes diverge and reconverge.

This is the core reason device-side coverage is a good forcing function for
rocjitsu DBI quality. If the DBI layer can prove and report these state effects
for coverage probes, the same machinery is valuable for tracing, profiling,
patching, and future DBT transformations.

### 7. Event Sinks and Observer Metrics

Frida separates transformation from event delivery. DynamoRIO and QEMU also
have distinct lifecycle and instrumentation hooks. RocFuzz should keep the hot
device path minimal, but DBI internals need a cold diagnostic channel.

Useful observer metrics include:

- decoded kernels, blocks, branches, and candidate sites;
- sites instrumented as previous-BB edges, fixed counters, block entries, or
  skipped;
- skip reasons: opaque instruction, relocation failure, EXEC/SCC hazard, branch
  range, register pressure, descriptor growth, loader risk, or device-state
  ABI limit;
- cave/trampoline allocation statistics;
- shadow module cache hits and invalidations;
- per-loader-path patch success and fallback events;
- device counter deltas and AFL map pressure.

These metrics are not just debugging aids. They are how we explain low edge
counts, define support boundaries, and make product claims credible.

## Architectural Gaps Exposed in Current rocjitsu DBI

The current RocFuzz prototype is valuable because it has forced real loader
and code-object paths to work. The survey suggests the next bottleneck is not
more one-off harnesses; it is making the DBI substrate explicit.

### Missing Stable Mutable AMDGCN DBI IR

rocjitsu can decode instructions and RocFuzz can patch selected sites, but
there is no mature `instrlist`/iterator/writer equivalent for AMDGCN DBI
clients. Without that layer, site selection, relocation, and probe emission
stay tightly coupled to the preload implementation.

The desired abstraction is small: decoded AMDGCN instructions with annotations,
block-level CFG metadata, app/meta instruction ownership, and structured edit
operations. It does not have to be a general compiler IR.

### Mutation Transactions Are Still Too Fuzzer-Local

The prototype patches raw ELFs, bundles, CCOB payloads, HSA-reader bytes, and
runtime shadows. That is enough to prove feasibility, but a product DBI layer
needs a transaction object that can preflight, reserve, emit, validate,
publish, roll back, and report in one place.

This is the AMDGPU analogue of fragments, Translation Blocks, or Stalker exec
blocks. The unit is not a CPU cache fragment; it is an instrumented code-object
variant with loader lifetime and container ownership.

### Relocation and Special-State Proofs Need to Be Centralized

Current support is intentionally conservative and adaptive. That is the right
product policy, but the proofs that decide "previous-BB edge", "fixed branch",
"block entry", or "skip" should live in rocjitsu. Otherwise every future DBI
client will need to relearn the same AMDGCN hazards.

The important missing product surface is an explainable legality result:

```text
site is patchable because ...
site was degraded because ...
site was skipped because ...
```

That result should be tied to decoded operands, liveness, EXEC/SCC/VCC state,
available scratch, branch reachability, and descriptor/resource changes.

### Device-State ABI Is Not Yet a General DBI Service

RocFuzz needs device counters and per-workitem `previous_bb` state. Other DBI
clients may need trace buffers, sampling buffers, or per-launch metadata. Today
the prototype has enough state handoff to run checked examples, but a product
layer needs a stable ABI for:

- process, stream, grid, and launch-scoped buffers;
- cleanup and reuse across persistent iterations;
- concurrent launches and multiple patched modules;
- map sizing and per-workitem state sizing;
- loader paths that cannot safely bake process-local pointers into code.

This is one of the clearest places where the fuzzer work benefits rocjitsu
outside fuzzing.

### Plugin/Client Boundary Is Underspecified

The existing preload is the only serious DBI client. The surveyed systems all
have a client-facing shape: events, transformers, plugin markers, or
instrumentation callbacks. rocjitsu should not expose a CPU-style API
verbatim, but it should expose semantic operations:

```text
on code object discovered
enumerate kernels and blocks
classify candidate sites
request probe resources
emit AMDGCN probe sequence
commit instrumented variant
observe launch/sync/lifetime events
```

That boundary would keep AFL++ policy and coverage semantics separate from the
binary mutation backend.

## Suggested rocjitsu DBI Product Shape

A minimal production-oriented design could have these components:

| Component | Responsibility |
| --- | --- |
| `CodeObjectSession` | Own loader/container identity, target gfxip, extracted device images, lifetime, and report sink. |
| `KernelRegion` | Own decoded text, CFG, stable block ids, liveness summaries, and patchable ranges for one kernel. |
| `DbiPassPipeline` | Run ordered analysis and mutation phases with explicit inputs and outputs. |
| `ProbeSitePlan` | Record candidate site facts, selected semantic, resource plan, clobbers, relocation needs, and fallback reason. |
| `AmdgcnWriter` | Emit structured AMDGCN probe fragments and branch islands through named operations. |
| `InstructionRelocator` | Prove and emit relocated app instructions, or return a structured failure reason. |
| `PatchTransaction` | Allocate caves/trampolines, patch descriptors/resources, preserve rollback data, validate ranges, and publish shadow variants. |
| `DeviceStateABI` | Allocate and bind device-visible state for counters, previous-BB data, diagnostics, and future DBI clients. |
| `DbiObserver` | Emit structured metrics for selected/degraded/skipped sites, loader paths, cache behavior, and coverage deltas. |

RocFuzz would then become a coverage-policy client over this substrate. It
would define the AFL map layout and probe semantics, while rocjitsu owns code
object mutation, resource legality, and publication.

## What to Avoid

- Do not import LLVM IR as the default DBI representation for this work. The
  non-LLVM projects show that native instruction IR plus strong metadata is
  enough and better aligned with prebuilt AMDGPU code objects.
- Do not model AMDGPU probes as generic function calls. Inline sequences and
  trampolines are often the safer primitive when device-call ABI, register
  pressure, and metadata growth are uncertain.
- Do not copy CPU DBI thread/TLS assumptions. The right scope is loader,
  module, stream, launch, grid, wave, and workitem state.
- Do not make coverage modes a public configuration matrix. Keep the product
  policy adaptive and use internal observer metrics to explain decisions.
- Do not treat code caves and branch islands as incidental details. They are
  part of the managed patch artifact and should be planned, cached, and
  invalidated explicitly.

## Near-Term Implications for RocFuzz

The survey reinforces the current README's production gaps, but sharpens their
DBI framing:

1. Create a stable AMDGCN block/mutation abstraction before expanding edge
   coverage aggressively.
2. Move relocation, scratch/resource planning, and special-state legality out
   of fuzzer-specific helpers and into rocjitsu DBI services.
3. Treat patched kernel variants as first-class transactions with manifest,
   rollback, invalidation, and structured reports.
4. Define a reusable device-state ABI for coverage counters and future DBI
   buffers.
5. Add observer metrics that explain patchability and edge-count quality across
   real ROCm library examples.

These steps are also the highest-leverage path toward broader device branch
coverage. More library fuzz targets will be useful once this substrate is
stronger, but the DBI architecture is the piece that determines whether those
targets can produce trustworthy coverage rather than only launch-level crash
checks.

## Glossary

This section uses the terms in their AMDGCN/ROCm code-object sense, not as
generic CPU DBI vocabulary.

| Term | AMDGCN-grounded meaning |
| --- | --- |
| AMDGCN | AMD's GPU instruction set architecture used by ROCm device code. RocFuzz works on AMDGCN instructions inside HSACO/code-object payloads, not on LLVM IR or HIP source. |
| Code object / HSACO | The ELF-like binary container consumed by the ROCm loader. It contains AMDGCN kernel text, metadata, symbols, and kernel descriptors. Instrumentation changes must leave this loadable by HIP/HSA. |
| Kernel descriptor | A data record associated with a kernel symbol that tells the loader/hardware about code entry, SGPR/VGPR usage, scratch/private-segment requirements, kernarg layout, and related dispatch metadata. If probes need more registers or scratch, descriptor fields may need to grow. |
| Text section | The executable AMDGCN instruction bytes for one or more kernels. RocFuzz patches text to redirect control flow into probes or to insert short inline probe sequences. |
| Basic block | A straight-line region of AMDGCN instructions with one entry and exits at branches, returns, or other control-flow terminators. Coverage ids should be stable over original code offsets, not over mutable internal objects created during patching. |
| Edge coverage | Feedback that records control flow from a previous basic block to the current block, commonly using `previous_bb ^ current_bb`. This is richer than saying only that a block executed, but it requires per-workitem or per-wave previous-block state. |
| Probe | Instrumentation code inserted into, or branched to from, an AMDGCN kernel. In RocFuzz, probes update device-visible coverage counters and sometimes per-workitem `previous_bb` state. |
| Patch site | The original instruction offset where rocjitsu decides instrumentation can be attached. A site may be a block entry, direct branch, or other safe insertion point. The planner may select, degrade, or skip it based on liveness, relocation, branch range, and special-state hazards. |
| Code cave | Unused or newly appended executable space in a code object where rocjitsu can place extra AMDGCN instructions. Caves matter because many probes are too large to fit at the original patch site. |
| Trampoline | A short branch sequence that transfers control from original kernel text to probe code in a cave and then returns to the correct app instruction or target. On AMDGCN this must respect branch range, scalar branch encoding, EXEC/SCC state, and any relocated instructions. |
| Branch island | A trampoline-like helper used when the final target is too far for the available branch encoding. It gives the patcher an intermediate reachable target. For AMDGCN this is a placement and reachability problem inside the code object. |
| Relocation | Re-emitting an original AMDGCN instruction somewhere else because the patch overwrote or moved it. Relocation is safe only if operands, PC-relative references, branch targets, literals, and special-state effects still mean the same thing after movement. |
| Opaque instruction | An instruction whose bytes are known but whose full semantics are not modeled by rocjitsu at the required level. Opaque instructions can often be copied or skipped over conservatively, but they should not be used to justify register availability or state preservation. |
| Liveness | Analysis that determines whether an SGPR/VGPR or special state is still needed by later instructions. Probe planning uses liveness to decide whether it can use a temporary register directly, must spill it, or must skip/degrade the site. |
| SGPR / VGPR | Scalar and vector general-purpose registers. SGPRs are shared across a wavefront for scalar code; VGPRs hold lane-specific values. Coverage probes may need both, and using additional registers can require descriptor/resource changes. |
| EXEC | The AMDGCN execution mask that indicates which lanes in the current wavefront are active. Instrumentation that changes or depends on EXEC can change SIMT control-flow semantics unless it saves/restores EXEC or proves the transformation is safe. |
| SCC / VCC | Scalar and vector condition state used by AMDGCN compare and branch sequences. A probe placed between a compare and its branch can break the program if it clobbers or fails to preserve the relevant condition state. |
| Wavefront | The group of lanes executing together under one EXEC mask. Current RDNA targets commonly use wave32 by default and can force wave64. Coverage semantics must not hardcode a CUDA warp model. |
| Workitem / lane | One logical GPU thread within a wavefront. Per-workitem `previous_bb` state lets divergent lanes retain distinct predecessor blocks while masked off. |
| Scratch / private segment | Per-workitem backing memory used when register values must be spilled or when the kernel needs private memory. If probes require scratch, the code object metadata and kernel descriptor must advertise enough private-segment usage. |
| Patched variant / shadow module | A modified copy of a code-object image or module that contains instrumentation while preserving the original payload for fallback or sibling kernels. RocFuzz uses shadow variants for several loader paths instead of mutating every loaded library image in place. |
| Patch transaction | The desired production abstraction for grouping all edits to a kernel variant: cave allocation, probe emission, relocated instructions, descriptor changes, branch fixups, validation, publish, and rollback data. |
| Device-state ABI | The mechanism that makes coverage buffers and `previous_bb` state visible to instrumented AMDGCN code. A product version must handle launches, streams, grids, persistent iterations, cleanup, and concurrent modules without relying on fragile ad hoc pointer embedding. |
| CCOB / fatbin / KPACK | ROCm library and compiler packaging layers that can wrap one or more device code objects. They are not instrumentation concepts, but they determine which bytes rocjitsu can safely patch and when kernel identity becomes known. |

## Source Pointers

The local project surveys used these primary anchors:

- DynamoRIO: `core/ir/*`, `core/lib/dr_events.h`, `ext/drmgr/*`,
  `ext/drreg/*`, `ext/drx/*`, `core/fragment.h`, `core/fcache.*`,
  `core/link.*`, and `api/samples/{stats,bbbuf,cbrtrace}.c`.
- QEMU TCG: `docs/devel/tcg*.rst`, `include/exec/translation-block.h`,
  `include/exec/translator.h`, `accel/tcg/translator.c`,
  `accel/tcg/translate-all.c`, `accel/tcg/plugin-gen.c`,
  `include/plugins/qemu-plugin.h`, `tcg/tcg.c`, and `tcg/optimize.c`.
- Frida Gum/Stalker: `gum/gumstalker.h`, `gum/gumstalker.c`,
  `gum/backend-*/gumstalker-*.c`, `gum/arch-*/gum*relocator.*`,
  `gum/arch-*/gum*writer.*`, `gum/gumevent.h`, `gum/gumeventsink.*`, and
  `tests/core/arch-*/stalker-*.c`.
- rocjitsu/RocFuzz anchors: [../README.md](../README.md),
  [device-side-instrumentation-api-boundary.md](device-side-instrumentation-api-boundary.md),
  [../afl-dbi/runtime/instrumentation_planner.h](../afl-dbi/runtime/instrumentation_planner.h),
  [../afl-dbi/runtime/instruction_relocator.h](../afl-dbi/runtime/instruction_relocator.h),
  and [../../docs/dbt-design.md](../../docs/dbt-design.md).

[cufuzz]: https://github.com/NVlabs/cuFuzz
[rocprofiler-pc-sampling]: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-pc-sampling.html
