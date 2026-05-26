# ROCm AFL DBI Coverage Prototype

This is a deliberately small end-to-end ROCm/HIP fuzzing prototype:

```text
AFL++ host run
  -> librocjitsu_afl_preload.so maps trace_bits
  -> hipModuleLoad is intercepted and the device ELF is patched
  -> selected entry or branch probes record device CFG coverage
  -> previous-BB probes update per-workitem previous_bb state
  -> one active lane increments the previous_bb ^ bb_id counter slot when safe
  -> hipModuleLaunchKernel records that a launch happened
  -> hipDeviceSynchronize copies and merges counters into trace_bits[32768..]
```

The sample kernels do not expose any coverage or fuzzer state argument. During
`hipModuleLoad`, the preload allocates the process-local device state, rewrites
the loaded code object, and bakes that state pointer into the probe sequence.
The state starts with AFL device counters and is followed by
per-workitem `previous_bb` entries.

The runtime also decodes the smoke target with rocjitsu `BasicBlock` CFG
utilities and patches a bounded set of block entries. Those trampolines compute
`previous_bb ^ bb_id`, update active lanes' `previous_bb`, and increment the
hashed device counter from one active lane. Because `previous_bb` is tracked per
workitem, multi-predecessor joins can produce distinct edge hashes without
patching every predecessor terminator.

Full `previous_bb` edge instrumentation is currently enabled for the supported
GFX11/GFX12 families. Older rocjitsu-decodable targets use the same preload and
device allocation path with a conservative entry-counter probe so we can keep
target support broad without specializing every optimal instruction sequence up
front.

## Layout

- `runtime/`: HIP preload and code-object patching runtime.
- `include/rocjitsu_fuzzer/`: shared coverage planning and probe utilities.
- `smoke/`: branchy two-kernel HIP target used by the end-to-end checks.
- `tools/`: small code-object packaging and inspection diagnostics.
- `tests/`: host-only coverage planning tests.

The smoke target keeps GPU work deliberately simple: kernel A fills a scratch
array, kernel B applies selector-dependent branches and writes one output value
per workitem, and the host performs the final reduction. That keeps the DBI
test focused on code-object patching and device coverage rather than GPU
reduction lowering.

## Build

```bash
cmake -S emulation/rocjitsu -B emulation/rocjitsu/build -G Ninja
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_preload \
  rocjitsu_afl_branchy_host \
  rocjitsu_afl_branchy_persistent_host \
  rocjitsu_afl_branchy_kernels_hsaco \
  rocjitsu_afl_edge_plan_unit \
  rocjitsu_afl_probe_decode_unit
```

Override the smoke target architecture with
`-DROCFUZZ_GPU_ARCH=<gfx arch>`. The default is `gfx1201`.

Current preload patching support is table-driven by the ELF gfxip machine ID:

| gfxip | rocjitsu arch | instrumentation tier |
| --- | --- | --- |
| gfx90a | CDNA2 | entry counter |
| gfx940, gfx941, gfx942 | CDNA3 | entry counter |
| gfx950 | CDNA4 | entry counter |
| gfx1010 | RDNA1 | entry counter |
| gfx1030 | RDNA2 | entry counter |
| gfx1100, gfx1150 | RDNA3/RDNA3.5 | `previous_bb` edges |
| gfx1200, gfx1201 | RDNA4 | `previous_bb` edges |

The GFX9/CDNA and GFX10/RDNA entry-counter tier uses the shared scalar-base
flat-global probe path. GFX11/GFX12 keep the fuller edge probes. Gfxip values
that rocjitsu does not expose through generated ELF machine constants and
decoders are intentionally left unsupported for now; add them to the table when
that rocjitsu infrastructure lands.

The preload uses rocjitsu's generated decoders to build CFGs. For CFG planning
only, unsupported compiler-emitted VOPD encodings are treated as opaque 8-byte
non-control instructions; opaque VOPD is not relocated when it is the first
instruction at an instrumentation site. Opaque instructions participate in
strict liveness as all-register uses, and strict mode disables fresh
descriptor-grown temporaries for any scope that still contains opaque
instructions. This blocks unsafe temporaries before the unknown instruction
without using unmodeled opaque code to justify larger register allocations. A
separate build target checks that the same smoke kernels can be compiled for
RDNA3:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_branchy_kernels_rdna3_hsaco
```

On hosts with a gfx1100 GPU, run the patched RDNA3 preload smoke with:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_rdna3_smoke
```

The normal smoke kernels use the compiler's default wavefront size, which is
wave32 on the current GFX11/GFX12 test targets. Force wave64 and run the same
preload checks with:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_wave64_smoke \
  rocjitsu_afl_rdna3_wave64_smoke
```

The wave64 targets are smoke checks: they prove the patched code object runs
and records device edges with 64-thread workgroups. The safe default uses the
same hybrid policy as wave32: previous-BB branch probes where the planner can
preserve EXEC/SCC state and fixed self-contained branch counters for sites whose
condition depends directly on EXEC. The forced previous-BB report smoke now
covers the full previous-BB-eligible branch set in the synthetic wave64 kernel.

For a local end-to-end coverage regression sweep across the currently supported
GFX11/GFX12 paths, run:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_gfx11_gfx12_coverage_matrix
```

This aggregate target does not select another public coverage mode. It runs the
default adaptive direct-module report, CCOB report, HSA-reader/CCOB report,
KPACK runtime-shadow reports, the branchy runtime-overhead smoke, and
forced-wave64 smokes so planner changes have one focused regression gate.

The target sets `HIP_VISIBLE_DEVICES=${ROCFUZZ_RDNA3_VISIBLE_DEVICES}`. The
default selector is `1` for the local mixed gfx1201/gfx1100 development host;
override `-DROCFUZZ_RDNA3_VISIBLE_DEVICES=<index>` when the RDNA3 device is
enumerated differently.

If the local TheRock rocBLAS gfx1100 payload is installed, the build also
enables a real-library lazy CCOB lifecycle smoke:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_lazy_ccob_report_smoke
```

This loads `ROCFUZZ_LAZY_CCOB_MODULE` through `hipModuleLoad`, discovers a
kernel symbol from the CCOB payload, calls `hipModuleGetFunction` twice, and
checks `ROCJITSU_AFL_PATCH_REPORT` for lazy CCOB module/function/release
records plus the repeated-lookup cache hit. The synthetic branchy CCOB report
smoke additionally launches both patched shadow functions, repeats one of those
launches, and checks `lazy_ccob_launch` event counts. Override
`-DROCFUZZ_LAZY_CCOB_MODULE=<path>` or
`-DROCFUZZ_LAZY_CCOB_VISIBLE_DEVICES=<selector>` when using a different CCOB
asset or RDNA3 device index.

Optional AFL++ build:

```bash
git clone https://github.com/AFLplusplus/AFLplusplus.git emulation/rocjitsu/third_party/AFLplusplus
git -C emulation/rocjitsu/third_party/AFLplusplus checkout a918a9ab647d86824d289f36014b9ca99f077984
make -C emulation/rocjitsu/third_party/AFLplusplus source-only -j$(nproc)
```

Code-object diagnostics:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_inspect_code_object_image

emulation/rocjitsu/build/fuzzer/afl-dbi/rocjitsu_afl_inspect_code_object_image \
  path/to/library_kernel.co
```

The inspector uses the same CCOB/offload-bundle parser as the preload. It
reports top-level container kind, extracted device-image count, target id, and
payload sizes. The printed `index` is the parser's stable extraction order; the
preload uses it internally when rebuilding sibling-preserving containers so
repeated adjacent payloads with the same display id still patch the exact image
that matched the current device. The inspector can dump an extracted HSACO with
`--dump-device <index> <input.co> <output.hsaco>` for `llvm-objdump` triage.
If the local rocBLAS/hipBLASLt/shared test assets are installed, the optional
inventory smoke validates the same parser over those real `.co` files:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_real_ccob_inventory_smoke
```

The target accepts both top-level CCOB and raw-ELF `.co` assets, requires every
input to yield a device image, and reports whether any multi-payload CCOB was
present. The current local inventory is single-payload, so it improves
real-container parser coverage but does not replace the still-open
multi-payload/RDC rebuild validation.

GPU-independent unit check:

```bash
ctest --test-dir emulation/rocjitsu/build \
  -R 'RocjitsuAflDbi\.(EdgePlanUnit|ProbeDecodeUnit)' \
  --output-on-failure
```

`ProbeDecodeUnit` decodes the emitted probe streams for every target in the
preload support table. It checks the portable entry-counter stream everywhere
and the fuller edge streams on targets whose tier enables previous-BB edges.

## Run

Standalone HIP check:

```bash
emulation/rocjitsu/build/fuzzer/afl-dbi/rocjitsu_afl_branchy_host \
  emulation/rocjitsu/build/fuzzer/afl-dbi/branchy_kernels.gfx1201.hsaco \
  emulation/rocjitsu/build/fuzzer/afl-dbi/seed
```

Preload/DBI check:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_dbi_smoke
```

The branch-coverage smoke targets set `ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1`.
The preload still returns a HIP error when a synchronization point sees no
device edge slots, and it also exits nonzero at process teardown if the harness
ignored those errors and no device edge was ever observed. Use that guard on
library harnesses when the intended signal is device branch coverage rather
than host launch interception.

AFL-visible map check, after building the AFL host target:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_showmap_edge
```

This runs two inputs through `afl-showmap`, filters upper-half device slots
after slot 0, and fails if the device edge coverage is identical. The current
default is best-effort and adaptive: ordinary HSACO uses entry-backed
previous-BB probes only when the persistent state SGPR can be made safe, and
otherwise switches to self-contained previous-BB branch terminator sites where
strict liveness can prove the stronger probe safe. Entry-unsafe Tensile, lazy
CCOB payloads, and gfx10+/RDNA kernels whose SGPR count comes only from metadata
can switch to self-contained previous-BB branch probes; MIOpen activation,
HSA-reader, runtime-shadow payloads, and larger wave64 branch plans use
self-contained fixed branch counters when kernel, loader, budget, or resource
evidence makes previous-BB state unsafe or unproven.
Within a self-contained previous-BB branch plan, individual sites can degrade to
fixed counters when strict liveness cannot prove the larger probe safe but can
prove the smaller counter probe safe. Ordinary payloads can also switch to a
self-contained branch strategy when preflight finds no block-entry edge sites
but does find branch edge sites.
For default self-contained branch plans, the preload asks the planner for an
adaptive per-kernel strategy: eligible direct branch sites use previous-BB
hashing by default, while EXEC-conditioned or otherwise unproven sites degrade
to fixed counters when the smaller probe is safe. The previous-BB logical-edge
budget is derived from the eligible writer candidate count, so every candidate
is attempted without adding another user-visible coverage mode.
JSONL reports expose `branch_edge_site_limit_auto` and
`previous_bb_branch_site_limit_auto` at the patch event level and
`branch_edge_candidate_edges`, `branch_edge_budget`,
`previous_bb_branch_edge_candidate_edges`,
`previous_bb_branch_site_candidate_sites`,
`previous_bb_branch_site_budget`, `branch_edge_budget_reason`,
`previous_bb_branch_site_budget_reason`,
`previous_bb_branch_edge_over_budget`, `previous_bb_branch_site_over_budget`,
`previous_bb_branch_edges_selected`,
`previous_bb_branch_edge_trampolines_planned`,
`previous_bb_branch_edge_trampoline_bytes`,
`previous_bb_branch_afl_map_pressure_ppm`,
`previous_bb_branch_code_growth_pressure_ppm`,
`previous_bb_branch_trampoline_avg_bytes_x100`,
`previous_bb_branch_overhead_status`, and degradation counts per kernel.
`branch_edge_candidate_edges` counts all direct branch edges available to the
hybrid planner; the `previous_bb_branch_*_candidate_*` counters count only the
subset still eligible to write previous-BB state after safety classification.
Per-kernel summaries include `previous_bb_branch_aggregate_limit_kind`,
`previous_bb_branch_aggregate_safety`, and
`fixed_counter_branch_edge_fallback_*` fields so reports state whether the
candidate previous-BB plan fit the candidate-derived budget, hit a
debug/configured cap, derived a fixed-counter fallback budget from the candidate
branch edge set, or used a non-previous-BB policy.
When either aggregate previous-BB cap is active, the JSONL report retains every
selected edge sample so the patch-offset minimizer can operate on the real
capped report. Uncapped reports keep a compact selected-edge sample.
Each patch event and kernel summary also exposes `coverage_signal`, a compact
classification of the actual device signal selected by the adaptive planner:
`previous-bb-edges`, `hybrid-previous-bb-and-fixed`,
`fixed-branch-counters`, `entry-counter`, or `none`. This is reporting only; it
does not introduce another user-selectable coverage mode.
The current planner proves individual sites through liveness/resource
selection, descriptor and private-segment patching, and trampoline placement.
After the minimizer split the isolated `previous-bb-set` profile from the older
opaque fresh-register diagnostic, and after EXEC-conditioned branches were split
into fixed-counter safety fallback, the default hybrid path selects all
previous-BB eligible MT128x64 writer sites in the rocBLAS smoke and still
produces device edges. The fixed-counter fallback budget is derived from the
candidate branch edge set and capped by the fixed-counter map, because fixed
probes are smaller and do not write previous-BB state. Fixed fallback edges
therefore do not consume the previous-BB writer budget; they only consume the
independent fixed-counter fallback budget. Patch reports now expose the static
overhead needed by a product policy gate: previous-BB branch AFL map pressure,
average trampoline size, appended/local previous-BB trampoline bytes, and
appended-code growth pressure against the input code object. Runtime overhead is
not inferred from these static counters. The benchmark target below compares the
same branchy host command without preload and with the AFL preload enabled,
emits `runtime_overhead_branchy.json`, and validates that the instrumented run
also produced patch-report evidence:

```bash
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_runtime_overhead_smoke
```

The smoke records medians and the instrumented/baseline ratio but does not
enforce a product threshold; that threshold needs data from representative
library and user kernels rather than a synthetic branchy target alone.
When previous-BB branch sites degrade to fixed counters, including budget-cap
fallback and placement-time trampoline fallback, per-kernel
`degradation_reason_counts` records which stronger-probe proof failed, which cap
was reached, or which placement/range check failed before the smaller probe was
accepted.
The patchability classifier is the front door for these default decisions: it
centralizes loader constraints, kernel filters, and prior unsafe-kernel
knowledge before liveness and relocation preflight select concrete sites.
Set `ROCJITSU_AFL_DEBUG_EDGE_LIMIT=N` to experiment with more block-entry sites.
The `rocjitsu_afl_kpack_ccob_safe_edge_report_smoke` target checks both loader
families under the current safe defaults: the lazy CCOB half must use
self-contained hashed branch probes and produce device edges, while the KPACK
half exercises HSA-reader and runtime-shadow loader-scoped fixed branch
counters. The old `rocjitsu_afl_kpack_ccob_high_edge_report_smoke`
target name remains as a compatibility alias, but the current safe policy no
longer promises above-default edge counts on these loader paths until more
relocation and liveness support lands.

Default terminator-edge patching check:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_branch_edge_smoke
```

Direct branch terminator patching is part of the normal gfx11/gfx12 best-effort
policy. When entry-backed kernels are safe they keep their previous-BB block-entry
probes, while branch terminators can use self-contained probes that load the
coverage state directly instead of relying on an entry-initialized SGPR to
survive through the original kernel body. Full self-contained branch plans can
use previous-BB hashing where safe and degrade individual sites to a flagless
fixed-slot counter probe when liveness cannot prove the larger previous-BB probe
safe. Fixed-counter branch sites use a stable site hash over the same
block/edge IDs instead of
allocating sequential slots per kernel, so the normal hybrid planner is
deterministic across launches and planning order. Those fixed slots still share
the AFL bitmap and may collide by design rather than starving later lazy
CCOB/HSA-reader payloads. Fixed-counter probes materialize literal offsets when
a slot is outside the small inline-immediate range, so this fallback can use
the full device counter half-map rather than only slots 1..16. Patch reports
expose the selected code-object `coverage_strategy`, its reason, per-kernel
coverage strategies and coverage signals, `branch_edge_slot_policy`,
`branch_edge_slot_policy_reason`, fixed/hashed site counts,
`fixed_slot_requests`, `fixed_slot_exhaustions`, `fixed_slot_collisions`,
`degradation_reason_counts`, the legacy inline-slot aliases, the collision
policy, and `failure_phase` for unsuccessful patch attempts. Arbitrary overwritten
instruction relocation still requires special-register preservation and
liveness-aware temporary allocation.

For the current smoke target, verbose mode should report nonzero `edge_slots`.
Coverage instrumentation has one product strategy: adaptive hybrid coverage. It
prefers previous-BB hashing, degrades individual branch sites to fixed hashed
counters when only the smaller probe is proven safe, and falls back to entry
counters only when edge coverage is unsupported or unproven. The following
variables are diagnostics for tests and minimizers, not user-selectable product
modes.
Set `ROCJITSU_AFL_DEBUG_DISABLE_EDGES=1` to use the original launch-counter-only
diagnostic path, where slot 0 increments once per launched kernel.
Set `ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1` to force the diagnostic branch-site-only
path where kernel descriptors keep their original entries and edge trampolines
load the coverage-state pointer directly. The default planner now selects this
path automatically for known entry-unsafe and loader-scoped payloads. Use
`ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed|hashed` only for ablation and
regression tests that need to pin one branch slot policy and disable the normal
per-site fallback. `ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1` forces the older
entry-backed branch-site diagnostic path for synthetic tests; pair it with an
explicit branch-edge slot policy so the diagnostic does not depend on product
fallback heuristics.
The adaptive planner considers VGPR scratch-spill fallback by default and enables
it only where the loader path is proven. Direct HSACO branch probes can grow private
scratch by patching both the kernel descriptor and loader-visible AMDGPU
metadata. Use `ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS=1` only when
isolating scratch-specific regressions. Loader-scoped CCOB, HSA-reader, and
runtime KPACK paths still
report `vgpr_scratch_spills_requested=true`,
`vgpr_scratch_spills_enabled=false`, and
`vgpr_scratch_spill_reason=loader-context-scratch-spills-unproven` because a
forced lazy-CCOB scratch run is not launch-safe yet. Block-entry scratch probes
remain disabled with the skip reason `scratch-backed block-entry probes require
loader-visible private segment growth` until they have a separate prologue-safe
plan. Use
`rocjitsu_afl_branch_edge_hashed_smoke` to exercise the explicit
branch-site-only hashed policy on the synthetic branchy kernels. Use
`rocjitsu_afl_branch_edge_hashed_strict_smoke` to pin the explicit strict
liveness configuration in reports.
Set `ROCJITSU_AFL_DEBUG_LAUNCH_ONLY=1` to leave loaded device code objects unpatched
while still allocating the preload device state, intercepting HIP launches, and
reporting the launch count. This is useful for library paths whose selected
payload format or ABI is not yet safe for DBI rewriting.
Set `ROCJITSU_AFL_PATCH_REPORT=/path/to/report.jsonl` to append a JSONL stream
of patch and loader-lifecycle records. `code_object_image` records summarize the
top-level loader input before patching, including raw ELF, clang bundle, CCOB,
device-image count, and whether raw-ELF bypass would drop sibling CCOB payloads.
`patch_device_elf` records the loader context and selected device image,
the extraction index when the image came from a parsed container,
candidate/selected edge counts, skip counters, descriptor updates,
`edge_instrumentation_reason`, cave placement, branch-range failures, aggregate
skip reason counts plus bounded skip offset, mnemonic, and raw-word samples,
active kernel filter,
active HIP fatbin/KPACK registration key, and a bounded sample of edge patch
failures. Lazy loader
records describe runtime KPACK shadow sources, shadow functions, shadow misses,
registration release, lazy CCOB modules, lazy CCOB functions, lazy CCOB
launches, and CCOB module release. HSA-reader records report whether the
preferred patched image was a rebuilt CCOB, whether a raw patched ELF fallback
was available, final/primary/fallback reader handles, and the final reader
status. `ccob_rebuild` records include before/after device-image counts and a
`sibling_payloads_preserved` flag for sibling-preserving container rewrites.
These records include
registration/module/function identifiers, kernel names,
cache-hit/insert/duplicate-cleanup state, miss reasons, launch geometry, and
CCOB payload summaries where available. Lazy CCOB HIP shadow reports explicitly
state the `raw_exact_kernel_elf` policy: HIP shadows trade sibling-payload
preservation for launch stability, while the lower HSA-reader path remains the
sibling-preserving CCOB rebuild path. This is intended for comparing KPACK,
CCOB, and direct module paths without scraping verbose stderr.
Use the summary tool to turn that JSONL stream into per-loader and per-kernel
coverage evidence:

```bash
emulation/rocjitsu/fuzzer/afl-dbi/tools/summarize_patch_report.py \
  --format markdown path/to/report.jsonl
```

The JSON output is intended for automation. It aggregates patch events,
candidate/selected/skipped sites, branch-edge counts and budgets, sampled
failure reasons, planner skip reasons, shadow-loader events, CCOB rebuild
state, and post-sync `device_edge_delta` records. The markdown output is
intended for per-example campaign reports.
For blocked branch-edge experiments, the internal minimizer can run the target
repeatedly with a bounded branch-edge limit and print patch-site,
device-edge-delta, and descriptor-resource growth summaries from each report.
Use `--diagnostic-profile=opaque-fresh` only when reducing the older
fresh-register-growth fault class:

```bash
emulation/rocjitsu/fuzzer/afl-dbi/tools/minimize_opaque_fresh_growth.py \
  --diagnostic-profile=opaque-fresh \
  --preload build/fuzzer/afl-dbi/librocjitsu_afl_preload.so \
  --kernel-include MT128x64 \
  --patch-text-offset 0x40 \
  --report-dir /tmp/rocfuzz-opaque-min \
  --limit-range 1 16 -- ./target args...
```

Use `--diagnostic-profile=previous-bb-set` for the MT128x64 aggregate
previous-BB investigation. That profile still forces bounded previous-BB branch
site selection and strict liveness, but it keeps opaque fresh-register growth
disabled so a reduced failing offset set is not confounded with the older
VOPD/Tensile fresh-register override. Successful minimizer runs do not change
the product default. Its descriptor summary calls out SGPR-only, SGPR+VGPR,
VGPR, and private-segment growth so resource-descriptor failures can be
identified without ad hoc JSONL parsing. When running a range, it also emits a
final
`opaque_fresh_minimization_summary` JSON object with the first failing limit,
the last passing limit before it, and the sampled patch/register tuple that was
visible in the failing report. Add `--force-fresh-sgprs` or
`--force-fresh-vgprs` to isolate SGPR and VGPR descriptor growth independently
from the ordinary allocated-register liveness path. Add
`--disable-vgpr-scratch-spills` when the scratch fallback would otherwise avoid
the descriptor-growth case being minimized. On gfx10+/RDNA kernels whose SGPR
count is metadata-backed, SGPR-forced runs patch `.sgpr_count`; the current
rocBLAS SGEMM MT128x64 diagnostic reaches high-site SGPR-only growth without
reproducing the older fresh-growth fault.
The maintained unit target for both minimizer profiles is
`rocjitsu_afl_edge_minimizer_unit`; `rocjitsu_afl_opaque_fresh_minimizer_unit`
is kept as a compatibility alias.
`--patch-text-offset` is a DEBUG-only site selector; it is intended for reducing
one already-selected branch probe, not for production coverage configuration.
Use `--scan-patch-offsets-from <report.jsonl>` to rerun the target once per
sampled selected `patch_text_offset` from an existing failing report. That mode
emits `opaque_fresh_patch_offset_scan_summary`, which distinguishes a
single-offset reproducer from a failure that requires a combination of selected
sites. For multi-site failures, use
`--minimize-patch-offset-set-from <report.jsonl>` to delta-reduce the selected
offset set through the debug-only
`ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSETS` filter. That emits
`opaque_fresh_patch_offset_set_minimization_summary` with the current reduced
offset set and the last failing run. Reports produced with
`ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS=1` retain every selected edge
sample so this scan can be used for stable first-site reduction; older or
non-debug reports expose `selected_edge_samples_truncated` when the sampled
offset list was capped.
Coverage is merged after successful `hipDeviceSynchronize`,
`hipStreamSynchronize`, `hipStreamSynchronize_spt`, `hipEventSynchronize`,
blocking `hipMemcpy`, persistent iteration end, and process teardown. These
hooks keep device branch feedback visible to AFL for harnesses and libraries
that do not end every iteration with a full device sync.
The preload also parses top-level CCOB (`.co`) code-object containers, extracts
AMDGPU ELF payloads from the decompressed offload bundle, and loads patched raw
ELF through the existing module path. For large HIP module-loaded CCOB payloads,
the preload can defer patching until `hipModuleGetFunction` names the launched
kernel, then load a second raw ELF patched only for that kernel and return the
patched function handle. This avoids spending branch-reachable local text caves
on unlaunched Tensile variants. If the original CCOB is not directly loadable by
HIP, the preload falls back to the eager raw-ELF bypass. HIP lazy shadow modules
are released either when the original HIP module unloads or during preload
teardown if the harness exits without unloading it. They currently keep using
raw patched ELF even when the original CCOB has multiple payloads; duplicate
rebuilt-CCOB shadow modules loaded successfully but were not launch-stable on
the branchy smoke. Lower HSA-reader CCOB interception uses the
preserving policy instead: it rebuilds the CCOB after replacing only the
selected AMDGPU ELF, tries that rebuilt container first, and keeps the raw
patched ELF as a fallback if the reader rejects the container. The
`rocjitsu_afl_hsa_multi_ccob_reader_report_smoke` target validates the
multi-payload memory-reader path. The
`rocjitsu_afl_hsa_rdc_multi_ccob_reader_report_smoke` target repeats that check
on a HIP-compiler-produced `-fgpu-rdc` multi-arch offload bundle wrapped in
CCOB, so the rebuild path sees real RDC code-object payloads instead of only
hand-assembled branchy HSACOs.

Compiler-registered HIP fatbins are covered through lower HSA code-object
reader interposition. The HIP registration hooks record `HIPF`/`HIPK`
registration and host/device function names while the HSA reader hook patches
the selected raw ELF. For HIP runtime launches with no explicit kernel filter,
the preload temporarily scopes HSA-reader DBI to the launched runtime-function
name; this keeps large KPACK payloads from being patched as one monolithic
module. By default, the preload caches the selected raw ELF from a lower
HSA-reader patch under the active HIP fatbin registration and redirects later
named `hipLaunchKernel` calls through per-kernel shadow modules patched from
that cached image. Set
`ROCJITSU_AFL_DEBUG_DISABLE_RUNTIME_SHADOW_MODULES=1` only when isolating
loader issues; regression targets that must override an inherited disable can
set `ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1`. Shadow sources and
functions are released when
`__hipUnregisterFatBinary` unregisters that fatbin, or during preload teardown
if unregister-time release did not run. Patch reports emit
`runtime_shadow_source`, `runtime_shadow_function`, and
`runtime_shadow_registration_release` lifecycle records for this path.
`runtime_shadow_source` records include the decoded kernel count for each
cached image. `runtime_shadow_function` records include the cached source count,
matching source count, ambiguity flag, attempted source count, source dedupe
key, and microsecond patch/load/publish timings for the shadow module, which
are the debugging and overhead hooks for future multi-source KPACK cases. The
KPACK report smoke forces a repeated runtime launch and checks that the second
launch reports a
`runtime_shadow_function` cache hit instead of creating another shadow module. A
separate concurrent smoke launches the same runtime kernel from two host
threads, widens the publication race, and checks that the losing duplicate
shadow module is unloaded while both launches use the published shadow function.
ROCclr internal helper kernels are left uninstrumented to avoid recursively
initializing the HIP runtime while its own helper code objects are still loading.
Clients that fetch the AMD loader extension table are covered by rewriting the
v1.02+ `hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`
table slot to the same preload wrapper. The report smoke also creates a reader
through that table slot for the branchy HSACO and asserts the resulting
`hsa_reader_patch` handle diagnostics:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_loader_table_smoke
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_loader_table_reader_report_smoke
```

Persistent-boundary check:

```bash
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_persistent_smoke
```

Persistent harnesses can call `rocjitsu_afl_persistent_begin()` before each
input to reset the device counters and per-workitem `previous_bb` state, then
call `rocjitsu_afl_persistent_end()` after launching work to synchronize and
merge the iteration's device coverage into AFL's `trace_bits`. Set
`ROCJITSU_AFL_PERSISTENT=1` or `COV_PERSISTENT=1` to keep ordinary
`hipDeviceSynchronize` calls from merging outside those explicit boundaries.

Set `ROCJITSU_AFL_DEBUG_FIXED_EDGE_SLOTS=1` to force the older fixed-slot edge probes.
That path is kept only as a diagnostic fallback for separating CFG/relocation
failures from previous-BB state bugs.
Set `ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1` for conservative high-edge
experiments where a planned site should be dropped unless probe temporaries are
liveness-safe. The planner first tries registers that are dead at every
selected insertion point, then tries fresh registers above the kernel's current
SGPR/VGPR allocation where the target's resource model is patchable. On
gfx10+/RDNA, LLVM leaves the descriptor SGPR granule field at zero and records
the real `.sgpr_count` in AMDGPU metadata, so RocFuzz sizes liveness from that
metadata count and patches `.sgpr_count` when a selected fresh SGPR probe needs
more scalar registers. Patch reports expose `fresh_registers`,
`fresh_register_probe_points`, `has_metadata_sgpr_count`,
`descriptor_sgpr_count_effective`, `sgpr_count_metadata_patch`, and
`descriptor_resource_failure_reason` so this path is visible.
`ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS=1` is an allocator diagnostic for
synthetic tests that need to isolate SGPR fresh-register selection while leaving
VGPR selection on the normal liveness path; it is not a coverage mode.
`ROCJITSU_AFL_DEBUG_FORCE_FRESH_VGPRS=1` is the matching diagnostic for forcing
fresh VGPR temporaries while leaving SGPR selection on the normal liveness path.
`ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS=1` is a dangerous minimization
hook for reproducing blocked VOPD/Tensile faults under bounded branch-edge
limits. Pair it with
`tools/minimize_opaque_fresh_growth.py --diagnostic-profile=opaque-fresh`; do
not use it for normal fuzzing or coverage claims. Strict
opaque-scope reports also run a planning-only fresh-register candidate and emit
`opaque_fresh_register_candidate_*` fields plus sampled candidate probe sites so
we can tell whether the blocked coverage would require SGPR growth, VGPR growth,
or both without dispatching unsafe instrumentation.
`ROCJITSU_AFL_DEBUG_EDGE_PATCH_TEXT_OFFSET=<offset>` is another minimizer-only
selector. It filters the final edge plan to one `patch_text_offset` after normal
planning, which is useful for reducing a failing branch-edge limit to one
candidate patch/register tuple.
liveness drops also carry resource-specific skip reasons such as missing state
SGPR pairs, saved-EXEC SGPR pairs, VGPR runs, or liveness-analysis failures.
Set `ROCJITSU_AFL_KERNEL_INCLUDE=<substring>` or
`ROCJITSU_AFL_KERNEL_EXCLUDE=<substring>` to scope entry and edge patching while
debugging large library code objects.

## Divergence Semantics

The unit check also models the next coverage contract: each lane/workitem owns a
`previous_bb` entry, and a basic-block hook updates only lanes active under the
current execution mask. The test drives an entry block, divergent even/odd paths,
and a reconverged join to prove that masked-off lanes keep their prior
`previous_bb` until their path executes.

The current DBI runtime allocates per-workitem `previous_bb` state immediately
after the device counter map and emits the cuFuzz-compatible `FirstActiveLane`
counter policy. The logical extension point is still in the edge hook: change
the coverage key / counter policy to include distinct previous-BB values,
active-lane counts, or the execution mask when we want divergence-sensitive
feedback.

## Prototype Limits

- The DBI prologue is currently a fixed probe plan. GFX9/GFX10/CDNA use a
  conservative shared entry-counter probe; full edge probes remain limited to
  GFX11/GFX12 until the portable flat-address edge helpers are built out.
- The device-state pointer is currently embedded as a process-local literal in
  patched code. A production path should pass state through the HSA dispatch
  layer, a code-object global, or another ABI-stable mechanism.
- The edge probe assumes `v0` is workitem-id x and `ttmp9` carries the
  workgroup-id-derived value used by the smoke target. GFX11/GFX12 previous-BB
  probes preserve `EXEC` and `SCC` around the injected first-active-lane update,
  and the maintained wave64 branch target covers the full previous-BB-eligible
  synthetic branch set. EXEC-conditioned branches still degrade to fixed
  counters until the DBI side has a dynamic-target-safe proof for probing an
  outcome whose original EXEC may be empty. The
  planner conservatively skips block entries whose first
  instruction is an ALU wait or EXEC save/transition until relocation can
  preserve those scheduling and mask semantics around injected probes.
- Edge patching currently handles only block entries whose first instruction can
  be copied into a trampoline without relocating embedded control flow or
  preserving status flags across the return branch.
- The current edge hook records incoming edges at successor block entry using
  per-workitem `previous_bb`; the branch-terminator patcher can cover a bounded
  set of direct predecessor terminators, including conditional taken/fallthrough
  edges, but does not yet handle arbitrary overwritten-instruction relocation.
- Selected edge trampolines are planned before text redirects or appended cave
  bytes are installed. Descriptor updates are deferred until entry and edge
  probes are selected, but arbitrary relocation still needs the fuller planner
  split described below.
- Local text-cave placement only consumes existing aligned zero-byte runs in
  `.text`. It is enough for the current rocBLAS SGEMM CCOB validation, but a
  product DBI planner should create or relocate branch-reachable caves instead
  of depending on incidental padding.
- Per-workitem `previous_bb` storage is a fixed-size side array in the same
  device allocation as the counters; large grids still need bounds-aware state
  sizing.
- `ROCJITSU_AFL_PATCH_REPORT` reports the current edge/cave decisions and emits
  post-sync `device_edge_delta` records with launch-path attribution for normal,
  runtime-shadow, and lazy CCOB launches. It covers device, stream, event,
  blocking memcpy, and persistent-end sync points, but does not yet cover every
  blocking HIP API or include spill/private-segment planning.
- The default edge limit is twelve sites for the smoke target. More sites can be
  selected, but some currently require stronger relocation and liveness support.
- Skip-entry edge probes can use liveness-selected SGPR/VGPR temporaries.
  Entry-only probes have a separate entry live-in calculation because the
  prologue runs before compiler-generated EXEC masking. Kernel descriptors are
  sized from the union of selected entry and edge probes, using the concrete
  resource requirements of the selected probe form instead of the full
  previous-BB high-water mark for every site. Scratch/private-segment spill
  planning is part of the adaptive planner. Direct HSACO branch probes can use
  it when descriptor and AMDGPU metadata growth both succeed, including
  metadata-backed `.sgpr_count` growth on gfx10+/RDNA and private-segment
  metadata growth for scratch spills. CCOB, HSA-reader, runtime KPACK, and
  block-entry scratch paths still report stable disabled reasons instead of
  attempting known-unproven launches. EXEC-empty
  fixed-counter fallbacks can also use scratch-backed counter VGPR preservation
  with a liveness-dead allocated scratch-address VGPR, so fully allocated VGPR
  files no longer require fresh address growth. Kernels with no liveness-dead
  address VGPR still fail closed.
- Unsupported code objects fall back to the original unpatched `hipModuleLoad`.
- Real multi-payload library CCOB artifacts still need artifact coverage. The
  inventory smoke reports when none are present locally; the checked-in smokes
  cover real single-payload library CCOBs and a compiler-produced multi-arch
  `-fgpu-rdc` offload bundle wrapped in CCOB.
