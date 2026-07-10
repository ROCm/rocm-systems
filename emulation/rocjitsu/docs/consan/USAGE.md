# ConSan Usage

This is the detailed runbook for ConSan in rocJITsu. For a shorter guided
walkthrough, start with [TUTORIAL.md](TUTORIAL.md). The currently implemented
flavors are `supercollider` and `moi`. The
SuperCollider flavor is the redundant-access check/trap proof path and is the
recommended mode for anyone trying the current snapshot. The MOI flavor
currently has a `record_replay` engine with dynamic access records,
experimental barrier records, HSA-tool-owned report buffers, host-side
exact-shadow replay, host-side release/acquire atomic modeling, and a narrow
experimental DBI atomic-record patch for selected RDNA4 no-SADDR
`flat_atomic*` instructions. It also has a first-class `sampled` engine whose
current implementation can publish compact sampled watchpoint entries directly
from DBI probes, without first writing full access records. Its
`inline_shadow` engine can also publish exact-shadow entries directly from
native dword LDS load/store probes. With `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` it can
also emit a first compact inline diagnostic for a prior non-empty,
different-owner conflict. ConSan is instrumentation-only on the local RDNA4 `gfx1201`
device; it is not meant to translate the code object to another architecture.

## Shortest Useful Snapshot Run

For a colleague-facing smoke run, start with SuperCollider mode:

```sh
export ROCJITSU_BUILD_DIR=/path/to/rocm-systems/emulation/rocjitsu/build
export ROCM_DIST_DIR=/path/to/rocm

export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_CONSAN_FLAVOR=supercollider
export RJ_CONSAN_LOG=1
export RJ_CONSAN_DELAY_MODE=sleep
export RJ_CONSAN_DELAY=1
export RJ_CONSAN_MAX_PATCHES=1

./app-or-test
```

Useful success signal:

```text
ConSan summary ... patches=... modified=true
```

If that works on a focused workload, add the stricter guard:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
```

That guard fails a code-object load when rocJITsu sees a supported site in the
active instrumentation scope but cannot actually rewrite any such site. It is
useful for demos and CI-like checks, but it is too strict for broad exploratory
runs where many loaded helper code objects may legitimately have no supported
sites.

Treat MOI as developer-facing for now. Use it to run the checked rocJITsu GPU
tests or to inspect record-buffer logs, not as the first external try-it path.

## Build

From the `rocm-systems` tree:

```sh
cmake --build emulation/rocjitsu/build --target rocjitsu_dbi_hooks
cmake --build emulation/rocjitsu/build --target rocjitsu_tests
```

The HSA tools library built by the first command is:

```text
emulation/rocjitsu/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so
```

The commands below assume these paths are set for the local checkout/builds:

```sh
export ROCJITSU_BUILD_DIR=/path/to/rocm-systems/emulation/rocjitsu/build
export HIP_MOI_BUILD_DIR=/path/to/hip-moi-build
export IREE_BUILD_DIR=/path/to/iree-build
export ROCM_DIST_DIR=/path/to/rocm
```

## How The Hook Is Loaded

Use the HSA tools path:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_DELAY=2 \
  ./app
```

`HSA_TOOLS_LIB` is the same HSA-level mechanism used by other ROCm tools:

- `projects/rocr-runtime/runtime/hsa-runtime/core/util/flag.h` reads
  `HSA_TOOLS_LIB` in the ROCr runtime.
- `projects/rocprofiler/doc/ROCProfiler_V1_API_spec.md` documents
  `HSA_TOOLS_LIB` as the library loaded by the HSA runtime for rocprofiler.
- `projects/rocr-debug-agent/docs/how-to/user-guide.rst` uses
  `HSA_TOOLS_LIB=/opt/rocm/lib/librocm-debug-agent.so.2 ./my_program`.
- The HSA API surface being wrapped is documented in
  `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa.h` and
  `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_api_trace.h`; the relevant
  calls are the code-object reader creation APIs and
  `hsa_executable_load_agent_code_object`.

The MVP uses this HSA tools loader path. A separate waitcheck-style
`LD_PRELOAD` path is not needed for the current ConSan work.

## Environment Variables

- `RJ_CONSAN_FLAVOR=supercollider|moi`: select the ConSan instrumentation
  flavor. There is intentionally no default; each run must opt in explicitly.
- `RJ_CONSAN_MOI_ENGINE=record_replay|inline_shadow|sampled`: select the MOI
  engine when `RJ_CONSAN_FLAVOR=moi`. The default is `record_replay`.
  `record_replay` inventories native `ds_*` and likely group/LDS `flat_*`
  candidates, defines the report ABI, and can patch supported native LDS or
  likely group/LDS zero-offset flat load/store sites to write first-light access
  records when a MOI report buffer and scratch VGPR are supplied. RDNA
  two-address DS sites lower to two ordinary access records. The checked GPU
  smoke test host-replays two DBI-written native-LDS records into an exact-shadow
  conflict diagnostic. `sampled` patches supported load/store sites to write
  packed sampled watchpoint entries directly into a compact report-buffer layout.
  HSA-tool-owned sampled report summaries scan those entries for lower-fidelity
  sampled conflicts. Host-side sampled packing/replay
  helpers still exist as semantic reference code, but the checked sampled GPU
  path no longer writes full access records first.
  `inline_shadow` patches native LDS `ds_store_b32`/`ds_write_b32` and
  `ds_load_b32`/`ds_read_b32` sites to publish packed exact-shadow entries
  directly from GPU code with RDNA4 `flat_atomic_swap_b64`. When
  `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` is provided, it also checks the returned prior
  shadow entry and emits one compact diagnostic for the first implemented
  conflict predicate: prior entry nonzero, prior owner different from the
  current owner, and not a read/read pair. It does not yet implement inline
  barrier epochs or wider LDS forms.
- `RJ_CONSAN_MOI_BACKEND=context|sampled_watchpoint`: legacy alias. It is used
  only when `RJ_CONSAN_MOI_ENGINE` is unset. `context` maps to `record_replay`;
  `sampled_watchpoint` maps to `sampled`.

MOI knob summary:

| Knob | Applies To | Notes |
| --- | --- | --- |
| `RJ_CONSAN_MOI_ENGINE` | all MOI runs | Public engine selector. |
| `RJ_CONSAN_MOI_BACKEND` | legacy all MOI runs | Ignored when `RJ_CONSAN_MOI_ENGINE` is set. |
| `RJ_CONSAN_MOI_REPORT_BUFFER`, `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE` | `record_replay`, `inline_shadow`, `sampled` | Explicit caller-owned report buffer. The layout depends on `RJ_CONSAN_MOI_ENGINE`. |
| `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE` | `record_replay`, `inline_shadow`, `sampled` | HSA-tool-owned report buffer, easiest for IREE. The HSA tool allocates the matching engine layout. For `record_replay`, teardown replays visible access/barrier/atomic records and logs the first diagnostic when one is emitted. For `inline_shadow`, teardown counts visible diagnostics and samples nonzero exact-shadow entries; the buffer must be at least 131,496 bytes with the default four diagnostic slots and one internal inline atomic release slot so the exact-shadow table covers the full 64 KiB LDS address range. |
| `RJ_CONSAN_MOI_REQUIRE_RECORDS` | `record_replay`, `inline_shadow`, `sampled` with auto report buffers | Demo guard: process exits nonzero at hook unload if all auto report buffers contain zero visible access/barrier/atomic/diagnostic/exact-shadow/sampled records. |
| `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS` | `record_replay`, `inline_shadow`, `sampled` with auto report buffers | Demo guard: process exits nonzero at hook unload if auto report buffers contain zero visible inline diagnostics, zero host-replay diagnostics, and zero sampled conflicts. The checked inline-shadow race and sampled-conflict controls use this guard. |
| `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS` | `record_replay`, `inline_shadow`, `sampled` with auto report buffers | Demo guard: process exits nonzero at hook unload if auto report buffers contain any visible inline diagnostic, host-replay diagnostic, or sampled conflict. The checked inline-shadow barrier-order control uses this guard. |
| `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT` | `record_replay` with auto report buffers | Demo guard: process exits nonzero at hook unload if auto-buffer host replay emits no conflict. |
| `RJ_CONSAN_TMP_VGPR` | `record_replay`, `inline_shadow`, `sampled` | Required for current access/barrier/sampled/inline-shadow DBI probes. Static access probes use three scratch VGPRs, dynamic access probes use eight, barrier probes use six, direct sampled probes use five, inline-shadow publication probes use seven, inline-shadow diagnostic probes use nine, and inline-shadow atomic-ordering probes use five starting at this VGPR. |
| `RJ_CONSAN_MOI_SAMPLE_STRIDE`, `RJ_CONSAN_MOI_SAMPLE_OFFSET` | `sampled` | Deterministic direct-sampled candidate selection. Defaults are stride 1, offset 0. |
| `RJ_CONSAN_MOI_TRACK_BARRIERS` | `record_replay`, `inline_shadow` | For `record_replay`, emit dynamic barrier records and let host replay advance epochs. For `inline_shadow`, trampoline supported RDNA4 barrier sites so the original barrier executes and then the configured epoch VGPR increments. Accepted with `sampled` but sampled replay does not yet consume barrier records. |
| `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS` | `record_replay` | Opt-in access-record append mode. Each active lane reserves its own access-record slot at runtime. Requires `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`. Dynamic probes preserve VCC and currently skip candidates immediately after `s_*_saveexec` while control-flow/liveness handling is still conservative. |
| `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` | `record_replay` dynamic access/barrier experiments, `inline_shadow` diagnostics/atomics | Explicit even normal SGPR-pair base. `record_replay` dynamic probes require an even pair in `0..104`. `inline_shadow` diagnostics require an even base in `0..100` without epoch tracking, or `0..98` with `RJ_CONSAN_MOI_EPOCH_VGPR`, because they use up to four SGPR pairs while narrowing the old-entry, owner-mismatch, same-epoch, and access-kind predicates. Inline-shadow atomic acquire probes require an even base in `0..100`. |
| `RJ_CONSAN_MOI_TRACK_ATOMICS` | `record_replay`, `inline_shadow` | For `record_replay`, enable the experimental atomic-record patch for a narrow RDNA4 no-SADDR `flat_atomic*` subset; host replay models supported release/acquire atomic events, and live GPU smoke coverage checks one same-address handoff. For `inline_shadow`, enable the first one-slot address-scoped release/acquire metadata patch for the same flat-atomic subset; release atomics publish owner/epoch/address into the internal slot, and acquire atomics import `release_epoch + 1` only when the address matches. |
| `RJ_CONSAN_MOI_OWNER_VGPR`, `RJ_CONSAN_MOI_EPOCH_VGPR`, `RJ_CONSAN_MOI_INIT_OWNER_EPOCH` | `record_replay`, `inline_shadow` experiments | For `inline_shadow`, `RJ_CONSAN_MOI_OWNER_VGPR` is required. The recommended prototype recipe is to set `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`, choose owner/epoch VGPRs below the scratch range, and let the DBI prologue initialize them. |
| `RJ_CONSAN_MOI_OWNER_SOURCE`, `RJ_CONSAN_MOI_OWNER_SGPR` | owner/epoch prologue experiments | Select how `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1` initializes the owner VGPR. Default `workitem_id` uses the original `workitem_id_x >> log2(wavefront_size)` estimate. `hw_id` reads RDNA4 `HW_ID1` with `s_getreg_b32`, uses its low 10 bits as a wave-uniform owner, and requires `RJ_CONSAN_MOI_OWNER_SGPR=<0..105>` as a scalar temp. ConSan grows the descriptor SGPR allocation for that temp but still does not prove scalar-register liveness or spill it. |
| `RJ_CONSAN_REQUIRE_PATCH` | all flavors | For MOI, currently guards supported access candidates for the selected engine. |

The hook warns when MOI-only knobs are provided while `RJ_CONSAN_FLAVOR` is not
`moi`, when both the new engine selector and legacy backend selector are set,
and when engine-specific knobs are provided for an engine that does not consume
them.
- `RJ_CONSAN_LOG=1`: emit compact info logs. Higher numeric values enable more
  verbose per-kernel diagnostics.
- `RJ_CONSAN_DUMP_DIR=/tmp/rj-dbi-dump`: write original memory-backed code objects,
  and patched code objects when a patch is produced, as `.hsaco` files for
  `llvm-readelf` / `llvm-objdump` inspection.
- `RJ_CONSAN_CHECK_TRAP_MODE=all|lds|flat`: restrict the primary check/trap
  instrumentation scope. The default is `all`: try native `ds_*` LDS
  instrumentation first, then let likely group/LDS `flat_*` instrumentation use
  any remaining patch budget when the existing DS patch ranges still map into
  the original code object. Use `lds` or `flat` only for targeted debugging.
- `RJ_CONSAN_DELAY=N`: configure the delay between the original LDS/flat access
  and the duplicate/readback check.
- `RJ_CONSAN_DELAY_MODE=nop|sleep|sleep_var`: choose the delay encoding. The
  default `nop` mode emits `N` simple `s_nop 0` instructions. The `sleep` mode
  emits one `s_sleep N` instruction when `N > 0`; `N` must fit the 16-bit
  `s_sleep` immediate. The `sleep_var` mode emits one `s_sleep_var` instruction
  when `N > 0`.
- `RJ_CONSAN_DELAY_VAR_SSRC=N`: scalar source operand encoding for
  `RJ_CONSAN_DELAY_MODE=sleep_var`. The default is `106`, the RDNA4 `vcc_lo`
  operand encoding already preserved by the injected native-DS compare path.
- `RJ_CONSAN_MAX_PATCHES=N`: bound how many native-DS or flat/VFLAT check/trap
  sites can be instrumented in a single code object. The default is `1`.
  Selection is file-ordered and rejects overlapping inline ranges, anchor
  rewrites, and local NOP caves.
- `RJ_CONSAN_TMP_VGPR=N`: force a scratch VGPR. Use this for hand-shaped tests
  whose kernel descriptor reserves the selected VGPR. Normal auto-selection
  first prefers descriptor-covered scratch and can grow the descriptor as a
  fallback for native-DS sites when no ordinary candidate is available.
- `RJ_CONSAN_REQUIRE_PATCH=1`: test guard. If a code object has a supported
  site in the active instrumentation scope but no patch is emitted, fail the
  load. For `supercollider`, this applies to the active check/trap scope. For
  `moi`, this applies to supported first-light access-record candidates.
  Code objects with no supported sites still pass through.
- `RJ_CONSAN_REPORT_BUFFER=0xADDR`: opt-in prototype report-buffer ABI. On a
  mismatch, the injected check writes one 32-bit marker to this caller-supplied
  device-visible address and continues instead of executing `s_trap 0`. rocJITsu
  does not allocate this buffer or pass it as a kernel argument; a test harness
  must provide an address that patched device code can legally store to. The
  hook refreshes this field at each code-object load, so a harness can allocate
  the buffer, export the address, and then launch the kernel whose code object
  will be patched.
- `RJ_CONSAN_REPORT_MARKER=N`: marker value written to
  `RJ_CONSAN_REPORT_BUFFER`. The default is `1`. This is intentionally just a
  sticky "a mismatch happened" flag, not a per-site or per-lane report record.
- `RJ_CONSAN_MOI_REPORT_BUFFER=0xADDR`: MOI report/state buffer address. The
  caller owns allocation and must pass a device-visible address. This is
  separate from `RJ_CONSAN_REPORT_BUFFER`, which is only the SuperCollider
  one-word marker path.
- `RJ_CONSAN_MOI_REPORT_BUFFER_SIZE=N`: byte size of the MOI report/state
  buffer. For `record_replay`, this must hold the versioned header plus at
  least one access record. For `sampled`, this must hold the versioned header
  plus at least one packed 64-bit sampled watchpoint entry. For
  `inline_shadow`, this must hold the versioned header, the small diagnostic
  section, and enough packed 64-bit exact-shadow entries for the LDS byte range
  being exercised. The current prototype does not yet emit an in-kernel bounds
  guard around exact-shadow publication. The header, access record, diagnostic
  record, exact-shadow, and sampled layouts are defined in `consan_moi.h`. When
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`, the current
  `record_replay` prototype derives equal access and barrier record capacities
  from the payload size.
- `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=N`: ask the HSA tool to allocate one
  MOI report buffer per code object and pass it into DBI patching. This is the
  easiest path for IREE-style runs because the application does not need to
  provide a kernel argument or exported device pointer. For `record_replay`,
  the teardown log prints access/barrier/atomic capacities, visible and dropped
  record counts, `event_counter`, host-replay totals, the first replay
  diagnostic when one is emitted, and a few sample records. For `sampled`, it
  prints sampled capacity/count, sampled conflict count, the first conflicting
  sampled pair when present, and up to a few decoded sampled entries. For
  `inline_shadow`, it prints exact-shadow capacity, visible nonzero
  exact-shadow entries, and up to a few decoded exact-shadow entries. With
  `RJ_CONSAN_MOI_TRACK_BARRIERS=1`, the `record_replay` auto buffer is split
  into access and barrier sections.
- `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1`: opt into dynamic access-record
  append for `record_replay`. The default access-record probe writes one static
  slot per patched site. Dynamic append atomically increments
  `access_record_count` per active lane, masks out lanes whose slot is beyond
  capacity, writes per-lane records, and restores EXEC. This mode requires
  `RJ_CONSAN_MOI_EXEC_SAVE_SGPR` and can consume report-buffer slots quickly.
- `RJ_CONSAN_MOI_REQUIRE_RECORDS=1`: demo guard for
  `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE`. At hook unload, fail the process with
  a nonzero exit code if every HSA-tool-owned MOI report buffer contains zero
  visible access/barrier/atomic/diagnostic/exact-shadow/sampled records. This
  is intentionally an end-of-process guard for IREE-style demos, not a
  per-dispatch assertion.
- `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1`: demo guard for
  `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE`. At hook unload, fail the process with
  a nonzero exit code if auto report buffers contain zero visible inline
  diagnostics, zero host-replay diagnostics, and zero sampled conflicts. This is
  the guard used by the checked inline-shadow two-wave race diagnostic and the
  checked sampled conflict smoke.
- `RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1`: demo guard for
  `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE`. At hook unload, fail the process with
  a nonzero exit code if auto report buffers contain any visible inline
  diagnostic, host-replay diagnostic, or sampled conflict. This is the guard
  used by the checked inline-shadow barrier-order clean control.
- `RJ_CONSAN_MOI_REQUIRE_REPLAY_CONFLICT=1`: stricter `record_replay` demo guard
  for auto report buffers. At hook unload, fail the process with a nonzero exit
  code if auto-buffer host replay emits no conflict. This is useful for small
  racy controls; do not use it for known-correct IREE e2e runs.
- `RJ_CONSAN_MOI_SAMPLE_STRIDE=N`: direct sampled candidate-selection stride.
  The default is `1`, which makes every supported sampled access candidate
  eligible. Values must be at least `1`.
- `RJ_CONSAN_MOI_SAMPLE_OFFSET=M`: direct sampled candidate-selection offset.
  The default is `0`; `M` must be smaller than
  `RJ_CONSAN_MOI_SAMPLE_STRIDE`. Candidate `i` is eligible when
  `i % stride == offset`. This is a deterministic static site filter, not a
  runtime random sampler.
- `RJ_CONSAN_MOI_OWNER_VGPR=N`: experimental MOI first-light hook. When set,
  the injected LDS probe copies this VGPR into the access record's owner field.
  When unset for a kernel-owned first-light site, the probe derives a prototype
  owner as `workitem_id_x >> log2(wavefront_size)`, using the kernel
  descriptor's wavefront-size bit, and writes that value to the record. With
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`, the kernel-entry prologue initializes this
  VGPR from `RJ_CONSAN_MOI_OWNER_SOURCE`. In direct sampled mode, an explicit
  owner VGPR value is packed into a 10-bit sampled field without additional
  in-kernel masking.
- `RJ_CONSAN_MOI_OWNER_SOURCE=workitem_id|hw_id`: controls only the
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1` prologue. `workitem_id` is the default and
  initializes owner as `workitem_id_x >> log2(wavefront_size)`. `hw_id` reads
  RDNA4 `HW_ID1` with `s_getreg_b32`, copies the low 10 bits into the owner
  VGPR, and is independent of 1D/2D/3D local invocation layout. The `hw_id`
  path requires `RJ_CONSAN_MOI_OWNER_SGPR`.
- `RJ_CONSAN_MOI_OWNER_SGPR=N`: scalar temp used by
  `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id`. `N` must be a normal SGPR in `0..105`.
  The prologue grows the kernel descriptor's SGPR allocation to cover it, but
  ConSan still does not prove that the original kernel has no live value there.
- `RJ_CONSAN_MOI_EPOCH_VGPR=N`: experimental MOI first-light hook. When set,
  the injected LDS probe copies this VGPR into the access record's epoch field.
  With `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`, the kernel-entry prologue initializes
  this VGPR to zero. The newer barrier-record path does not require this VGPR;
  host replay advances epochs from emitted barrier records. In direct sampled
  mode, an explicit epoch VGPR value is packed into a 10-bit sampled field
  without additional in-kernel masking.
- `RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1`: experimental MOI entry-prologue patch.
  Requires `RJ_CONSAN_MOI_OWNER_VGPR` and `RJ_CONSAN_MOI_EPOCH_VGPR`. The patch
  appends a tiny RDNA4 stub to `.text`, redirects each kernel descriptor to it,
  initializes the owner/epoch VGPRs, and branches back to the original entry.
  It grows the kernel descriptor's VGPR allocation, and for
  `RJ_CONSAN_MOI_OWNER_SOURCE=hw_id` also the SGPR allocation, when needed. It
  does not yet prove those registers are semantically unused by the original
  kernel.
- `RJ_CONSAN_MOI_TRACK_BARRIERS=1`: experimental MOI barrier record patch.
  Requires a MOI report buffer, either explicit or
  `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE`, plus `RJ_CONSAN_TMP_VGPR`. The
  dynamic append prototype also requires `RJ_CONSAN_MOI_EXEC_SAVE_SGPR=N`,
  where `N` is an even normal SGPR-pair base in `0..104` that the original
  kernel is not using. Each decoded 32-bit barrier is rewritten to branch
  through an appended `.text` trampoline. The trampoline narrows record writes
  to one representative lane, dynamically reserves a barrier-record slot,
  skips only the record writes on overflow, restores EXEC, executes the
  original barrier, and branches back. Raw records represent barrier arrivals;
  host replay coalesces a contiguous same-workgroup run of arrivals into one
  logical epoch advance.
- `RJ_CONSAN_MOI_TRACK_ATOMICS=1`: experimental MOI atomic patching.
  Requires a MOI report buffer, either explicit or
  `RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE`, plus `RJ_CONSAN_TMP_VGPR`. The
  current DBI path supports only a narrow RDNA4 no-SADDR `flat_atomic*` subset.
  In `record_replay`, it writes `ConSanMoiAtomicRecord` entries that host replay
  consumes as release/acquire events; the current live GPU smoke covers one
  same-address flat-atomic handoff and a wrong-address control. In
  `inline_shadow`, it writes one internal release slot and lets acquire atomics
  import the releasing epoch only when the 64-bit flat address matches. The
  inline-shadow path also requires `RJ_CONSAN_MOI_OWNER_VGPR`,
  `RJ_CONSAN_MOI_EPOCH_VGPR`, and `RJ_CONSAN_MOI_EXEC_SAVE_SGPR`.
- `RJ_CONSAN_FAULT_DROP_BARRIER=1`: demo-only synchronization fault injection.
  After the primary proof/instrumentation pass, rewrite one decoded
  `s_barrier*` instruction to `s_nop 0`.
- `RJ_CONSAN_FAULT_BARRIER_INDEX=N`: select which decoded barrier to drop in a
  code object. The default is `0`.
- `RJ_CONSAN_FAIL_CLOSED=1`: debugging guard that rejects unsupported kernels
  rather than quietly skipping them.

The default check/trap scope is combined but conservative. For a given code
object, rocJITsu tries native DS first, then lets flat/VFLAT consume any
remaining `RJ_CONSAN_MAX_PATCHES` budget when the existing DS patch ranges can
be mapped in the original code object and the flat ranges do not overlap them.
If the DS pass used appended `.text` growth, flat composition skips that object
instead of guessing at shifted ELF offsets. Barrier fault injection is
composable because it runs after the selected check/trap path. The bring-up-only
probes below are explicit debug overrides of the default check/trap path.

Native-DS scratch selection first stays inside the AMDHSA kernel descriptor's
allocated VGPR count. If that prevents all inline or local-cave native-DS
patches for a code object, rocJITsu may select liveness-free scratch above the
original allocation and grow
`COMPUTE_PGM_RSRC1.GRANULATED_WORKITEM_VGPR_COUNT`. That growth is
fallback-only; an ordinary in-descriptor patch wins whenever one exists.

Additional bring-up-only probes also exist:

- `RJ_CONSAN_PROBE_NOP=1`
- `RJ_CONSAN_PROBE_TRAMPOLINE_NOP=1`
- `RJ_CONSAN_PROBE_ENDPGM=1`
- `RJ_CONSAN_PROBE_LDS_ENDPGM=1`
- `RJ_CONSAN_PROBE_FLAT_TRAP=1`

Those are useful for debugging code-object mutation but are not the race-check
mode. `RJ_CONSAN_PROBE_TRAMPOLINE_NOP=1` skips ROCclr runtime helper-only code
objects and code objects with no supported DBI candidate sites.
`RJ_CONSAN_PROBE_FLAT_TRAP=1` rewrites likely group/LDS `flat_load/store` sites
in place as `s_trap 0; s_nop 0; s_nop 0`; it is intentionally destructive and
should make an executing patched site fail.

## Coverage Signal

With `RJ_CONSAN_LOG=1`, look for:

```text
ConSan summary reader=... kernels=... candidates=... skips=... rejects=... supported_lds_sites=... flat_sites=... patches=... modified=...
```

Interpretation:

- `modified=true` and `patches>0`: this code object was actually patched.
- `inline-barrier-nop-rewrite`: demo-only sync-fault injection rewrote one
  selected decoded barrier to `s_nop 0`. It intentionally skips ROCclr
  runtime-helper-only code objects.
- `supported_lds_sites=0`: the current native-DS MVP found no supported
  instructions in this code object.
- `local-cave-lds-load-check-trap` / `local-cave-lds-store-check-trap` patch
  logs mean a compact native DS site was redirected through uncovered local NOP
  slack and returned to the original fallthrough. The focused IREE WMMA
  ROCm/HIP e2e object patches one `ds_load_2addr_b64` site this way and passes
  under `RJ_CONSAN_REQUIRE_PATCH=1`.
- `flat_sites>0`: flat/generic memory sites were decoded and logged at
  `RJ_CONSAN_LOG=2`.
- `function_flat_maybe_group_hints>0`: helper-function flat sites are likely
  LDS/shared accesses. The destructive flat proof can patch these with
  `RJ_CONSAN_PROBE_FLAT_TRAP=1`. Padded likely group flat loads and stores are
  checked by the default combined scope when patch budget remains after native
  DS selection; ordinary unpadded hip-moi helper sites can be checked through
  conservative local NOP caves when they are reachable.
- `local-cave-flat-load-check-trap` / `local-cave-flat-store-check-trap` patch
  logs mean an unpadded flat helper site was redirected through uncovered local
  NOP slack and returned to the original fallthrough. The focused hip-moi
  `NoPipelineProd16x8` object patches one such site and passes cleanly.
- `skips>0` with `modified=false`: the hook observed the code object but chose
  pass-through for policy or coverage reasons.
- `rejects>0`: the hook found something it considered unsafe or unsupported.
- Proof-mode warnings such as `skipped ROCclr runtime helper code object` and
  `skipped code object without supported DBI candidate sites` mean the proof
  mode intentionally avoided an expensive or unsafe trampoline search.
- For flat/VFLAT selection, an unpadded skip warning includes
  `supported_candidates=...`, `scratchable_candidates=...`,
  `max_observed_padding_words=...`, `append_cave_reachable_candidates=...`,
  `uncovered_nop_caves=...`, `max_uncovered_nop_cave_words=...`, and
  `local_cave_reachable_candidates=...` so the local-cave/code-growth gap is
  measurable. In the focused hip-moi `NoPipelineProd16x8` object, appended
  end-of-text caves are not directly reachable, but conservative uncovered NOP
  caves are reachable for the likely group flat helper candidates.

The broad hip-moi matmul kernels use flat/generic and scratch forms rather than
native `ds_load/store_*`. The focused IREE e2e kernels expose compact native DS
sites. The hook can identify and destructively trap likely group/LDS
flat helper-function sites. The non-destructive race-checking paths cover padded
native DS sites, compact native DS sites through local or appended caves, native
DS descriptor-growth fallback cases, padded likely group flat sites, and
unpadded likely group flat sites when conservative local NOP caves are
reachable.

## What The MVP Instruments

The live-safe path instruments these native non-atomic LDS instructions:

- `ds_load_b32`
- `ds_load_b64`
- `ds_load_b128`
- `ds_load_2addr_b32`
- `ds_load_2addr_b64`
- `ds_load_2addr_stride64_b32`
- `ds_load_2addr_stride64_b64`
- `ds_load_u16_d16`
- `ds_load_u16_d16_hi`
- `ds_store_b32`
- `ds_store_b64`
- `ds_store_b128`

It also instruments likely group/LDS VFLAT instructions:

- likely group/LDS `flat_load_b32`
- likely group/LDS `flat_load_b64`
- likely group/LDS `flat_load_b128`
- likely group/LDS `flat_store_b32`
- likely group/LDS `flat_store_b64`
- likely group/LDS `flat_store_b128`

For loads, the patch duplicates the load after the requested delay and reports
if the two values differ. For stores, the patch reads back the stored LDS value
after the requested delay and reports if it differs from the original stored
value. By default, reporting is `s_trap 0`. With `RJ_CONSAN_REPORT_BUFFER`, the
report action writes `RJ_CONSAN_REPORT_MARKER` to the supplied buffer address
instead. The injected compares preserve `vcc_lo` by saving it to a
liveness-selected SGPR. For native DS and likely group flat helper sites, the
patch may use trailing padding, redirect through conservative uncovered local
NOP caves, or use an appended `.text` cave for native DS when that is the safe
placement. Native DS can also grow the descriptor VGPR allocation as a fallback
when no ordinary in-descriptor patch candidate exists. All of these paths are
bounded by `RJ_CONSAN_MAX_PATCHES`.

## Repeatable Local Tests

rocJITsu positive/negative regression:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R 'ConSanLdsTest' \
  --parallel 1 \
  --output-on-failure
```

That regression includes two non-trapping marker-buffer cases:

- `ConSanLdsTest.DbiPaddedCleanStoreReportBufferStaysZero`
- `ConSanLdsTest.DbiPaddedRacyStoreReportsMarker`

Both tests allocate a device report word in the HIP test process, set
`RJ_CONSAN_REPORT_BUFFER` to that device address before launching the padded
LDS kernel, and run with `RJ_CONSAN_REQUIRE_PATCH=1`. The clean case proves that
the report word is not spuriously written. The racy case proves that the same
instrumented site can report a mismatch by writing `RJ_CONSAN_REPORT_MARKER`
and allowing the dispatch to complete.

MOI dynamic access-record and host-replay smoke:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R 'ConSanMoiHipTest' \
  --parallel 1 \
  --output-on-failure
```

That regression runs with `RJ_CONSAN_FLAVOR=moi` through the HSA tools hook. It
allocates a MOI report buffer in the HIP test process, patches two padded
native-LDS sites, verifies two DBI-written access records with distinct
prototype owners, replays them into a record/replay conflict diagnostic, and verifies
that advancing the consumer epoch suppresses the diagnostic on the same dynamic
records.

MOI inline-shadow publication smoke:

```sh
env HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=262144 \
  RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
  RJ_CONSAN_REQUIRE_PATCH=1 \
  RJ_CONSAN_TMP_VGPR=104 \
  RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1 \
  RJ_CONSAN_MOI_OWNER_VGPR=102 \
  RJ_CONSAN_MOI_EPOCH_VGPR=103 \
  RJ_CONSAN_MOI_EXEC_SAVE_SGPR=30 \
  RJ_CONSAN_MAX_PATCHES=4 \
  RJ_CONSAN_LOG=1 \
  "$ROCJITSU_BUILD_DIR/tests/hip_consan_lds_test" \
  '--gtest_filter=ConSanLdsTest.PaddedCleanStoreCompletes'
```

This is a clean-control publication test with inline diagnostics enabled. The
HSA-tool teardown log should show `visible_exact_shadow` greater than zero,
`visible_diagnostics=0`, and sample decoded exact-shadow entries with `kind=2`,
`generation=1`, and the patched instruction offset.

MOI inline-shadow live race diagnostic:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" -j8 \
  -R 'ConSanInlineShadowTest.DbiReportsCrossWaveRace' \
  --output-on-failure
```

That CTest launches a 128-thread block whose two waves write the same LDS dword
and runs with `RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1`. The HSA-tool teardown log
should show `visible_diagnostics=1`.

MOI inline-shadow barrier-order clean control:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" -j8 \
  -R 'ConSanInlineShadowTest.DbiBarrierEpochOrdersCrossWaveAccesses' \
  --output-on-failure
```

That CTest launches a 128-thread block where one wave writes an LDS dword before
a full-workgroup barrier and another wave writes the same LDS dword after the
barrier. It runs with `RJ_CONSAN_MOI_TRACK_BARRIERS=1` and
`RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1`, so the process fails if inline-shadow
epoch tracking reports a false conflict across the barrier.

MOI cross-engine semantic smoke:

```sh
"$ROCJITSU_SOURCE_DIR/tests/dbi/consan_cross_engine_smoke.sh" \
  "$ROCJITSU_BUILD_DIR"
```

This runs the focused record/replay, inline-shadow, and sampled race controls,
plus record/replay and inline-shadow barrier-order clean controls. The script
uses `ctest -j8`.

Selected hip-moi smoke:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_DELAY=2 \
  ctest --test-dir "$HIP_MOI_BUILD_DIR" \
    -R 'JakubRdna4MatmulReference|HipMoiRdna4Pingpong' \
    --parallel 8 \
    --output-on-failure
```

Focused hip-moi flat local-cave smoke:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_DELAY=1 \
  ctest --test-dir "$HIP_MOI_BUILD_DIR" \
    -R NoPipelineProd16x8 \
    --parallel 1 \
    --output-on-failure -V
```

Focused IREE ROCM/ROCDL smoke:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_DELAY=2 \
  ctest --test-dir "$IREE_BUILD_DIR" \
    -R 'iree/compiler/plugins/target/ROCM/test/smoketest.mlir.test|iree/compiler/plugins/target/ROCM/test/smoketest_hsaco.mlir.test|iree/compiler/Codegen/LLVMGPU/test/convert_to_rocdl_gfx1201.mlir.test|iree/compiler/Codegen/LLVMGPU/test/ROCDL/config_tile_and_fuse_gfx1201.mlir.test|iree/compiler/Codegen/LLVMGPU/test/ROCDL/pipeline_full_smoketests.mlir.test' \
    --parallel 8 \
    --output-on-failure
```

IREE HIP/ROCm e2e patch-required inventory:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  RJ_CONSAN_FLAVOR=supercollider \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_DELAY_MODE=sleep \
  RJ_CONSAN_DELAY=1 \
  RJ_CONSAN_MAX_PATCHES=4 \
  RJ_CONSAN_REQUIRE_PATCH=1 \
  ctest --test-dir "$IREE_BUILD_DIR" \
    -R '^iree/tests/e2e/(encoding|linalg|math|matmul|rocm_specific|stablehlo_ops)/.*(rocm_hip|rocm-rocm)' \
    --parallel 8 \
    --output-on-failure
```

IREE TileAndFuse inline-shadow `hw_id` owner-source smoke:

```sh
env \
  HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  RJ_CONSAN_FLAVOR=moi \
  RJ_CONSAN_MOI_ENGINE=inline_shadow \
  RJ_CONSAN_MOI_AUTO_REPORT_BUFFER_SIZE=262144 \
  RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
  RJ_CONSAN_REQUIRE_PATCH=1 \
  RJ_CONSAN_TMP_VGPR=240 \
  RJ_CONSAN_MOI_INIT_OWNER_EPOCH=1 \
  RJ_CONSAN_MOI_OWNER_SOURCE=hw_id \
  RJ_CONSAN_MOI_OWNER_SGPR=100 \
  RJ_CONSAN_MOI_OWNER_VGPR=250 \
  RJ_CONSAN_MOI_EPOCH_VGPR=251 \
  RJ_CONSAN_MAX_PATCHES=1 \
  ctest --test-dir "$IREE_BUILD_DIR" \
    -R 'iree/tests/e2e/matmul/e2e_matmul_rocm_.*rdna4_tileandfusewmma.*rocm_hip' \
    --parallel 8 \
    --output-on-failure
```

Keep GPU test fanout near 8.

## Limitations

- Flat/generic LDS provenance is conservative. The hook can identify likely
  group/LDS helper-function flat sites and destructively trap them. The
  non-destructive check/trap path can patch eligible sites through padding or
  local NOP caves, bounded by `RJ_CONSAN_MAX_PATCHES`, but it still does not
  prove that every decoded flat access is an LDS access.
- SuperCollider does not have shadow memory or structured non-trapping report
  records yet. Its default signal is `s_trap`; `RJ_CONSAN_REPORT_BUFFER`
  provides only a one-word sticky marker. MOI has a separate report ABI with
  dynamic access records, experimental barrier records, host-side replay, and a
  narrow `inline_shadow` exact-shadow publication path. Inline shadow has a
  first different-owner diagnostic with read/read suppression, but not full
  GPU-side race diagnostics yet.
- Same-value races and unlucky schedules can be missed.
- Appended trampoline/code-growth patching remains bring-up-only. Proof NOP
  mode is gated to candidate-bearing non-ROCclr code objects; use the padded or
  local-cave check/trap paths for MVP race checks.
- Scratch selection is liveness-based and fail-closed. The patcher can grow the
  native-DS kernel descriptor's VGPR allocation as a fallback, but it does not
  spill arbitrary live registers; a site with no free scratch VGPR/SGPR run is
  skipped.
- Fence-heavy, global-memory, async-copy, broad atomic coverage, and complex
  barrier-epoch cases are unsupported and should be skipped or rejected with
  diagnostics.
