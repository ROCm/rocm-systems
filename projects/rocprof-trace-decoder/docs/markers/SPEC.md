# SQTT Instrument Pass -- Technical Specification

## Overview

SQTTInstrumentPass is an LLVM pass plugin for AMDGPU that injects
`s_ttracedata` / `s_ttracedata_imm` markers into shader code at compile time.
These markers appear in SQTT (Shader Queue Thread Trace) hardware trace buffers,
enabling correlation of trace timestamps with source-level events such as
function entry/exit, barrier synchronization, and user-defined annotations.

The system consists of:

- **`libsqttinstrumentpass.so`** -- LLVM pass plugin loaded via `-fpass-plugin=`
- **`markers.hpp`** -- Device-side header for user markers
- **`sqtt_flamegraph.py`** -- Post-processing tool: reads SQTT traces and
  `.sqtt_funcmap` sections to generate flamegraphs
- **`sqtt_perfetto.py`** -- Post-processing tool: exports SQTT traces as
  Chrome JSON for viewing in the Perfetto UI (per-wave timelines)
- **`sqtt_decode_funcmap.py`** -- Post-build tool to extract the ID-to-name map

## Building

```bash
cmake -B build
cmake --build build
```

Requires ROCm LLVM (tested with ROCm 7.x). The plugin links no LLVM libraries;
it gets symbols from the host compiler process.

## Usage

```bash
# Minimal (barriers + user markers only)
SQTT_INSTRUMENT_BARRIERS=1 \
hipcc -DSQTT_ENABLED=1 -fpass-plugin=build/lib/libsqttinstrumentpass.so \
      -I include/rocprof_trace_decoder/cxx/ my_kernel.hip -o my_kernel

# Full (function entry/exit with threshold, barriers, scope filtering)
SQTT_INSTRUMENT_FUNCTIONS=10 SQTT_INSTRUMENT_BARRIERS=1 \
SQTT_SCOPE_CU=0x3 \
hipcc -DSQTT_ENABLED=1 -fpass-plugin=build/lib/libsqttinstrumentpass.so \
      -I include/rocprof_trace_decoder/cxx/ my_kernel.hip -o my_kernel
```

All configuration is read from environment variables at compile time and baked
into the generated IR.

---

## Marker Encoding

Every marker is a 32-bit value written via `s_ttracedata` (m0-based, all
architectures) or an 8-bit immediate via `s_ttracedata_imm` (gfx10+, faster,
no m0 setup).

### Bit layout

```
  31                    2     1       0
 +------------------------+-------+-------+
 |       ID (30 bits)     | enter | exit  |
 |                        | scope | prev  |
 +------------------------+-------+-------+
```

| Bit(s) | Name       | Meaning                                           |
|--------|------------|---------------------------------------------------|
| 0      | exit_prev  | Pop the top scope from the stack                   |
| 1      | enter      | Push scope `id` onto the stack                     |
| 31:2   | id         | Marker identifier (30 bits)                        |

The decoder determines whether an ID is a function, user marker, barrier, or
memory event by looking it up in the funcmap — no type bits in the encoding.

### Encoding formula

```
encoded = (id << 2) | flags
```

Where `flags` is the OR of applicable bits: `EXIT_PREV=1`, `ENTER=2`.

### Instruction selection

- If `encoded <= 0xFF`, the target supports it, and shader-clock packing is
  inactive: emit `s_ttracedata_imm` (8-bit immediate, no m0 setup, ~1 cycle faster)
- Otherwise: emit `s_ttracedata` through M0. The pass explicitly emits
  `s_mov_b32 m0, <value>` + `s_nop 0` + `s_ttracedata` on every target.

With the 2-bit encoding, IDs 1-63 fit in `s_ttracedata_imm` (previously
limited to IDs 1-15 with 4-bit flags).

SQTT hardware only captures the low 8 bits from `s_ttracedata_imm`, so the
8-bit limit is a hardware constraint, not an ISA one.

### GFX12 Shader Clock Packing

For the supported gfx12 targets, marker headers include a truncated shader clock.
gfx1250 uses `s_get_shader_cycles_u64`; gfx1200 and gfx1201 use
`s_getreg_b32 hwreg(HW_REG_SHADER_CYCLES_LO, shift, bits)`. With a 12-bit
setting at the default shift, the window is source bits 4-15
(`SQTT_SHADER_CLOCK_SHIFT=4`, `SQTT_SHADER_CLOCK_BITS=12`),
stored in marker bits 31-20, leaving 18 marker ID bits plus the two low flags:

```python
clock_bits = 12
exit_prev  = val & 1
enter      = (val >> 1) & 1
id         = (val >> 2) & ((1 << (30 - clock_bits)) - 1)
clock      = val >> (32 - clock_bits)
```

Only headers created by the pass use this packed form. Numeric point and enter
markers retain their legacy values so arbitrary user shaderdata is not
reinterpreted as a packed header. Bare exits use the packed form too: their
value is intrinsically indistinguishable from a pass-generated exit.
`att_tool.py` restores packed headers to the legacy form by default;
`--no-decode-markers` preserves the raw value.

The clock field is the target-specific shader-clock sample, not the realtime
clock associated with `s_sendmsg`. The funcmap records the active layout with
`M:shader_clock_bits=N;shader_clock_shift=S`. When `SQTT_SHADER_CLOCK_BITS=0`,
gfx12 uses the no-clock full-ID layout and may use `s_ttracedata_imm`.
At compile time, an omitted `SQTT_SHADER_CLOCK_SHIFT` uses its default value of
4; an `M:` row that omits `shader_clock_shift` is malformed and uses the
no-clock full-ID layout.

Any marker with `R:extra_payload_count > 0`, including address traces and
named `sqtt_marker_data()` records, requires `SQTT_SHADER_CLOCK_BITS=0`.
Clock packing is disabled by default; an explicit nonzero setting is an error.

The sampled window lets a decoder compensate for the variable delay between
the trace instruction issuing and its shaderdata record appearing in SQTT.
Shaderdata and shader-clock timestamps have the same rate but an unknown fixed
phase per physical SIMD. Do not compare them directly. For each **marker
header** (never an `R:`-declared raw payload), choose any header `r` in the
same physical SIMD/layout only as a coordinate origin and calculate:

```text
window_mask = (1ULL << (clock_bits + shader_clock_shift)) - 1
half_window = (window_mask + 1) / 2
bucket_size = 1ULL << shader_clock_shift
sampled_window = packed_clock_field << shader_clock_shift
relative = (((record_time - r.record_time) -
             (sampled_window - r.sampled_window) + half_window) & window_mask) - half_window
```

Within each `(shader engine, CU, SIMD, clock_bits, shader_clock_shift)` domain,
the implementation assumes the relative delta stays within the signed sampled
window. Collect the complete domain before selecting `minimum_relative`, then
shift every header earlier by
`max(0, relative - minimum_relative - (bucket_size - 1))`. The coordinate
origin is not a chronological anchor: a later lower value corrects earlier
headers retroactively.

`att_tool.py` applies this correction while writing JSON by default;
`--no-decode-markers` preserves the packed values and arrival timestamps. It
rewrites each corrected header to the legacy `(id << 2) | flags` form. For an
externally produced packed payload block, JSON output may normalize a
recognized header but does not apply timestamp correction.

### Semantic rules

| enter | exit_prev | id       | Meaning                                  |
|-------|-----------|----------|------------------------------------------|
| 1     | 0         | any      | Enter scope `id` (push)                  |
| 0     | 1         | 0        | Exit (pop top scope)                     |
| 1     | 1         | any      | Exit previous scope, enter scope `id`    |
| 0     | 0         | any      | Point marker (no scope change)           |
| any   | any       | 0        | No-op (reserved), except exit (0x1)      |

All exit markers use `exit_prev` semantics (value `0x01`). Enter/exit markers
are always at function scope boundaries and are properly nested, so exiting
always pops the top of the stack.

### Exit+enter fusion

When the pass detects adjacent `sqtt_marker_exit("A")` followed by
`sqtt_marker_enter("B")` in the same basic block, it fuses them into a
single `s_ttracedata` with the enter ID and `exit_prev=true`:
`encoded = (B_id << 2) | 0x3`. This halves the marker overhead for scope
transitions.

### Decoding

```python
exit_prev = (val >> 0) & 1
enter     = (val >> 1) & 1
id        = val >> 2  # or masked to 30 - shader_clock_bits when M: is present
```

The marker type (function, user scope, point event) is resolved from the
funcmap, not from encoding bits.

### C++ decoder marker encoding

`<rocprof_trace_decoder/cxx/code_printing.hpp>` exposes the parsed funcmap and
its `MarkerEncoding`:

```cpp
const auto& encoding = codeobjs.getFuncmap(codeobj_id).marker_encoding;
const auto marker = codeobjs.decodeMarkerValue(codeobj_id, raw_shaderdata_value);

uint32_t id = marker.id;
uint32_t marker_id_mask = encoding.marker_id_mask();
uint32_t packed_clock_mask = encoding.packed_shader_clock_mask();
uint32_t source_clock_mask = encoding.shader_clock_source_mask();

// The sampled field is right-aligned.
uint32_t sampled_lo = 0;
if (encoding.has_shader_clock() && encoding.is_valid())
    sampled_lo = encoding.decode_shader_clock(raw_shaderdata_value) << encoding.shader_clock_shift;
```

`marker_id_mask()` and `packed_shader_clock_mask()` apply to the raw 32-bit
shaderdata value. `shader_clock_source_mask()` describes the matching source
low-word window for timestamp comparison (`HW_REG_SHADER_CYCLES_LO` on gfx1200/1201).
`decode_shader_clock()` is meaningful only after the record has been identified
as a packed header.

`decode_marker_value(raw, funcmap)` (and the `CodeobjMap::decodeMarkerValue()`
wrapper) uses the packed layout for IDs known to the funcmap and for bare
exits; unregistered numeric point/enter values otherwise remain legacy. A
numeric value whose low packed ID collides with a registered ID is inherently
ambiguous from the 32-bit record alone. Call
`decode_marker_value(raw, encoding)` directly when the record is known to be a
packed header.

With no `M:` row, `MarkerEncoding` uses the no-clock full-ID layout:
`marker_id_mask()` is `0xFFFFFFFC` and the two shader-clock masks and decoded
sampled field are zero.

---

## ID Allocation

All IDs are allocated from a single counter (`NextEventID`, starting at 1).
There are no reserved ID ranges — IDs are assigned dynamically at compile
time based on which features are enabled.

### Allocation order

When early function instrumentation runs, its function and already-resolved
named-marker IDs are compacted together by emission count (ties by original ID)
to enable `s_ttracedata_imm` coverage. System event IDs follow:

1. Early function and named marker IDs (compacted together)
2. Barrier IDs (if `SQTT_INSTRUMENT_BARRIERS=1`)
3. Memory op IDs (if `SQTT_INSTRUMENT_MEMORY` set)
4. Late-resolved named markers and address trace IDs, as encountered during late instrumentation

### Funcmap format

The `.sqtt_funcmap` ELF section uses type prefixes to distinguish marker types:

```
F:id:name[@source_loc]  — function (enter/exit scope marker)
K:name[@source_loc]     — kernel (for vaddr lookup, not instrumented)
U:id:name               — user scope marker (enter/exit)
P:id:name               — point marker (barrier, memory op, user point)
P:id:name@source_loc    — point marker with source location (addr trace ops)
W:N                     — wave size (32 or 64), present when a qualifying address operation is traced
R:id:extra_payload_count=N
                        — optional record metadata; number of following
                          s_ttracedata records consumed by this marker header
M:shader_clock_bits=N;shader_clock_shift=S
                        — marker headers reserve high bits for gfx12 shader clock
```

`source_loc`, when present, follows rocprofiler-sdk's inline-chain format:
`<innermost_file>:<line> -> <next_outer_file>:<line> -> ...`. For `F:` and
`K:` entries the chain has one element (the function/kernel definition
site). For `P:` address-trace entries the chain reflects how the memory
op got to its eventual call site through any inlining.

The decoder uses the funcmap type to determine behavior:
- `F:` and `U:` markers use enter/exit scope semantics
- `P:` markers are point events (1-cycle sample, no push/pop)
- `P:` markers with `addr_trace_` prefix trigger address block parsing
- `K:` entries are not instrumented; they provide vaddr lookup for kernels
- `W:` entries communicate the wave size from the compiler to the decoder
- `R:` entries attach optional metadata to a marker ID. Missing `R:` rows
  imply `extra_payload_count=0`, which is the behavior for function markers,
  user markers, barriers, and normal memory point markers.
- `M:` entries communicate non-legacy marker encoding metadata. Missing `M:`
  rows imply the original `id = value >> 2` decoder behavior.
- User markers (`U:`) and the deduplicated point markers for barriers /
  vmem are intentionally source-loc-less: their IDs are shared across all
  call sites so any single source loc would be misleading.

### Barrier marker placement

Barrier IDs are allocated dynamically. Placement rules:

| Barrier pattern                | Marker placement                       |
|--------------------------------|----------------------------------------|
| `s_barrier` (standalone)      | Point marker inserted **before**       |
| `s_barrier_signal` (unpaired) | Point marker inserted **after**        |
| `s_barrier_wait` (unpaired)   | Point marker inserted **before**       |
| `signal` + `wait` (consecutive, same BB) | Single point marker **between** them |

### Memory operation markers

Memory ops (global, buffer, flat, atomics — not LDS or private/scratch) are
instrumented with point markers. Atomics are classified as stores. Configured
via `SQTT_INSTRUMENT_MEMORY=N:M`:
- **N** = number of memory ops per marker (chunk size)
- **M** = max instruction gap between ops in the same sequence

Two IDs are allocated: `vmem_load` and `vmem_store`. Loads and stores are
never mixed in the same group — a load/store transition always breaks the
sequence, so each marker accurately reflects the operation type.

Example with `N=2, M=5` and sequence `load load store load load load`:
- Loads 1-2 → `vmem_load` (chunk of 2)
- Store 3 → `vmem_store` (type change breaks group, remainder gets marker)
- Loads 4-5 → `vmem_load` (chunk of 2)
- Load 6 → `vmem_load` (remainder gets marker)

### Function entry/exit markers

For a function assigned id=N:

- **Entry**: `encoded = (N << 2) | ENTER = (N << 2) | 2`
- **Exit**: `encoded = 0x1` (pure exit_prev; `s_ttracedata_imm` on RDNA
  when shader-clock packing is inactive)

---

## Configuration (Environment Variables)

All variables are read at **compile time** by the pass plugin.

### Feature toggles

| Variable                   | Default | Description                                    |
|----------------------------|---------|------------------------------------------------|
| `SQTT_INSTRUMENT_BARRIERS` | `0`     | Instrument barrier intrinsics with markers     |
| `SQTT_INSTRUMENT_FUNCTIONS`| `0`     | Function entry/exit threshold (0=disabled)     |
| `SQTT_INSTRUMENT_MEMORY`   | off     | Memory op markers. Format: `N:M` (N=ops per marker, M=max gap) |
| `SQTT_MEM_BARRIER`         | `fence` | Reordering boundary around markers (`none`/`asm`/`fence` or `0`/`1`/`2`) |
| `SQTT_SHADER_CLOCK_BITS`   | `0`     | Gfx12 shader clock bits. Set a nonzero value to enable packing; it is invalid with any payload-bearing marker. |
| `SQTT_SHADER_CLOCK_SHIFT`  | `4`     | Source bit offset in the shader-clock low word for packing |

### Marker reorder boundary (`SQTT_MEM_BARRIER`)

Selects the strength of the reordering boundary planted before AND after every
emitted `s_ttracedata` / `s_ttracedata_imm`. Default is `fence` — chosen for
**marker accuracy**, not minimum overhead. Power users running tight kernels
can opt down.

| Value | Synonyms | Boundary emitted | Machine cost | Drift protection |
|-------|----------|------------------|--------------|------------------|
| `fence` | `2`, `on`, `hw` | `fence syncscope("workgroup") acq_rel, !mmra !{!"amdgpu-synchronize-as", !"local"}` | target-dependent local/LDS fence code; no marker-generated global cache invalidation | Strongest — survives post-RA sinking, block placement, and machine scheduling |
| `asm`   | `1`, `compiler`, `clobber` | empty inline asm `~{memory}` | 0 (IR/MIR-only constraint) | Stops IR/MIR memory reorderings, does **not** constrain post-RA sinking |
| `none`  | `0`, `off` | nothing | 0 | Only the cheap `sched_barrier(0)` hints survive — markers may drift in LDS-pipelined regions |

**Why workgroup-scope `acq_rel`:** the marker fence is a compiler-visible
boundary, not a data-sharing operation. It exists so optimizers and backend
schedulers cannot move `s_ttracedata` away from the memory operation or barrier
it is meant to timestamp. Symmetric `acq_rel` on both sides is used (rather
than `release`-then-`acquire`) so the marker is a *true* boundary regardless of
which direction the optimizer would otherwise sink instructions.

**Why `amdgpu-synchronize-as:local`:** SQTT markers themselves do not access
global memory. The AMDGPU MMRA tag lets codegen avoid treating this marker-only
fence as a global-memory synchronization point, preventing marker-generated
`global_inv` while keeping the IR fence in place for marker placement.

The `none` mode is provided for kernels where SQTT timing fidelity is less
important than peak throughput (e.g. measuring only call-graph shape, not
event timing).

### Function threshold

`SQTT_INSTRUMENT_FUNCTIONS` controls the minimum function size (in IR
instructions) for a device function to receive entry/exit markers:

- `SQTT_INSTRUMENT_FUNCTIONS=0` -- disabled (no function instrumentation)
- `SQTT_INSTRUMENT_FUNCTIONS=10` -- instrument functions with >10 instructions
- `SQTT_INSTRUMENT_FUNCTIONS=cost:50` -- use weighted cost model (memory ops
  count more, matrix ops count 16x, etc.)

### Scope filtering (bitmasks)

| Variable         | Default        | Bits  | Description                   |
|------------------|----------------|-------|-------------------------------|
| `SQTT_SCOPE_WAVE`| `0xFFFFFFFF`   | 32    | Which wave IDs emit markers   |
| `SQTT_SCOPE_SIMD`| `0xF`          | 4     | Which SIMD units emit markers |
| `SQTT_SCOPE_CU`  | `0x3`          | varies| Which CU/WGP IDs emit markers |
| `SQTT_SCOPE_WG`  | `0xFFFFFFFF`   | 32    | Which workgroup IDs emit      |

Set to `-1` or `0xFFFFFFFF` to disable filtering for that dimension. When any
mask is not all-ones, the pass inserts a runtime scope check (`s_getreg` +
bit-test) that conditionally skips the marker emission. The check uses
architecture-appropriate HW_ID register encodings (HW_ID for gfx9, HW_ID1/HW_ID2
for RDNA). The scope check result is computed once per function at the entry
block and cached.

When scope filtering is active, `sched_barrier(0)` is **not** emitted around
markers. The conditional branch provides equivalent ordering, and `sched_barrier`
inside tiny conditional basic blocks creates `s_nop` stalls with no useful work
to schedule.

**Exception — sync-adjacent markers.** When the marker insertion point sits
immediately before a workgroup sync (`s_barrier`, `s_barrier_signal/wait`,
`__syncthreads`, IR `fence`), the scope-check CFG split would otherwise let
late codegen schedule unrelated instructions between the trace block and the
sync, drifting the marker away from the event it is supposed to time. In that
case the pass plants a single `sched_barrier(0)` at the head of the `sqtt.skip`
basic block (right before the sync), pinning the sync as the first real
instruction. This applies to:

- Barrier markers whose insertion point is immediately before the sync
  (wait/full and paired signal+wait); an unpaired signal is pinned only when
  it is immediately followed by a sync
- User markers (`sqtt_marker_enter/exit/point/data`) whose immediate next
  non-sentinel instruction is a workgroup sync (e.g. wrapping a
  `__syncthreads()` call)

Function entry/exit markers, point markers, and ordinary user markers not
adjacent to a sync are unaffected.

---

## Two-Phase Pass Architecture

Function instrumentation uses a two-phase design to correctly handle inlining:

### Phase 1: Early (before inliner)

- Registered at `PipelineEarlySimplificationEP` (skipped at `-O0`)
- Force-inlines wrappers around named marker sentinels
  (`__sqtt_named_marker_enter`, `__sqtt_named_marker_exit`,
  `__sqtt_named_marker_point`, `__sqtt_named_marker_data`), then resolves
  sentinel calls to bare
  `s_ttracedata` intrinsics. Adjacent exit+enter pairs in the same basic block
  are fused into a single `s_ttracedata` with `exit_prev=true`.
- When `SQTT_INSTRUMENT_FUNCTIONS` is nonzero, instruments eligible non-kernel
  device functions with bare `s_ttracedata` markers (no scope checks or
  barriers) at entry and all return points
- Tags each function with `!sqtt.func.id` metadata
- Stores the typed marker ledger (function/user kind, ID, name, source
  location, pre-opt size, and payload count) in `!sqtt.markers.early`
  module metadata

When functions are inlined by LLVM, the markers travel with the code into the
caller.

### Phase 2: Late (after inliner)

- Registered at `OptimizerLastEP` (or `PipelineStartEP` at `-O0`)
- Reads the `!sqtt.markers.early` ledger
- For each function that still exists: measures post-optimization size,
  compares to threshold. Below threshold: removes markers from the **entire
  module** (including inlined copies in callers)
- For fully inlined functions (no longer in module): uses stored pre-opt size
  for threshold comparison, removes markers module-wide if below threshold
- Reassigns compact contiguous IDs starting from 1 to enable
  `s_ttracedata_imm` for small ID sets (IDs 1-63 fit in 8-bit immediate)
- Converts function exit markers to `exit_prev` (value `0x01`)
- Wraps all surviving `s_ttracedata` / `s_ttracedata_imm` calls with scope
  checks (if scope filtering is active)
- Adds `sched_barrier(0)` and optional memory barriers around surviving
  early-pass markers (skipped when scope check provides equivalent ordering)
- Processes any remaining named marker sentinels, including data markers
  (with exit+enter fusion)
- Allocates barrier and memory op IDs from the unified counter
- Instruments barrier intrinsics and memory operations
- Emits the `.sqtt_funcmap` ELF section with type prefixes (`F:`, `U:`, `P:`, `K:`)

### `-O0` behavior

At `-O0`, only the Late phase runs. It instruments functions directly (no
early/late split needed since there is no inlining).

---

## User Marker API (`markers.hpp`)

Include with `#include "markers.hpp"` and compile with `-DSQTT_ENABLED=1`.
When `SQTT_ENABLED` is 0 or undefined, all calls compile to nothing.

### Functions

```cpp
// --- Enter (scope open) ---

// Named enter marker (requires pass plugin)
// The pass replaces the sentinel call with s_ttracedata + scope checks.
// Duplicate strings get the same ID automatically.
__device__ void sqtt_marker_enter(const char *name);

// Numeric enter marker (no pass plugin required)
// Emits sched_barrier + s_ttracedata((data << 2) | ENTER_FLAG) + sched_barrier.
__device__ void sqtt_marker_enter(uint32_t data);

// --- Exit (scope close) ---

// Named exit marker (requires pass plugin)
__device__ void sqtt_marker_exit(const char *name);

// Numeric exit marker (no pass plugin required)
// Emits s_ttracedata with EXIT_PREV (value 1). The data parameter is
// kept for API compatibility but ignored -- exit always pops the top.
__device__ void sqtt_marker_exit(uint32_t data);

// --- Point (no scope change) ---

// Named point marker (requires pass plugin)
// Recorded in trace but does not push/pop the stack.
__device__ void sqtt_marker_point(const char *name);

// Numeric point marker
// Emits s_ttracedata with (data << 2) (no flags).
__device__ void sqtt_marker_point(uint32_t data);

// Named point marker followed by one raw 32-bit payload record.
// The funcmap stores P:id:name plus R:id:extra_payload_count=1.
__device__ void sqtt_marker_data(const char *name, uint32_t data);
```

All numeric markers wrap the s_ttracedata with `sched_barrier(0)` on each
side. The pass plugin separately plants the `SQTT_MEM_BARRIER`-controlled
reorder boundary (`none` / `asm` / `fence`) around every s_ttracedata it
sees in IR, so there is no per-call ordering knob in the header.

User markers have enter/exit semantics for scoped regions.
`sqtt_marker_enter()` opens a scope (sets `ENTER` flag),
`sqtt_marker_exit()` pops the top scope. Matching enter/exit pairs
produce nested frames in flamegraphs.

Point markers (`sqtt_marker_point()` and `sqtt_marker_data()`) record an event
without pushing or popping the call stack. **They are dropped by the flamegraph
tool** -- a point event has no meaningful cycle attribution (start/end are the
same instant), so attributing synthetic weight to it would distort the time
visualization. Inspect point events via `sqtt_decode_funcmap.py` and the raw
shaderdata records. `sqtt_marker_data()` emits the point marker header followed
by one raw payload record; decoders skip that payload as part of the header
using the marker's `R:id:extra_payload_count=1` metadata.

### Named marker processing

1. User writes `sqtt_marker_enter("my_label")`, `sqtt_marker_exit("my_label")`,
   `sqtt_marker_point("my_label")`, or `sqtt_marker_data("my_label", value)`
2. The compiler emits calls to `__sqtt_named_marker_enter(const char*)` and
   the matching named marker sentinel respectively
3. The Early pass force-inlines the wrappers, resolves the string literal,
   assigns a unique ID (starting at 1, with deduplication within each
   scope/point/data marker kind), and replaces each call with a bare
   `s_ttracedata` intrinsic
4. Adjacent exit+enter sentinel calls in the same basic block are fused into
   a single `s_ttracedata` with `exit_prev=true`, halving marker overhead
5. The ID-to-name mapping is stored in `.sqtt_funcmap` as `U:id:name`
   (scope markers) or `P:id:name` (point markers). Data markers additionally
   emit `R:id:extra_payload_count=1`.

If you forget to load the pass plugin, you get a linker error
("undefined reference to `__sqtt_named_marker_enter`") -- no silent
miscompilation.

When `SQTT_ENABLED=0` (the default), named markers compile to nothing.

---

## Extracting Data from Compiled Binaries

### The `.sqtt_funcmap` ELF section

The pass embeds a `.sqtt_funcmap` section in each AMDGPU code object. This
section contains a null-terminated string with newline-delimited entries:

```
F:ID:mangled_function_name@file.hip:42         -- auto-instrumented function (enter/exit scope)
K:mangled_kernel_name@file.hip:80              -- kernel entry point (for vaddr lookup)
U:ID:marker_name                               -- named user scope marker (enter/exit)
P:ID:marker_name                               -- point marker (barrier, memory op, user point)
P:ID:addr_trace_load@kernel.hip:59 -> hip_runtime.h:264
                                               -- address trace op with inline chain source location
R:ID:extra_payload_count=130                   -- following payload records for that header
W:64                                           -- wave size (present when a qualifying address operation is traced)
M:shader_clock_bits=12;shader_clock_shift=4    -- gfx12 marker header clock packing
```

The section is added to `llvm.used` to prevent linker stripping.

### `code.json` marker metadata

`att_tool.py` adds embedded `sqtt_funcmap` data to `code.json` using the
established six-column row shape. Two additive top-level tables carry metadata
that cannot be represented in those rows:

```json
"sqtt_funcmap_layout": [[codeobj_id, shader_clock_bits, shader_clock_shift]],
"sqtt_funcmap_payloads": [[codeobj_id, marker_id, extra_payload_count]]
```

`sqtt_funcmap_layout` contains rows only for code objects with shader-clock
packing; `sqtt_funcmap_payloads` contains nonzero payload counts. Readers that
do not know these optional tables can continue using the unchanged
`sqtt_funcmap` rows. For a packed row, an absent payload count means zero.

### Extracting the funcmap

**Method 1: `sqtt_decode_funcmap.py` (recommended)**

```bash
# Extract code object from the fat binary first
hipcc --offload-arch=gfx942 ... -o my_app
# The code object is embedded; extract it:
roc-obj-extract my_app -o my_app.co

# Decode the funcmap
python3 scripts/sqtt_decode_funcmap.py my_app.co --demangle
```

Output:
```
  Type        ID               Vaddr  Name
  ----       ---               -----  ----
kernel         -  0x0000000000001000  _Z8kernelAi  @ kernel.hip:80
  user         1                   -  load_input
  user         2                   -  compute_start
  func         3  0x0000000000001200  _Z4syncv  @ kernel.hip:42
  func         4                   -  _Z12helper_funcv  @ helper.hpp:17  (inlined)
 point         5                   -  barrier_signal
 point         6                   -  barrier_wait
 point         7                   -  barrier
 point         8                   -  vmem_load
 point         9                   -  vmem_store
```

With address tracing, each memory op has its own entry with source location.
When the op is reached through inlining, the source location is an inline
chain (innermost first, then outward call sites separated by ` -> `):
```
  wave_size: 64
 point         1                   -  addr_trace_load  @ hip_runtime.h:264 -> kernel.hip:59
 point         2                   -  addr_trace_load  @ kernel.hip:59
 point         3                   -  addr_trace_lds_store  @ kernel.hip:59
 point         4                   -  addr_trace_lds_load  @ kernel.hip:45
 point         5                   -  addr_trace_lds_store  @ kernel.hip:45
 point         6                   -  addr_trace_lds_load  @ kernel.hip:49
 point         7                   -  addr_trace_store  @ kernel.hip:62
```

The script uses `llvm-objcopy --dump-section` (preferred) or `llvm-readelf -p`
as fallback. It also resolves ELF virtual addresses via `llvm-nm`.

**Method 2: Manual extraction**

```bash
# Dump the raw section
llvm-objcopy --dump-section=.sqtt_funcmap=/dev/stdout my_app.co

# Or view with readelf
llvm-readelf -p .sqtt_funcmap my_app.co
```

### Generating flamegraphs from SQTT traces

**`sqtt_flamegraph.py`** reads rocprofv3 `--att` output (shaderdata JSON,
occupancy JSON, code objects) and generates interactive SVG flamegraphs:

```bash
# Capture a trace
rocprofv3 --att -d trace_output -- ./my_app

# Generate flamegraph (folded stacks to stdout, SVG to disk)
python3 scripts/sqtt_flamegraph.py trace_output/ --demangle --show
```

Each `ui_*` directory under the trace output is treated as an independent
time domain (SQTT time resets between collections). Folded stack counts are
merged after per-directory processing.

The flamegraph shows:
- User marker scopes (`sqtt_marker_enter`/`sqtt_marker_exit` pairs)
- Auto-instrumented function entry/exit

Point markers (barriers, memory ops, user points) are **not** rendered --
they have no meaningful cycle attribution. Use `sqtt_decode_funcmap.py`,
the raw shaderdata records, or the Perfetto exporter (next section) to
inspect them.

### Exporting traces to Perfetto

**`sqtt_perfetto.py`** converts SQTT shaderdata into the Chrome JSON
Trace Event Format, viewable at <https://ui.perfetto.dev>:

```bash
# One *.perfetto.json per ui_* (sibling-of-source by default)
python3 scripts/sqtt_perfetto.py trace_output/ --demangle

# Collect outputs into one directory; convert cycles to ns
python3 scripts/sqtt_perfetto.py trace_output/ -o /tmp/perfetto_out \
    --clock-rate-ghz 2.1
```

Each `ui_*` directory becomes a separate file -- each is its own SQTT
time domain and we don't fake offsets across collections. Inside a file:

- `pid` = `dispatch_id` -- one Perfetto "process" per dispatch
- `tid` = `(cu << 20) | (simd << 16) | (wave_id << 8) | instance` --
  one thread per wave instance, named `CU{cu}/SIMD{simd}/W{wid}#{inst}`
- Function entry/exit pairs render as duration slices (`B`/`E`)
- Point markers render as instant events (`i`) with thread scope
- Categories: `sqtt.function`, `sqtt.user`, `sqtt.point`

Without `--clock-rate-ghz`, timestamps are SQTT cycles labelled as ns --
proportions and orderings are correct but absolute values are nominal.

### Decoding markers from SQTT trace data

When reading an SQTT trace buffer, each `s_ttracedata` / `s_ttracedata_imm`
token appears as a user data marker in the trace. To decode:

```python
def decode_marker(val: int) -> dict:
    """Decode a 32-bit SQTT marker value."""
    return {
        "exit_prev": bool(val & 0x1),
        "enter":     bool(val & 0x2),
        "id":        val >> 2,
        "raw":       val,
    }
```

The marker type (function, user, point) is determined by looking up the ID
in the funcmap, not from encoding bits.

### Complete decode workflow

1. **Compile** with the pass plugin and desired env vars
2. **Capture** an SQTT trace (via `rocprofv3 --att`)
3. **Generate** flamegraph: `python3 scripts/sqtt_flamegraph.py trace_dir/ --demangle`
4. Or **export to Perfetto**: `python3 scripts/sqtt_perfetto.py trace_dir/ --demangle`
5. Or: **Extract** the code object, **read** `.sqtt_funcmap`, and **decode**
   trace tokens manually using the bit layout above

---

## Instruction Overhead

Each marker emits the following instruction sequence (worst case):

| Component             | Instructions | Condition                  |
|-----------------------|-------------|----------------------------|
| `s_sched_barrier 0`   | 1           | No scope check active      |
| reorder boundary      | 0-1         | depends on `SQTT_MEM_BARRIER` (see below) |
| `s_ttracedata`        | 1 (+ m0 setup) | id > 63, gfx9, or clock packing active |
| `s_ttracedata_imm`    | 1           | id <= 63, gfx10+, and clock packing inactive |
| reorder boundary      | 0-1         | depends on `SQTT_MEM_BARRIER` (see below) |
| `s_sched_barrier 0`   | 1           | No scope check active      |

With scope filtering, a branch-over block is added (4-8 scalar instructions for
the `s_getreg` + bit-test + conditional branch, computed once per function),
and the marker is only emitted when the scope check passes. When scope
filtering is active, `sched_barrier` is omitted from the marker sequence
because the conditional branch provides equivalent scheduling constraints.

With `SQTT_MEM_BARRIER=none` on gfx10+ with a small ID, the overhead is a
single `s_ttracedata_imm` instruction (plus `sched_barrier` if no scope check).

A *pure* exit marker is `s_ttracedata_imm 1` on RDNA when clock packing is
inactive (no m0 setup needed), regardless of function ID. A *fused* exit+enter (see below) encodes
as `(id << 2) | 0x3`, so it only fits in `s_ttracedata_imm` when the entered
ID is `<= 63`; larger entry IDs fall back to `s_ttracedata` with the usual
m0 setup.

### Exit+enter fusion

Adjacent `sqtt_marker_exit()` + `sqtt_marker_enter()` calls in the same basic
block are fused into a single `s_ttracedata` with `exit_prev=true`, halving
the per-transition overhead.

---

## Cost Model

The weighted cost model (`SQTT_INSTRUMENT_FUNCTIONS=cost:N`) assigns different
weights to different IR instruction types:

| Instruction type              | Weight |
|-------------------------------|--------|
| PHI, alloca, debug, unreachable| 0     |
| `llvm.lifetime.*`, `llvm.dbg.*`| 0    |
| MFMA/WMMA matrix ops          | 16    |
| LDS intrinsics (`llvm.amdgcn.ds.*`)| 4 |
| LDS load/store (addr space 3) | 4     |
| Global/flat load/store        | 10    |
| Everything else               | 1     |

The plain instruction count mode (`SQTT_INSTRUMENT_FUNCTIONS=N`) counts all
non-trivial instructions equally (weight 1), skipping PHI, alloca, debug, and
unreachable.

---

## Architecture Support

| Architecture | `s_ttracedata` | `s_ttracedata_imm` | HW_ID register |
|--------------|---------------|--------------------:|----------------|
| gfx9 (CDNA)  | Yes           | No                 | HW_ID (reg 4)  |
| gfx10 (RDNA1) | Yes          | Yes                | HW_ID1 (23), HW_ID2 (24) |
| gfx11 (RDNA3) | Yes          | Yes                | HW_ID1 (23), HW_ID2 (24) |
| gfx12 (RDNA4) | Yes          | Yes                | HW_ID1 (23), HW_ID2 (24) |

The pass auto-detects the target from the function's `target-cpu` attribute.
Non-AMDGPU modules are skipped entirely.

---

## Address Tracing

When `SQTT_TRACE_ADDRESSES` is set, the pass emits per-lane virtual addresses
for every qualifying memory operation. Each memory op gets its own unique
marker ID with source location information, enabling downstream analysis such
as cache line utilization, stride detection, and memory coalescing analysis.

### Configuration

| Variable | Format | Default | Description |
|---|---|---|---|
| `SQTT_TRACE_ADDRESSES` | `memory`, `lds`, or `memory,lds` | off | Trace per-lane addresses. `memory` = flat/global (AS=0/1), buffer, and matching atomics. `lds` = LDS (AS=3), LDS atomics, ds_permute/ds_bpermute. |

`SQTT_TRACE_ADDRESSES` and `SQTT_INSTRUMENT_MEMORY` are mutually exclusive —
setting both produces an error. Address tracing already emits a header marker
for every memory op, so `SQTT_INSTRUMENT_MEMORY` is redundant.

### Address width by type

| Category | Address space | Width | Tokens/lane | Fixed tokens | Total data tokens (wave32) |
|---|---|---|---|---|---|
| `memory` (global/flat) | 0, 1 | 64-bit | 2 (lo+hi) | 0 | 64 |
| `atomic` (global/flat) | 0, 1 | 64-bit | 2 (lo+hi) | 0 | 64 |
| `lds` | 3 | 32-bit | 1 (lo) | 0 | 32 |
| `lds_atomic` | 3 | 32-bit | 1 (lo) | 0 | 32 |
| `buffer` (raw) | buffer | 32-bit | 1 (voffset) | 3 (rsrc_lo, rsrc_hi, soffset) | 35 |
| `struct_buffer` | buffer | 32-bit | 2 (voffset+vindex) | 3 (rsrc_lo, rsrc_hi, soffset) | 67 |
| `ds_permute`/`ds_bpermute` | LDS hw | 32-bit | 1 (index) | 0 | 32 |

Pointer address spaces other than 0, 1, and selected LDS AS=3 (including
private/scratch AS=5) are excluded. All categories also emit exec_lo + exec_hi
(2 tokens) and a header marker (1 token) before the data tokens shown above.

### Wave size

The wave size is derived from the ISA target and stored in the funcmap as
a `W:N` entry:

- gfx9 (CDNA): wave64 (64 lanes) → `W:64`
- gfx10+ (RDNA): wave32 (32 lanes) → `W:32`

The decoder uses this entry to determine how many lane addresses to read
per trace block. No sentinel or lane count token is needed — the decoder
trusts the thread trace data to be complete.

### Framing protocol

#### Memory / LDS / atomic operations

For load, store, and atomic operations on global/flat/LDS memory:

```
s_ttracedata  <header>      ← point marker with per-op unique ID
s_ttracedata  exec_lo       ← EXEC mask low 32 bits
s_ttracedata  exec_hi       ← EXEC mask high 32 bits (0 on wave32)
s_ttracedata  lane0_lo      ← address low 32 bits (readlane loop)
s_ttracedata  lane0_hi      ← address high 32 bits (omitted for LDS/32-bit)
...
s_ttracedata  laneN_lo      ← N = wave_size - 1
s_ttracedata  laneN_hi
```

#### Buffer intrinsics (component-based protocol)

Buffer intrinsics (`raw.buffer.*`, `struct.buffer.*`, and their `ptr` variants)
emit raw descriptor components instead of computed addresses:

```
s_ttracedata  <header>      ← point marker (addr_trace_buffer_* or addr_trace_struct_buffer_*)
s_ttracedata  exec_lo       ← EXEC mask
s_ttracedata  exec_hi
s_ttracedata  rsrc_lo       ← buffer descriptor word 0 (base address [31:0])
s_ttracedata  rsrc_hi       ← buffer descriptor word 1 (base address [47:32] + stride)
s_ttracedata  soffset       ← scalar offset (uniform SGPR)
s_ttracedata  lane0_voffset ← per-lane byte offset (readlane loop)
...
s_ttracedata  laneN_voffset
```

For struct buffer operations, an additional readlane loop emits per-lane vindex:

```
s_ttracedata  lane0_vindex  ← per-lane buffer index (readlane loop)
...
s_ttracedata  laneN_vindex
```

The decoder reconstructs addresses as: `addr = (rsrc_hi:rsrc_lo & 0xFFFFFFFFFFFF) + soffset + voffset`.

#### ds_permute / ds_bpermute

Lane shuffle intrinsics emit a per-lane index (byte offset into LDS):

```
s_ttracedata  <header>      ← point marker (addr_trace_ds_permute or addr_trace_ds_bpermute)
s_ttracedata  exec_lo       ← EXEC mask
s_ttracedata  exec_hi
s_ttracedata  lane0_index   ← per-lane byte offset / source lane * 4 (readlane loop)
...
s_ttracedata  laneN_index
```

The header marker is encoded as a point marker (no scope change) with a
unique ID from the funcmap. The decoder identifies address trace blocks by
looking up the header ID in the funcmap and checking for the `addr_trace_`
prefix.

All lanes (0 to wave_size-1) are emitted regardless of the EXEC mask.
Inactive lanes contain garbage VGPR data. The decoder reads the EXEC mask
and filters to only report active lane addresses.

### Per-op unique IDs and source locations

Each memory operation receives its own unique marker ID, allocated from the
unified ID counter. The funcmap records each ID with:

- The operation kind (see table below)
- Source location (if debug info is available): appended as `@file:line`

| Funcmap name prefix | Instruction type | Gated by |
|---|---|---|
| `addr_trace_load` | Flat/global (AS=0/1) load | `memory` |
| `addr_trace_store` | Flat/global (AS=0/1) store | `memory` |
| `addr_trace_atomic` | Flat/global (AS=0/1) atomic (AtomicRMW, CmpXchg) | `memory` |
| `addr_trace_lds_load` | LDS load | `lds` |
| `addr_trace_lds_store` | LDS store | `lds` |
| `addr_trace_lds_atomic` | LDS atomic | `lds` |
| `addr_trace_buffer_load` | Raw buffer load intrinsic | `memory` |
| `addr_trace_buffer_store` | Raw buffer store intrinsic | `memory` |
| `addr_trace_buffer_atomic` | Raw buffer atomic intrinsic | `memory` |
| `addr_trace_struct_buffer_load` | Struct buffer load intrinsic | `memory` |
| `addr_trace_struct_buffer_store` | Struct buffer store intrinsic | `memory` |
| `addr_trace_struct_buffer_atomic` | Struct buffer atomic intrinsic | `memory` |
| `addr_trace_ds_permute` | `ds_permute` lane shuffle | `lds` |
| `addr_trace_ds_bpermute` | `ds_bpermute` / `ds_bpermute_fi_b32` lane shuffle | `lds` |

Example funcmap entries:

```
W:64
P:5:addr_trace_lds_store@test_auto.cpp:45
R:5:extra_payload_count=66
P:6:addr_trace_lds_load@test_auto.cpp:49
R:6:extra_payload_count=66
P:7:addr_trace_store@test_auto.cpp:62
R:7:extra_payload_count=130
P:8:addr_trace_atomic@test_auto.cpp:70
R:8:extra_payload_count=130
P:9:addr_trace_buffer_load@test_auto.cpp:85
R:9:extra_payload_count=69
```

This allows the decoder to correlate each address trace back to the exact
source line and distinguish between multiple memory operations of the same
kind.

The `extra_payload_count` value excludes the header marker itself and counts
only subsequent records in the same `(cu, simd, wave_id)` stream. Existing
parsers that do not understand `R:` rows can ignore them; all markers without
an `R:` row have an implicit count of zero.

### IR generation

For each qualifying memory operation, the pass emits (before the op):

1. Allocate a unique ID from `NextEventID++`
2. Record the ID, kind name, and source location in the marker ledger
3. Header point marker via `emitBareTrace()` with the unique ID
4. EXEC mask via inline asm: `s_mov_b32 m0, exec_lo` / `exec_hi`, then
   `s_nop 0` and `s_ttracedata`.
5. For memory/LDS/atomics: `ptrtoint` of the pointer operand (i64 for memory, i32 for LDS)
   For buffers: extract rsrc_lo/hi, soffset as scalar values
   For permutes: extract the index operand (arg 0 of the intrinsic)
6. A readlane loop over all lanes using `llvm.amdgcn.readlane`:
   - PHI node for lane counter (0 to wavesize-1)
   - For 64-bit addresses: truncate to i32 (lo), lshr+trunc (hi), two `s_ttracedata`
   - For 32-bit addresses / buffer voffset / permute index: single `s_ttracedata`
   - For struct buffers: a second readlane loop for vindex
   - Loop metadata with `llvm.loop.unroll.disable` to prevent unrolling

The loop is wrapped in a scope-check conditional branch when scope filtering
is active.

### Buffer intrinsics

Buffer intrinsics (`llvm.amdgcn.raw.buffer.*`, `llvm.amdgcn.struct.buffer.*`,
and `raw.ptr.buffer.*` / `struct.ptr.buffer.*` variants) use a component-based
trace protocol. Instead of computing full virtual addresses inline, the pass
emits the raw buffer descriptor components (rsrc base, soffset, per-lane
voffset, and vindex for struct buffers). The decoder reconstructs addresses
in post-processing.

The pass handles operand layout differences across load, store, atomic, and
cmpswap variants. The rsrc operand may be `<4 x i32>` (legacy) or
`ptr addrspace(8)` (modern) — both are supported.
`*.buffer.load.lds` is excluded because its LDS-destination operand layout is
not this protocol.

### Atomic operations

`AtomicRMWInst` and `AtomicCmpXchgInst` are classified by address space the
same way as loads/stores (AS=3 → LDS, AS=0/1 → memory, all others excluded).
They receive distinct funcmap names (`addr_trace_atomic`, `addr_trace_lds_atomic`)
rather than being folded into load/store categories, so the decoder can
distinguish atomic access patterns.

Buffer atomics (`raw.buffer.atomic.*`, `struct.buffer.atomic.*`) use the buffer
component protocol with `addr_trace_buffer_atomic` / `addr_trace_struct_buffer_atomic`
funcmap names.

### ds_permute / ds_bpermute

The `ds_permute`, `ds_bpermute`, and `ds_bpermute_fi_b32` (gfx12+) intrinsics
are lane shuffle operations that route through LDS hardware. They are gated by
`traceLDS` and emit a per-lane readlane loop over the index operand (arg 0),
which is a byte offset into LDS (source lane ID * 4). These intrinsics are
`IntrNoMem`, so they are detected by intrinsic ID rather than by `LoadInst`/
`StoreInst` classification.

### Decoder integration

Address trace blocks are extracted from the record stream by a preprocessing
step (`preprocess_records()` in `sqtt_data.py`) before `build_stacks()` runs.

Because SQTT hardware captures `s_ttracedata` from all active waves into a
single time-ordered buffer, records from concurrent waves are interleaved.
The decoder demultiplexes records by `(cu, simd, wave_id)` before parsing
address blocks, ensuring each wave's multi-record sequence (header + exec +
addresses) is processed contiguously.

For each wave's record stream, the decoder:

1. Recognizes address trace headers by resolving the marker ID against the
   funcmap and checking for the `addr_trace_` prefix
2. Reads `exec_lo` and `exec_hi` from the next two records
3. Determines the protocol from the funcmap name prefix:
   - Memory/LDS/atomic/permute: reads `wave_size` lane tokens (2 per lane for 64-bit, 1 for 32-bit)
   - Buffer: reads 3 fixed tokens (rsrc_lo, rsrc_hi, soffset), then `wave_size` voffset tokens
     (plus `wave_size` vindex tokens for struct buffers)
4. Filters addresses by the EXEC mask, keeping only active lanes
5. For buffer ops, reconstructs addresses: `base + soffset + voffset[lane]`

The flamegraph builder never sees address trace records. The memory trace tool
(`sqtt_memory_trace.py`) consumes the extracted `AddressTrace` objects and
outputs JSON with per-lane addresses, source locations, and hierarchical
summaries for analysis.
