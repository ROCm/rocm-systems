# rocjitsu Plugins

Execution plugins that hook into rocjitsu's simulation model. Each plugin
implements the `ExecutionPlugin` interface and receives callbacks for
wavefront dispatches, memory instructions, register reads, barriers, etc.

## Plugins

| Plugin | Location | Description |
|---|---|---|
| `RaceDetectorPlugin` | `race_detector/` | Reports core register findings and checks LDS ordering. Reports violations with disassembly traces. See [race-detector.md](race-detector.md). |
| `KernelLoggingPlugin` | `logging/` | Logs kernel dispatches and detects MMA instruction usage. |
| `ThroughputPlugin` | `throughput/` | Reports per-dispatch and aggregate wave-instruction MIPS with an exclusive instruction-family breakdown. |
| `InstructionMixPlugin` | `instruction_mix/` | Records which ISA mnemonics a run executed and how often, per dispatch and in aggregate, for instruction-mix and instruction-coverage reporting. |
| `PerfsimPlugin` | `perfsim/` | Adapts gfx1250 execution observations to an external Perfsim backend implementing FFM observer APIs v8 through v13. Built only when explicitly enabled. See the [Perfsim adapter README](../lib/rocjitsu/src/rocjitsu/vm/plugins/perfsim/README.md). |

The race plugin owns LDS analysis (`race_detector/core/`) and report presentation
(`race_detector/plugin.h`). Register readiness belongs to the emulator's core
scoreboard; the plugin observes its findings rather than register accesses.

Race detection is opt-in through `plugins.race`. Without that plugin, no
workgroup race detector, per-wave race state, or LDS byte history is allocated.
Core register-wait diagnostics are separate and default to warnings. For
uninstrumented emulation, omit the plugins and set `wait_checking` to `off` to
disable those diagnostics too. A register-finding observer such as `race`
still requests core dependency tracking even when the warnings are disabled.

### Throughput Plugin

The throughput plugin counts one instruction whenever a wavefront reaches the
synchronous before-execute or asynchronous issue hook. Counts are **wave instructions**, not
active-lane operations. It reports one JSON object per completed dispatch and
one aggregate object at shutdown using the `rocjitsu.throughput.v2` JSONL
schema. Each object contains wall time, total wave instructions, MIPS, and an
exclusive breakdown into `scalar`, `vector`, `matrix`, `lds`, `global`,
`control`, and `other`; the family counts always sum to the total. Each family
reports `untimed_instructions` and `execution_timing_valid`. A family containing
untimed async work sets validity to `false` and emits null `execution_seconds`
and `execution_mips`; counts and `dispatch_mips` remain valid. Otherwise it
reports `execution_seconds` measured between this plugin's before- and
after-execute callbacks, `execution_mips` using only that family-local time, and
`dispatch_mips` using the complete dispatch time. Scheduler gaps, dispatch
setup, runtime work, and time spent executing other families are not included
in `execution_seconds`. Program terminators close their interval in the
wave-halt hook because they intentionally have no after-execute callback.
When other execution plugins are enabled, their before/after hooks can fall
inside this interval depending on registration order; run `throughput` alone
when comparing family timing. The timestamps themselves are outside the
measured interval but still add observer overhead to the run.

For a summary record, `wall_seconds` is the inclusive span from the earliest
dispatch begin to the latest dispatch end, including idle gaps between
dispatches. `dispatch_seconds_sum` is the sum of completed dispatch durations.
Dispatches that never reach the execution-end callback are omitted from both
per-dispatch output and the summary.

Memory instructions take precedence over their scalar or vector encoding. The
`lds` and `global` families describe the instruction's pre-routing pipeline tag
or mnemonic fallback: `lds` covers DS/LDS local-memory instructions, while
`global` covers global, scalar-memory, flat, buffer, typed-buffer, image,
tensor, and scratch instructions. A later shared-aperture FLAT-to-LDS remap is
therefore still reported as `global`. Their `execution_seconds` measure
synchronous instruction execution/address generation, not later routing,
deferred pipeline completion, or stalls charged to wait instructions. Matrix
includes MFMA, SMFMAC, WMMA, and SWMMAC instructions. Control covers branches,
waits, barriers, termination, no-ops, sleeps, and delays, including the LDS-pipe
`ds_nop` and the GWS barrier/semaphore instructions, which move no data.

The mnemonic fallback is not just for synthetic instructions: whole prefix
families never set the memory-op flag — RDNA4 `ds_direct_load` and `tbuffer_*`,
RDNA3.5 `lds_direct_load`, the image encodings, the CDNA5 tensor transfers, and
every scalar-memory encoding, since `smem` stopped setting the flag when the
decoded memory-issue metadata landed — and some mnemonics carry it on one
architecture but not another. Classification is by flag *and* by prefix so that
a mnemonic reports the same family whichever architecture executed it.

Where neither applies, the decoded memory-issue metadata is consulted last: it
names the completion domain, so an encoding that follows no naming convention
is still placed. The CDNA5 `cluster_load_*` set is the live case. The metadata
is checked after the prefix so that generic FLAT, which reports both LDS and
VMEM obligations, keeps the `global` bucket described above.

For a machine-readable report:

```json
{
  "plugins": { "throughput": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/rocjitsu-throughput" }
}
```

The report is written to `/tmp/rocjitsu-throughput/throughput.log`.

### Instruction Mix Plugin

The instruction-mix plugin answers "which instructions did this run actually
execute, and how many of each?". It counts one execution whenever a wavefront
reaches the before-execute hook, so counts are executed **wave instructions**,
not active-lane operations — the same convention as the throughput plugin. Both
plugins classify through the shared `plugins/instruction_family.h`, so the two
reports join on mnemonic and agree family for family.

The plugin subscribes only to before-execute and async-issue. Offloaded
instructions count exactly as synchronous instructions, without requiring
synchronous execution. Other plugins can request additional hooks; throughput,
for example, also subscribes to after-execute for timing. None of these
counting plugins enables register-access observation.

It emits one JSON object per line using the `rocjitsu.instruction_mix.v1`
schema: one `"record":"dispatch"` object per completed dispatch and one
`"record":"summary"` object at shutdown. Each carries `wave_instructions`,
`unique_mnemonics`, a per-family breakdown (`scalar`, `vector`, `matrix`,
`lds`, `global`, `control`, `other` — the same exclusive families the
throughput plugin uses), and a `mnemonics` object mapping each executed
mnemonic to its execution count, the
encoding format id and opcode of its first sighting, its encoding size in
bytes, its family, and the first dispatch that reached it. Mnemonics are
emitted in sorted order so two runs of the same workload produce
byte-comparable output.

Two mnemonics need care when joining the report against an ISA inventory. A
VOPD instruction reports its *pair*, not either operation: the generated CDNA5
`Vopd` constructor joins the two slot names with `" :: "`, so a record key looks
like `v_dual_add_f32 :: v_dual_mov_b32`. An exact-name
lookup matches neither half and silently loses both. Split on `" :: "` and look
the two slot operations up separately — while still counting the record as the
one wave instruction it is, since the pair issues together.

A mnemonic absent from the summary was not executed — provided the summary says
`"complete": true`. The summary also carries `dispatches` (how many reached
execution-end) and `incomplete_dispatches` (how many were still in flight when
the run stopped, as a bounded run via `rj_vm_request_exit` leaves them). An
unfinished dispatch still contributes its mnemonics to the summary's union, but
gets no `"record":"dispatch"` line, because its per-dispatch totals are not
final. When `incomplete_dispatches` is non-zero the summary is a subset:
wavefronts that reached neither halt nor cancellation keep their counts in
wavefront-local state that the plugin cannot reach at shutdown, so absence no
longer proves non-execution.

Turning coverage into a *percentage* needs a denominator the plugin
deliberately does not supply: it observes decoded instructions, not the target
that produced them, so it reports no architecture name. The harness that chose
the config knows the architecture and joins the report against the
per-architecture instruction set.

```json
{
  "plugins": { "instruction-mix": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/rocjitsu-instruction-mix" }
}
```

The report is written to `/tmp/rocjitsu-instruction-mix/instruction-mix.log`.

Note on mnemonic storage: `Instruction::mnemonic()` is documented as pointing to
static storage, but the generated FLAT encoding on every architecture and VOPD
on gfx11/gfx12 and CDNA5 point it at a per-instruction `std::string` member
instead. The returned `std::string_view` therefore does not outlive the
instruction, and this plugin copies each mnemonic on first sight rather than
storing the view. A plugin that keys a long-lived map on the view will read
freed memory;
`InstructionMixPluginTest.OwnsMnemonicStorageWhenTheSourceIsNotStatic` pins the
behaviour. See [Retaining instruction information](#retaining-instruction-information)
for the other borrowed parts of an `Instruction`.

### Kernel Logging Plugin

The logging plugin records kernel dispatch metadata and detects MMA
(matrix multiply-accumulate) instruction usage:

- **Kernel dispatches**: entry PC, grid dimensions, workgroup dimensions,
  register counts, and kernel name (when available from the code object).
- **MMA detection**: reports the first MFMA or WMMA instruction seen in
  each dispatch.

### Perfsim compatibility plugin

The optional Linux-only Perfsim adapter loads a separately built Perfsim backend at
runtime. RocJITsu does not build, link, vendor, or install Perfsim and does not
write a persistent Perfsim trace; Perfsim continues to own its `GPUCSIM_*`
configuration and reports. See the [Perfsim adapter README](../lib/rocjitsu/src/rocjitsu/vm/plugins/perfsim/README.md)
for the runtime-first setup, independent build boundary, compatibility
requirements, limitations, and validation procedure.

## Enabling plugins

Plugins are compiled into standalone shared objects named
`librocjitsu_plugin_<name>.so` and discovered at runtime through the
standard dynamic-linker search path (`librocjitsu_plugin_*.so` are
installed next to the interposer, and the launcher adds that directory to
`LD_LIBRARY_PATH`).

A plugin is enabled by listing it in the `plugins` section of the
rocjitsu config file. The key is the plugin name (the `<name>` in
`librocjitsu_plugin_<name>.so`) and the value is a JSON object with the
plugin's configuration:

```json
{
  "plugins": {
    "race": {},
    "logging": {},
    "throughput": {}
  }
}
```

Plugin loading is best-effort by default: an unavailable or invalid entry is
logged and skipped. Set the optional top-level `"require_all_plugins": true`
when every configured plugin is required; startup then fails if any entry
cannot be loaded. With strict loading, `plugins` must be an object when present,
though omitting it is valid and creates an empty plugin group.

The default bundled plugins are `race` (`RaceDetectorPlugin`), `logging`
(`KernelLoggingPlugin`), `throughput` (`ThroughputPlugin`), and
`instruction-mix` (`InstructionMixPlugin`). `perfsim` (`PerfsimPlugin`) is
available only in builds configured with `ROCJITSU_ENABLE_PERFSIM_PLUGIN=ON`.

### Enabling plugins from the mirage CLI

When launching a workload through mirage, plugins can be selected on the
command line with `--plugin <name>` instead of editing a config file.
mirage injects each selected plugin into the rocjitsu config it synthesises
for the run (and, for containerised profiles, bind-mounts the plugin's
`.so` next to the interposer). The flag is repeatable and merges with any
plugins the profile already enables:

```bash
# Enable the race detector and the kernel logger for a single run.
mirage run --plugin race --plugin logging -- ./my_app

# Same, when starting a session.
mirage session start --profile mi350x --plugin race
```

Each `--plugin` enables the plugin with its schema defaults. Plugins that
take required arguments, or runs that need custom sink settings, are
configured through a profile or an explicit `--config <file>`.

### Plugin loader boundary

Plugins are repository-owned components built and shipped with rocJitsu. The
loader boundary does not provide compatibility or versioning for independently
built plugins; the host and plugins must always be rebuilt together.

Each plugin `.so` exports three `extern "C"` functions:

- `const PluginMetadata *rocjitsu_plugin_metadata()` — returns a pointer
  to static metadata: `name` and a `config_schema` JSON string.
- `PluginHandle rocjitsu_plugin_create(const char *config_json)` —
  constructs the plugin from its resolved JSON configuration string and
  returns an opaque handle.
- `void rocjitsu_plugin_destroy(PluginHandle handle)` — destroys an
  instance previously returned by `rocjitsu_plugin_create`.

Allocation and deallocation stay on the plugin side of the boundary: the
host destroys each instance through the plugin's own
`rocjitsu_plugin_destroy` export. Use the `ROCJITSU_DEFINE_PLUGIN` macro
from `plugin_exports.h` to emit all three functions. The host validates the
required exports.

### Config schema

The `config_schema` string describes the accepted config keys. Each key
maps to an object with a `type` (`string`, `number`, or `boolean`), an
optional `description`, an optional `default`, and an optional boolean
`optional`. A missing key with a `default` receives that value, including when
`optional` is also `true`. A missing key with `optional: true` and no default
remains absent. All other keys are required. Example:

```json
{
  "argname": { "type": "string", "description": "does something important", "default": "defaultvalue" },
  "requiredarg": { "type": "number" },
  "optionalarg": { "type": "boolean", "optional": true }
}
```

The loader validates supplied values first, applies defaults to missing keys,
omits missing optional keys, checks for required keys, and passes the resolved
JSON object to `rocjitsu_plugin_create`.

## Plugin output

Plugins write reports and logs through a configurable sink system rather than
directly to stderr.
This makes output testable and redirectable.

### Sink configuration

Sinks are configured from an optional top-level `sinks` object in the
rocjitsu config (the same file that lists the `plugins`). There are no
sink-related environment variables.

| Key | Default | Description |
|---|---|---|
| `types` | `["stderr"]` | Array of sink types: `stderr`, `stdout`, `file` |
| `dir` | *(none)* | Directory for file sinks. Required when `file` is in `types` |

When `file` is in `types`, each plugin writes to
`<dir>/<plugin_name>.log`. Plugin names are fixed:
`race` for `RaceDetectorPlugin`, `logging` for `KernelLoggingPlugin`,
`throughput` for `ThroughputPlugin`, `instruction-mix` for
`InstructionMixPlugin`, and `perfsim` for adapter diagnostics. Perfsim's
own report remains controlled by its `GPUCSIM_*` configuration.

### Examples

Interactive use — output goes to stderr (the default):

```json
{ "plugins": { "race": {} } }
```

```bash
rocjitsu --config my_config.json -- ./my_app
```

Save race reports to files (for test harnesses):

```json
{
  "plugins": { "race": {} },
  "sinks": { "types": ["file"], "dir": "/tmp/output" }
}
```

```bash
rocjitsu --config my_config.json -- ./my_app
# Race reports are in /tmp/output/race.log
```

Send output to both stderr and a file simultaneously:

```json
{
  "plugins": { "race": {} },
  "sinks": { "types": ["stderr", "file"], "dir": "/tmp/output" }
}
```

> Note: plugins can also be selected on the mirage command line with
> `mirage run --plugin <name>` (see "Enabling plugins from the mirage
> CLI" above). Sink selection is still driven entirely by the config file
> shown here.

### Writing a plugin that uses sinks

Plugins inherit a sink from `ExecutionPlugin`. Use `sink().write(msg)`
for all output instead of `fprintf(stderr, ...)` or `std::cerr`:

```cpp
class MyPlugin : public ExecutionPlugin {
public:
  explicit MyPlugin(const char *config_json) : ExecutionPlugin("myplugin") {
    (void)config_json;
  }

  void onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) override {
    sink().write(std::format("[myplugin] dispatch {}\n", info.dispatch_id));
  }
};
```

The sink is assigned by the `ExecutionPluginGroup` when the plugin is
added. Writes through sinks assigned by one group are serialized at the
group's fanout boundary, including writes from asynchronous plugin workers.
If no group configures a sink, the default is stderr.

`KernelDispatchInfo` reports the effective LDS allocation in
`lds_size_bytes`, the descriptor-selected `wave_size`, the configured
execution `code_target`, and the workgroup-cluster dimensions. The code target
describes the simulator ISA selected for execution; it is not source-code-object
or translation provenance.

## How it works

The `ExecutionPlugin` interface (`execution_plugin.h`) defines hooks
that the compute unit and command processor call during execution.
Multiple plugins can be active simultaneously via `ExecutionPluginGroup`.

### Workgroup initialization

The workgroup-dispatched hook supplies dispatch and workgroup identities and
the initialized waves. A wave's allocation counts and its CU's physical
register block sizes are distinct quantities; consumers query the wave or CU
for the metadata they need.

### VGPR observation precision

`onAmdgpuWriteVgprLanes` observes instruction-level VGPR destinations rather
than VM/runtime storage writes. Reads and writes carry the architectural lane
and byte masks for masked, sub-dword, and multi-register operands. Internal
storage operations used to preserve unaffected register state do not produce
additional architectural callbacks.

Asynchronous memory operations are modeled separately. The core scoreboard
records their register dependencies at issue. A later completion updates storage
without emitting the same instruction-level write again. Architectural waits and
ordering guarantees release dependencies; eager storage writeback is not proof
of readiness. Plugins needing register hazards should consume the structured
findings below rather than reconstructing this analysis from access hooks.

### Scalar register identity

`onAmdgpuReadScalarRegister` and `onAmdgpuWriteScalarRegister` carry a
`RegisterRef` whose class and index identify the architectural register. This
keeps plugins independent of encoded selector values and of the simulator's
physical storage layout. In particular, ordinary SGPRs and wave-private TTMPs
have distinct identities even though both can appear in scalar operand fields.

The older `onAmdgpuReadSgpr` hook continues to expose physical SGPR indices for
compatibility. New consumers that need architectural identity should use the
typed scalar-register hooks. Scalar writes have only the typed callback.

### Core register findings

`observes_register_hazards()` defaults to false. Opting in requests ordinary
core register-dependency checking even when `memory_wait_diagnostics` or the
global `wait_checking` setting disables stderr warnings. It does not enable
register-access hooks. XCNT replay-source checking still requires its existing
explicit setting; when enabled, its findings use the same hook.

`onAmdgpuRegisterHazard(observation, wave)` runs on the issuer before the
consumer executes, including before its before-execute or async-issue callback.
The `RegisterHazardObservation` owns the producer and consumer instruction
encodings when available and gives their PCs, overlapping register range, lane
and byte masks, read/overwrite kind, counter, and sufficient wait threshold.
Register indices are wave-relative physical indices, including selected VGPR
banks. Scalar findings use an all-ones lane mask. An encoding can be absent for
a synthetic producer or a dependency recorded before observer attachment;
reporters must not invent bytes or discard the finding for that reason. The
wave reference supplies current dispatch/workgroup context and is borrowed only
for the callback.

The core retains dependencies after reporting while any register-finding observer
is attached. Waits and other architectural guarantees still retire them. The
observer owns deduplication and output limits; it receives findings independently
of the core stderr warning budget. It must not mutate the scoreboard or execute
or replay an instruction from the callback. This is a finding interface, not a
complete register-access stream: shadow-negative checks do not enumerate accesses.
Eager memory writeback is not a wait proof, and register readiness does not prove
cross-wave LDS synchronization. See [memory wait diagnostics](memory-wait-diagnostics.md).

### Dispatch threading

Callback policy is derived from the plugins contained by an
`ExecutionPluginGroup`. An empty group returns before dispatch or locking. The
group divides hooks by frequency and synchronization cost:

- Lifecycle, dispatch, workgroup, wavefront, and barrier callbacks are
  infrequent. The group takes one recursive mutex before iterating its plugins,
  so two infrequent callbacks cannot overlap across simulation partitions. With
  the default hot-hook policy, an infrequent callback can still overlap a
  high-frequency callback. Recursive acquisition lets a callback synchronously
  read registers and fire register-observation hooks without deadlocking.
- Instruction before/after, memory-routing, tensor-DMA issue, tensor-DMA
  memory, register-finding, and register-access callbacks use the hot-hook policy and run concurrently
  with both other high-frequency callbacks and infrequent callbacks by default.
  Each callback is scoped to a wavefront below the simulation's shader-engine
  partition granularity. During the before-instruction callback, a memory
  instruction exposes its decoded wait-counter obligations and completion-order
  metadata through `amdgpu_memory_issue_info()`, before address or store-data
  operands are read. This metadata describes operations routed through the
  scalar, vector, and local memory pipelines; it is not a complete inventory of
  non-memory events, such as messages and timestamp queries, that hardware wait
  counters may also track.

A plugin whose high-frequency callbacks reach shared mutable state may override
`requires_serial_hot_hooks()` to return `true`. The group samples that stable
policy once when the plugin is added and then takes the same group mutex around
every high-frequency callback, serializing it with the infrequent callbacks
without a per-instruction scan of the plugin list. Plugins that protect their
own shared state should retain the parallel default.

Every high-frequency hook is opt-in. Override its callback and return `true`
from the corresponding interest method:
`observes_before_execute_instruction()`,
`observes_after_execute_instruction()`,
`observes_async_instruction_issued()`,
`observes_memory_instruction_routing()`, `observes_vgpr_reads()`,
`observes_vgpr_writes()`, `observes_sgpr_reads()`,
`observes_scalar_register_writes()`, `observes_memory_routing()`,
`observes_tensor_dma_memory_access()`, `observes_tensor_dma_issue()`, or
`observes_register_hazards()`. All default to `false`; overriding a callback
alone does not subscribe to it. The group samples these stable flags once when
the plugin is added and avoids both fanout and upstream observation work when
no contained plugin consumes a hook. Lifecycle callbacks need no interest
declaration. A no-op or lifecycle-only plugin enables no hot paths and leaves
async execution enabled.

For finer-grained selection, `observes_hot_hooks_for_wavefront(wf)` controls
all high-frequency callbacks for one wavefront. After every plugin's
`onAmdgpuWavefrontDispatched()` callback completes, the group samples and caches
that predicate on the wavefront until halt or cancellation, including rollback
before a retry in the same slot. Reentrant hooks during dispatch, halt or
cancellation, hooks without a wavefront, and hooks on a resident wave after live
plugin-group replacement use the current group's live predicate instead. The predicate
must therefore be lock-free, thread-safe, and stable for the subscribed portion
of a wavefront incarnation, ending at its matching halt or cancellation.
Lifecycle, dispatch, workgroup, wavefront, and barrier callbacks are not filtered
by this predicate.

Live replacement refreshes subscriptions for resident waves, but does not
replay their dispatch callbacks or migrate or clear their wave-local plugin
state. A newly attached stateful plugin therefore has no dispatch-initialized
state for those waves, while a replacement that reuses a plugin slot can see
state left by the previous plugin. Stateful plugin groups should be replaced at
a quiescent boundary unless they explicitly tolerate both cases.

### Observing memory accesses

There are two memory hooks, and they see different things.
`onAmdgpuRouteMemoryInstruction` fires before routing decides anything: the
instruction still carries the address space it decoded as and the addresses it
computed. `onAmdgpuMemoryAccessRouted` fires once routing has settled and
reports a `MemoryAccessObservation` describing the access the memory system is
actually about to be asked for — the pipeline it was issued to, its routing-selected
counter, and its effective addresses. The scalar `wait_counter` field does not
describe the complete wait requirement: mixed FLAT retains its VMEM counter here,
while the resolved issue carries both VMEM and LDS obligations. Use
`amdgpu::resolved_memory_issue_info(inst)` from `vm/amdgpu/mem_state.h` to borrow
the same descriptor as the memory pipeline, then read its `counter_obligations()`.
Without a routing override it returns the decoded metadata. A present, empty
override means the resolved issue contributes no obligations; it does not fall
back to the decoded domains. The pointer is valid only while the instruction
and its current dynamic state remain unchanged.

When all requesting FLAT lanes use the shared aperture, routing selects the
LDS pipeline and sets `normalized_to_local` to `true`. This flag describes the
uniform local route, not whether addresses were normalized.

Mixed FLAT requests retain `MemoryRoute::GLOBAL`. The global pipeline separates the lanes by backing store, completing
retryable global/scratch work before accessing LDS. These
observations have `normalized_to_local == false`. On both routes, `addresses`
reports effective ordinary LDS addresses, which can still be outside the backing, and
`pre_routing_addresses` preserves the original aperture addresses. Normalizing
shared lanes during routing gives execution and observers the same addresses.
CDNA5 DDS lanes are identified separately. Their reported values also include
LDS-base arithmetic, but identify neither a DDS backing location nor a valid
LDS allocation address. Requesting DDS lanes emit a warning naming the instruction
and PC, then fault before any memory effects because DDS backing is unimplemented.
On other architectures the shared aperture remains LDS; out-of-backing accesses
read zero and discard writes.

For RDNA4 FLAT LDS requests, the offset wraps to 17 bits before adding the
workgroup base. `flat_local_allocation_end` captures the exclusive modeled
`lds_base + lds_size` limit and remains fixed across retries. This includes the
current 256-byte scheduler rounding; PM4 sizes may already be 512-byte multiples
before that rounding. Target allocation and encoding granules remain unmodeled.
A lane whose first memory element does not fit is rejected in full and appears
in `flat_local_oob_lane_mask`. Other lanes retain only complete elements below
the allocation end (DWORDs for multiword transfers); atomics require their whole
element. Rejected reads return zero and rejected writes are discarded. These
bounds leave `valid_lane_mask`, `request_lane_mask` and nominal counter
participation unchanged, so those masks describe requests rather than bytes
actually accessed.

Observers must inspect `flat_local_lane_mask` and `flat_dds_lane_mask` to account
for LDS and DDS lanes; choosing a backing store
from `route` alone misses the shared-aperture portion of a mixed request.

`decoded_space` preserves the instruction's original address-space family
independently of its effective `route`, so explicit SCRATCH remains distinct
from a FLAT access that resolves to scratch or LDS. When routing changes an
address, `pre_routing_addresses` contains the original per-lane span and
`addresses` contains the effective span. The former is empty when no address
changed; when present, both spans have `wavefront_size` entries and only lanes
in `valid_lane_mask` are meaningful.

Resolved memory views are built only when a subscribed plugin requests them
through `observes_memory_routing()`.

`onAmdgpuMemoryAccessRouted(access, inst, wf)` supplies the resolved observation,
instruction and wave together. Observers can inspect decoded metadata and use
wave-local state directly. The instruction and wave already reflect the selected
route and are borrowed only for the callback.

The observation's spans borrow execution-owned storage and are valid only for
the duration of the callback. A plugin that keeps one must copy them.

Tensor DMA transfers execute directly rather than through the ordinary memory
routing path. `onAmdgpuTensorDmaMemoryAccess` reports them separately after the
instruction and any descriptor-requested atomic-barrier arrival return normally.
Its `addresses` member is a callback-lifetime `TensorDmaAddressView` over the
in-bounds global element bases whose requests were attempted, in execution order
with duplicates preserved. Empty and fully masked transfers produce no callback,
so a callback's transfer fields always come from a validated descriptor. `pc` is
the tensor instruction's PC, also when a deferred transfer completes after the
wavefront has moved past the instruction. Consumers must opt in through
`observes_tensor_dma_memory_access()`. A consumer that retains addresses must
use `size()` to budget and allocate the event, then call `copy_to()` during the
callback; a failed copy means the observation is malformed and must be rejected.
`onAmdgpuTensorDmaMemoryAccess(access, wf)` also supplies the issuing wave for
wave-local state. The wave is borrowed only for the callback; its current PC can
differ from the observation's PC when the transfer was deferred.

`onAmdgpuTensorDmaIssued(issue, wf)` is the tensor DMA counterpart of
`onAmdgpuMemoryAccessRouted`. It fires once for every tensor DMA instruction, in
the issuing wavefront's instruction step, after the descriptor is read and,
unless it is NULL, validated. That includes an instruction that moves no data:
a NOP tile (`tile_dim0 = 0`), a gather with no valid indices or a store whose
tile lies wholly outside the tensor still completes on TENSORcnt, and the
emulator also counts a NULL descriptor (`count = 0`) on TENSORcnt, an assumption
since hardware behavior for that encoding is not established. A transfer that
waits for memory is reported when it issues and not again when it resumes. When
the hook fires, a store has already sampled its LDS source bytes, so changing
LDS in the callback does not change what the store writes to global memory; no
global request has been made and a load has written no LDS. Its `lds_ranges`
member is a callback-lifetime span of the LDS byte runs the transfer will write
(loads) or read (stores), with adjacent elements merged and padding bytes
excluded. Bytes outside the wave's LDS allocation are not transferred and are
omitted, so a range can end mid-element when an element straddles the end of
the allocation; consumers must treat ranges as byte ranges, not as whole
elements. A load covers every element of its tile, zero-filling those outside
the tensor; a store covers only in-bounds elements. Empty `lds_ranges` mean
only that the transfer touches no LDS payload bytes, not that it makes no global
accesses: for a tile wholly past the allocation, a load still reads its
in-bounds tensor elements from global memory and can fault there, and a store
still writes zeros to its global destination. Consumers must opt in through
`observes_tensor_dma_issue()`.

SGPR owner resolution is skipped when no contained plugin observes scalar
register reads. `observes_sgpr_reads()` enables both
`onAmdgpuReadScalarRegister` and legacy `onAmdgpuReadSgpr` callbacks.

Pass the complete sink configuration to the group constructor and add plugins
before publishing the group to simulation components. `add()` is not
thread-safe, and the group must remain immutable while callbacks may dispatch
concurrently.

### Retaining instruction information

The `Instruction` passed to instruction and memory hooks is borrowed and is
valid only during the callback. So are the views it returns: `raw_encoding()`
points to encoding words owned by the decoder or the instruction, and
`mnemonic()` can view a member of the instruction (see the note on mnemonic
storage above). Copy anything you keep before the callback returns.
`Instruction::disassemble()` is different: it formats a new `std::string` on
each call, and the caller owns the result, so a plugin may keep it.

Formatting text for every executed instruction is expensive when little of it
is ever printed. A plugin that needs text only when it reports something can
instead copy the instruction's PC and encoding as it executes, and disassemble
the copy when the report is written. Validate the encoding before copying it:
`raw_encoding()` is null when the instruction has no encoding, and the copy
needs `size()` to be a positive multiple of four bytes that fits the buffer.
The CU normally fetches four words starting at the PC. GFX10 MIMG with
NSA=3 uses a fifth word. The fetch-aware hook receives all five for that
instruction. The example stores up to five words, as the race detector does,
and rejects invalid sizes at capture. Perfsim's FFM event format has its own
four-word encoding limit.

```cpp
// Set for each wave in onAmdgpuWavefrontDispatched() with wf.set_plugin_state().
struct LastInstruction : WavefrontState {
  uint64_t pc = 0;
  std::array<uint32_t, 5> words{};
  size_t word_count = 0; // Zero until the wave executes an instruction.
};

bool observes_before_execute_instruction() const override { return true; }
bool supports_async_instructions() const override { return false; }

void onAmdgpuBeforeExecuteInstruction(uint64_t pc, const Instruction &inst,
                                      amdgpu::Wavefront &wf) override {
  auto &last = *static_cast<LastInstruction *>(wf.plugin_state(slot_index()));
  const int bytes = inst.size();
  if (!inst.raw_encoding() || bytes <= 0 || bytes % sizeof(uint32_t) != 0 ||
      static_cast<size_t>(bytes) > sizeof(last.words)) {
    if (!wf.instruction_execution_failed())
      wf.report_instruction_execution_error(
          amdgpu::InstructionExecutionError::PluginFailure, "instruction has no valid encoding");
    return;
  }
  last.pc = pc;
  last.word_count = bytes / sizeof(uint32_t);
  std::copy_n(inst.raw_encoding(), last.word_count, last.words.begin());
}

// Later, when a report about this wave needs text:
if (last.word_count != 0) {
  util::FailureOr<std::string> text = wf.disassemble_instruction(
      std::span(last.words).first(last.word_count), last.pc);
  if (text.failed()) {
    if (!wf.instruction_execution_failed())
      wf.report_instruction_execution_error(
          amdgpu::InstructionExecutionError::PluginFailure, "cannot disassemble instruction");
    return;
  }
  // Use text.value().
}
```

`Wavefront::disassemble_instruction` decodes the copy with the wave's CU target
decoder, so a standalone plugin module can use it without linking the
simulator. Call it on the CU execution thread, for example from one of that
wave's synchronous hooks. It returns `FailureOr<std::string>` and emits a diagnostic
through the optional `DiagnosticEmitter` if the encoding is empty, truncated, or
cannot be decoded. Because the text comes from the copied bytes,
it describes the instruction that executed even if the code was later
overwritten or unmapped. The race detector formats its instruction traces this
way.

Plugin callbacks, policy queries, constructors, destructors, and output sinks must
not throw. This is a semantic contract; callback signatures do not require
`noexcept`. From synchronous instruction hooks, report expected instruction or
reporting failures through
`Wavefront::report_instruction_execution_error()` and preserve any error already
recorded on the wave. Every accepted asynchronous instruction must still receive
its normal completion accounting before cancellation or shutdown.

`onAmdgpuWavefrontHalted` is a terminal notification. The wave's instruction-error
state is discarded when this hook returns. Use `sink().write()` to report problems
from this hook.

## Adding a new plugin

1. Implement `ExecutionPlugin` in a new subdirectory. The plugin class can
   provide a `const char *config_json` constructor, or a default constructor and
   `util::Result initialize(const char *, util::DiagnosticEmitter)`.
   The export macro uses the `const char *` constructor when available; it uses
   the checked initializer only when that constructor is absent. A constructor
   taking `const char * = nullptr` still takes precedence, so remove it when
   migrating to the checked form. If initialization fails, the factory logs
   `create failed: <diagnostic>`, destroys the candidate, and returns a null handle.
2. Add a `plugin_export.cpp` that calls
   `ROCJITSU_DEFINE_PLUGIN(MyPlugin, "myname", schema)`.
3. In `CMakeLists.txt`, add the object library and a
   `rj_add_plugin_so(myname <object_lib> <export_src>)` call so it builds
   `librocjitsu_plugin_myname.so`.
4. Use `sink().write()` for all output — never write to stderr directly.
5. Audit shared mutable state reached by high-frequency hooks against both
   high-frequency and infrequent callbacks. Override
   `requires_serial_hot_hooks()` when that state cannot be protected within the
   plugin.
6. Return `true` from the per-hook `observes_*()` methods for every high-frequency
   hook the plugin consumes. Use `observes_hot_hooks_for_wavefront()` when subscription
   also depends on the dispatched wavefront; keep that predicate lock-free,
   thread-safe, and stable after dispatch.
7. Enable it by adding `"myname": { ... }` to the `plugins` section of
   the config file.
8. Return `false` from `supports_async_instructions()` when the plugin requires
   synchronous state inspection or the before/after pair for every instruction.
   One such plugin disables offload for the entire group. An instruction
   observer that supports async must subscribe to accepted-issue notifications
   as well as synchronous callbacks, and accommodate concurrent same-wave
   helper register hooks when it subscribes to register accesses.
9. Copy any instruction data that the plugin keeps beyond a callback; see
   [Retaining instruction information](#retaining-instruction-information).

## Asynchronous arithmetic

`supports_async_instructions()` defaults to `true`, leaving the scheduler's
offload policy unchanged. A plugin requiring synchronous inspection returns
`false`. The group samples this requirement on `add()`; any such plugin
disables offload for the group. This policy is independent of subscriptions:
`observes_async_instruction_issued()` requests the accepted-issue notification,
which replaces the ordinary before/after pair for offloaded instructions.

The notification runs on the issuer after a helper accepts the instruction;
helper register hooks may already have run. Issue callbacks may read metadata,
not register values: even a source can alias a destination being written.
Callbacks must not retain instruction or wave references. There is no completion
notification. Holding the callback mutex does not provide a complete
architectural snapshot.

Plugins that need to report instruction failures from register hooks must keep
`supports_async_instructions()` false. Helper register hooks can run while the
issuer clears the same wave's instruction error field for its next instruction;
neither reading nor writing that field from a helper is safe. Issuer-thread issue
callbacks may report failures. The CU checks the retained error after the full
issue callback fanout and drains accepted jobs before cancelling the wave.

Throughput, instruction mix, kernel logging, and the
[simulator race detector](race-detector.md) support this contract. Race captures
instruction evidence on the issuer without observing helper register accesses;
the core checks register dependencies before async publication. A plugin that
requires synchronous inspection still disables offload for the whole group.
Plugins and the host must be rebuilt together, as for other execution-plugin
interface changes.

Plugin support does not make an instruction eligible for offload. The current
arithmetic handlers require issuer-owned error state, so the core executes
them synchronously even when helpers are requested. See
[asynchronous instructions](async-instructions.md) for the current policy.

Sampler instructions, regardless of filter mode, report one `onAmdgpuMemoryAccessRouted` callback per
instruction. Samples with a prepared footprint use `image_sample_footprints`
and leave `addresses` empty. A null resource or unbacked sampler completes as
an empty request: the callback carries the ordinary per-lane `addresses` span,
an empty footprint span and a zero `request_lane_mask`. Its counter obligation
still identifies the sampled request on targets with SAMPLECNT. Each footprint has
`wavefront_size` entries and its lane mask is a subset of `request_lane_mask`.
Border-only and zero-weight taps remain present with those lanes cleared.
`bytes_per_lane()` is the size of one texel per footprint, including gather
instructions that return four selected components. Consumers must visit every
footprint without counting another issued instruction or wait-counter event.
