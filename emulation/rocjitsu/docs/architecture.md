# Architecture

rocjitsu is a full-system GPU simulator organized into layered components.
From bottom up:

```
┌─────────────────────────────────────────────┐
│  HIP / ROCR / RCCL  (unmodified binaries)   │
├─────────────────────────────────────────────┤
│  KMD Layer (kmd/linux/)                     │  ┌──────────────────────────┐
│    Interposer ── RemoteDriver ── RPC ───────┼──┤  Daemon (tools/rocjitsu) │
│    SimulatedDriver ── EventState            │  │  Unix socket RPC         │
├─────────────────────────────────────────────┤  └──────────────────────────┘
│  VM Layer (vm/amdgpu/)                      │
│    GPU SOC and component blocks             │
│    GpuVm (address spaces and translation)   │
│    GpuMemory (physical sparse backing)      │
│    Cache and memory models                  │
├─────────────────────────────────────────────┤
│  ISA Layer (isa/)                           │
│    Decoder ── Instruction ── Execute        │
│    GFX architecture targets                 │
├─────────────────────────────────────────────┤
│  Code Layer (code/)                         │
│    Executable ── AmdGpuCodeObject           │
│    BasicBlock ── DBT (binary translator)    │
├─────────────────────────────────────────────┤
│  Simdojo Engine (simdojo/)                  │
│    PDES ── Components ── Topology           │
│    Events ── Ports ── Links                 │
├─────────────────────────────────────────────┤
│  Config (config/)                           │
│    JSON ── FlatBuffers ── load_config       │
└─────────────────────────────────────────────┘
```

## Components

### Simdojo (`lib/simdojo/`)

Parallel Discrete Event Simulation (PDES) framework. Provides the
component model, topology builder, event queue, and multi-threaded
execution engine with barrier-based LBTS synchronization.

See [simdojo.md](simdojo.md) for the full design.

### VM — GPU Hardware Model (`vm/amdgpu/`)

See [vm-design.md](vm-design.md) for the full hardware model design.
Models the GPU hardware pipeline:

- **CommandProcessor** — Monitors compute doorbells, fetches AQL and supported
  PM4 packets from ring buffers, parses kernel descriptors, and dispatches
  workgroups to CUs. It does not own or execute SDMA queues.
- **ComputeUnit** — Executes wavefronts. Manages SGPR/VGPR register
  files, LDS, and scratch memory. Supports functional and cycle-accurate
  modes.
- **CompletionTracker** — Tracks per-dispatch workgroup retirement.
  Fires completion signals in submission order. Writes queue-inactive
  signal on HQD idle. Guest-visible publication is journaled across transient
  backing stalls; only the affected queue pauses while independent queues keep
  fetching and dispatching.
- **ShaderEngine / XCD / IOD** — Hierarchical GPU topology matching
  real hardware (shader engines contain CU arrays, XCDs contain SEs).
- **GpuVm** — Frontend-neutral owner of address-space identity, lifetime,
  translation policy, permissions, invalidation epochs, and typed access
  outcomes. Legacy KFD/interposer and PCI/VFIO queues retain the same
  generation-checked address-space handles and feed the same CP, SDMA, and CU
  execution models. Queue execution owners also retain a `GpuVmBindingLease`,
  which prevents address-space teardown while they can still take new immutable
  access snapshots; this lifetime pin is separate from frontend queue-reference
  accounting.
- **LegacyGpuVmAdapter / LegacyAddressSpace** — KFD-owned compatibility layer
  for the interposer path. Each process registration gets an isolated
  `LegacyAddressSpace` that adapts legacy page tables, passthrough,
  client-process access, fault delivery, MTYPE resolution, copies, and atomics
  to the generic `GpuVm` translator and backing interfaces.
- **GpuMemory** — Physical sparse backing bytes only. It has no VMID routing,
  translation policy, process mappings, or transport endpoint. HBM controllers
  adapt simdojo memory requests into `GpuVm` accesses or direct physical
  backing operations. A SoC-owned `GpuMemoryPhysicalAccess` adapter and
  handle-only identity address space route standalone model queues through the
  same `GpuVmAccess` interface without claiming VMID 0 from PCI/VFIO. The SoC
  installs this backing as configuration state: repeated installation of the
  same object is idempotent, but replacing it with a different backing is not a
  supported runtime operation.
- **SdmaQueueScheduler** — SoC-owned SDMA scheduler with one dedicated worker. It owns
  every SDMA ring consumer, provides round-robin service with bounded root-packet
  turns, retries temporary backing failures, and synchronously quiesces a queue
  before detaching it. Individual long-running SDMA packets retain resumable
  continuation state but are not yet preempted by an intra-packet work quantum.
- **SdmaRingConsumer** — Transport-neutral owner of one SDMA root ring's device
  cursor, immutable VM snapshot, wrapped fetch progress, typed semantic
  continuation, retirement publication, and terminal state. Legacy KFD and PCI/MES
  create queues through `GpuQueueRegistry` and the same SDMA binding factory; the command
  processor is not part of the SDMA path.
- **Packet processors** — `AqlPacketProcessor`, `Pm4PacketProcessor`, and
  `SdmaPacketProcessor` implement the same compile-time, one-head-packet
  processing contract while retaining protocol-specific request and diagnostic
  state. An SDMA request carries caller-owned opaque continuation state because
  an SDMA packet may block after a partial effect; AQL and PM4 retries are
  restartable and require no continuation. Packet processors
  do not own queue registration, scheduling, ring lifetime, or completion
  tracking. Expected guest-packet and descriptor failures are returned as typed
  per-queue results; they do not escape as simulator exceptions or stop unrelated
  queues.
- **PCI/VFIO adapters** — PCI configuration, BAR/MMIO, DMA, interrupts, and
  transport-session lifetime. These adapt accesses into `GpuVm` and the shared
  block models; they do not contain alternate CP, MES, SDMA, or shader models.
- **Cache hierarchy** — L1 vector cache, L1 scalar cache, L2 cache,
  memory-side cache. MTYPE-aware (UC, CC, RW).
- **Execution plugins** — Pluggable hooks for race detection, kernel
  logging, and more. See [plugins.md](plugins.md).

### PCI/VFIO control-path boundary

`VfioDeviceHost` owns only the libvfio-user transport boundary. It translates
VFIO callbacks into PCI configuration, BAR/MMIO, DMA, and interrupt operations;
it does not implement GPU queues or execution engines. A `PciTransportSession`
adds a generation-checked lifetime around the active DMA/IRQ endpoints, and
operation leases keep an in-flight callback bound to the session generation it
captured even if the transport is detached or replaced concurrently.

`GpuPciDevice` owns the device-facing control plane. MMIO writes enqueue
deferred work into a FIFO, and reset advances the device epoch so stale work is
discarded rather than applied to a replacement session. Register-block
observers publish queue changes through `GpuQueueRegistry`; command-processor queue
registration is delivered through the CP inbox on the CP's execution context.
Observer callbacks, reset, and deferred-work draining must not run while the
libvfio-user `vfu_mutex_` is held, because those paths can re-enter transport or
simulation services.

The PCI models are adapters over the shared `GpuVm`, `CommandProcessor`, MES,
`SdmaQueueScheduler`, and CU objects. They must not grow alternate CP/MES/SDMA
execution models; new PCI-visible behavior should terminate at the narrow MMIO,
DMA, interrupt, address-space, or queue-registry interface owned by the
corresponding core model.

The firmware-free non-AQL compute ring follows the same boundary. MES
maps the queue and selects its `GpuVm` address space, then registers it with the
selected command processor through `GpuQueueRegistry`. The registry binding is
only a lifetime and notification adapter. The CP-owned `Pm4QueueController`
holds each queue's ring traversal, cursor publication, retry state, and concrete
`Pm4PacketProcessor`, which implements the exact NOP/SET_UCONFIG_REG subset. The
MMIO model supplies only the supported register-write sink. This is deliberately
not a general PM4 implementation.

The PM4 controller leases a queue under its management lock, then releases that
lock before VM access or register callbacks. Removal and reconfiguration wait for
an active lease to quiesce, so frontend callbacks cannot stall unrelated queue
registration or notification and queue state cannot be destroyed during service.
The CP invokes PM4 service outside its AQL queue lock, and each PM4 queue turn is
bounded so a large PM4 ring cannot monopolize the CP event.
Each service transaction pins both its `GpuVmAccess` and producer boundary; work
announced after a root replacement is deferred to a new snapshot rather than
being appended to an older blocked transaction.
Ordinary queue removal is prepared before `GpuQueueRegistry` revokes the handle.
If retirement has not yet been published, PM4 reports the removal as temporarily
unavailable, keeps the same queue generation live, and completes publication
before a later removal retry. Device reset and lost-session teardown use the
separate force-cancel path and never wait indefinitely on inaccessible backing.
The AQL binding follows the same graceful-close rule: it remains registered while
cursor, completion-signal, or queue-idle publication is retryable, while reset
and rollback retain an explicit force-cancel path.

SDMA follows the same composition rule. KFD and PCI/MES decode queue
configuration and producer notifications, then submit typed operations through
`GpuQueueRegistry` to the shared SDMA binding factory. The SoC-owned `SdmaQueueScheduler` schedules
those queues on its worker, while `SdmaRingConsumer` owns root-ring traversal and
delegates packet semantics to `SdmaPacketProcessor`. A service attempt retains one
`GpuVmAccess` for the head packet through incremental fetch, execution, and
read-pointer publication. It fetches one framing dword first and expands only to
the processor-requested packet extent, so an unavailable later ring page cannot
block an already available head packet. A completed conditional packet whose
skip extends beyond the current producer cursor is retained without re-evaluation
and retired only after that extent becomes producer-visible. Temporary backing or
transport unavailability resumes that exact state; malformed requests and permanent
faults terminate the queue without replaying already-retired packet effects.

Queue lifecycle is shared without forcing unrelated protocols into one execution
engine. AQL, PM4, and SDMA use `GpuQueueRegistry`, reusable `QueueBindingFactory` objects,
and unique per-registration `QueueBinding` objects. PM4 and SDMA share
`ConsumerCursorJournal` for retry-safe consumer publication; SDMA and the
restricted PM4 compute-queue path share `CircularRingReader` for wrap-safe
fetches. AQL, PM4, and SDMA expose the same `PacketProcessResult` envelope
and core processor operation. The common layer validates only
protocol-independent status,
retirement, and input-growth invariants. `required_bytes` is the fetch extent
needed to process the head packet, while `retirement_bytes` is the cursor advance
requested after processing and may include protocol-defined skipped commands. The
ring owner validates that retirement extent against its capacity and the
producer-visible cursor. All three remain concrete protocol
implementations because their request, scheduling, retry, and completion
semantics differ; SDMA continuation lifetime remains with its ring consumer.

```text
KFD / MES / PCI MMIO
        |
        v
GpuQueueRegistry + shared SDMA binding factory
        |
        v
SoC-owned SdmaQueueScheduler worker
        |
        v
SdmaRingConsumer / SdmaPacketProcessor
        |
        v
immutable GpuVmAccess -> GpuVm translation -> physical backing
```

### ISA — Instruction Set Architecture (`isa/`)

Instruction decoding and execution for AMD GPU architectures plus
RISC-V. Most files are autogenerated from the MR ISA XML spec by the
`amdisa` codegen pipeline.

- **Decoder** — Decodes variable-length instructions from code objects.
- **Concrete target legality** — Variant-aware decoders combine immutable
  target capabilities with generated instruction-and-encoding requirements;
  architecture-only lookup uses an explicit fail-closed default.
- **Instruction** — Per-encoding instruction structs with typed fields.
- **Execute** — Instruction semantics (shared templates across ISAs
  where possible).
- **Operand resolution** — SGPR, VGPR, literal, inline constant,
  FLAT_SCRATCH encoding.

Hand-written files handle address calculation (`addr_calc_flat.h`),
matrix math execution (`mma_exec.h`), and ISA-specific traits.

See [codegen.md](codegen.md) for the full codegen pipeline and
regeneration commands. See [isa-target-providers.md](isa-target-providers.md)
for target identity, static provider composition, and model-only linkage.

### KMD — Kernel Mode Driver Emulation (`kmd/linux/`)

Emulates the AMDKFD kernel driver via LD_PRELOAD interposition:

- **Interposer** — Intercepts `open`, `close`, `ioctl`, `mmap`,
  `munmap`, `fopen`, `dup2`, `fork` to route `/dev/kfd` operations.
- **SimulatedDriver** — Handles KFD ioctls (create_queue,
  alloc_memory, map_memory, wait_events, etc.). Manages per-process
  state (page tables, doorbells, events).
- **RemoteDriver** — Client-side RPC stub for daemon mode. Forwards
  ioctls over a Unix socket to the daemon's SimulatedDriver.
- **EventState** — KFD event lifecycle (create, set, reset, wait,
  destroy). Signal page shared via memfd between daemon and client.

### Code — Executable Loading & Analysis (`code/`)

- **Executable** — Loads x86 HIP fat binaries, extracts Clang offload
  bundles, parses AMD GPU HSA device ELFs.
- **BasicBlock** — Control-flow basic block construction from decoded
  instruction streams.
- **DBT** — Dynamic Binary Translation between ISA pairs. Uses
  autogenerated legalization tables and encoding translators.
  See [dbt-design.md](dbt-design.md) for the full design.
- **DBI** — Dynamic Binary Instrumentation. Instruments existing
  binaries with probe functions for tracing, software counters,
  and more. See [dbi-design.md](dbi-design.md) for the current state.

Probe code and the destination code object must resolve to the same concrete
GPU target. Instrumentation rejects cross-variant insertion because no
concrete-target legalization contract proves that every copied instruction is
legal for the destination target.

#### Loading and inspecting a code object (C API)

```c
rj_code_executable_t *exec = NULL;
rj_code_executable_create("kernels/matmul_naive.o", &exec);

rj_code_object_t *obj = NULL;
rj_code_executable_get_code_object(exec, ROCJITSU_CODE_TARGET_GFX942, 0, &obj);

rj_code_basic_block_list_t *bbs = NULL;
rj_code_basic_block_list_create(obj, ROCJITSU_CODE_TARGET_GFX942, &bbs);

uint32_t num_blocks = rj_code_basic_block_list_size(bbs);
for (uint32_t i = 0; i < num_blocks; ++i) {
    rj_code_basic_block_t *bb = NULL;
    rj_code_basic_block_list_get(bbs, i, &bb);

    for (const rj_code_inst_t *inst = rj_code_basic_block_first_inst(bb);
         inst != NULL; inst = rj_code_inst_next(inst)) {
        char buf[256];
        rj_code_inst_disassemble(inst, buf, sizeof(buf));
        printf("%s\n", buf);
    }

    rj_code_basic_block_destroy(bb);
    rj_code_basic_block_release(bb);
}

rj_code_basic_block_list_destroy(bbs);
rj_code_object_destroy(obj);
rj_code_object_release(obj);
rj_code_executable_destroy(exec);
```

### Analysis (`code/analysis/`)

Register liveness and def-use chain analysis over GPU kernel CFGs.
Shared by DBT (for temporary register allocation during instruction
expansion) and DBI (for spill slot planning).

- **LivenessAnalysis** — Backward dataflow over `BasicBlock` CFG scoped
  to a single kernel. Provides `live_before(inst)`, `find_free_run()`,
  `find_free_sgpr_pair()` for safe register allocation in injected code.
- **DefUseChain** — Per-instruction def-use relationships including
  implicit operands (e.g., FLAT `saddr` SGPR pairs).

### Config (`config/`)

JSON configuration validated against FlatBuffers schemas. The
`load_config` functions build the full GPU topology (SoC, XCDs, SEs,
CUs, caches, memory) from a single JSON file.

See [configuration.md](configuration.md) for the config format.

### CLI (`tools/rocjitsu/`)

Three execution modes:

- **Local** — In-process simulation via LD_PRELOAD
- **Daemon** — Fork a daemon server, then exec the application
- **Attach** — Connect to a running daemon

See [rocjitsu-cli.md](rocjitsu-cli.md) for the daemon RPC protocol.
