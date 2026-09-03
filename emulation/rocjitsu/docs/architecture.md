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
│    GpuMemory (legacy-compatible backing)    │
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

- **CommandProcessor** — Monitors doorbells, fetches AQL packets from
  ring buffers, parses kernel descriptors, dispatches workgroups to CUs.
  Handles SDMA packets (copy, fill, GCR, HDP flush).
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
  execution models.
- **GpuMemory** — Backing storage and legacy compatibility implementation. It
  retains the older per-VMID mappings, passthrough mode, and shared-memfd
  behavior for the interposer path, but does not own PCI/VFIO routing or the
  GFX12 page-table policy.
- **SdmaQueueRunner** — Transport-neutral owner of one SDMA root ring's device
  cursor, immutable VM snapshot, wrapped fetch progress, resumable packet
  executor, retirement publication, and terminal state. Both the legacy CP and
  PCI SDMA block are thin adapters over this runner; neither owns a second SDMA
  fetch or retry state machine.
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
observers publish queue changes through `QueueService`; command-processor queue
registration is delivered through the CP inbox on the CP's execution context.
Observer callbacks, reset, and deferred-work draining must not run while the
libvfio-user `vfu_mutex_` is held, because those paths can re-enter transport or
simulation services.

The PCI models are adapters over the shared `GpuVm`, `CommandProcessor`, MES,
SDMA, and CU objects. They must not grow alternate CP/MES/SDMA execution
backends; new PCI-visible behavior should terminate at the narrow MMIO, DMA,
interrupt, address-space, or queue-service interface owned by the corresponding
core model.

The firmware-free non-AQL compute startup ring follows the same boundary. MES
selects its `GpuVm` address space and adapts the one permitted MMIO write, while
the transport-neutral `Pm4BootstrapExecutor` owns ring traversal and the exact
NOP/SET_UCONFIG_REG packet subset. It is deliberately not a general PM4 backend;
normal compute execution remains on the shared command processor.

SDMA follows the same composition rule. MMIO and MES decode queue configuration
and doorbells, while `SdmaQueueRunner` owns root-ring traversal and delegates
packet semantics to `SdmaExecutor`. A service attempt retains one `GpuVmAccess`
through fetch, execution, and read-pointer publication. Temporary backing or
transport unavailability resumes that exact state; malformed requests and
permanent faults terminate the queue without replaying already-retired packet
effects. The legacy CP uses the same runner and differs only in how it obtains
the producer cursor and schedules retries.

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
