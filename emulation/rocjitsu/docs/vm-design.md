# AMDGPU Virtual Machine Design

An AMDGPU virtual machine built on the simdojo simulation framework. Models
a complete SoC hierarchy, from command processors and shader engines down to
compute units, wavefronts, and register files. Runs within simdojo's unified
PDES epoch loop, supporting both interactive stepping and continuous
execution.

## File Overview

| File | Purpose |
|------|---------|
| `soc.h/cpp` | SoC: root of one GPU's hardware hierarchy, owning XCDs, IODs, and shared GPU memory. Installed as the topology root only by a caller driving the config loader directly, as the tests do |
| `virtual_machine.h/cpp` | VirtualMachine: the runtime topology root, owns the SoCs and the simulated KFD |
| `kmd/linux/simulated_kfd.h/cpp` | SimulatedKfd: the KMD interface, serving the KFD ioctl surface |
| `amdgpu/iod.h/cpp` | IOD: I/O die with memory-side cache and HBM controllers |
| `amdgpu/memory_side_cache.h/cpp` | Memory-side cache component between L2 and HBM |
| `amdgpu/hbm_controller.h` | HBM memory controller wrapping GpuMemory |
| `rj_vm.cpp` | C API: create, step, run, checkpoint |
| `amdgpu/command_processor.h/cpp` | CP: dispatch packets, doorbell loop |
| `amdgpu/compute_unit.h/cpp` | CU: wavefront slots, register files, execution |
| `amdgpu/shader_engine.h/cpp` | SE: container of compute units |
| `amdgpu/xcd.h/cpp` | XCD: CP + shader engines |
| `amdgpu/wavefront.h/cpp` | Wavefront: ISA-specific thread state |
| `amdgpu/gpu_memory.h` | GpuMemory: flat address space wrapper |

---

## Component Hierarchy

```
SimulationEngine                           (simdojo - owns topology)
└── Topology
    └── VirtualMachine                     (CompositeComponent - topology root)
        ├── SimulatedKfd driver_            (owned member, not a child component)
        └── gpu[0..G] ("gpu0"..)           (CompositeComponent, one per GPU)
            ├── SoC ("soc")                 (owned, but its children were adopted
            │                                by the gpuN wrapper above, so it is
            │                                empty in the tree; every SoC is named
            │                                "soc", and find_child returns the first
            │                                match, so the wrapper is what keeps the
            │                                paths unique)
            ├── GpuMemory ("memory")        (shared across all XCDs)
            ├── Iod[0..I] ("iod0"..)       (CompositeComponent - memory-side cache + HBM controllers)
            └── Xcd[0..N] ("xcd0"..)       (CompositeComponent)
                ├── CommandProcessor ("cp") (Component - event-driven dispatch)
                └── ShaderEngine[0..M]      (CompositeComponent)
                    └── ComputeUnit[0..K]   (CompositeComponent - register files + wavefront slots)
```

The `VirtualMachine` is a `simdojo::CompositeComponent` set as the topology root,
and it owns the simulated KFD alongside its SoCs. The simulation infrastructure
(engine, topology, partitioning) is managed by `SimulationEngine`; the SoC
represents the hardware being modeled.

---

## Driver

The Driver models the kernel-mode driver (KMD) interface presented to a
user-mode driver (e.g., rocr). It is an abstract interface rather than a member
of `SoC`; the concrete implementation is the simulated KFD, which the
interception layer routes a guest process's driver syscalls to.

It therefore exposes a syscall surface rather than a dispatch entry point:

- `open()` / `close()` - open and close the driver device.
- `ioctl(request, arg)` - the KFD ioctl surface, including the queue creation
  path described under *Queue ownership and XCD fan-out* below.
- `mmap()` / `munmap()` - map driver-owned memory, such as the doorbell page.

Work does not reach the hardware through this interface. A process creates a HW
queue through `ioctl`, and thereafter submits by writing that queue's ring and
ringing its doorbell, which the owning XCD's command processor polls.

---

## Command Processor

The CP reads dispatch packets and distributes wavefronts across registered
compute units in round-robin order. Each XCD has its own CP, and a CP is wired
only to its own XCD's compute units.

### Queue ownership and XCD fan-out

`SoC::assign_queue_owner_cp()` rotates HW queues across the XCDs. The XCD it
returns *owns* the queue: it alone reads the ring, advances the read pointer, and
holds each dispatch's completion signal. It is not the only XCD that runs the
work.

`HwQueue::xcd_fanout` is the switch. A queue that sets it is replicated onto
every XCD at registration; a queue that does not keeps the whole grid on the CP
it was registered against. The KFD path sets it for compute queues and leaves it
clear for SDMA, which belongs to one engine; a test queue opts in through
`AqlQueue(..., xcd_fanout=true)` and leaves it clear otherwise. The creation path
is not itself the switch — either path can produce either kind of queue.

Each dispatch on a fanned-out queue is split so that **XCD i runs the grid chunks
congruent to i modulo the XCD count** — round-robin, one workgroup at a time. The
chunk is one workgroup, except for a clustered dispatch where it is a whole
cluster, so cluster peers stay co-resident on the XCD whose LDS they share.

For those dispatches, the rank is the XCD's own index rather than its position
relative to the queue's owner, so the workgroup-to-XCD mapping does not depend on
which XCD the queue landed on. That matters because kernels swizzle their
workgroup index for cache locality assuming exactly this permutation. A dispatch
that is not fanned out makes no such claim: it runs wholly on its owner.

A grid with fewer chunks than XCDs is still split; the XCDs that get nothing take
an empty share. Those empty shares are what keep every XCD's copy of the queue in
step, because `barrier_satisfied()` reads ordering from the entries sitting ahead
of a barrier'd packet, and an XCD that never heard about a packet would start the
next one early. An empty share is still a *kernel* dispatch -- `is_non_kernel()`
is false for it, since the packet kind is recorded rather than inferred from the
workgroup count -- and it completes immediately because its `total_wgs` is zero.
Like every other shard it is then held at the head until the grid retires.

Every packet on a fanned-out queue reaches every XCD; what differs is how. A
kernel dispatch is **divided** -- each XCD takes the chunks described above. A
packet that runs no shader has no grid to divide, so it is **copied** whole. The
full set as it stands:

| Packet | On a fanned-out queue |
|---|---|
| AQL kernel dispatch | Divided: XCD i takes the chunks congruent to i |
| AMD extended kernel dispatch (clustered) | Divided, with a whole cluster as the chunk |
| BarrierAND / BarrierOR | Copied whole to every XCD |
| AMD barrier-value | Copied whole to every XCD |
| AMD PM4 IB | Copied whole to every XCD |

`replicate_non_kernel_entry()` does the copying. A replica's entry list is
therefore the owner's whole sequence rather than a subsequence of it, and that is
what makes the ordering `barrier_satisfied()` reads from the entries sitting ahead
of a barrier'd packet a device-wide ordering rather than a local one. Adding a
packet type means deciding which column it belongs in; a type that is neither
divided nor copied would silently let a replica run ahead of the owner.

A copy carries no completion signal and no dispatch-level callbacks. A packet is
owed exactly one of each however many XCDs end up running it, and the XCD that
read the packet keeps that duty -- for a copied packet exactly as for a shard of a
divided one.

Replicas never read the ring and never poll a doorbell; shards arrive from the
owning XCD through the engine's cross-thread event queue, so the handoff is safe
when `partition_topology_by_xcds` has put each XCD on its own worker thread. A
packet's acquire fence travels with the shard and each XCD applies it to its own
caches on its own thread, since one XCD may not touch another's.

### Cross-XCD completion

A fanned-out dispatch retires once, after the last workgroup anywhere on the
device. Each XCD counts its own share, flushes its own caches, then publishes the
share to a `GridCompletion` counter shared by all shards. The owning XCD holds the
head of its queue until that counter covers the grid, then fires the completion
signal. Publishing releases and the owner's check acquires, so no XCD's results
are still sitting in its caches when the signal is written.

An XCD parked on a share it has already finished re-arms a re-check for as long as
the grid is still outstanding. Which timer carries that re-check depends on whether
the CP has a doorbell poll thread, and only one of the two paths uses the engine's
event queue: a CP that polls its own host-accessible queues just sets a pending flag
that the poll thread re-reads at its 100us cadence, while a CP with no poll thread --
an internal test queue, or a fan-out replica, which holds host-accessible queues but
never polls them -- schedules an event on its own queue instead. That event backs off
exponentially, up to a cap, and resets as soon as work arrives; at one tick it is a
spin, and with one replica per XCD waiting on every grid it was a 12x slowdown on
real workloads. The XCD that retires
the grid does wake every XCD, but that wake travels the engine's cross-thread
async queue, which neither contributes to LBTS nor counts as outstanding work
when the engine tests for termination: with one partition per XCD, every
partition can go quiescent in the same epoch the wake is deposited, and the run
ends before the next epoch delivers it. The re-check keeps the waiting
partition's next-event time finite, which leaves the wake an optimization rather
than the only thing standing between the grid retiring and the signal being
written.

A peer shard carries no completion signal and does not report the queue idle. The
packet-scoped plugin callbacks are emitted once for the packet rather than once per
share, but by two different mechanisms. `onAmdgpuDispatchPacketProcessed` and
`onAmdgpuDispatchExecutionEnd` belong to the owner: a peer skips them outright.
`onAmdgpuDispatchExecutionBegin` is not owner-only -- it is emitted by whichever XCD
first places a workgroup, and `GridCompletion::claim_execution_begin()` suppresses
every later caller. The distinction is load-bearing when the owner's own share is
empty: the callback is necessarily emitted by a peer there, and an owner-only rule
would drop it. The workgroup and wavefront callbacks are not skipped either: every
XCD still reports the work it actually ran.

Destroying a fan-out queue discards any share that has not yet been published:
the teardown runs on the caller's thread and so cannot flush a partition's compute
units, and publishing without that write-back would let the owner signal with an
XCD's results still cached. Dropping them is sound **only because a fan-out queue
is always destroyed on every XCD at once** — the KFD paths sweep every command
processor and an owner cascades to its replicas — so no XCD is ever left holding a
grid that can no longer retire. A future change that tears one XCD's copy down
alone would strand the owner.

A shard still sitting in a peer's inbox when its replica is destroyed is dropped
for the same reason and on the same argument. It has not run and its XCD's caches
have not been written back, so crediting it would be worse than losing it: the
owner would retire the grid and fire the completion signal for workgroups that
never executed. KFD teardown is what reaches this window, since it removes
replicas in XCD order while a later owner is still registered.

### Event-Driven Dispatch

The CP is event-driven, and work reaches it only through a registered queue --
there is no submit entry point on the CP itself:

- `register_queue(HwQueue)` / `unregister_queue(...)` - attach and detach a HW
  queue. A host-accessible queue also starts the doorbell poll thread -- unless it
  is a fan-out replica. Peer copies of a fanned-out queue are host-accessible too,
  but their work arrives as dispatch shards from the owning XCD; a replica that read
  the ring would dispatch the same packets once per XCD, so only the owner polls.
- `handle_doorbell(...)` - the doorbell event handler. It fetches newly written
  packets from each queue's ring, then runs the dispatch loop.
- `step()` - one engine step of that same dispatch loop, for the internal test
  queues that are driven by `run()`/`step()` rather than by a poll thread.

A producer writes an AQL packet into the ring and rings the doorbell; the poll
thread notices the change and fires the doorbell event. For each workgroup in
this XCD's share -- the whole grid unless the queue fans out, in which case the
owner hands each peer its shard through `accept_fanout_shard()` and every CP
walks only its own -- the CP calls `dispatch_wf()` on the next CU in round-robin
order. `dispatch_wf()`
self-schedules the CU's tick via `schedule_work()`; there is no separate
`activate()` call.

Each CU runs its dispatched wavefronts independently. The CU is
self-driving: `dispatch_wf()` calls `schedule_work()`, which schedules a
tick event (only when the CU has runnable wavefronts) that calls
`execute_quantum()`. A quantum executes up to `kFunctionalQuantum`
instructions, but may yield early when a wavefront requests it (e.g.
`s_sleep`, a vendor-dependency retry). The tick reschedules itself at
`now + max(1, last_quantum_executed_)` — i.e. by the work actually
executed, so an early yield resumes promptly instead of leaping a full
quantum, while `max(1, ...)` keeps the event strictly in the future.
Since functional mode is 1 CPI, ticks advance proportionally to
instruction count. The quantum allows CU events to interleave,
guaranteeing forward progress for inter-CU synchronization patterns such
as spin-locks or semaphore acquire/release on global memory.

A wavefront that reaches `s_endpgm` halts: it frees its SGPR/VGPR
resources immediately and notifies the CP of workgroup completion (there
is no separate lazy retirement pass). When a CU has no resident
wavefronts it stops scheduling and fires its `on_idle` callback. When all
CUs are idle and no packets remain, the CP signals completion via
`engine()->primary_release()`.

---

## Dispatch Packet Flow

```
producer writes an AQL packet into the ring, then rings the doorbell
  └── doorbell poll thread observes the change -> doorbell event
        └── cp->handle_doorbell():
              fetch_from_queue() reads the packet and builds a DispatchEntry
                (a fanned-out packet also hands each peer XCD its shard here)
              then, for each workgroup of this XCD's share:
                cu = next CU (round-robin)
                cu->dispatch_wf(wg_id, pc, sgprs, vgprs)
                  └── find idle slot, allocate SGPR/VGPR blocks
                      initialize wavefront state (pc, wg_id)
                      schedule_work() -> tick event (self-driving)
```

The in-flight record the CP builds from that packet is a `DispatchEntry`, which
carries among other things:
- `kernel_entry_pc` - byte address of kernel code in GPU memory
- `total_wgs` - workgroups to launch, narrowed to this XCD's share once sharded
- `wfs_per_workgroup` - wavefronts per workgroup
- `sgprs_per_wf` / `vgprs_per_wf` - register requirements (from code object)

Wavefronts are distributed round-robin across CUs within the XCD. Each CU
allocates a contiguous block in its physical SGPR and VGPR files for the
wavefront. For a fanned-out dispatch this walk covers only the XCD's own share
of the grid; see *Queue ownership and XCD fan-out* above.

---

## Memory Hierarchy and Coherence

Each CU has private L1 scalar (K$) and L1 vector (V$) caches backed by a
shared L2 per XCD. The memory type (Mtype), derived from instruction
encoding bits (sc0/sc1/nt), controls caching behavior:

| Mtype | sc1 | sc0 | L1 Behavior | L2 Behavior |
|-------|-----|-----|-------------|-------------|
| RW    |  0  |  0  | Cached, write-through | Write-back |
| CC    |  0  |  1  | Invalidate-on-read, write-through | Write-through to HBM |
| UC    |  1  |  0  | Bypass | Bypass |
| NT    |  0  |  0+nt| Bypass L1 | Cached |

**CC (coherently cacheable)** loads invalidate the L1 line before
refetching from L2, matching real SC0/GLC hardware behavior. This
ensures that stores from other CUs (which write through L1 to L2)
are visible to polling loops.

**Atomic operations** (`flat_atomic_*`) bypass L1 entirely and perform
read-modify-write at L2. The old value is returned to vdst when
SC0/GLC is set. The L1 line is invalidated after the atomic to prevent
stale reads. Supported integer atomics: swap, cmpswap, add, sub,
smin/umin, smax/umax, and, or, xor, inc, dec.

**Cache management instructions:**
- `s_dcache_inv` / `s_dcache_inv_vol` — invalidate the L1 scalar cache
- `s_gl1_inv` — invalidate the L1 vector cache

---

## Execution Modes

### Interactive Stepping (`rj_vm_step`)

Synchronous, single-threaded. The caller drives execution one tick at a
time:

```
rj_vm_step(vm, &active)
  └── engine.step()         process all events at next timestamp
        └── CP doorbell event fires → cp->step() drains dispatch queue
            CU tick events fire → execute one instruction per wavefront
```

Returns `active=1` while any wavefront is still executing. The engine is
built during `rj_vm_create()`, not on first step.

### Continuous Execution (`rj_vm_run`)

The simulation thread runs `engine.run()`, which drains the event queue
continuously. The main thread injects work via `schedule_event_async()`:

```
Main thread                          Simulation thread
───────────                          ─────────────────
                                     engine.run()
                                       └── epoch loop processes events
                                             │
write ring + ring doorbell ────────►  doorbell event → CP fetches the packet
  │                                          │
  │                                     handle_doorbell() dispatches wavefronts
  │                                     CU tick events execute instructions
  │                                          │
rj_vm_request_exit()     ──────────►  done_ set, workers stop
  │                                        (ends the epoch loop; SimulatedKfd::
  │                                         close() only tears down KFD process
  │                                         state and does not stop the engine)
  │
sim_thread.join()  ◄─────────────────────    engine shuts down components
  │
engine.shutdown()
```

In single-threaded mode, the engine drains events in timestamp order
without LBTS synchronization. The CP schedules doorbell events during
`startup()` for pre-loaded packets and via `schedule_event_async()` for
external submissions.

---

## Simulation Integration

The hardware hierarchy plugs into simdojo's `SimulationEngine` directly. The C
API layer (`rj_vm.cpp`) owns the engine and wires that hierarchy into the
topology, under a `VirtualMachine` root:

1. **Construction** - The config loader parses a declarative JSON topology
   config and creates a `SoC` with engine configuration. What becomes the
   topology root depends on the entry point: `create_from_loaded()` wraps the
   loaded SoC -- or SoCs, for a multi-GPU config -- in a `VirtualMachine` and
   installs that via `set_root()`, which is what owns `SimulatedKfd`. A caller
   driving the config loader itself, as the tests do, may instead install its
   `SoC` as the root directly.

2. **`create()`** - Partitions topology, initializes all components, and
   sets up the engine.

3. **`run()`** - Starts all components and drains events continuously.
   In single-threaded mode, events are processed
   in timestamp order until the queue empties and termination conditions are
   met. In multi-threaded mode, workers run LBTS-synchronized epoch loops.

4. **`step()`** - Processes all events at the next timestamp (one tick step).
   Returns whether the simulation can continue.

5. **`shutdown()`** - Calls `shutdown()` on all components, tears down engine.

After `run()` returns, `last_exit()` provides a `ExitStatus` with
the termination reason (`COMPLETED`, `EXIT_REQUEST`, `INTERRUPTED`), the
simulation tick, and a human-readable message. Components can call
`engine()->request_exit(reason, code)` to stop the simulation.

---

## KMD Emulation (`kmd/`)

The `kmd/linux/` layer makes a real ROCm stack (ROCR + libhsakmt) run against the
simulated GPU without any kernel driver. It is Linux-only and activated via
`LD_PRELOAD`.

### Architecture

```
ROCm application
  └── ROCR / HIP runtime
        └── libhsakmt
              ├── open("/dev/kfd")    ──►  interposer.cpp intercepts
              ├── ioctl(kfd_fd, …)   ──►  SimulatedKfd::ioctl()
              ├── mmap(kfd_fd, …)    ──►  SimulatedKfd::mmap()
              ├── fopen("/sys/…")    ──►  interposer.cpp redirects → Sysfs temp dir
              └── close(kfd_fd)      ──►  SimulatedKfd::close()
```

### Components

| File | Purpose |
|------|---------|
| `interposer.cpp` | LD_PRELOAD shim: intercepts `open`, `ioctl`, `mmap`, `munmap`, `fopen`, `close` via syscall |
| `simulated_kfd.h/cpp` | `SimulatedKfd`: handles all KFD ioctls, owns doorbell/event pages |
| `sysfs.h/cpp` | `Sysfs`: generates a per-process `/tmp/rocjitsu_topology_*` directory that ROCR reads instead of the real `/sys/devices/virtual/kfd/kfd/topology` |

### KFD ioctl surface

| ioctl | Handler | Notes |
|-------|---------|-------|
| `GET_VERSION` | `get_version_ioctl` | Returns KFD_IOCTL_MAJOR/MINOR_VERSION |
| `GET_PROCESS_APERTURES_NEW` | `get_process_apertures_ioctl` | Returns `gpu_apertures(ordinal)` — LDS/scratch shifted by `kApertureStride` per GPU, with per-instance `gpu_id` |
| `ACQUIRE_VM` | `acquire_vm_ioctl` | No-op (VM is always acquired) |
| `ALLOC_MEMORY_OF_GPU` | `alloc_memory_ioctl` | Allocates host memory, assigns GPU VA from a linear bump allocator |
| `FREE_MEMORY_OF_GPU` | `free_memory_ioctl` | Frees host memory, removes VA mapping |
| `MAP_MEMORY_TO_GPU` / `UNMAP` | map/unmap ioctls | No-op (host pointers serve as GPU VAs) |
| `CREATE_QUEUE` | `create_queue_ioctl` | Registers an AQL ring with the CP; deferred until doorbell page is mapped |
| `DESTROY_QUEUE` | `destroy_queue_ioctl` | Unregisters the ring from the CP |
| `CREATE_EVENT` | `create_event_ioctl` | Allocates a slot in the memfd-backed signal page |
| `DESTROY_EVENT` | `destroy_event_ioctl` | Removes event slot; wakes any WAIT_EVENTS callers |
| `SET_EVENT` | `set_event_ioctl` | Marks slot non-zero with `memory_order_release`; notifies waiters |
| `WAIT_EVENTS` | `wait_events_ioctl` | Waits up to 100ms then returns; simulates `wake_up_interruptible` |

### Signal event page

libhsakmt expects a memfd-backed page at the KFD mmap offset
`KFD_MMAP_TYPE_EVENTS | gpu_id`. Each 64-bit slot corresponds to one
`event_id`. libhsakmt polls `signal_page[event_id]` directly; non-zero means
the event is pending. `close()` sets all slots to 1 to unblock any polling
threads during shutdown.

---

## C API (`rj_vm.h`)

The C API wraps the VM behind an opaque `rj_vm_t` handle:

`rj_vm_t` is reference-counted (extends `RefCounted`). Use `rj_vm_retain`
and `rj_vm_release` to manage shared ownership; `rj_vm_destroy` is a
convenience wrapper that releases the last reference and tears down the VM.

| Function | Description |
|----------|-------------|
| `rj_vm_create()` | Load config from JSON file, build VM and engine |
| `rj_vm_create_from_string()` | Load config from JSON string |
| `rj_vm_retain()` | Increment reference count |
| `rj_vm_release()` | Decrement reference count; destroys when it reaches zero |
| `rj_vm_destroy()` | Tear down VM |
| `rj_vm_step()` | One interactive step |
| `rj_vm_run()` | Run to completion via driver open/close |
| `rj_vm_save_checkpoint()` | Serialize VM state to FlatBuffer |
| `rj_vm_restore_checkpoint()` | Restore VM from checkpoint file |

### Example

```c
#include <rocjitsu/rocjitsu.h>

rj_vm_t *vm = NULL;
rj_vm_create("configs/gfx950_mi355x.json", &vm);

uint64_t ticks = 0;
rj_vm_run(vm, &ticks);

rj_vm_destroy(vm);
```

Internal C++ code (tests, GUI) accesses the `SoC` directly via the
config loader (`config::load_config()` / `config::load_config_from_string()`).
The `rj_vm.h` header is a pure C API with opaque handles.
