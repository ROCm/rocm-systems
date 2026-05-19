# AMDGPU Virtual Machine Design

An AMDGPU virtual machine built on the simdojo simulation framework. Models
a complete SoC hierarchy, from command processors and shader engines down to
compute units, wavefronts, and register files. Runs within simdojo's unified
PDES epoch loop, supporting both interactive stepping and continuous
execution.

## File Overview

| File | Purpose |
|------|---------|
| `soc.h/cpp` | SoC: top-level topology root, owns XCDs, IODs, and shared GPU memory |
| `driver.h/cpp` | Driver: KMD interface, lifecycle state machine |
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
SimulationEngine                       (simdojo - owns topology)
└── Topology
    └── SoC ("gpu_soc")               (CompositeComponent - topology root)
        ├── Driver driver_             (value member, not a child component)
        ├── GpuMemory ("memory")       (shared across all XCDs)
        ├── Iod[0..I] ("iod0"..)      (CompositeComponent - memory-side cache + HBM controllers)
        └── Xcd[0..N] ("xcd0"..)      (CompositeComponent)
            ├── CommandProcessor ("cp") (Component - event-driven dispatch)
            └── ShaderEngine[0..M]     (CompositeComponent)
                └── ComputeUnit[0..K]  (CompositeComponent - register files + wavefront slots)
```

The SoC is a `simdojo::CompositeComponent` set as the topology root. The
simulation infrastructure (engine, topology, partitioning) is managed by
`SimulationEngine`; the SoC represents the hardware being modeled.

---

## Driver

The Driver models the kernel-mode driver (KMD) interface presented to a
user-mode driver (e.g., rocr). It is owned as a value member of
SoC (`Driver driver_{*this}`) - the driver is part of the hardware, not
external software.

The Driver is a thin bridge between the application and the simulation
engine. It exposes two methods:

- `submit(DispatchPacket, uint32_t xcd_idx)` - enqueues a dispatch packet
  on the target XCD's command processor doorbell queue.
- `close()` - shuts down the simulation by calling
  `engine()->request_exit()`.

---

## Command Processor

The CP reads dispatch packets and distributes wavefronts across registered
compute units in round-robin order.

### Event-Driven Dispatch

The CP is event-driven. During `startup()`, it schedules a doorbell event
for any pre-loaded packets. Each doorbell event triggers `step()`, which
processes one dispatch packet: for each workgroup, it calls `dispatch_wf()`
on the next CU in round-robin order, then calls `activate()` on that CU.

- `enqueue(packet)` - pre-loads packets without triggering dispatch (used
  by config loader and tests)
- `submit(packet)` - thread-safe; appends packet and schedules an async
  doorbell event via `schedule_event_now()`
- `activate()` - registers the CP as a primary component and begins
  processing doorbell events

Each CU runs its dispatched wavefronts independently. In functional mode,
a work event calls `advance()`, which executes up to `functional_quantum`
instructions (or all remaining if quantum is 0) before yielding back to
the event loop. A non-zero quantum allows CU events to interleave,
guaranteeing forward progress for inter-CU synchronization patterns such
as spin-locks or semaphore acquire/release on global memory. When a CU
becomes idle, it fires its `on_idle` callback. When all CUs are idle
and no packets remain, the CP signals completion via
`engine()->primary_release()`.

---

## Dispatch Packet Flow

```
Config loader / Driver::submit()
  └── cp->enqueue(packet)  or  cp->submit(packet) [+ async doorbell event]
        └── cp->step() processes one packet:
              for each workgroup:
                cu = next CU (round-robin)
                cu->dispatch_wf(wg_id, pc, sgprs, vgprs)
                  └── find idle slot, allocate SGPR/VGPR blocks
                      initialize wavefront state (pc, wg_id)
                cu->activate()
                  └── schedule work event (functional) or resume clock (clocked)
```

A `DispatchPacket` specifies:
- `kernel_entry_pc` - byte address of kernel code in GPU memory
- `workgroup_count` - number of workgroups to launch
- `wfs_per_workgroup` - wavefronts per workgroup
- `sgprs_per_wf` / `vgprs_per_wf` - register requirements (from code object)

Wavefronts are distributed round-robin across CUs within the XCD. Each CU
allocates a contiguous block in its physical SGPR and VGPR files for the
wavefront.

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
driver().submit(packet)  ──────────►  async event → CP doorbell fires
  │                                          │
  │                                     cp->step() dispatches wavefronts
  │                                     CU tick events execute instructions
  │                                          │
driver().close()         ──────────►  done_ set, workers stop
  │                                        (close() calls engine()->request_exit())
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

The SoC plugs into simdojo's `SimulationEngine` directly. The C API layer
(`rj_vm.cpp`) owns the engine and wires the SoC into the topology:

1. **Construction** - The config loader parses a declarative JSON topology
   config and creates a `SoC` with engine configuration. The C API sets the
   SoC as the topology root via `engine.topology().set_root(std::move(soc))`.

2. **`build()`** - Partitions topology, initializes all components, and
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
              ├── ioctl(kfd_fd, …)   ──►  SimulatedDriver::ioctl()
              ├── mmap(kfd_fd, …)    ──►  SimulatedDriver::mmap()
              ├── fopen("/sys/…")    ──►  interposer.cpp redirects → Sysfs temp dir
              └── close(kfd_fd)      ──►  SimulatedDriver::close()
```

### Components

| File | Purpose |
|------|---------|
| `interposer.cpp` | LD_PRELOAD shim: intercepts `open`, `ioctl`, `mmap`, `munmap`, `fopen`, `close` via syscall |
| `simulated_driver.h/cpp` | `SimulatedDriver`: handles all KFD ioctls, owns doorbell/event pages |
| `sysfs.h/cpp` | `Sysfs`: generates a per-process `/tmp/rocjitsu_topology_*` directory that ROCR reads instead of the real `/sys/devices/virtual/kfd/kfd/topology` |

### KFD ioctl surface

| ioctl | Handler | Notes |
|-------|---------|-------|
| `GET_VERSION` | `get_version_ioctl` | Returns KFD_IOCTL_MAJOR/MINOR_VERSION |
| `GET_PROCESS_APERTURES_NEW` | `get_apertures_ioctl` | Returns `default_apertures_` with per-instance `gpu_id` |
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

### Known limitations (P1)

- `WAIT_EVENTS` always returns via timeout (100ms cap). The CP does not yet
  fire an interrupt to drive `set_event_ioctl`. Signal completion is detected
  by ROCR via direct memory polling of the signal value instead.
- `AVAILABLE_MEMORY` ioctl returns uninitialized output.
- `wait_events` ignores specific event IDs — wakes on any event.

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

Internal C++ code (tests, GUI) accesses the `SoC` directly via the
config loader (`config::load_config()` / `config::load_config_from_string()`).
The `rj_vm.h` header is a pure C API with opaque handles.
