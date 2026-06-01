# Rocjitsu CLI and Transport Layer Design

## Overview

The `rocjitsu` CLI is the primary entry point for running applications on
the simulated GPU. It operates in two modes:

- **Local mode**: Sets `LD_PRELOAD` and `execve`s the target application.
  The simulator runs in-process via the interposer.
- **Daemon mode**: Spawns a daemon child process that hosts the simulation
  engine, then `LD_PRELOAD` + `execve`s the target. Client processes
  communicate with the daemon via RPC over a Unix domain socket. GPU memory
  is shared using memfds passed via SCM_RIGHTS.

Daemon mode supports vLLM's multiprocessing spawn, torchrun, torch.distributed,
and NCCL --- workloads where multiple processes share a single simulated GPU.

## Architecture

```
Client Process (HIP application)          Daemon Process
+-----------------------------------+     +-----------------------------------+
| HIP Runtime                       |     | SimulationEngine (background)     |
| ROCR (HSA Runtime)                |     |   VirtualMachine (topology root)  |
| libhsakmt (KFD thunk)             |     |     SoC -> XCD -> CP, CUs, L2     |
|   |                               |     |   SimulatedDriver (KFD ioctls)    |
|   v                               |     |     EventState, allocations       |
| LD_PRELOAD interposer             |     +-----------------------------------+
|   |-- open("/dev/kfd")            |                    |
|   |     RemoteDriver (RPC stub) --+----- Unix socket --+
|   |-- ioctl(kfd_fd, ...)          |     (RPC_IOCTL)
|   |-- mmap(drm_fd, offset)        |     (RPC_MMAP + SCM_RIGHTS)
|   |-- close(kfd_fd)               |     (RPC_CLOSE)
|   `-- madvise/mprotect/munmap     |
+-----------------------------------+
```

### Ownership Chain

```
rj_vm_t (C API handle)
  |-- engine (unique_ptr<SimulationEngine>)
  |     `-- topology
  |           `-- root (unique_ptr<VirtualMachine>)
  |                 |-- SoC (adopted children: XCD, GpuMemory, ...)
  |                 `-- driver_ (unique_ptr<SimulatedDriver>)
  `-- engine_config
```

No circular ownership. Destroying `rj_vm_t` destroys the engine, which
destroys the topology root (VirtualMachine), which destroys the driver.

### Process Modes

| Mode | Entry Point | Driver | Engine |
|------|------------|--------|--------|
| Local | `rocjitsu -- ./app` | SimulatedDriver (in-process) | Background thread |
| Daemon (server) | `rocjitsu --daemon -- ./app` | SimulatedDriver (in-process) | Background thread |
| Daemon (client) | LD_PRELOAD interposer | RemoteDriver (RPC stub) | In daemon process |

## RPC Protocol

### Message Format

All messages use a fixed 16-byte header followed by an opcode-specific payload:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|    opcode     |   reserved    |          request_id           |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                        payload_bytes                          |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                           result                              |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

### Opcodes

| Opcode | Direction | Payload | Memfd |
|--------|-----------|---------|-------|
| `RPC_HANDSHAKE` | C->D->C | `RpcHandshakeResponse` + topology path | No |
| `RPC_CLOSE` | C->D->C | None | No |
| `RPC_MMAP` | C->D->C | `RpcMmapRequest` / `RpcMmapResponse` | Optional |
| `RPC_MUNMAP` | C->D->C | `RpcMunmapRequest` | No |
| `RPC_IOCTL` | C->D->C | `RpcIoctlRequest` + raw args + inlined arrays | Optional |

### Embedded Pointer Serialization

KFD ioctls like `WAIT_EVENTS`, `MAP_MEMORY_TO_GPU`, and
`GET_PROCESS_APERTURES_NEW` contain pointer fields that reference inline
arrays. The client serializes these by inlining the pointed-to array after
the struct. The daemon reconstructs the pointer to its local buffer. The
client saves and restores original pointers across the RPC round-trip.

### File Descriptor Passing

Memfds are passed via `sendmsg()`/`recvmsg()` with `SCM_RIGHTS` ancillary
data. This is the standard Unix mechanism for cross-process file descriptor
sharing (same pattern as QEMU vhost-user and CRIU).

Two RPC operations carry memfds:
- `RPC_IOCTL` (ALLOC_MEMORY response) --- allocation backing memfd
- `RPC_MMAP` (response) --- doorbell and event page memfds

### Thread Safety

ROCR is heavily multithreaded. The `rpc_mutex_` in `RemoteDriver` serializes
all send+recv pairs. `WAIT_EVENTS` uses client-side polling with `ppoll` on
a shutdown `eventfd` between daemon timeout=0 polls, avoiding mutex starvation.

## Memory Sharing (Same-VA Architecture)

Both client and daemon map every memfd at the **same virtual address** using
MAP_FIXED. GPU VAs are in the GPUVM aperture (`0x1000000000` - `0x3FFFFFFFFFFF`),
which doesn't collide with either process's normal address space.

This means `reinterpret_cast<void*>(gpu_va)` works in both processes ---
AQL packets, kernarg pointers, signal handles, ring buffers, and rptr/wptr
all resolve correctly without VA translation.

### Allocation Flow

```
Client                                     Daemon
------                                     ------
1. ROCR: fmm_allocate()
   mmap(MAP_PRIVATE|MAP_NORESERVE)

2. ROCR: hsaKmtAllocMemory()
   RemoteDriver::send_ioctl(ALLOC)  ---->  alloc_memory_ioctl()
                                            memfd_create + ftruncate + fallocate
                                            MAP_FIXED memfd at VA (proactive)
                                    <----  response + memfd (SCM_RIGHTS)
   [USERPTR: mincore+copy, MAP_FIXED]

3. ROCR: fmm_map_to_cpu()
   mmap(drm_fd, MAP_FIXED, offset)
   interposer routes DRM fd     ---->     mmap() -> already mapped, return
                                    <----  response
```

### Proactive Mapping

The daemon MAP_FIXED maps all allocations at ALLOC_MEMORY time, not at
mmap time. This prevents a race where the CP thread dereferences kernarg
pointers before the client's fmm_map_to_cpu RPC arrives.

### USERPTR Sharing

For USERPTR allocations (ring buffer, rptr/wptr), the client receives the
daemon's memfd, copies committed pages via mincore, then MAP_FIXED replaces
the anonymous pages with the shared memfd. The interposer suppresses
MADV_HUGEPAGE and MADV_DONTFORK on GPUVM addresses to prevent kernel 6.17
shmem fault failures.

## Transport Layer

### Abstract Interface

```cpp
class Transport {
public:
  virtual bool send(const void *data, size_t len) = 0;
  virtual bool recv(void *data, size_t len) = 0;
  virtual ssize_t send_with_handle(const void *data, size_t len, int handle) = 0;
  virtual ssize_t recv_with_handle(void *data, size_t len,
                                   int *handle_out, size_t *num_handles) = 0;
  virtual void close() = 0;
};
```

### Unix Implementation

`UnixTransport` wraps a Unix domain socket with SCM_RIGHTS for handle passing.
Static factory methods:
- `UnixTransport::listen(endpoint)` --- create a listening socket
- `UnixTransport::connect(endpoint)` --- connect to a daemon
- `transport->accept()` --- accept a client connection

The endpoint is `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock` by default.

### Platform Portability

The transport abstraction enables future Windows support:
- Unix: Unix domain socket + SCM_RIGHTS
- Windows: Named pipe + DuplicateHandle

The RPC message format (header + payload) is platform-independent. Only the
transport and shared-memory handle mechanism change per platform.

## Event System

### Auto-Reset Semantics in Daemon Mode

In daemon mode, WAIT_EVENTS uses timeout=0 polling via RPC. Multiple client
threads poll the same event independently. Auto-reset is skipped on timeout=0
polls to prevent one poller from "stealing" the event signal from others.
This matches the real kernel's behavior where `wake_up_all()` wakes all
waiters simultaneously.

### Shutdown Sequence

1. ROCR calls SET_EVENT on wake signals during `AsyncEventsControl::Shutdown()`
2. All client-side pollers see the event (no auto-reset on polls)
3. ROCR's async threads exit their loops, thread joins succeed
4. ROCR calls DESTROY_QUEUE, DESTROY_EVENT, close(kfd_fd)
5. `RemoteDriver::close()` sends RPC_CLOSE to daemon
6. Daemon calls `SimulatedDriver::close()` -> `notify_closing()`

## Fd Management

### Fd Reservation (DMTCP Pattern)

The driver reserves a contiguous fd range (4000-4255) for internal memfds.
`claim_fd(real_fd)` moves a real fd into this range via `dup2`. The
interposer's `close()` checks `owns_reserved_fd(fd)` to protect internal
fds from ROCR's normal fd lifecycle.

### Fd Protection

ROCR opens and closes many fds during init (topology files, /proc, DRM
render nodes). Without protection, a transient file's fd number could
collide with an allocation memfd. The `owned_fds_` set in `SimulatedDriver`
tracks all internal memfds, and the interposer suppresses close() on
protected fds.

## Limitations

### Single Client

The daemon uses an atomic guard to reject concurrent connections. Same-VA
mapping prevents multiple clients from using overlapping GPUVM ranges.
Multi-client support requires per-client VA translation tables in the CP.

### Kernel 6.17 shmem Bug

MAP_SHARED at GPUVM addresses can trigger SIGBUS on kernel 6.17 due to a
shmem large folio regression. The interposer suppresses MADV_HUGEPAGE and
MADV_DONTFORK on GPUVM addresses as a workaround. Upstream fixes are in
kernel 6.12.59+ and 6.15+.

## File Map

| File | Role |
|------|------|
| `rpc.h` | Protocol types, send/recv helpers |
| `transport.h/.cpp` | Abstract transport + Unix socket implementation |
| `remote_driver.h/.cpp` | Client-side RPC stub |
| `simulated_driver.h/.cpp` | KFD ioctl dispatch, allocation table, events |
| `interposer.cpp` | LD_PRELOAD syscall intercepts |
| `events.h/.cpp` | KFD event subsystem (create, set, wait, destroy) |
| `tools/rocjitsu/main.cpp` | CLI entry point, RPC dispatch loop (daemon mode) |
| `tests/daemon_test.cpp` | Gtest fixture for daemon end-to-end tests |
| `virtual_machine.h/.cpp` | VM component owning SoC and driver |

## Future Work

- **Multi-client**: Per-client state isolation, VA translation tables
- **Windows**: Named pipe transport, KMD escape driver backend
- **RDMA**: Rack-scale transport for distributed simulation
- **Platform-independent commands**: Abstract command enum for cross-platform
