# Rocjitsu CLI and Transport Layer Design

## Overview

The `rocjitsu` CLI is the primary entry point for running applications on
the simulated GPU. It supports these modes:

- **Local mode**: Sets `LD_PRELOAD` and `execve`s the target application.
  The simulator runs in-process via the interposer.
- **Preload-only mode**: Preloads the interposer for a Python application,
  which selects its simulation configuration with `rocjitsu.enable()` before
  GPU discovery.
- **Daemon mode**: Spawns a daemon child process that hosts the simulation
  engine, then `LD_PRELOAD` + `execve`s the target. Client processes
  communicate with the daemon via RPC over a Unix domain socket. GPU memory
  is shared using backing descriptors passed via SCM_RIGHTS.
- **Attach mode**: Launches an application against an already-running daemon.
- **VFIO-user mode**: Serves each configured local GPU as a PCI function to an
  external VMM through its own AF_UNIX socket. This mode is available only in
  builds configured with `ROCJITSU_ENABLE_VFIO=ON`.

Daemon mode supports vLLM's multiprocessing spawn, torchrun, torch.distributed,
and NCCL --- workloads where multiple processes share a single simulated GPU.

Daemon mode currently rejects dma-buf export with `EOPNOTSUPP`; its RPC protocol
does not transfer exported descriptors. HIP VMM allocation (`hipMemCreate`)
depends on that export even with `hipMemHandleTypeNone`, so ROCr may report it
as out of memory. Use local mode for VMM allocation and its shareable-handle APIs.
Ordinary daemon `hipMalloc`, copies, and kernel dispatch remain supported.

For HIP IPC between daemon clients (`hipIpcGetMemHandle` and
`hipIpcOpenMemHandle`), set `HSA_ENABLE_IPC_MODE_LEGACY=1` before launching the
clients to select the supported legacy KFD IPC path. The current runtime's
default dma-buf IPC path fails at handle creation in both local and daemon
mode; switching to local mode does not resolve that limitation.

### The fork boundary

A child created with `fork()` before the parent has started a GPU backend can
initialize its own rocJITsu context. This supports Python forkserver workers.
The child handler replaces unused interposer bookkeeping and the host mapping
lock, while preserving the invocation metadata used to find the configuration
and daemon. It does not acquire or destroy inherited locks.

**After the parent starts a backend, forked children must `exec` before using
rocJITsu.** The child may inherit locks held by threads that no longer exist, so
GPU endpoint opens fail with `ENODEV` until `exec` initializes a fresh context.
This restriction applies to local and daemon clients, including after their GPU
descriptors have been closed. Inheriting real GPU descriptors also prevents
unused-context initialization.

`vfork()` and `posix_spawn()` do not run this child handler and require `exec`
before GPU access. The usual fork/spawn followed by exec, including Python's
`subprocess`, continues to work.

Use daemon mode (`--daemon`) when multiple processes need to share a simulated
GPU. Each client still needs a fresh context: a process forked after its parent
used a backend must exec before attaching to the daemon. These restrictions are
a cooperative API contract; interposition does not prevent raw syscalls or
receiving real GPU descriptors over a Unix socket.

## Command-Line Options

```
Usage: rocjitsu --config <config.json> [--daemon|--attach] -- <app> [args...]
       rocjitsu --preload-only -- <app> [args...]
       rocjitsu --config <config.json> [--node <id>] --vfio-socket <path> [--vfio-socket <path> ...]
```

| Option | Description |
|--------|-------------|
| `--config <path>` | Path to simulation config JSON (required except with `--preload-only`) |
| `--node <id>` | Select a local node from a multi-node system manifest. Legacy `vm.gpu.num_gpus` configs need no node selector. |
| `--preload-only` | Preload the interposer for local simulation without a config. The application must call `rocjitsu.enable(config)` in Python before GPU discovery; the Python bootstrap `python -m rocjitsu` uses this mode. Cannot be combined with configuration options, `--daemon`, or `--attach`. |
| `--daemon` | Run in daemon mode: fork a daemon process hosting the simulation engine, then launch the application with the interposer. Without `-- <app>`, runs the daemon server only. |
| `--attach` | Attach to a running daemon. The socket path is resolved as `$ROCJITSU_RUNTIME_DIR/daemon.sock`, then `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock`, falling back to `/tmp/rocjitsu-<uid>/daemon.sock`. |
| `--vfio-socket <path>` | Serve a local GPU as a PCI function over VFIO-user. Repeat once per local GPU in ordinal order for a multi-GPU node. The number of sockets must match the selected node's local GPU count, and paths must be distinct. The VMM must share guest RAM through mmap-able file descriptors. |
| `--check-vfio-user` | Exit successfully only when this binary was built with VFIO-user support. |
| `--help`, `-h` | Print usage and exit |
| `--version`, `-v` | Print version, Git revision, commit date, and commit title, then exit |
| `--` | Separator between rocjitsu options and the target application command line |

### Usage Examples

```bash
# Local mode: in-process simulation
rocjitsu --config configs/gfx950_mi355x_kmd.json -- ./app

# Python startup activation: app.py calls rocjitsu.enable(config)
rocjitsu --preload-only -- python app.py

# Daemon mode: fork daemon + launch app
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json -- ./app args...

# Daemon-only: run server (no app launched)
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json

# Attach to running daemon
rocjitsu --attach --config configs/gfx950_mi355x_kmd.json -- ./app

# Serve the gfx1250 PCI function to a VMM
rocjitsu --config configs/gfx1250_mi455x.json \
  --vfio-socket /tmp/rocjitsu-vfio/vfio-user.sock

# Argument-order example: this fixture has synthetic PCI IDs and is not a guest profile
rocjitsu --config tests/fixtures/fabric_node_projection.json --node tray7 \
  --vfio-socket /tmp/rocjitsu-vfio/gpu0.sock \
  --vfio-socket /tmp/rocjitsu-vfio/gpu1.sock
```

`--preload-only` and `enable()` create a local simulator in each interpreter.
Daemon devices are configured when the daemon starts; `--attach` clients reuse
that simulator. To use a daemon from Python, launch with `--daemon` or `--attach`
and `--config` before `-- python app.py`, and omit `enable()` in the application.
The Python `Session` API exposes the same modes through `mode="daemon"` and
`mode="attach"`; it does not allocate or reconfigure devices in a running daemon.

See [QEMU VFIO-user compute](qemu-vfio.md) for the supported guest contract and
launcher.

The preload-only example is equivalent to `python -m rocjitsu app.py`. In both
cases, `app.py` imports `rocjitsu` and calls `rocjitsu.enable(config)` before
GPU discovery or framework imports. The CLI preloads the interposer; the Python
application chooses the config. Applications configured with `--config` already
have simulation enabled and must not call `enable()`.

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
| VFIO-user server | `rocjitsu --config ... --vfio-socket ...` (repeat per local GPU) | Guest AMDGPU driver through PCI/VFIO | rocJITsu process |

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

| Opcode | Direction | Payload | Backing descriptor |
|--------|-----------|---------|--------------------|
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

Backing descriptors (memfds or fallback socket tokens) are passed via
`sendmsg()`/`recvmsg()` with `SCM_RIGHTS` ancillary data.
This is the standard Unix mechanism for cross-process file descriptor
sharing (same pattern as QEMU vhost-user and CRIU).

Two RPC response forms carry backing descriptors with different lifetimes:

- `RPC_IOCTL` (`ALLOC_MEMORY` and `IPC_IMPORT_HANDLE` responses): the client
  stores the allocation or import descriptor until free or driver cleanup.
- `RPC_MMAP` responses: temporary aliases of allocation, doorbell, or event-page
  backing. `RemoteDriver::mmap` closes the received alias on every exit, including
  size-check and mapping failures. If the reply has no descriptor, it borrows
  the stored allocation or import descriptor and leaves that descriptor open.

### Thread Safety

ROCR is heavily multithreaded. The `rpc_mutex_` in `RemoteDriver` serializes
all send+recv pairs. `WAIT_EVENTS` uses client-side polling with `ppoll` on
a shutdown `eventfd` between daemon timeout=0 polls, avoiding mutex starvation.

## Memory Sharing (Same-VA Architecture)

Both client and daemon map shared backing at the **same virtual address** using
`map_memory_backing` with `MAP_FIXED`. GPU VAs are in the GPUVM aperture (`0x1000000000` - `0x3FFFFFFFFFFF`),
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
                                            create_memory_backing via rocjitsu-memory
                                            MAP_FIXED backing at VA (proactive)
                                    <----  response + backing fd (SCM_RIGHTS)
   [store fd until free or driver cleanup]
   [USERPTR: mincore+copy, MAP_FIXED]

3. ROCR: fmm_map_to_cpu()
   mmap(drm_fd, MAP_FIXED, offset)
   interposer routes DRM fd     ---->     mmap() -> already mapped, return
                                    <----  response + temporary backing fd (SCM_RIGHTS)
   [map backing; close reply fd on every exit]
```

The private helper sizes ordinary memfds when its inherited hard file limit
allows. Larger allocations use System V shared RAM retained by Unix socket
tokens, which travel through the same `SCM_RIGHTS` path. The mapping and metadata
adapters support both representations; tokens require those adapters and do not
support `pread`/`pwrite`. The PCI `VramStore` uses separate file-backed storage.
See [memory backing](memory-backing.md) for keeper-lifetime and credential
constraints, the same-UID trust assumption, and shared-memory budgets.

### Proactive Mapping

The daemon MAP_FIXED maps all allocations at ALLOC_MEMORY time, not at
mmap time. This prevents a race where the CP thread dereferences kernarg
pointers before the client's fmm_map_to_cpu RPC arrives.

### USERPTR Sharing

For USERPTR allocations (ring buffer, rptr/wptr), the client receives the
daemon's backing descriptor, copies committed pages via mincore, then MAP_FIXED
replaces the anonymous pages with shared backing. The interposer suppresses
MADV_HUGEPAGE and MADV_DONTFORK on GPUVM addresses to prevent kernel 6.17
shmem fault failures.

### Trust Model

The RPC protocol assumes a trusted, version-matched client (i.e., the
same rocjitsu build on both sides). The handshake rejects version
mismatches, but payload contents are not validated beyond size checks.
Untrusted or adversarial clients are out of scope.

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

The driver reserves a contiguous fd range (4000-4255) for internal backing descriptors.
`claim_fd(real_fd)` moves a real fd into this range via `dup2`. The
interposer's `close()` checks `owns_reserved_fd(fd)` to protect internal
fds from ROCR's normal fd lifecycle.

### Fd Protection

ROCR opens and closes many fds during init (topology files, /proc, DRM
render nodes). Without protection, a transient file's fd number could
collide with an allocation backing descriptor. The `owned_fds_` set in
`SimulatedKfd` tracks internal backing descriptors, including socket tokens,
and the interposer suppresses close() on protected fds.

## Limitations

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
| `simulated_kfd.h/.cpp` | KFD ioctl dispatch, allocation table, events |
| `interposer.cpp` | LD_PRELOAD syscall intercepts |
| `events.h/.cpp` | KFD event subsystem (create, set, wait, destroy) |
| `tools/rocjitsu/main.cpp` | CLI entry point, RPC dispatch loop (daemon mode) |
| `tests/daemon_test.cpp` | Gtest fixture for daemon end-to-end tests |
| `virtual_machine.h/.cpp` | VM component owning SoC and driver |

## Future Work

- **Windows**: Named pipe transport, KMD escape driver backend
- **RDMA**: Rack-scale transport for distributed simulation
- **Platform-independent commands**: Abstract command enum for cross-platform

### Programmatic activation environment

`--preload-only` sets `ROCJITSU_PROGRAMMATIC=1`. Interposed KFD/DRM device
and discovery paths refuse access until the application activates a config.
A discovery attempt prevents later activation because frameworks may cache the
failed result. Ordinary configured launches set the variable to `0`.

After successful activation, Python exports `ROCJITSU_INVOCATION_DIR` pointing
to its private config handoff and sets `ROCJITSU_PROGRAMMATIC=2` for fresh exec
children. The interpreter that calls `enable()` owns this handoff and must
remain alive until those children finish their GPU work.

The inherited mode (`2`) permits an identical `enable` call when multiprocessing
re-imports a module in a spawn or forkserver child. The native parser compares
the effective configs, including any CPU thread budget override; a different
config is refused. Configured CLI launches (`0`) still reject activation.
These environment values and the native entry point are private implementation
details of the Python binding.
