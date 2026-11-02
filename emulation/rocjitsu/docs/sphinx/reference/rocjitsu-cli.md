---
myst:
    html_meta:
        "description": "rocjitsu CLI reference covering command-line options, execution modes, environment variables, socket path resolution, RPC opcodes, and daemon lifecycle."
        "keywords": "rocJITsu, CLI, ROCm, daemon, attach, local, RPC, environment variables, socket, GPU emulation"
---

# rocjitsu CLI reference

The `rocjitsu` command is the primary entry point for running applications on a simulated GPU. It supports local, daemon, attach, and VFIO-user modes and manages the relevant simulation engine and transport.

For details on how KMD emulation and the interposer layer work, see [GPU virtual machine design](/conceptual/gpu-vm-design.md).

## Command-line options

``` text
rocjitsu --config <config.json> [--daemon|--attach] [--] <app> [args...]
rocjitsu --config <config.json> [--node <id>] --vfio-socket <path> [--vfio-socket <path> ...]
```

| Option | Description |
| --- | --- |
| `--config <path>` | Path to the simulation configuration JSON file. Required for all modes. |
| `--node <id>` | Select a local node from a multi-node system manifest. Legacy `vm.gpu.num_gpus` configs need no node selector. |
| `--daemon` | Run in daemon mode: fork a daemon process hosting the simulation engine, then launch the application with the interposer. Without `-- <app>`, runs the daemon server only (no application is launched). |
| `--attach` | Attach to a running daemon. The socket path is resolved as described in [Environment variables and socket path resolution](#socket-path-resolution). |
| `--vfio-socket <path>` | Serve a local GPU as a PCI function over VFIO-user. Repeat once per local GPU in ordinal order for a multi-GPU node. The socket count must match the selected node's local GPU count, and paths must be distinct. Requires a build configured with `ROCJITSU_ENABLE_VFIO=ON` and a VMM that shares guest RAM through mmap-able file descriptors. |
| `--check-vfio-user` | Exit successfully only when the selected binary includes VFIO-user support. |
| `--help`, `-h` | Print usage information and exit. |
| `--version`, `-v` | Print the version string and exit. |
| `--` | Separator between rocJITsu options and the target application command line. |

## Execution modes

### Local mode

``` bash
rocjitsu --config configs/gfx950_mi355x_kmd.json -- ./app
```

The simulator runs in-process. `rocjitsu` sets `LD_PRELOAD` and calls `execve` on the target application. The interposer routes `/dev/kfd` operations to a `SimulatedDriver` within the same process, and a background thread runs the simulation engine.

This mode corresponds to `RJ_VM_MODE_LOCAL` in the C API.

### Daemon mode

``` bash
# Fork daemon + launch application
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json -- ./app args...

# Daemon-only (no application launched)
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json
```

A child daemon process is forked to host the simulation engine and `SimulatedDriver`. The parent then `execve`'s the target application with `LD_PRELOAD` set. Client processes communicate with the daemon over a Unix domain socket using the RPC protocol described below.

GPU memory allocations use backing descriptors shared between the daemon and client processes via `SCM_RIGHTS`. These descriptors are ordinary memfds, or socket tokens retaining System V shared RAM when an allocation exceeds the private memory helper's inherited hard file-size limit. Both sides use rocjitsu's mapping adapters to map the backing at the same virtual address (`MAP_FIXED`), so GPU virtual addresses resolve correctly in both processes without translation.

This mode corresponds to `RJ_VM_MODE_DAEMON` in the C API. It supports multi-process workloads such as PyTorch, `torchrun`, `torch.distributed`, and RCCL.

The daemon hosts one shared simulation engine. A fatal instruction decode or execution failure stops that engine and disconnects every client, including clients whose GPU work is valid. The affected queue cancels its outstanding dispatches without publishing successful kernel or barrier completion. Runtime acceptance of a queue-error notification does not recover the engine; a debugger-owned illegal-instruction stop can still be repaired and resumed.

After a successful KFD open, any daemon disconnect terminates the attached native process via `_Exit(EXIT_FAILURE)`. This includes an orderly daemon stop or restart while clients are idle or doing only host work, even after their GPU work has finished. `_Exit` skips `atexit` handlers and stdio flushing. Disconnect monitoring also interrupts clients waiting on shared-memory completion signals, so those waits cannot remain blocked after engine shutdown. The faulting client's HIP queue-error callback may abort it before the interposer handles the disconnect.

### Attach mode

``` bash
rocjitsu --attach --config configs/gfx950_mi355x_kmd.json -- ./app
```

Connects to an already-running daemon. The socket path is resolved using the environment variables described in [Environment variables and socket path resolution](#socket-path-resolution).

### VFIO-user mode

``` bash
rocjitsu --config configs/gfx1250_mi455x.json \
  --vfio-socket /tmp/rocjitsu-vfio/vfio-user.sock

# Argument-order example: this fixture has synthetic PCI IDs and is not a guest profile
rocjitsu --config tests/fixtures/fabric_node_projection.json --node tray7 \
  --vfio-socket /tmp/rocjitsu-vfio/gpu0.sock \
  --vfio-socket /tmp/rocjitsu-vfio/gpu1.sock
```

rocJITsu exposes each configured local GPU as a PCI function to an external
VMM. The guest AMDGPU driver owns queue setup through BAR and DMA operations;
the PCI adapter routes that state into the same GPU VM, command processor, MES,
SDMA scheduler, and compute units used by the other front ends. See
[Run a QEMU VFIO-user compute guest](/how-to/qemu-vfio.md) for the complete
host and guest contract.

(socket-path-resolution)=
## Environment variables and socket path resolution

### Environment variables

| Variable | Description |
| --- | --- |
| `ROCJITSU_RUNTIME_DIR` | If set, the daemon socket path is `$ROCJITSU_RUNTIME_DIR/daemon.sock`. |
| `XDG_RUNTIME_DIR` | If `ROCJITSU_RUNTIME_DIR` is not set, the socket path is `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock`. |

### Socket path resolution order

When `--attach` is used or when the interposer connects to a daemon, the socket path is resolved in this order:

1.  `$ROCJITSU_RUNTIME_DIR/daemon.sock`
2.  `$XDG_RUNTIME_DIR/rocjitsu/daemon.sock`
3.  `/tmp/rocjitsu-<uid>/daemon.sock` (fallback)

For additional environment variables that control simulation plugins, see [Environment variable reference](/reference/environment-variables.md).

## RPC protocol

Daemon mode uses a fixed 16-byte message header followed by an opcode-specific payload. All communication flows over a Unix domain socket.

### Message header

``` text
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

### RPC opcodes

| Opcode | Direction | Payload | Backing descriptor |
| --- | --- | --- | --- |
| `RPC_HANDSHAKE` | C→D→C | `RpcHandshakeResponse` + topology path | No |
| `RPC_CLOSE` | C→D→C | None | No |
| `RPC_MMAP` | C→D→C | `RpcMmapRequest` / `RpcMmapResponse` | Optional |
| `RPC_MUNMAP` | C→D→C | `RpcMunmapRequest` | No |
| `RPC_IOCTL` | C→D→C | `RpcIoctlRequest` + raw args + inlined arrays | Optional |

Backing descriptors (memfds or fallback socket tokens) are passed via `sendmsg()`/`recvmsg()` with `SCM_RIGHTS` ancillary data. Two RPC response forms carry backing descriptors with different lifetimes:

-   `RPC_IOCTL` (`ALLOC_MEMORY` and `IPC_IMPORT_HANDLE` responses): the client stores the allocation or import descriptor until free or driver cleanup.
-   `RPC_MMAP` responses: temporary aliases of allocation, doorbell, or event-page backing. `RemoteDriver::mmap` closes the received alias on every exit, including size-check and mapping failures. If the reply has no descriptor, it borrows the stored allocation or import descriptor and leaves that descriptor open.

## Daemon startup and shutdown sequence

### Startup

1.  `rocjitsu --daemon` forks a child process.
2.  The child creates the VM with `RJ_VM_MODE_DAEMON`, which initializes the simulation engine, topology, driver, and shared GPU memory backing.
3.  The child creates a listening Unix domain socket at the resolved socket path.
4.  The child begins accepting client connections and dispatching RPC messages.
5.  The parent sets `LD_PRELOAD` and `execve`'s the target application (if one was specified with `--`).

### Shutdown

During normal client shutdown:

1.  The HSA runtime calls `SET_EVENT` on wake signals during its async control shutdown.
2.  All client-side pollers observe the event (auto-reset is skipped on `timeout=0` polls).
3.  The runtime's async threads exit, and thread joins succeed.
4.  The runtime calls `DESTROY_QUEUE`, `DESTROY_EVENT`, and `close(kfd_fd)`.
5.  The client's `RemoteDriver` sends `RPC_CLOSE` to the daemon.
6.  The daemon closes the client connection and, when all clients have disconnected, shuts down the simulation engine.

Stopping the daemon while clients remain connected closes every client socket. After a successful KFD open, those clients immediately terminate via `_Exit(EXIT_FAILURE)`, including idle clients; their `atexit` handlers and stdio flushing are skipped. A fatal instruction failure also stops the shared engine and disconnects every client, as described in [Daemon mode](#daemon-mode).
