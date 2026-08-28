---
myst:
    html_meta:
        "description": "rocjitsu CLI reference covering command-line options, execution modes, environment variables, socket path resolution, RPC opcodes, and daemon lifecycle."
        "keywords": "rocJITsu, CLI, ROCm, daemon, attach, local, RPC, environment variables, socket, GPU emulation"
---

# rocjitsu CLI reference

The `rocjitsu` command is the primary entry point for running applications on a simulated GPU. It supports three execution modes---local, daemon, and attach---and manages the simulation engine, interposer setup, and RPC transport on your behalf.

For details on how KMD emulation and the interposer layer work, see [GPU virtual machine design](/conceptual/gpu-vm-design.md).

## Command-line options

``` text
rocjitsu --config <config.json> [--daemon|--attach] [--] <app> [args...]
```

| Option | Description |
| --- | --- |
| `--config <path>` | Path to the simulation configuration JSON file. Required for all modes. |
| `--daemon` | Run in daemon mode: fork a daemon process hosting the simulation engine, then launch the application with the interposer. Without `-- <app>`, runs the daemon server only (no application is launched). |
| `--attach` | Attach to a running daemon. The socket path is resolved as described in [Environment variables and socket path resolution](#socket-path-resolution). |
| `--help`, `-h` | Print usage information and exit. |
| `--version`, `-v` | Print the version string and exit. |
| `--` | Separator between rocJITsu options and the target application command line. |

## Execution modes

### Local mode

``` bash
rocjitsu --config configs/amdgpu_cdna4_kmd.json -- ./app
```

The simulator runs in-process. `rocjitsu` sets `LD_PRELOAD` and calls `execve` on the target application. The interposer routes `/dev/kfd` operations to a `SimulatedDriver` within the same process, and a background thread runs the simulation engine.

This mode corresponds to `RJ_VM_MODE_LOCAL` in the C API.

### Daemon mode

``` bash
# Fork daemon + launch application
rocjitsu --daemon --config configs/amdgpu_cdna4_kmd.json -- ./app args...

# Daemon-only (no application launched)
rocjitsu --daemon --config configs/amdgpu_cdna4_kmd.json
```

A child daemon process is forked to host the simulation engine and `SimulatedDriver`. The parent then `execve`'s the target application with `LD_PRELOAD` set. Client processes communicate with the daemon over a Unix domain socket using the RPC protocol described below.

GPU memory allocations are backed by `memfd` objects and shared between the daemon and client processes via `SCM_RIGHTS`. Both sides map every `memfd` at the same virtual address (`MAP_FIXED`), so GPU virtual addresses resolve correctly in both processes without translation.

This mode corresponds to `RJ_VM_MODE_DAEMON` in the C API. It supports multi-process workloads such as PyTorch, `torchrun`, `torch.distributed`, and RCCL.

### Attach mode

``` bash
rocjitsu --attach --config configs/amdgpu_cdna4_kmd.json -- ./app
```

Connects to an already-running daemon. The socket path is resolved using the environment variables described in [Environment variables and socket path resolution](#socket-path-resolution).

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

| Opcode | Direction | Payload | Memfd |
| --- | --- | --- | --- |
| `RPC_HANDSHAKE` | C→D→C | `RpcHandshakeResponse` + topology path | No |
| `RPC_CLOSE` | C→D→C | None | No |
| `RPC_MMAP` | C→D→C | `RpcMmapRequest` / `RpcMmapResponse` | Optional |
| `RPC_MUNMAP` | C→D→C | `RpcMunmapRequest` | No |
| `RPC_IOCTL` | C→D→C | `RpcIoctlRequest` + raw args + inlined arrays | Optional |

File descriptors (`memfd` handles) are passed via `sendmsg()`/`recvmsg()` with `SCM_RIGHTS` ancillary data. Two operations carry `memfd` handles:

-   `RPC_IOCTL` (`ALLOC_MEMORY` response) --- allocation backing `memfd`
-   `RPC_MMAP` (response) --- doorbell and event page `memfd` handles

## Daemon startup and shutdown sequence

### Startup

1.  `rocjitsu --daemon` forks a child process.
2.  The child creates the VM with `RJ_VM_MODE_DAEMON`, which initializes the simulation engine, topology, driver, and `memfd`-backed GPU memory.
3.  The child creates a listening Unix domain socket at the resolved socket path.
4.  The child begins accepting client connections and dispatching RPC messages.
5.  The parent sets `LD_PRELOAD` and `execve`'s the target application (if one was specified with `--`).

### Shutdown

1.  The HSA runtime calls `SET_EVENT` on wake signals during its async control shutdown.
2.  All client-side pollers observe the event (auto-reset is skipped on `timeout=0` polls).
3.  The runtime's async threads exit, and thread joins succeed.
4.  The runtime calls `DESTROY_QUEUE`, `DESTROY_EVENT`, and `close(kfd_fd)`.
5.  The client's `RemoteDriver` sends `RPC_CLOSE` to the daemon.
6.  The daemon closes the client connection and, when all clients have disconnected, shuts down the simulation engine.

