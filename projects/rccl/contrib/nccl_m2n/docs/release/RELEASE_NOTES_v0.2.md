# NCCL M2N v0.2 Release Notes

This v0.2 release supersedes v0.1. It requires NCCL 2.30.5 or newer and a
Linux build. Rebuild all C, C++, and generated-binding consumers against
`nccl_m2n.h`.

## New Features

- `ncclReshard` is the primary reshard operation. It provides library-managed
  copy and staging transfers.
- `ncclReshard` accepts blocking and non-blocking NCCL communicators.
- `ncclReshard` optimizes transfers when trainer and generator ranks share an
  NVLink domain.
- Split hierarchical communicators improve scalability on eligible RING paths.
- `ncclM2nInit` and `ncclM2nFinalize` provide explicit runtime handles.
  Passing `NULL` to `ncclReshard` uses a default handle.
- `ncclMesh_t` describes topology, while `ncclDistTensor_t` carries tensor
  placement. Initialize public descriptors with their initializer macros.
- `ncclM2nGroupStart`, `ncclM2nGroupEnd`, and `ncclM2nGroupAbort` support
  grouped submission with documented ordering and indexed error semantics.
- A standalone CMake build installs `libnccl_m2n` and `nccl_m2n.h` and exports
  the `NCCL::m2n` target for CMake consumers.
- The new PIPE copy algorithm supports fine-grained pipelining with
  device and host-RMA modes (beta).

## Fixes

- Descriptor and mesh validation reject unsupported layouts, invalid rank
  intervals, and arithmetic overflow before work is submitted.
- Strict configuration parsing prevents malformed values from being applied;
  invalid values retain the applicable default.
- Runtime error reporting includes thread-local M2N detail through
  `ncclM2nGetLastError`.
- GIN signal accounting and internal-stream ordering are hardened for
  asynchronous execution.
- `NCCL_HOME` accepts a pip-installed NCCL wheel.

## Known Limitations

| Limitation | Description |
|---|---|
| Tensor and mesh rank | Tensors support at most 3 dimensions. Meshes support 1 or 2 dimensions. |
| Placement | Each source or destination layout supports at most one SHARD axis. |
| Window entry point | `ncclReshardWithWindow` ignores its window argument and follows `ncclReshard` transport selection. |
| Communicator concurrency | Sequential calls can be stream-ordered. Concurrent host calls on the same communicator are unsupported. |
| Finalization | `ncclM2nFinalize` does not synchronize CUDA streams. Complete submitted work before finalization or resource destruction. |
| Peer caps | Current copy and staging paths support at most 16 sources and 64 targets per rank peer list. |
| Failure recovery | An error after PACK host-RMA protocol entry is fail-stop for its communicator and runtime epoch. |
