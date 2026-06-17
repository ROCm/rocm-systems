# Changelog

All notable changes to rocjitsu are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-06-23

### Changed
- Serve the amdgpu/DRM ioctl ABI natively so upstream libdrm_amdgpu runs
  unmodified over the simulator; DRM device discovery and queries are handled at
  the syscall layer instead of by intercepting libdrm library symbols
- **BREAKING:** `rj_vm_device_open()` now takes a connecting client's PID:
  `rj_vm_device_open(vm, pid_t client_pid, uint32_t *process_id)`. Pass 0 in
  local mode; a nonzero PID enables daemon-mode process reuse and cross-process
  memory access

### Added
- Daemon-mode multi-process client support: per-client process reuse,
  client-PID identification via SO_PEERCRED, and cross-process GPU memory access
- Vendored kernel DRM UAPI headers (drm.h, amdgpu_drm.h, drm_mode.h) for native
  amdgpu ioctl struct fidelity

## [0.1.0] - 2026-06-04

### Added
- Daemon-based GPU simulator via LD_PRELOAD interposition
- ISA execution engine with codegen pipeline (10 AMDGPU targets + RISC-V)
- AQL packet dispatch with doorbell-monitored hardware queues
- Multi-GPU topology with per-GPU command processors
- KFD driver emulation (ioctls, mmap, event system, VMID page tables)
- HIP kernel tests (vector_add, memcpy, GEMM)
- RCCL collective tests (AllReduce, Broadcast, AllGather, ReduceScatter, SendRecv)
- CLI with local, daemon, and attach modes
- JSON-configurable SoC topology
