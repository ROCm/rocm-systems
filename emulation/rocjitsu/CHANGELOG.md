# Changelog

All notable changes to rocjitsu are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
