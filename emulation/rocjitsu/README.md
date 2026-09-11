# rocjitsu: ROCm&trade; Just-in-Time Suite

[![rocjitsu CI](https://github.com/ROCm/rocm-systems/actions/workflows/rocjitsu-ci.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocjitsu-ci.yml)

An emulation toolkit for running AMD GPU applications on simulated or
real hardware. Supports three execution strategies:

- **Simulation** — Full ISA emulation on simulated GPU hardware via
  LD_PRELOAD interposition. No physical GPU or kernel module required.
- **Dynamic Binary Translation (DBT)** — Cross-ISA translation enabling
  applications compiled for one GPU architecture to run on another.
- **Dynamic Binary Instrumentation (DBI)** — Runtime instrumentation
  of GPU kernels for profiling, tracing, and analysis.

## Supported architectures

| Architecture | GFX Target | ISA Family | Simulation |
|---|---|---|---|
| CDNA1&trade; | gfx908 | GFX9 | Experimental |
| CDNA2&trade; | gfx90a | GFX9 | Experimental |
| CDNA3&trade; | gfx94x | GFX9 | Beta |
| CDNA4&trade; | gfx950 | GFX9 | Beta |
| CDNA5&trade; | gfx1250 | GFX12.5 | Beta |
| RDNA1&trade; | gfx1010 | GFX10 | Experimental |
| RDNA2&trade; | gfx1030 | GFX10 | Experimental |
| RDNA3&trade; | gfx110x | GFX11 | Beta |
| RDNA3.5&trade; | gfx1151 | GFX11.5 | Experimental |
| RDNA4&trade; | gfx120x | GFX12 | Beta |
| RISC-V | RV64I | RV | Experimental |

CDNA4&trade; to CDNA3&trade; dynamic binary translation is experimental.

<!-- \NPI new GPU: add a row to the supported-architectures table above. -->

## Project layout

```
lib/
  simdojo/              Simulation engine (PDES framework)
  rocjitsu/
    include/rocjitsu/   Public C API
    src/rocjitsu/
      vm/amdgpu/        GPU hardware model (CP, CU, caches, memory)
      isa/arch/amdgpu/  Hand-written ISA support; generated/ holds generated ISA files
      kmd/linux/        KFD driver emulation + LD_PRELOAD interposer
      code/             Code object loader, basic block analysis
      code/dbt/         Dynamic binary translator
      code/patch/       Code object patcher, spill manager
      code/analysis/    Register liveness and def-use analysis
      config/           JSON/FlatBuffers configuration
  util/                 Shared utilities
  python/amdisa/        ISA codegen pipeline
tools/rocjitsu/         CLI (local, daemon, attach modes)
configs/                GPU topology JSON files
schemas/                FlatBuffers schemas
tests/                  Test suite
docs/                   Design documents and guides
```

## Building

```bash
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

See [docs/building.md](docs/building.md) for CMake options, sanitizers,
and formatting setup.

## Running tests

```bash
ctest --test-dir build
```

HIP kernel tests and RCCL daemon tests require an ROCm installation.
They are disabled automatically when `hipcc` or `libhsa-runtime64` is
not found.

## Running HIP workloads

```bash
# Local mode (in-process simulation)
rocjitsu --config configs/gfx950_mi355x_kmd.json -- ./my_hip_app

# Daemon mode (separate daemon process)
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json -- ./my_hip_app
```

See [docs/rocjitsu-cli.md](docs/rocjitsu-cli.md) for all CLI modes.

## VGPR observation contract

Execution plugins observe architectural instruction-level VGPR reads and
writes. Callback lane and byte masks describe the architectural register
effect, including masked and sub-dword operations. Internal storage operations
used to preserve unaffected register state are not reported as additional
instruction effects.

Asynchronous memory operations use a separate lifecycle. The race detector
records register dependencies when the operation is issued. Later completion
updates register storage without reporting the same instruction effect again.
See
[docs/plugins.md](docs/plugins.md) for the plugin contract.

## Running PyTorch

```bash
rocjitsu --daemon --config configs/gfx950_mi355x_kmd.json -- \
  python3 -c "import torch; x = torch.randn(4, 4, device='cuda'); print(x @ x)"
```

See [docs/building.md](docs/building.md) for container setup with PyTorch.

## Documentation

### Getting started

| Document | Description |
|---|---|
| [Building](docs/building.md) | Build options, sanitizers, container setup |
| [Benchmarking](docs/benchmarking.md) | Reproducible performance and memory measurement protocol |
| [Configuration](docs/configuration.md) | JSON config format and topology |
| [CLI & Transport](docs/rocjitsu-cli.md) | CLI modes, daemon RPC protocol |
| [Race Detector](docs/race-detector.md) | Race detection tutorial and internals |
| [Debugging with ROCgdb](docs/rocgdb-debugging.md) | Debug emulated GPU kernels with ROCgdb: breakpoints, watchpoints, faults, multi-wave |

### Design

| Document | Description |
|---|---|
| [Architecture](docs/architecture.md) | Component overview and layer map |
| [Plugins](docs/plugins.md) | Execution plugin system and sink API |
| [VM Design](docs/vm-design.md) | GPU hardware model (CP, CU, caches, KMD) |
| [Simdojo Engine](docs/simdojo.md) | PDES simulation framework |
| [DBT Design](docs/dbt-design.md) | Binary translator architecture |
| [DBI Design](docs/dbi-design.md) | Binary instrumentation (in progress) |
| [Codegen](docs/codegen.md) | ISA codegen pipeline and regen commands |
| [ISA Target Providers](docs/isa-target-providers.md) | Static target registration and per-component subsets |

### Reference

| Document | Description |
|---|---|
| [Utilities](lib/util/README.md) | Shared utility library (`lib/util/`) |
| [ISA Gap Audit](docs/isa-gap-audit.md) | Workflow for auditing manual, XML, and rocjitsu ISA semantic gaps |
| [rj_dbt_translate](docs/rj_dbt_translate.md) | Standalone DBT translation tool |
| [CHANGELOG](CHANGELOG.md) | Release history |

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting changes. It
covers where new code should live, which existing libraries to use and
extend, the codegen workflow, and code style conventions.

## License

[MIT License](LICENSE.md) &mdash; Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
