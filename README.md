# ROCm Systems

Welcome to the ROCm Systems super-repo. This repository consolidates multiple ROCm systems projects into a single repository to streamline development, CI, and integration. The first set of projects focuses on requirements for building PyTorch.

# ROCm Systems Project Directory

| Library | Description | Component CI Status | Documentation |
|---|---|---|---|
| [`amdsmi`](https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi) | GPU management, monitoring, and control through APIs and a CLI. | — | [AMD Docs](https://rocm.docs.amd.com/projects/amdsmi/en/latest/) |
| [`aqlprofile`](https://github.com/ROCm/rocm-systems/tree/develop/projects/aqlprofile) | Low-level GPU profiling and hardware performance-counter support. | [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/aqlprofile-continuous_integration.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/aqlprofile/en/latest/) |
| [`clr`](https://github.com/ROCm/rocm-systems/tree/develop/projects/clr) | Compute Language Runtimes for HIP and OpenCL. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/clr/README.md) |
| [`cuid`](https://github.com/ROCm/rocm-systems/tree/develop/projects/cuid) | Consistent identifiers for hardware components across systems. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/cuid/README.md) |
| [`hip`](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip) | GPU runtime API and C++ kernel programming language. | — | [AMD Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/) |
| [`hip-tests`](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip-tests) | Unit and conformance tests for HIP. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/hip-tests/README.md) |
| [`hipfile`](https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile) | Direct-to-GPU storage I/O library. | — | [AMD Docs](https://rocm.docs.amd.com/projects/hipFile/en/latest/) |
| [`hipother`](https://github.com/ROCm/rocm-systems/tree/develop/projects/hipother) | HIP support for non-AMD backends. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/hipother/README.md) |
| [`hrr`](https://github.com/ROCm/rocm-systems/tree/develop/projects/hrr) | HIP API capture and replay for reproducing and validating issues. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/hrr/README.md) |
| [`rccl`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl) | Multi-GPU and multi-node collective communication library. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rccl/en/latest/) |
| [`rccl-tests`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl-tests) | Correctness and performance tests for RCCL. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/rccl-tests/README.md) |
| [`rdc`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rdc) | GPU fleet monitoring and administration for data centers. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rdc/en/latest/) |
| [`rocdbgapi`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdbgapi) | Low-level AMD GPU debugging API. | None | [AMD Docs](https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/) |
| [`rocdecode`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode) | GPU-accelerated video decoding APIs and utilities. | [![Media Libs CI](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/rocDecode/en/latest/) |
| [`rocjpeg`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg) | GPU-accelerated JPEG decoding APIs and samples. | [![Media Libs CI](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/media-libs-ci.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/rocJPEG/en/latest/) |
| [`rocm-core`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocm-core) | Core ROCm package metadata and installation information. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocm-core/README.md) |
| [`rocm-smi-lib`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocm-smi-lib) | Legacy GPU system monitoring and management library. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rocm_smi_lib/en/latest/) |
| [`rocminfo`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocminfo) | Utility for listing GPU agents and ROCm system capabilities. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rocminfo/en/latest/) |
| [`rocprof-trace-decoder`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprof-trace-decoder) | Decodes GPU wave-trace data for performance-analysis tools. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprof-trace-decoder/README.md) |
| [`rocprofiler`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler) | Legacy GPU profiling and performance-counter tooling. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rocprofiler/en/latest/) |
| [`rocprofiler-compute`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute) | Kernel-level profiling and performance analysis. | [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-continuous-integration.yml/badge.svg?event=schedule)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-continuous-integration.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-formatting.yml) <br> [![rhel](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-rhel.yml) <br> [![Sanitizers](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-sanitizers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-sanitizers.yml) <br> [![ubuntu jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-compute-ubuntu-jammy.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/) |
| [`rocprofiler-register`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-register) | Coordinates profiler registration with runtime API tables. | [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-register-continuous-integration.yml) | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-register/README.md) |
| [`rocprofiler-sdk`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-sdk) | Tracing and profiling infrastructure for GPU applications. | [![Code Coverage Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-code_coverage.yml) <br> [![CodeQL](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-codeql.yml) <br> [![Continuous Integration](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-continuous_integration.yml) <br> [![Documentation](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-docs.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-python.yml) <br> [![Restrictions](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-restrictions.yml) <br> [![Release Compatibility](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-sdk-rocm_release_compatibility.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/) |
| [`rocprofiler-systems`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems) | System-wide CPU/GPU application tracing and profiling. | [![Containers](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-containers.yml) <br> [![rocprofiler-systems GHCR Packages for CI Images](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml/badge.svg)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ghcr.yml) <br> [![Formatting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-formatting.yml) <br> [![Python Linting](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-python.yml) <br> [![RedHat Linux](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-redhat.yml) <br> [![Ubuntu Jammy](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-jammy.yml) <br> [![Ubuntu Noble](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/rocprofiler-systems-ubuntu-noble.yml) | [AMD Docs](https://rocm.docs.amd.com/projects/rocprofiler-systems/en/latest/) |
| [`rocr-debug-agent`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-debug-agent) | Reports GPU wavefront state to diagnose execution faults. | None | [AMD Docs](https://rocm.docs.amd.com/projects/rocr_debug_agent/en/latest/) |
| [`rocr-runtime`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocr-runtime) | HSA runtime and kernel-driver interface for AMD GPUs. | — | [Project README](https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-runtime/README.md) |
| [`rocshmem`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocshmem) | GPU-centric, OpenSHMEM-style communication library. | — | [AMD Docs](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/) |
| [`roctracer`](https://github.com/ROCm/rocm-systems/tree/develop/projects/roctracer) | Legacy runtime tracing and annotation APIs. | — | [AMD Docs](https://rocm.docs.amd.com/projects/roctracer/en/latest/) |


# TheRock CI Status

Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.

[![The Rock CI](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml/badge.svg?branch%3Adevelop+event%3Apush)](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush)

---

## Nomenclature

Project names have been standardized to match the casing and punctuation of released packages. This removes inconsistent camel-casing and underscores used in legacy repositories.

## Structure

The repository is organized as follows:

```
projects/
  amdsmi/
  aqlprofile/
  clr/
  hip/
  hipfile/
  hipother/
  hip-tests/
  rccl/
  rdc/
  rocdbgapi/
  rocdecode/
  rocjpeg/
  rocm-core
  rocminfo/
  rocmsmilib/
  rocprofiler/
  rocprofiler-compute/
  rocprofiler-register/
  rocprofiler-sdk/
  rocprofiler-systems/
  rocr-debug-agent/
  rocrruntime/
  rocshmem/
  roctracer/
```

- Each folder under `projects/` corresponds to a ROCm systems project that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple projects, but does not produce its own distinct packages in previous ROCm releases.

## Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Getting Started

To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This super-repo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.
📄 Refer to the header notice in individual files outside `projects/` or `shared/` folders for their specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
