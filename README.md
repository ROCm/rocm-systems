<<<<<<< HEAD
# ROCm Systems

Welcome to the ROCm Systems super-repo. This repository consolidates multiple ROCm systems projects into a single repository to streamline development, CI, and integration. The first set of projects focuses on requirements for building PyTorch.
=======
# AQLProfile: Architected Queuing Language Profiling Library

AQLProfile is an open source library that enables advanced GPU profiling and tracing on AMD platforms. It works in conjunction with [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk) to support profiling methods such as performance counters (PMC) and SQ thread trace (SQTT). AQLProfile provides the foundational mechanisms for constructing AQL packets and managing profiling operations across multiple AMD GPU architecture families.
>>>>>>> 77ec2dd8ee (publish branch)

# Super-repo Status and CI Health

<<<<<<< HEAD
This table provides the current status of the migration of specific ROCm systems projects as well as a pointer to their current CI health.
=======
AQLProfile builds on concepts from the Heterogeneous System Architecture (HSA) and Architected Queuing Language (AQL), which define the foundations for GPU command processing and profiling on AMD platforms. For further reading:
>>>>>>> 77ec2dd8ee (publish branch)

**Key:**
- **Completed**: Fully migrated and integrated. This super-repo should be considered the source of truth for this project. The old repo may still be used for release activities.
- **In Progress**: Ongoing migration, tests, or integration. Please refrain from submitting new pull requests on the individual repo of the project, and develop on the super-repo.
- **Pending**: Not yet started or in the early planning stages. The individual repo should be considered the source of truth for this project.

| Component              | Source of Truth | Migration Status | Azure CI Status                       | Component CI Status                   |
|------------------------|-----------------|------------------|---------------------------------------|---------------------------------------|
| `amdsmi`               | EMU             | Pending          |                                       |                                       |
| `aqlprofile`           | EMU             | Pending          |                                       |                                       |
| `clr`                  | EMU             | Pending          |                                       |                                       |
| `hip`                  | EMU             | Pending          |                                       |                                       |
| `hipother`             | EMU             | Pending          |                                       |                                       |
| `hip-tests`            | EMU             | Pending          |                                       |                                       |
| `rccl`                 | Public          | Pending          |                                       |                                       |
| `rdc`                  | EMU             | Pending          |                                       |                                       |
| `rocm-core`            | EMU             | Pending          |                                       |                                       |
| `rocminfo`             | EMU             | Pending          |                                       |                                       |
| `rocm-smi-lib`         | EMU             | Pending          |                                       |                                       |
| `rocprofiler`          | Public          | Completed        |                                       |                                       |
| `rocprofiler-compute`  | Public          | Completed        |                                       |                                       |
| `rocprofiler-register` | Public          | Completed        |                                       |                                       |
| `rocprofiler-sdk`      | EMU             | Pending          |                                       |                                       |
| `rocprofiler-systems`  | Public          | Completed        |                                       |                                       |
| `rocr-runtime`         | EMU             | Pending          |                                       |                                       |
| `rocshmem`             | Public          | Pending          |                                       |                                       |
| `roctracer`            | Public          | Completed        |                                       |                                       |

<<<<<<< HEAD

## Tentative migration schedule
=======
AQLProfile is a companion library to [rocprofiler-sdk](https://github.com/ROCm/rocprofiler-sdk). 
It provides the low-level mechanisms required by rocprofiler-sdk to enable advanced GPU profiling and tracing capabilities on AMD platforms. The development and evolution of AQLProfile are closely aligned with the needs of rocprofiler-sdk, ensuring compatibility and feature support for new GPU architectures and profiling requirements.

AQLProfile abstracts the complexity of constructing and managing AQL (Architected Queuing Language) packets, command buffers, and register programming. These components are essential for orchestrating profiling operations such as performance counter collection and thread tracing. The library supports a range of AMD GPU architecture families such as GFX9, GFX10, GFX11, GFX12 and so on.
It provides the necessary infrastructure for rocprofiler-sdk to interact with hardware-level profiling features.
>>>>>>> 77ec2dd8ee (publish branch)

| Component              | Tentative Date |
|------------------------|----------------|
| `aqlprofile`           | 8/7            |
| `rocprofiler-sdk`      | 8/7            |
| `rdc`                  | 8/8            |
| `rocm-smi-lib`         | 8/8            |
| `rocminfo`             | 8/11           |
| `rocr-runtime`         | 8/11           |
| `rocm-core`            | 8/12           |
| `clr`                  | 8/21           |
| `hip`                  | 8/21           |
| `hipother`             | 8/21           |
| `hip-tests`            | 8/21           |

<<<<<<< HEAD
*Remaining schedule to be determined.
=======
- Profiling AQL packets for GPU workloads.
- Performance counters (PMC) and SQ thread traces (SQTT).
- Support for GFX9, GFX10, GFX11 and GFX12 architecture families.
- Verbose tracing and error logging capabilities.
- Thread trace binary data generated by AQLProfile can be decoded using [rocprof-trace-decoder](https://github.com/ROCm/rocprof-trace-decoder/releases).
>>>>>>> 77ec2dd8ee (publish branch)

# TheRock CI Status

<<<<<<< HEAD
Note TheRock CI performs multi-component testing on top of builds leveraging [TheRock](https://github.com/ROCm/TheRock) build system.
=======
The AQLProfile library supports profiling and tracing GPU workloads across multiple architectures.<br>
Below is a summary of the counter blocks supported for each architecture:
>>>>>>> 77ec2dd8ee (publish branch)

[![The Rock CI](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml/badge.svg?branch%3Adevelop+event%3Apush)](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml?query=branch%3Adevelop+event%3Apush)

---

## Nomenclature

Project names have been standardized to match the casing and punctuation of released packages. This removes inconsistent camel-casing and underscores used in legacy repositories.

## Structure

<<<<<<< HEAD
The repository is organized as follows:

```
projects/
  amdsmi/
  aqlprofile/
  clr/
  hip/
  hipother/
  hip-tests/
  rccl/
  rdc/
  rocm-core
  rocminfo/
  rocmsmilib/
  rocprofiler/
  rocprofiler-compute/
  rocprofiler-register/
  rocprofiler-sdk/
  rocprofiler-systems/
  rocrruntime/
  rocshmem/
  roctracer/
=======
### Building AQLProfile

You can build AQLProfile using either the provided build script (recommended for most users) or by manually invoking CMake for custom builds.

#### Option 1: Using the Build Script (Recommended)

This will configure and build the project with default settings:

```bash
./build.sh
>>>>>>> 77ec2dd8ee (publish branch)
```

- Each folder under `projects/` corresponds to a ROCm systems project that was previously maintained in a standalone GitHub repository and released as distinct packages.
- Each folder under `shared/` contains code that existed in its own repository and is used as a dependency by multiple projects, but does not produce its own distinct packages in previous ROCm releases.

## Goals

- Enable unified build and test workflows across ROCm libraries.
- Facilitate shared tooling, CI, and contributor experience.
- Improve integration, visibility, and collaboration across ROCm library teams.

## Getting Started

<<<<<<< HEAD
To begin contributing or building, see the [CONTRIBUTING.md](./CONTRIBUTING.md) guide. It includes setup instructions, sparse-checkout configuration, development workflow, and pull request guidelines.

## License

This super-repo contains multiple subprojects, each of which retains the license under which it was originally published.

📁 Refer to the `LICENSE`, `LICENSE.md`, or `LICENSE.txt` file within each `projects/` or `shared/` directory for specific license terms.

> **Note**: The root of this repository does not define a unified license across all components.

## Questions or Feedback?

- 💬 [Start a discussion](https://github.com/ROCm/rocm-systems/discussions)
- 🐞 [Open an issue](https://github.com/ROCm/rocm-systems/issues)

We're happy to help!
=======
cd /path/to/aqlprofile
mkdir build
cd build
cmake ..
make -j
```

### Debug Trace Mode (optional; for debugging only)

To enable debug tracing, set the following environment variable before running CMake:

```bash
export CMAKE_DEBUG_TRACE=1
```

This enables verbose debug output of the command packets while this library executes

### Installation

After building, install the AQLProfile libraries with:

```bash
cd build
sudo make install
```

## Support

For issues or questions, please report them in the GitHub Issues section or contact AMD support at <dl.ROCm-Profiler.support@amd.com>.

## License

AQLProfile is open source and distributed under the MIT License. See the LICENSE file for more details.
>>>>>>> 77ec2dd8ee (publish branch)
