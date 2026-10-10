# RCCL

ROCm Communication Collectives Library

[![RCCL](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Frccl?repoName=ROCm%2Frccl&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=107&repoName=ROCm%2Frccl&branchName=develop)
[![TheRock CI](https://github.com/ROCm/rccl/actions/workflows/therock-ci.yml/badge.svg?branch=develop&event=push)](https://github.com/ROCm/rccl/actions/workflows/therock-ci.yml)

> **Note:** The published documentation is available at [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/index.html) in an organized easy-to-read format that includes a table of contents and search functionality. The documentation source files reside in the [rccl/docs](https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl/docs) folder in this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Introduction

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations.  It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

Additionally, RCCL supports zero-CU (zero-CTA) collectives for selected operations,
including all-gather, all-to-all, gather, and scatter. These collectives use SDMA/Copy
Engine (CE) hardware to offload data movement, enabling communication without consuming
GPU compute units. This path is opt-in: initialize the communicator with
NCCL_CTA_POLICY_ZERO and register symmetric buffers via ncclCommWindowRegister
with NCCL_WIN_COLL_SYMMETRIC (single-node only; requires ROCm 7.12+). On MI350,
CE collectives can outperform CU-based collectives for large message sizes and improve
overlap when computation and communication run concurrently.

## Requirements

1. ROCm supported GPUs
2. ROCm stack installed on the system (HIP runtime & HIP-Clang)

## Quickstart RCCL Build

RCCL directly depends on HIP runtime plus the HIP-Clang compiler, which are part of the ROCm software stack.
For ROCm installation instructions, see https://github.com/ROCm/ROCm.

The root of this repository has a helper script `install.sh` to build and install RCCL with a single command. It hard-codes configurations that can be specified through invoking cmake directly, but it's a great way to get started quickly and can serve as an example of how to build/install RCCL.

### To build the library using the install script:

```shell
./install.sh
```

For the current build options and defaults, use:

```shell
./install.sh --help
```

The ACCL timing-decomposition profiler is included by default on Linux. To omit
it, pass `--cmake-options "-DBUILD_PROFILER_ACCL=OFF"`. See the
[profiler documentation](plugins/profiler/accl/README.md) for usage and validation.

By default, RCCL builds for all GPU targets defined in `DEFAULT_GPUS` in `CMakeLists.txt`. To target specific GPU(s), and potentially reduce build time, use `--amdgpu_targets` as a `;` separated string listing GPU(s) to target.

## Manual build

### To build the library using CMake:

```shell
$ git clone --recursive https://github.com/ROCm/rccl.git
$ cd rccl
$ mkdir build
$ cd build
$ cmake ..
$ make -j 16      # Or some other suitable number of parallel jobs
```
If you have already cloned, you can check out the remaining git submodules manually. rocSHMEM is **not** a submodule; to build RCCL with rocSHMEM from CMake, set `ROCSHMEM_INSTALL_DIR` or `ROCSHMEM_SOURCE_DIR` as described under [rocSHMEM support](#rocshmem-support) below.
```shell
$ git submodule update --init --recursive --depth=1
```
You may substitute an installation path of your own choosing by passing `CMAKE_INSTALL_PREFIX`. For example:
```shell
$ cmake -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install -DCMAKE_BUILD_TYPE=Release ..
```
Note: ensure rocm-cmake is installed, `apt install rocm-cmake`.

### To build the RCCL package and install package :

Assuming you have already cloned this repository and built the library as shown in the previous section:

```shell
$ cd rccl/build
$ make package
$ sudo dpkg -i *.deb
```

RCCL package install requires sudo/root access because it installs under `/opt/rocm/`. This is an optional step as RCCL can instead be used directly by including the path containing `librccl.so`.

## Docker build

Refer to [docker/README.md](docker/README.md "docker/README.md")

Python wheel :
```shell
$ # Install uv to create the Python wheel (uv manages Python deps in a venv)
$ # See: https://docs.astral.sh/uv/getting-started/installation/
$ curl -LsSf https://astral.sh/uv/install.sh | sh
$ # Build NCCL Python wheel (this also builds the .txz archive as an intermediate)
$ make pkg.python_wheel.build
$ ls build/pkg/python_wheel/
```

## Tests

There are rccl unit tests implemented with the Googletest framework in RCCL.  The rccl unit tests require Googletest 1.10 or higher to build and execute properly (installed with the -d option to install.sh).
To invoke the rccl unit tests, go to the build folder, then the test subfolder, and execute the appropriate rccl unit test executable(s).

rccl unit test names are now of the format:

    CollectiveCall.[Type of test]

Filtering of rccl unit tests should be done with environment variable and by passing the `--gtest_filter` command line flag, for example:

```shell
UT_DATATYPES=ncclBfloat16 UT_REDOPS=prod ./rccl-UnitTests --gtest_filter="AllReduce.C*"
```

will run only AllReduce correctness tests with float16 datatype. A list of available filtering environment variables appears at the top of every run. See "Running a Subset of the Tests" at https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests for more information on how to form more advanced filters.

There are also other performance and error-checking tests for RCCL.  These are maintained separately at https://github.com/ROCm/rccl-tests.
See the rccl-tests README for more information on how to build and run those tests.

## rocSHMEM support

RCCL can use rocSHMEM's GPU Direct Async (GDA) backend to accelerate the **AllToAll** collective on supported multi-node setups. This is the only collective that currently uses rocSHMEM GDA inside RCCL.

Please consult the [rocSHMEM documentation](https://rocm.docs.amd.com/projects/rocSHMEM/en/latest/install.html#gda-nic-dependencies) to see which NICs and drivers are required for GDA alltoall support.

**Building with rocSHMEM**

- Using the install script:
  ```shell
  ./install.sh --rocshmem
  ```
  By default (without `ROCSHMEM_INSTALL_DIR`), the script creates a sparse git worktree of the mono-repo at a pinned commit and passes that rocSHMEM tree to CMake as `ROCSHMEM_SOURCE_DIR`, so RCCL builds rocSHMEM via CMake `ExternalProject`. To use an already-built rocSHMEM instead, set `ROCSHMEM_INSTALL_DIR` to its install prefix before running the script.

- **Manual CMake (without `install.sh`)**
  You need InfiniBand Verbs development libraries on the system (`libibverbs`; e.g. `rdma-core` / `libibverbs-dev` on Debian/Ubuntu). Then enable rocSHMEM and supply **either** a pre-built install prefix **or** a path to the rocSHMEM CMake source tree (the directory that contains rocSHMEM’s top-level `CMakeLists.txt`, e.g. `projects/rocshmem` in the rocm-systems mono-repo):

  ```shell
  # Option A — link against an existing rocSHMEM installation
  cmake -DENABLE_ROCSHMEM=ON -DROCSHMEM_INSTALL_DIR=/path/to/rocshmem/prefix ..

  # Option B — build rocSHMEM from source as part of the RCCL build
  cmake -DENABLE_ROCSHMEM=ON -DROCSHMEM_SOURCE_DIR=/path/to/rocshmem/source ..
  ```

  If neither `ROCSHMEM_INSTALL_DIR` (with a successful `find_package(rocshmem_static)`) nor `ROCSHMEM_SOURCE_DIR` is set, configuration fails with an error directing you to set `ROCSHMEM_SOURCE_DIR` (or use `install.sh --rocshmem`).

**Runtime behavior**

Users must set the following environment variables:

- **`RCCL_ROCSHMEM_ENABLE`** (default: `1`): Set to `0` to disable rocSHMEM usage in RCCL.
- **`RCCL_ROCSHMEM_THRESHOLD`** (default: `262144` bytes): Maximum AllToAll message size (in bytes) for which the GDA path is used. The GDA path is only considered when this value is ≤ 1 MiB (1048576); larger thresholds fall back to the standard AllToAll implementation.

The GDA AllToAll path is selected only when all of the following hold: rocSHMEM is enabled at build and runtime, the GPU architecture is gfx942 (e.g. MI300X), the job is multi-node with 8 GPUs per node, and the AllToAll message size is ≤ `RCCL_ROCSHMEM_THRESHOLD`.

## Library and API Documentation

Please refer to the [RCCL Documentation Site](https://rocm.docs.amd.com/projects/rccl/en/latest/) for current documentation.

### How to build documentation

Run the steps below to build documentation locally.

```shell
cd docs
pip3 install -r sphinx/requirements.txt
python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

## Copyright

All source code and accompanying documentation is copyright (c) 2015-2025, NVIDIA CORPORATION. All rights reserved.

All modifications are copyright (c) 2019-2025 Advanced Micro Devices, Inc. All rights reserved.
