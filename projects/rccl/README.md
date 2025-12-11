# RCCL

ROCm Communication Collectives Library

[![RCCL](https://dev.azure.com/ROCm-CI/ROCm-CI/_apis/build/status%2Frccl?repoName=ROCm%2Frccl&branchName=develop)](https://dev.azure.com/ROCm-CI/ROCm-CI/_build/latest?definitionId=107&repoName=ROCm%2Frccl&branchName=develop)
[![TheRock CI](https://github.com/ROCm/rccl/actions/workflows/therock-ci.yml/badge.svg?branch=develop&event=push)](https://github.com/ROCm/rccl/actions/workflows/therock-ci.yml)

> **Note:** The published documentation is available at [RCCL](https://rocm.docs.amd.com/projects/rccl/en/latest/index.html) in an organized easy-to-read format that includes a table of contents and search functionality. The documentation source files reside in the [rccl/docs](https://github.com/ROCm/rccl/tree/develop/docs) folder in this repository. As with all ROCm projects, the documentation is open source. For more information, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Introduction

RCCL (pronounced "Rickle") is a stand-alone library of standard collective communication routines for GPUs, implementing all-reduce, all-gather, reduce, broadcast, reduce-scatter, gather, scatter, and all-to-all. There is also initial support for direct GPU-to-GPU send and receive operations.  It has been optimized to achieve high bandwidth on platforms using PCIe, xGMI as well as networking using InfiniBand Verbs or TCP/IP sockets. RCCL supports an arbitrary number of GPUs installed in a single node or multiple nodes, and can be used in either single- or multi-process (e.g., MPI) applications.

The collective operations are implemented using ring and tree algorithms and have been optimized for throughput and latency. For best performance, small operations can be either batched into larger operations or aggregated through the API.

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

For more info on build options/flags when using the install script, use `./install.sh --help`
```shell
./install.sh --help
RCCL build & installation helper script
 Options:
       --address-sanitizer     Build with address sanitizer enabled
    -c|--enable-code-coverage  Enable code coverage
    -d|--dependencies          Install RCCL dependencies
       --debug                 Build debug library
       --enable_backtrace      Build with custom backtrace support
       --disable-colltrace     Build without collective trace
       --disable-msccl-kernel  Build without MSCCL kernels
       --enable-mscclpp        Build with MSCCL++ support
       --enable-mscclpp-clip   Build MSCCL++ with clip wrapper on bfloat16 and half addition routines
       --disable-roctx         Build without ROCTX logging
    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)
    -h|--help                  Prints this help message
    -i|--install               Install RCCL library (see --prefix argument below)
    -j|--jobs                  Specify how many parallel compilation jobs to run ($nproc by default)
    -l|--local_gpu_only        Only compile for local GPU architecture
       --amdgpu_targets        Only compile for specified GPU architecture(s). For multiple targets, separate by ';' (builds for all supported GPU architectures by default)
       --no_clean              Don't delete files if they already exist
       --npkit-enable          Compile with npkit enabled
       --log-trace             Build with log trace enabled (i.e. NCCL_DEBUG=TRACE)
       --openmp-test-enable    Enable OpenMP in rccl unit tests
    -p|--package_build         Build RCCL package
       --prefix                Specify custom directory to install RCCL to (default: `/opt/rocm`)
       --run_tests_all         Run all rccl unit tests (must be built already)
    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)
       --static                Build RCCL as a static library instead of shared library
    -t|--tests_build           Build rccl unit tests, but do not run
       --time-trace            Plot the build time of RCCL (requires `ninja-build` package installed on the system)
       --verbose               Show compile commands
```

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
If you have already cloned, you can checkout the external submodules manually.
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
