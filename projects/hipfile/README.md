# hipFile

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE.md)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11458/badge)](https://www.bestpractices.dev/projects/11458)
[![Coverity](https://scan.coverity.com/projects/hipFile/badge.svg)](https://scan.coverity.com/projects/hipFile)
[![clang-format](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-clang-format.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-clang-format.yml)
[![cmakelint](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-cmakelint.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-cmakelint.yml)
[![codespell](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-codespell.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-codespell.yml)
[![pylint](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-pylint.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-pylint.yml)
[![shellcheck](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-shellcheck.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-shellcheck.yml)

hipFile is AMD's Infinity Storage library that provides direct-to-GPU I/O for the ROCm platform. The library provides C and Python APIs for synchronous, asynchronous, and batch I/O operations. hipFile automatically falls back to POSIX I/O when operations are unable to use the direct-to-GPU path.

> [!NOTE]
> The published documentation is available at [hipFile](https://rocm.docs.amd.com/projects/hipFile/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in `projects/hipfile/docs` in this repository. As with all ROCm projects, the documentation is open source.

## Installing and Using hipFile

See [INSTALL.md](INSTALL.md) in the project root for a list of supported hardware and compilers as well as build and install instructions.

### hipify support

The `amd-develop` branch of [ROCm/HIPIFY](https://github.com/ROCm/HIPIFY) now has
support for hipFile. The hipFile changes are not yet in a public release.

A cuFile --> hipFile API map can be found [here](https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/cuFile_API_supported_by_HIP.md)

### fio support

We've created a fork of [axboe/fio](https://github.com/axboe/fio) at
[ROCm/fio](https://github.com/ROCm/fio). Changes to support a
libhipfile engine can be found in the `hipFile` branch. We package
unofficial releases of this branch [here](https://github.com/ROCm/fio/releases).
