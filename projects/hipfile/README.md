# hipFile

> [!CAUTION] 
> This release is an *early-access* software technology preview. Running production workloads is *not* recommended.

[![MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/ROCm/hipFile/develop/LICENSE.md)
[![CI](https://github.com/ROCm/hipFile/actions/workflows/root-ci.yml/badge.svg)](https://github.com/ROCm/hipFile/actions/workflows/root-ci.yml)
[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/11458/badge)](https://www.bestpractices.dev/projects/11458)
[![CodeQL](https://github.com/ROCm/hipFile/actions/workflows/github-code-scanning/codeql/badge.svg)](https://github.com/ROCm/hipFile/actions/workflows/github-code-scanning/codeql)
[![Coverity](https://scan.coverity.com/projects/hipFile/badge.svg)](https://scan.coverity.com/projects/hipFile)
[![clang-format](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-clang-format.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-clang-format.yml)
[![cmakelint](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-cmakelint.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-cmakelint.yml)
[![codespell](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-codespell.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-codespell.yml)
[![pylint](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-pylint.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-pylint.yml)
[![shellcheck](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-shellcheck.yml/badge.svg?branch=develop)](https://github.com/ROCm/rocm-systems/actions/workflows/hipfile-shellcheck.yml)

AMD Infinity Storage library that supports IO directly to the GPU.

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
