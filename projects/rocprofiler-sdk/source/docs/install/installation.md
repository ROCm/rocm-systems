---
myst:
    html_meta:
        "description": "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
        "keywords": "ROCprofiler-SDK installation, Install ROCprofiler-SDK, Build ROCprofiler-SDK"
---

# ROCprofiler-SDK installation

This document provides information required to install ROCprofiler-SDK from source.

## Supported systems

ROCprofiler-SDK is supported only on Linux. The following distributions are tested:

- Ubuntu 20.04
- Ubuntu 22.04
- OpenSUSE 15.4
- RedHat 8.8

ROCprofiler-SDK might operate as expected on other [Linux distributions](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems), but has not been tested.

### Identifying the operating system

To identify the Linux distribution and version, see the `/etc/os-release` and `/usr/lib/os-release` files:

```shell
$ cat /etc/os-release
NAME="Ubuntu"
VERSION="20.04.4 LTS (Focal Fossa)"
ID=ubuntu
...
VERSION_ID="20.04"
...
```

The relevant fields are `ID` and the `VERSION_ID`.

## Build requirements

Install [CMake](https://cmake.org/) version 3.21 (or later).

:::{note}
If the `CMake` installed on the system is too old, you can install a new version using various methods. One of the easiest options is to use PyPi (Python’s pip).
:::

```bash
pip install --user 'cmake==3.22.0'
export PATH=${HOME}/.local/bin:${PATH}
```

## Building ROCprofiler-SDK

```bash
git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
cmake                                         \
      -B rocprofiler-sdk-build                \
      -D ROCPROFILER_BUILD_TESTS=ON           \
      -D ROCPROFILER_BUILD_SAMPLES=ON         \
      -D CMAKE_INSTALL_PREFIX=/opt/rocm       \
       rocprofiler-sdk-source

cmake --build rocprofiler-sdk-build --target all --parallel 8
```

## Installing ROCprofiler-SDK

To install ROCprofiler-SDK from the `rocprofiler-sdk-build` directory, run:

```bash
cmake --build rocprofiler-sdk-build --target install
```

## Testing ROCprofiler-SDK

To run the built tests, `cd` into the `rocprofiler-sdk-build` directory and run:

```bash
ctest --output-on-failure -O ctest.all.log
```
