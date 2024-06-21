# Installation

## Operating System

ROCprofiler is only supported on Linux. The following distributions are tested:

- Ubuntu 20.04
- Ubuntu 22.04
- OpenSUSE 15.4
- RedHat 8.8

Other OS distributions may be supported but have yet to be tested.

### Identifying the Operating System

If you are unsure of the operating system and version, the `/etc/os-release` and `/usr/lib/os-release` files contain
operating system identification data for Linux systems.

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

## Installing ROCprofiler from source

### Build Requirements

ROCprofiler needs a CMake (https://cmake.org/) version 3.21 or higher.

***If the system installed 'CMake' is too old, installing a new version can be done through several methods. One of the easiest options is to use PyPi (i.e., python’s pip):***

```bash
pip install --user 'cmake==3.22.0'
export PATH=${HOME}/.local/bin:${PATH}
```

### Building ROCprofiler

```bash
git clone https://git@github.com:ROCm/rocprofiler-sdk.git rocprofiler-sdk-source  
cmake                                         \
      -B rocprofiler-sdk-build                \
      -D ROCPROFILER_BUILD_TESTS=ON           \
      -D ROCPROFILER_BUILD_SAMPLES=ON         \
      -D ROCPROFILER_BUILD_DOCS=ON            \
      -D CMAKE_INSTALL_PREFIX=/opt/rocm       \
       rocprofiler-sdk-source

cmake --build rocprofiler-sdk-build --target all --parallel 8  
```

### Testing ROCprofiler

To run the built tests, cd into the `rocprofiler-sdk-build` directory and run:

```bash
ctest --output-on-failure -O ctest.all.log
```

### Installing ROCprofiler

To install ROCprofiler from the `rocprofiler-sdk-build` directory, run:

```bash
cmake --build rocprofiler-sdk-build --target install
```
