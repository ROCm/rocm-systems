# Installation

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Quick Start (Latest Release, Binary Installer)

TODO: Installation quick start

## Operating System

TODO: supported OSes

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

The relevent fields are `ID` and the `VERSION_ID`.

## Installing ROCprofiler from source

### Build Requirements

TODO: build reqs

### Building ROCprofiler

TODO: cmake build

### Testing ROCprofiler

TODO: ctest

### Installing ROCprofiler

TODO: `make install` and/or `cpack`
