# Installation

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Operating System

Omnitrace is only supported on Linux.

- Ubuntu 18.04
- Ubuntu 20.04
- OpenSUSE 15.2
- OpenSUSE 15.3
- Other OS distributions may be supported but are not tested

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

## Architecture

At present, only amd64 (x86_64) architectures are tested but Dyninst supports several more architectures.
Thus, omnitrace should support other CPU architectures such as aarch64, ppc64, etc.

## Installing omnitrace from binary distributions

Every omnitrace release provides binary installer scripts of the form:

```shell
omnitrace-{VERSION}-{OS_DISTRIB}-{OS_VERSION}[-ROCm-{ROCM_VERSION}[-{EXTRA}]].sh
```

E.g.:

```shell
omnitrace-1.0.0-ubuntu-18.04-OMPT-PAPI-Python3.sh
omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI-Python3.sh
...
omnitrace-1.0.0-ubuntu-20.04-ROCm-50000-OMPT-PAPI-Python3.sh
```

Any of the EXTRA fields with a cmake build option (e.g. PAPI, see below) or no link requirements (e.g. OMPT) have
self-contained support for these packages.

### Download the appropriate binary distribution

```shell
wget https://github.com/AMDResearch/omnitrace/releases/download/v<VERSION>/<SCRIPT>
```

### Create the target installation directory

```shell
mkdir /opt/omnitrace
```

### Run the installer script

```shell
./omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI.sh --prefix=/opt/omnitrace --exclude-subdir
```

## Installing Omnitrace from source

### Build Requirements

Omnitrace needs a GCC compiler with full support for C++17 and CMake v3.16 or higher.
The Clang compiler may be used in lieu of the GCC compiler if Dyninst is already installed.

- GCC compiler v7+
  - Older GCC compilers may be supported but are not tested
  - Clang compilers are generally supported for [Omnitrace](https://github.com/AMDResearch/omnitrace) but not Dyninst
- [CMake](https://cmake.org/) v3.16+

> ***If the system installed cmake is too old, installing a new version of cmake can be done through several methods.***
> ***One of the easiest options is to use PyPi (i.e. python's pip):***
>
> ```python
> pip install --user 'cmake==3.18.4'
> export PATH=${HOME}/.local/bin:${PATH}`
> ```

### Required Third-Party Packages

- [DynInst](https://github.com/dyninst/dyninst) for dynamic or static instrumentation
  - [TBB](https://github.com/oneapi-src/oneTBB) required by Dyninst
  - [ElfUtils](https://sourceware.org/elfutils/) required by Dyninst
  - [LibIberty](https://github.com/gcc-mirror/gcc/tree/master/libiberty) required by Dyninst
  - [Boost](https://www.boost.org/) required by Dyninst
  - [OpenMP](https://www.openmp.org/) optional by Dyninst
- [libunwind](https://www.nongnu.org/libunwind/) for call-stack sampling

All of the third-party packages required by [DynInst](https://github.com/dyninst/dyninst) and
[DynInst](https://github.com/dyninst/dyninst) itself can be built and installed
during the build of omnitrace itself. In the list below, we list the package, the version,
which package requires the package (i.e. omnitrace requires Dyninst
and Dyninst requires TBB), and the CMake option to build the package alongside omnitrace:

| Third-Party Library | Minimum Version | Required By | CMake Option                              |
|---------------------|-----------------|-------------|-------------------------------------------|
| Dyninst             | 10.0            | Omnitrace   | `OMNITRACE_BUILD_DYNINST` (default: OFF)  |
| Libunwind           |                 | Omnitrace   | `OMNITRACE_BUILD_LIBUNWIND` (default: ON) |
| TBB                 | 2018.6          | Dyninst     | `DYNINST_BUILD_TBB` (default: OFF)        |
| ElfUtils            | 0.178           | Dyninst     | `DYNINST_BUILD_ELFUTILS` (default: OFF)   |
| LibIberty           |                 | Dyninst     | `DYNINST_BUILD_LIBIBERTY` (default: OFF)  |
| Boost               | 1.67.0          | Dyninst     | `DYNINST_BUILD_BOOST` (default: OFF)      |
| OpenMP              | 4.x             | Dyninst     |                                           |

### Optional Third-Party Packages

- [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation_new.html)
  - HIP
  - Roctracer for HIP API and kernel tracing
  - ROCM-SMI for GPU monitoring
- [PAPI](https://icl.utk.edu/papi/)
- MPI
  - `OMNITRACE_USE_MPI` will enable full MPI support
  - `OMNITRACE_USE_MPI_HEADERS` will enable wrapping of the dynamically-linked MPI C function calls
    - By default, if an OpenMPI MPI distribution cannot be found, omnitrace will use a local copy of the OpenMPI mpi.h
- Several optional third-party profiling tools supported by timemory (e.g. [Caliper](https://github.com/LLNL/Caliper), [TAU](https://www.cs.uoregon.edu/research/tau/home.php), CrayPAT, etc.)

| Third-Party Library | CMake Enable Option                        | CMake Build Option                   |
|---------------------|--------------------------------------------|--------------------------------------|
| PAPI                | `OMNITRACE_USE_PAPI` (default: ON)         | `OMNITRACE_BUILD_PAPI` (default: ON) |
| MPI                 | `OMNITRACE_USE_MPI` (default: OFF)         |                                      |
| MPI (header-only)   | `OMNITRACE_USE_MPI_HEADERS` (default: ON)  |                                      |

### Installing DynInst

#### Building Dyninst alongside Omnitrace

The easiest way to install Dyninst is to configure omnitrace with `OMNITRACE_BUILD_DYNINST=ON`. Depending on the version of Ubuntu, the apt package manager may have current enough
versions of Dyninst's Boost, TBB, and LibIberty dependencies (i.e. `apt-get install libtbb-dev libiberty-dev libboost-dev`); however, it is possible to request Dyninst to install
it's dependencies via `Dyninst_BUILD_<DEP>=ON`, e.g.:

```shell
git clone https://github.com/AMDResearch/omnitrace.git omnitrace-source
cmake -B omnitrace-build -DOMNITRACE_BUILD_DYNINST=ON -DDyninst_BUILD_{TBB,ELFUTILS,BOOST,LIBIBERTY}=ON omnitrace-source
```

where `-DDyninst_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON` is expanded by the shell to `-DDyninst_BUILD_TBB=ON -DDyninst_BUILD_BOOST=ON ...`

#### Installing Dyninst via Spack

[Spack](https://github.com/spack/spack) is another option to install Dyninst and it's dependencies:

```shell
git clone https://github.com/spack/spack.git
source ./spack/share/spack/setup-env.sh
spack compiler find
spack external find --all --not-buildable
spack spec -I --reuse dyninst
spack install --reuse dyninst
spack load -r dyninst
```

### Installing omnitrace

Omnitrace has cmake configuration options for supporting MPI (`OMNITRACE_USE_MPI` or `OMNITRACE_USE_MPI_HEADERS`), HIP kernel tracing (`OMNITRACE_USE_ROCTRACER`),
sampling ROCm devices (`OMNITRACE_USE_ROCM_SMI`), OpenMP-Tools (`OMNITRACE_USE_OMPT`), hardware counters via PAPI (`OMNITRACE_USE_PAPI`), among others.
Various additional features can be enabled via the [`TIMEMORY_USE_*` CMake options](https://timemory.readthedocs.io/en/develop/installation.html#cmake-options).
Any `OMNITRACE_USE_<VAL>` option which has a corresponding `TIMEMORY_USE_<VAL>` option means that the support within timemory for this feature has been integrated
into omnitrace's perfetto support, e.g. `OMNITRACE_USE_PAPI=<VAL>` forces `TIMEMORY_USE_PAPI=<VAL>` and the data that timemory is able to collect via this package
is passed along to perfetto and will be displayed when the `.proto` file is visualized in [ui.perfetto.dev](https://ui.perfetto.dev).

```shell
OMNITRACE_ROOT=/opt/omnitrace
git clone https://github.com/AMDResearch/omnitrace.git omnitrace-source
cmake                                       \
    -B omnitrace-build                      \
    -D CMAKE_INSTALL_PREFIX=/opt/omnitrace  \
    -D OMNITRACE_USE_HIP=ON                 \
    -D OMNITRACE_USE_ROCM_SMI=ON            \
    -D OMNITRACE_USE_ROCTRACER=ON           \
    -D OMNITRACE_USE_PYTHON=ON              \
    -D OMNITRACE_USE_OMPT=ON                \
    -D OMNITRACE_USE_MPI_HEADERS=ON         \
    -D OMNITRACE_BUILD_PAPI=ON              \
    -D OMNITRACE_BUILD_LIBUNWIND=ON         \
    -D OMNITRACE_BUILD_DYNINST=ON           \
    -D DYNINST_BUILD_TBB=ON                 \
    -D DYNINST_BUILD_BOOST=ON               \
    -D DYNINST_BUILD_ELFUTILS=ON            \
    -D DYNINST_BUILD_LIBIBERTY=ON           \
    omnitrace-source
cmake --build omnitrace-build --target all --parallel 8
cmake --build omnitrace-build --target install
source /opt/omnitrace/share/omnitrace/setup-env.sh
```

#### MPI Support within Omnitrace

[Omnitrace](https://github.com/AMDResearch/omnitrace) can have full (`OMNITRACE_USE_MPI=ON`) or partial (`OMNITRACE_USE_MPI_HEADERS=ON`) MPI support.
The only difference between these two modes is whether or not the results collected via timemory and/or perfetto can be aggregated into a single
output file during finalization. The primary benefits of partial or full MPI support are the automatic wrapping of MPI functions and the ability
to label output with suffixes which correspond to the `MPI_COMM_WORLD` rank ID instead of using the system process identifier (i.e. PID).
In general, it is recommended to use partial MPI support with the OpenMPI headers as this is the most portable configuration.
If full MPI support is selected, make sure your target application is built against the same MPI distribution as omnitrace,
i.e. do not build omnitrace with MPICH and use it on a target application built against OpenMPI.
If partial support is selected, the reason the OpenMPI headers are recommended instead of the MPICH headers is
because the `MPI_COMM_WORLD` in OpenMPI is a pointer to `ompi_communicator_t` (8 bytes), whereas `MPI_COMM_WORLD` in MPICH,
it is an `int` (4 bytes). Building omnitrace with partial MPI support and the MPICH headers and then using
omnitrace on an application built against OpenMPI will cause a segmentation fault due to the value of the `MPI_COMM_WORLD` being narrowed
during the function wrapping before being passed along to the underlying MPI function.

## Post-Installation Steps

### Configure the environment

If environment modules are available and preferred:

```shell
module use /opt/omnitrace/share/modulefiles
module load omnitrace/1.0.0
```

Alternatively, once can directly source the `setup-env.sh` script:

```shell
source /opt/omnitrace/share/omnitrace/setup-env.sh
```

### Test the executables

Successful execution of these commands indicates that the installation does not have any issues locating the installed libraries:

```shell
omnitrace --help
omnitrace-avail --help
```

> ***NOTE: If ROCm support was enabled, you may have to add the path to the ROCm libraries to `LD_LIBRARY_PATH`, e.g. `export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}`***
