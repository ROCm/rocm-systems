# Installation

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

- Ubuntu 18.04 or Ubuntu 20.04
  - Other OS distributions may be supported but are not tested
- GCC compiler v7+
  - Older GCC compilers may be supported but are not tested
  - Clang compilers are generally supported for [Omnitrace](https://github.com/AMDResearch/omnitrace) but not Dyninst
- [CMake](https://cmake.org/) v3.15+
- [DynInst](https://github.com/dyninst/dyninst) for dynamic or static instrumentation
  - [TBB](https://github.com/oneapi-src/oneTBB) required by Dyninst
  - [ElfUtils](https://sourceware.org/elfutils/) required by Dyninst
  - [LibIberty](https://github.com/gcc-mirror/gcc/tree/master/libiberty) required by Dyninst
  - [Boost](https://www.boost.org/) required by Dyninst
  - [OpenMP](https://www.openmp.org/) optional by Dyninst
- [ROCm](https://rocmdocs.amd.com/en/latest/Installation_Guide/Installation-Guide.html#ubuntu) (optional)
  - HIP
  - Roctracer for HIP API and kernel tracing
  - ROCM-SMI for GPU monitoring
- [PAPI](https://icl.utk.edu/papi/)
- [libunwind](https://www.nongnu.org/libunwind/) for call-stack sampling
- Several optional third-party profiling tools supported by timemory (e.g. TAU, Caliper, CrayPAT, etc.)

## Installing omnitrace from binary distributions

Every omnitrace release provides binary installer scripts of the form:

```shell
omnitrace-{VERSION}-{OS_DISTRIB}-{OS_VERSION}[-ROCm-{ROCM_VERSION}[-{EXTRA}]].sh
```

E.g.:

```shell
omnitrace-0.0.5-Ubuntu-18.04.sh
omnitrace-0.0.5-Ubuntu-18.04-ROCm-4.3.0.sh
omnitrace-0.0.5-Ubuntu-18.04-ROCm-4.5.0.sh
...
omnitrace-0.0.5-Ubuntu-20.04-ROCm-4.5.0-PAPI.sh
omnitrace-0.0.5-Ubuntu-20.04-ROCm-4.5.0-PAPI-MPICH.sh
omnitrace-0.0.5-Ubuntu-20.04-ROCm-4.5.0-PAPI-OpenMPI.sh
```

The EXTRA fields such as PAPI, MPICH, and OpenMPI are built against the libraries provided by the
OS package manager, e.g. `apt-get install libpapi-dev` for Ubuntu.

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
./omnitrace-0.0.5-Ubuntu-18.04-ROCm-4.3.0-PAPI-MPICH.sh --prefix=/opt/omnitrace
```

### Configure the environment

```shell
source /opt/omnitrace/share/omnitrace/setup-env.sh
```

### Test the executables

```shell
omnitrace --help
omnitrace-avail --help
```

## Installing Omnitrace from source

### Installing CMake

If using Ubuntu 20.04, `apt-get install cmake` will install cmake v3.16.3. If using Ubuntu 18.04, the cmake version via apt is too old (v3.10.2). In this case,
follow the instructions [here](https://apt.kitware.com/) to add the CMake apt package repository; or alternatively (if root access is not available),
specific versions of CMake can be easily installed via the Python pip package manager:

```shell
python3 -m pip install 'cmake==3.18.4'
export PATH=${HOME}/.local/bin
```

> NOTE: be wary of using `python3 -m pip install cmake`. If pip installs a cmake version with a `.post<N>` suffix, it will be necessary to
> specify the root path when cmake is invoked.

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
spack external find
spack install dyninst
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
OMNITRACE_ROOT=${HOME}/sw/omnitrace
git clone https://github.com/AMDResearch/omnitrace.git omnitrace-source
cmake                                           \
    -B omnitrace-build                          \
    -DOMNITRACE_USE_MPI_HEADERS=ON              \
    -DCMAKE_INSTALL_PREFIX=${OMNITRACE_ROOT}    \
    omnitrace-source
cmake --build omnitrace-build --target all --parallel 8
cmake --build omnitrace-build --target install
source ${OMNITRACE_ROOT}/share/omnitrace/setup-env.sh
```

#### MPI Support within Omnitrace

[Omnitrace](https://github.com/AMDResearch/omnitrace) can have full (`OMNITRACE_USE_MPI=ON`) or partial (`OMNITRACE_USE_MPI_HEADERS=ON`) MPI support.
The only difference between these two modes is whether or not the results collected via timemory can be aggregated into one output file. The primary
benefits of partial or full MPI support are the automatic wrapping of MPI functions and the ability to label output with suffixes which correspond to the
`MPI_COMM_WORLD` rank ID instead of using the system process identifier (i.e. PID).
In general, it is recommended to use partial MPI support with the OpenMPI headers as this is the most portable configuration.
If full MPI support is selected, make sure your target application is built against the same MPI distribution as omnitrace,
i.e. do not build omnitrace with MPICH and use it on a target application built against OpenMPI.
If partial support is selected, the reason the OpenMPI headers are recommended instead of the MPICH headers is
because the `MPI_COMM_WORLD` in OpenMPI is a pointer to `ompi_communicator_t` (8 bytes), whereas `MPI_COMM_WORLD` in MPICH,
it is an `int` (4 bytes). Building omnitrace with partial MPI support and the MPICH headers and then using
omnitrace on an application built against OpenMPI will cause a segmentation fault due to the value of the `MPI_COMM_WORLD` being narrowed
during the function wrapping before being passed along to the underlying MPI function.
