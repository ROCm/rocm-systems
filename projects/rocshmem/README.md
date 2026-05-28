# ROCm OpenSHMEM (rocSHMEM)

The ROCm OpenSHMEM (rocSHMEM) runtime is part of an AMD and AMD Research
initiative to provide GPU-centric networking through an OpenSHMEM-like interface.
This intra-kernel networking library simplifies application
code complexity and enables more fine-grained communication/computation
overlap than traditional host-driven networking.
rocSHMEM uses a single symmetric heap that is allocated on GPU memories.

There are currently three backends for rocSHMEM;
IPC, Reverse Offload (RO), and GDA.
The backends primarily differ in their implementations of intra-kernel networking.

The IPC backend implements communication primitives using load/store operations issued from the GPU.

The Reverse Offload (RO) backend has the GPU runtime forward rocSHMEM networking operations
to the host-side runtime, which calls into a traditional MPI or OpenSHMEM
implementation. This forwarding of requests is transparent to the
programmer, who only sees the GPU-side interface.

The GPU Direct Async (GDA) backend allows for rocSHMEM to issue communication operations to the NIC directly from the device-side code, without involving a CPU proxy.
within the GPU.
During initialization we prepare network resources for each NIC vendor using the vendor-appropriate
Direct Verbs APIs.
When calling the device-side rocSHMEM API, the device threads are used to construct Work Queue Entries (WQEs) and post the communication to the send queues of the NIC directly.
Completion Queues (CQs) are polled from the device-side code as well.

The RO and GDA backend is provided as-is with limited support from AMD or AMD Research.

## Requirements

rocSHMEM base requirements:
* ROCm v6.2.2 onwards
    *  May work with other versions, but it has not been tested

* The following AMD GPUs have been fully tested for compatibility with rocSHMEM:
  * MI250X
  * MI300X
  * MI350X (Requires ROCm 7.0 or later)

  > **Note:** Other AMD GPUs might function with unknown limitations. For the complete list of supported hardware, see [ROCm System Requirements](https://rocm.docs.amd.com/projects/install-on-linux-internal/en/latest/reference/system-requirements.html).

* ROCm-aware Open MPI and UCX as described in
  [Building the Dependencies](#building-the-dependencies)

rocSHMEM only supports HIP applications. There are no plans to port to
OpenCL.

## Building and Installation

rocSHMEM uses the CMake build system. The CMakeLists file contains
additional details about library options.

To create an out-of-source build for the IPC backend for single-node use-cases:

```
mkdir build
cd build
../scripts/build_configs/ipc_single
```

To create an out-of-source build for the RO backend for multi-node use-cases that can also utilize the IPC mechanisms for certain intra-node operations:

```
mkdir build
cd build
../scripts/build_configs/ro_ipc
```

To create an out-of-source build for the GDA backend, we do the following.
Ensure you select the correct NIC vendor script

```
mkdir build
cd build
../scripts/build_configs/gda_<vendor>
```

The build script passes configuration options to CMake to setup canonical builds.
There are other scripts in `./scripts/build_configs`
directory but currently, only `ipc_single` and `ro_ipc` is supported.

By default, the library is installed in `~/rocshmem`. You may provide a
custom install path by supplying it as an argument. For example:

```
../scripts/build_configs/ipc_single /path/to/install
```

If you have built dependencies in a non-standard path (for example using instructions from [Building the Dependencies](#building-the-dependencies)), you may have to set the following variables to find the dependencies:

```
MPI_ROOT=/path/to/openmpi UCX_ROOT=/path/to/ucx CMAKE_PREFIX_PATH="/path/to/rocm:$CMAKE_PREFIX_PATH" ../scripts/build_configs/ipc_single /path/to/install
```

## Compiling/Linking and Running with rocSHMEM

rocSHMEM is built as a library that can be statically
linked to your application during compilation using `hipcc`.

During the compilation of your application, include the rocSHMEM header files
and the rocSHMEM library when using hipcc.
Since rocSHMEM depends on MPI you will need to link to an MPI library.
The arguments for MPI linkage must be added manually
as opposed to using mpicc.

When using hipcc directly (as opposed to through a build system), we
recommend performing the compilation and linking steps separately.
At the top of the examples files (`./examples/*`),
example compile and link commands are provided:

```
# Compile
hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc \
  -I/opt/rocm/include                                \
  -I$ROCSHMEM_INSTALL_DIR/include                    \
  -I$OPENMPI_UCX_INSTALL_DIR/include/

# Link
hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test \
  $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                       \
  $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                        \
  -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

```

If your project uses cmake,
you may find the
[Using CMake with AMD ROCm](https://rocmdocs.amd.com/en/latest/conceptual/cmake-packages.html)
page useful.

## Runtime Parameters
rocSHMEM has the following enviroment variables:

```
    ROCSHMEM_HEAP_SIZE (default : 1 GB)
                        Defines the size of the rocSHMEM symmetric heap
                        Note the heap is on the GPU memory.

    ROCSHMEM_RO_DISABLE_MIXED_IPC (default : 0)
                        Disables IPC support for the RO or GDA backends.

    ROCSHMEM_MAX_NUM_CONTEXTS (default : 1024)
                        Maximum number of contexts used in library.

    ROCSHMEM_MAX_NUM_TEAMS (default : 40)
                        Maximum number of teams supported by the library.

    ROCSHMEM_GDA_ALTERNATE_QP_PORTS (default : 1)
                        Enables/Disables having QPs alternate their mappings
                        across rocSHMEM contexts.
```

## Examples

rocSHMEM is similar to OpenSHMEM and should be familiar to programmers who
have experience with OpenSHMEM or other PGAS network programming APIs in the
context of CPUs.
The best way to learn how to use rocSHMEM is to read the functions described in
headers in the dirctory `./include/rocshmem/`,
or to look at the provided example code in the `./example/` directory.
The examples can be run like so:

```
mpirun --map-by numa --mca pml ucx --mca osc ucx -np 2 ./build/examples/rocshmem_getmem_test
```

## Tests
rocSHMEM is shipped with a functional and unit test suite for the supported rocSHMEM API.
They test Puts, Gets, nonblocking Puts,
nonblocking Gets, Quiets, Atomics, Tests, Wait-untils, Broadcasts, Reductions, and etc.
To run the tests, you may use the driver scripts provided in the `./scripts/` directory:

```
# Run Functional Tests
./scripts/functional_tests/driver.sh ./build/tests/functional_tests/rocshmem_functional_tests all <log_directory>

# Run Unit Tests
./scripts/unit_tests/driver.sh ./build/tests/unit_tests/rocshmem_unit_tests all
```

## Code Coverage
rocSHMEM targets 80% code coverage in both unit and functional tests.  To check the coverage report for your
changes, we have a helper script you can use to build, test and generate the coverage report in a single step.

Because we need to build all 3 of `ipc`, `ro_net` and `ro_ipc`, the `codecov` script is run from the context of
the `build/` directory and will create and build to the 3 directories, with instrumented code. It will then start
a python http server where you can navigate to the link to view the coverage report.

```
cd rocSHMEM
mkdir build && cd build
../scripts/build_configs/codecov
```

## Building the Dependencies

rocSHMEM requires a ROCm-Aware Open MPI and UCX.
Other MPI implementations, such as MPICH,
_should_ be compatible with rocSHMEM but it has not been thoroughly tested.

## Building the Dependencies using our Helper Script
We have a script to install dependencies.
However, it is not guaranteed to work and perform optimally on all platforms.
Configuration options are platform dependent.

```
BUILD_DIR=/path/to/not_rocshmem_src_or_build/dependencies /path/to/rocshmem_src/sripts/install_dependencies.sh
```

After compiling and installing UCX and Open MPI, please update your `PATH` and `LD_LIBRARY_PATH`
to point to the installation locations.
The exact locations will be printed if the script ran successfully.

## Building the Dependencies from Source
To build and configure ROCm-Aware UCX (1.17.0 or later), you need to:

```
git clone https://github.com/openucx/ucx.git -b v1.17.x
cd ucx
./autogen.sh
./configure --prefix=<ucx_install_dir> --with-rocm=<rocm_path> --enable-mt
make -j 8
make -j 8 install
```

Then, you need to build Open MPI (5.0.6 or later) with UCX support.

```
git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
cd ompi
./autogen.pl
./configure --prefix=<ompi_install_dir> --with-rocm=<rocm_path> --with-ucx=<ucx_install_dir>
make -j 8
make -j 8 install
```

After compiling and installing UCX and Open MPI, please update your PATH and LD_LIBRARY_PATH to point to the installation locations, e.g.

```
export PATH=<ompi_install_dir>/bin:$PATH
export LD_LIBRARY_PATH=<ompi_install_dir>/lib:<ucx_install_dir>/lib:$LD_LIBRARY_PATH
```

For more information on OpenMPI-UCX support, please visit:
https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html
