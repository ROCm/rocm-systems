.. meta::
   :description: Instruction on how to install the RCCL library for collective communication primitives using the quick start install script
   :keywords: RCCL, ROCm, library, API, install

.. _install:

*****************************************
Installing RCCL using the install script
*****************************************

To quickly install RCCL using the install script, follow these steps.
For instructions on building RCCL from the source code, see :doc:`building-installing`.
For additional tips, see :doc:`../how-to/rccl-usage-tips`.

Requirements
============

The following prerequisites are required to use RCCL:

1. ROCm-supported GPUs
2. The ROCm stack must be installed on the system, including the :doc:`HIP runtime <hip:index>` and the HIP-Clang compiler.

Quick start RCCL build
======================

RCCL directly depends on the HIP runtime plus the HIP-Clang compiler, which are part of the ROCm software stack.
For ROCm installation instructions, see the :doc:`package manager installation guide <rocm-install-on-linux:install/install-methods/package-manager-index>`.

Use the `install.sh helper script <https://github.com/ROCm/rccl/blob/develop/install.sh>`_,
located in the root directory of the RCCL repository,
to build and install RCCL with a single command. It uses hard-coded configurations that can be specified directly
when using cmake. However, it's a great way to get started quickly and provides an
example of how to build and install RCCL.

Building the library using the install script:
----------------------------------------------

To build the library using the install script, use this command:

.. code-block:: shell

    ./install.sh

For more information on the build options and flags for the install script, run the following command:

.. code-block:: shell

    ./install.sh --help

The RCCL build and installation helper script options are as follows:

.. code-block:: shell

       --address-sanitizer     Build with address sanitizer enabled
    -c|--enable-code-coverage  Enable code coverage
    -d|--dependencies          Install RCCL dependencies
       --debug                 Build debug library
       --debug-fast            Build debug library with lto optimization disabled (fast build times)
       --enable_backtrace      Build with custom backtrace support
       --disable-colltrace     Build without collective trace
       --enable-msccl-kernel   Build with MSCCL kernels
       --dump-asm              Disassemble code and dump assembly with inline code
       --enable-mscclpp        Build with MSCCL++ support
       --enable-mscclpp-clip   Build MSCCL++ with clip wrapper on bfloat16 and half addition routines
       --disable-roctx         Build without ROCTX logging
    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)
    -h|--help                  Prints this help message
    -i|--install               Install RCCL library (see --prefix argument below)
    -j|--jobs                  Specify how many parallel compilation jobs to run (128 by default)
       --kernel-resource-use   Dump GPU kernel resource usage (e.g., VGPRs, scratch, spill) at link stage
    -l|--local_gpu_only        Only compile for local GPU architecture
       --amdgpu_targets        Only compile for specified GPU architecture(s). For multiple targets, separate by ';' (builds for all supported GPU architectures by default)
       --no_clean              Don't delete files if they already exist
       --npkit-enable          Compile with npkit enabled
       --log-trace             Build with log trace enabled (i.e. NCCL_DEBUG=TRACE)
       --enable-mpi-tests      Enable MPI-based tests (requires --debug and MPI installation; set MPI_PATH if not in /opt/ompi)
       --openmp-test-enable    Enable OpenMP in rccl unit tests
    -p|--package_build         Build RCCL package
       --prefix                Specify custom directory to install RCCL to (default: `/opt/rocm`)
       --run_tests_all         Run all rccl unit tests (must be built already)
    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)
       --static                Build RCCL as a static library instead of shared library
    -t|--tests_build           Build rccl unit tests, but do not run
       --time-trace            Plot the build time of RCCL (requires `ninja-build` package installed on the system)
       --verbose               Show compile commands
       --force-reduce-pipeline Force reduce_copy sw pipeline to be used for every reduce-based collectives and datatypes
       --generate-sym-kernels  Generate symmetric memory kernels
    -q|--quiet-warnings        Suppress majority of compiler warnings (not recommended)
       --rocshmem              Build with rocSHMEM support
       --cmake-options         Pass additional CMake options (e.g. --cmake-options "-DFOO=BAR -DBAZ=ON")

  Available RCCL-specific CMake options for --cmake-options:
    -DBUILD_EXT_EXAMPLES=ON               Build ext-{net,tuner,profiler} example plugins (default: OFF)
    -DENABLE_MSCCLPP_EXECUTOR=ON          Enable MSCCL++ Executor (default: OFF)
    -DENABLE_MSCCLPP_FORMAT_CHECKS=ON     Enable formatting checks in MSCCL++ (default: OFF)
    -DMSCCLPP_APPLY_PATCHES=OFF           Disable source code patches for MSCCL++ (default: ON)
    -DENABLE_IFC=ON                       Enable indirect function call (default: OFF)
    -DPROFILE=ON                          Enable profiling (default: OFF)
    -DTIMETRACE=ON                        Enable time-trace during compilation (default: OFF)
    -DFAULT_INJECTION=OFF                 Disable fault injection (default: ON)
    -DDWORDX4_INTRINSICS=OFF              Disable dwordx4 intrinsics (default: ON)
    -DENABLE_COMPRESS=OFF                 Disable GPU code compression (default: ON)
    -DRCCL_ROCPROFILER_REGISTER=OFF       Disable rocprofiler-register support (default: ON)

  Environment variables:
    ONLY_FUNCS                 Build only specified collective functions (debug builds only).
                               Restricts GPU kernel generation to the listed collectives, significantly
                               reducing build time during development. Use '|' to separate multiple functions.
                               Example: ONLY_FUNCS="AllReduce|SendRecv" ./install.sh --debug -t
                               Available: AllReduce, Broadcast, Reduce, AllGather, ReduceScatter,
                                          AlltoAllPivot, SendRecv, AlltoAllGda, AlltoAllvGda
                               Advanced: Specify algo, protocol, redop, and type per collective.
                                 ONLY_FUNCS="AllReduce RING SIMPLE Sum f32|SendRecv"

.. tip::

    By default, the RCCL install script builds all the GPU targets that are defined in ``DEFAULT_GPUS`` in `CMakeLists.txt <https://github.com/ROCm/rccl/blob/develop/CMakeLists.txt>`_.
    To target specific GPUs and potentially reduce the build time, use ``--amdgpu_targets`` along with
    a semicolon (``;``) separated string list of the GPU targets.
