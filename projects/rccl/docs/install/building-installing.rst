.. meta::
   :description: Information on how to build the RCCL library from source code
   :keywords: RCCL, ROCm, library, API, build, install

.. _building-from-source:

*************************
Building RCCL from source
*************************

To build RCCL as part of the ROCm Core SDK, see `TheRock build instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build RCCL standalone using the following
instructions.

To build RCCL directly from the source code, follow these steps. This guide also includes
instructions explaining how to test the build.
For information on using the quick start install script to build RCCL, see :doc:`installation`.

Requirements
============

The following prerequisites are required to build RCCL:

1. ROCm-supported GPUs
2. Having the ROCm stack installed on the system, including the :doc:`HIP runtime <hip:index>` and the HIP-Clang compiler.

Build and install RCCL using the install script
===============================================

To quickly install RCCL using the install script, follow these steps.
For additional tips, see :doc:`../how-to/rccl-usage-tips`.

Quick start RCCL build
----------------------

RCCL directly depends on the HIP runtime plus the HIP-Clang compiler, which are part of the ROCm software stack.
For ROCm installation instructions, see the :doc:`package manager installation guide <rocm-install-on-linux:install/install-methods/package-manager-index>`.

Use the `install.sh helper script <https://github.com/ROCm/rccl/blob/develop/install.sh>`_,
located in the root directory of the RCCL repository,
to build and install RCCL with a single command. It uses hard-coded configurations that can be specified directly
when using cmake. However, it's a great way to get started quickly and provides an
example of how to build and install RCCL.

Run the install script
----------------------

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
       --dump-asm              Disassemble code and dump assembly with inline code
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

Building the library using CMake
================================

To build the library from source, follow these steps:

.. code-block:: shell

    git clone --recursive https://github.com/ROCm/rccl.git
    cd rccl
    mkdir build
    cd build
    cmake ..
    make -j 16      # Or some other suitable number of parallel jobs

If you have already cloned the repository, you can checkout the external submodules manually.

.. code-block:: shell

    git submodule update --init --recursive --depth=1

You can substitute a different installation path by providing the path as a parameter
to ``CMAKE_INSTALL_PREFIX``, for example:

.. code-block:: shell

    cmake -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install -DCMAKE_BUILD_TYPE=Release ..

.. note::

    Ensure ROCm CMake is installed using the command ``apt install rocm-cmake``. By default,
    CMake builds the component in debug mode unless ``DCMAKE_BUILD_TYPE`` is specified.


Building the RCCL package and install package:
----------------------------------------------

After you have cloned the repository and built the library as described in the previous section,
use this command to build the package on Debian-based distros:

.. code-block:: shell

    cd rccl/build
    make package
    sudo dpkg -i *.deb

.. note::
   
   The RCCL package install process requires ``sudo`` or root access because it creates a directory
   named ``rccl`` in ``/opt/rocm/``. This is an optional step. RCCL can be used directly by including the path containing ``librccl.so``.

Testing RCCL
============

The RCCL unit tests are implemented using the Googletest framework in RCCL. These unit tests require Googletest 1.10
or higher to build and run (this dependency can be installed using the ``-d`` option for ``install.sh``).
To run the RCCL unit tests, go to the ``build`` folder and the ``test`` subfolder,
then run the appropriate RCCL unit test executables.

The RCCL unit test names follow this format:

.. code-block:: shell

    CollectiveCall.[Type of test]

Filtering of the RCCL unit tests can be done using environment variables
and by passing the ``--gtest_filter`` command line flag:

.. code-block:: shell

    UT_DATATYPES=ncclBfloat16 UT_REDOPS=prod ./rccl-UnitTests --gtest_filter="AllReduce.C*"

This command runs only the ``AllReduce`` correctness tests with the ``float16`` datatype.
A list of the available environment variables for filtering appears at the top of every run.
See the `Googletest documentation <https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests>`_
for more information on how to form advanced filters.

There are also other performance and error-checking tests for RCCL. They are maintained separately at `<https://github.com/ROCm/rccl-tests>`_.

.. note::

    For more information on how to build and run rccl-tests, see the `rccl-tests README file <https://github.com/ROCm/rccl-tests/blob/develop/README.md>`_ .
