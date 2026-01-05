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
    -d|--dependencies          Install RCCL dependencies
       --debug                 Build debug library
       --enable_backtrace      Build with custom backtrace support
       --disable-colltrace     Build without collective trace
       --disable-msccl-kernel  Build without MSCCL kernels
       --enable-mscclpp        Build with MSCCL++ support
    -f|--fast                  Quick-build RCCL (local gpu arch only, no backtrace, and collective trace support)
    -h|--help                  Prints this help message
    -i|--install               Install RCCL library (see --prefix argument below)
    -j|--jobs                  Specify how many parallel compilation jobs to run ($nproc by default)
    -l|--local_gpu_only        Only compile for local GPU architecture
       --amdgpu_targets        Only compile for specified GPU architecture(s). For multiple targets, separate by ';' (builds for all supported GPU architectures by default)
       --no_clean              Don't delete files if they already exist
       --npkit-enable          Compile with npkit enabled
       --openmp-test-enable    Enable OpenMP in rccl unit tests
       --roctx-enable          Compile with roctx enabled (example usage: rocprof --roctx-trace ./rccl-program)
    -p|--package_build         Build RCCL package
       --prefix                Specify custom directory to install RCCL to (default: `/opt/rocm`)
       --rm-legacy-include-dir Remove legacy include dir Packaging added for file/folder reorg backward compatibility
       --run_tests_all         Run all rccl unit tests (must be built already)
    -r|--run_tests_quick       Run small subset of rccl unit tests (must be built already)
       --static                Build RCCL as a static library instead of shared library
    -t|--tests_build           Build rccl unit tests, but do not run
       --time-trace            Plot the build time of RCCL (requires `ninja-build` package installed on the system)
       --verbose               Show compile commands

.. tip::

    By default, the RCCL install script builds all the GPU targets that are defined in ``DEFAULT_GPUS`` in `CMakeLists.txt <https://github.com/ROCm/rccl/blob/develop/CMakeLists.txt>`_.
    To target specific GPUs and potentially reduce the build time, use ``--amdgpu_targets`` along with
    a semicolon (``;``) separated string list of the GPU targets.
