.. meta::
   :description: Installation instructions for hipfile, AMD's direct-to-GPU I/O library for the ROCm platform.
   :keywords: hipfile, installation, ROCm, build, GPU, direct I/O, AMD, NVIDIA, CUDA

=====================
Installation overview
=====================

Prerequisites
*************

hipfile requires the following software to build and run:

- **ROCm** with ``hsa-runtime64`` and HIP runtime
- **CMake** 3.21 or later
- **C++17 compiler** (Clang or GCC)
- **libmount** (from ``util-linux``)

For the NVIDIA platform, the following additional dependencies are required:

- **CUDA Toolkit**
- **cuFile library**

For Python bindings, you also need:

- **Python 3** with development headers
- **Cython**
- **scikit-build-core**

See :doc:`/install/python-bindings` for details on building and installing the Python package.

.. warning::

   hipfile is currently an early-access software technology preview. Running production workloads is not recommended.

Install the ROCm Core SDK
*************************

hipfile is included with the ROCm Core SDK. For the most complete installation, use the ``amdrocm-core-sdk`` meta package. This ensures that all required runtime libraries and development headers are available.

For full ROCm installation instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`.

Install on Linux
****************

To install just the hipfile library group as a subset of the ROCm Core SDK, use the appropriate ``amdrocm-`` meta package. The package name follows this format:

.. code-block:: text

   amdrocm-<group><-dev/-devel><rocm_version><-llvm_target>

Where:

- ``-dev`` is the development suffix for Debian-based distributions
- ``-devel`` is the development suffix for RPM-based distributions
- ``<rocm_version>`` is an optional version pin (for example, ``6.4.0``)
- ``<-llvm_target>`` is an optional single GPU architecture target

.. note::

   Verify the exact package name for hipfile with the latest ROCm release documentation, as package group names may vary between releases.

.. tab-set::

   .. tab-item:: Debian-based (Ubuntu)

      .. code-block:: shell

         sudo apt-get update
         sudo apt-get install amdrocm-<group>-dev

   .. tab-item:: RHEL-based

      .. code-block:: shell

         sudo dnf install amdrocm-<group>-devel

   .. tab-item:: SLES

      .. code-block:: shell

         sudo zypper install amdrocm-<group>-devel

Building from source
********************

Source download
---------------

hipfile is part of the ROCm libraries monorepo. Use a sparse checkout to download only the hipfile project:

.. code-block:: shell

   git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries.git
   cd rocm-libraries
   git sparse-checkout set projects/hipfile

Library dependencies
--------------------

The AMD backend requires the following libraries, which are found automatically when ROCm is installed:

- ``hsa-runtime64`` (found via CMake config)
- ``hip`` (found via CMake config)
- ``libmount`` (from ``util-linux``)

The NVIDIA backend requires:

- ``CUDAToolkit`` (found via CMake)
- ``cuFile`` library (found in the CUDA Toolkit library directory)

Build commands
--------------

Configure, build, and install hipfile using standard CMake workflow:

.. code-block:: shell

   cd projects/hipfile
   cmake -B build -S .
   cmake --build build
   sudo cmake --install build

To target the NVIDIA platform instead of the default AMD platform:

.. code-block:: shell

   cmake -B build -S . -DCMAKE_HIP_PLATFORM=nvidia
   cmake --build build
   sudo cmake --install build

By default, hipfile installs to the ROCm path (``/opt/rocm``). You can override this with ``-DCMAKE_INSTALL_PREFIX``:

.. code-block:: shell

   cmake -B build -S . -DCMAKE_INSTALL_PREFIX=/custom/path

For more details, see :doc:`/install/build-from-source`.

CMake options
-------------

The following CMake options are available when building hipfile:

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``BUILD_SHARED_LIBS``
     - ``ON``
     - Build shared libraries instead of static libraries
   * - ``CMAKE_HIP_PLATFORM``
     - ``amd``
     - HIP platform to build with (``amd`` or ``nvidia``)
   * - ``AIS_CXX_STANDARD``
     - ``17``
     - C++ standard to build with (``17`` or ``20``)
   * - ``CMAKE_BUILD_TYPE``
     - ``RelWithDebInfo``
     - Build type (``Debug``, ``Release``, ``RelWithDebInfo``, ``MinSizeRel``, or ``None``)
   * - ``BUILD_TESTING``
     - ``ON``
     - Enable building of tests (via CTest)
   * - ``AIS_INSTALL_EXAMPLES``
     - ``ON``
     - Install example programs
   * - ``AIS_INSTALL_TOOLS``
     - ``ON``
     - Install tool programs (AMD platform only)
   * - ``AIS_USE_CODE_COVERAGE``
     - ``OFF``
     - Build with LLVM code coverage flags
   * - ``ROCM_PATH``
     - ``/opt/rocm``
     - Path to the ROCm installation
   * - ``ROCM_VERSION``
     - Auto-detected
     - ROCm version to build with
   * - ``AIS_CAPABLE_DIR``
     - ``/tmp``
     - Directory to use for end-to-end tests
