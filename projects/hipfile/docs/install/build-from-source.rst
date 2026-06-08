.. meta::
  :description: Step-by-step instructions for building hipFile from source using CMake on AMD and NVIDIA platforms.
  :keywords: hipFile, build from source, CMake, ROCm, install, direct-to-GPU I/O, AMD, NVIDIA

*************************
Build hipFile from source
*************************

Prerequisites
*************

Before building hipFile, make sure you have the following software installed:

- ``ROCm`` (provides ``hsa-runtime64`` and HIP). See :doc:`Install AMD ROCm <rocm:install/rocm>` for current requirements.
- ``CMake`` >= 3.21
- C++17 compiler (Clang or GCC)
- ``libmount`` (from ``util-linux``)

For the NVIDIA platform, you also need:

- CUDAToolkit
- cuFile library

Install the ROCm Core SDK
*************************

hipFile is included with the ROCm Core SDK. For the most complete installation, use the ``amdrocm-core-sdk`` meta package. Refer to :doc:`Install AMD ROCm <rocm:install/rocm>` for detailed installation instructions.

Install on Linux
****************

To install only the hipFile library group as a subset of the ROCm Core SDK, use the appropriate ``amdrocm-*`` meta package. The package name follows this format:

.. code-block:: text

   amdrocm-<group><-dev/-devel><rocm_version><-llvm_target>

Where:

- ``-dev`` is used for Debian-based distributions
- ``-devel`` is used for RPM-based distributions
- ``<rocm_version>`` is an optional version pin
- ``<-llvm_target>`` is an optional single GPU architecture specifier

.. note::

   Make sure the exact package name for hipFile with the latest ROCm release documentation, as the group name may vary between releases.

.. tab-set::

   .. tab-item:: Debian-based

      .. code:: shell

         sudo apt-get update
         sudo apt-get install amdrocm-<group>-dev

   .. tab-item:: RHEL-based

      .. code:: shell

         sudo dnf install amdrocm-<group>-devel

   .. tab-item:: SLES

      .. code:: shell

         sudo zypper install amdrocm-<group>-devel

Building from source
********************

Source download
---------------

Clone the repository using a sparse checkout from the ROCm monorepo:

.. code:: shell

   git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
   cd rocm-systems
   git sparse-checkout set projects/hipfile

Library dependencies
--------------------

hipFile requires the following libraries during configuration and linking:

- ``hsa-runtime64``: HSA runtime (found via CMake config)
- ``hip``: HIP runtime (found via CMake config)
- ``libmount``: mount information parsing (from ``util-linux``; located via ``find_library``)

On the NVIDIA platform, these additional dependencies are required:

- CUDAToolkit: CUDA runtime and headers
- cuFile: NVIDIA GPUDirect Storage library

Build commands
--------------

Create a build directory, configure with CMake, build, and install:

.. code:: shell

   cd rocm-systems/projects/hipfile
   cmake -B build \
       -DCMAKE_INSTALL_PREFIX=/opt/rocm \
       -DCMAKE_HIP_PLATFORM=amd
   cmake --build build
   sudo cmake --install build

For an NVIDIA platform build, set ``CMAKE_HIP_PLATFORM`` to ``nvidia`` and make sure
``CUDAToolkit`` and ``cuFile`` are discoverable:

.. code:: shell

   cmake -B build \
       -DCMAKE_HIP_PLATFORM=nvidia \
       -DCMAKE_INSTALL_PREFIX=/usr/local
   cmake --build build
   sudo cmake --install build

CMake options
-------------

The following table lists the CMake options available when configuring hipFile. For the complete reference table, see :doc:`/reference/cmake-options`.

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Option
     - Default
     - Description
   * - ``CMAKE_HIP_PLATFORM``
     - ``amd``
     - Target HIP platform. Set to ``amd`` or ``nvidia``.
   * - ``ROCM_PATH``
     - ``/opt/rocm``
     - Path to the ROCm installation. Can also be set via the ``ROCM_PATH`` environment variable.
   * - ``ROCM_VERSION``
     - Auto-detected
     - ROCm version string. If not set, it is read from the version file at ``ROCM_PATH``.
   * - ``CMAKE_INSTALL_PREFIX``
     - ``${ROCM_PATH}``
     - Installation directory for hipFile. Defaults to the ROCm path.
   * - ``BUILD_SHARED_LIBS``
     - ``ON``
     - Build shared libraries when ``ON``, or static libraries when ``OFF``.
   * - ``AIS_CXX_STANDARD``
     - ``17``
     - C++ standard to build with. Allowed values are ``17`` and ``20``.
   * - ``AIS_INSTALL_EXAMPLES``
     - ``ON``
     - Install example programs (such as ``aiscp`` and API examples).
   * - ``AIS_INSTALL_TOOLS``
     - ``ON``
     - When ``ON``, builds ``ais-stats`` on AMD platform builds. ``ais-check`` is always installed; ``ais-stats`` is built but not installed by default.
   * - ``AIS_BUILD_DOCS``
     - See :doc:`/reference/cmake-options`
     - Build documentation.
   * - ``BUILD_TESTING``
     - ``ON``
     - Enable building of tests via CTest.
   * - ``AIS_USE_CODE_COVERAGE``
     - ``OFF``
     - Build with LLVM code coverage instrumentation flags.
   * - ``CMAKE_BUILD_TYPE``
     - ``RelWithDebInfo``
     - Build type. Valid values are ``Debug``, ``Release``, ``RelWithDebInfo``, ``MinSizeRel``, and ``None``.
