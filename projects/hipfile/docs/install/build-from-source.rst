.. meta::
   :description: Step-by-step instructions for building hipFile from source using CMake on AMD ROCm platforms.
   :keywords: hipFile, build from source, CMake, ROCm, install, direct-to-GPU I/O, AMD

**************************************
Build and install hipFile from source
**************************************

To build hipFile as part of the ROCm Core SDK, see `TheRock build instructions <https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the supported path for full-stack ROCm source builds.

Alternatively, you can build hipFile standalone from ``rocm-systems`` using the
steps below. For ROCm meta packages instead of compiling, see :doc:`./install`.

Prerequisites
=============

hipFile targets Linux with ROCm HIP. For supported GPUs and stack components,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

- CMake 3.21 or later
- A C++ compiler that matches the ``AIS_CXX_STANDARD`` value you select. The
  default is C++20.
- HIP and HSA packages from ROCm so CMake can locate ``hip`` and ``hsa-runtime64``
- ``libmount`` from ``util-linux`` for mount metadata parsing
- A Linux kernel that exposes the peer-to-peer DMA (P2PDMA) paths hipFile expects for peer
  transfers on AMD builds


Build and install
=================

1. hipFile lives under ``projects/hipfile`` in the `ROCm rocm-systems repository <https://github.com/ROCm/rocm-systems>`__. Clone with a sparse checkout so you only fetch that subtree.

   .. code:: shell

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/hipfile

2. Check out the branch you intend to build, then enter the hipFile tree.

   .. code:: shell

      git checkout develop
      cd projects/hipfile

3. Configure, compile, and install with CMake. The defaults wire ``CMAKE_INSTALL_PREFIX``
   to ``ROCM_PATH`` so libraries and headers land next to your ROCm install.

   .. code:: shell

      cmake -B build -DCMAKE_HIP_PLATFORM=amd
      cmake --build build -j
      sudo cmake --install build

   After installation, ``libhipfile.so`` and ``hipfile.h`` appear under the ``lib``
   and ``include`` directories beneath ``CMAKE_INSTALL_PREFIX``, typically
   ``/opt/rocm`` when you don't override ``ROCM_PATH``.

4. Optional CTest pass from the build tree:

   .. code:: shell

      cd build
      ctest --output-on-failure

   You can also run ``cmake --build . --target test`` from ``build`` if you
   prefer Make-driven test targets.

CMake options
=============

The table lists the most common CMake cache entries for hipFile. Other flags
exist for sanitizers, clang-tidy, and documentation generation. Inspect
``CMakeCache.txt`` after the first configure pass for the full set.

.. list-table::
   :header-rows: 1
   :widths: 30 18 52

   * - Option
     - Default
     - Description
   * - ``CMAKE_HIP_PLATFORM``
     - ``amd``
     - HIP platform selector. Use ``amd``.
   * - ``ROCM_PATH``
     - ``/opt/rocm`` or derived from ``ROCM_VERSION``
     - ROCm root used for ``CMAKE_PREFIX_PATH`` and the default install prefix.
   * - ``ROCM_VERSION``
     - Read from ``${ROCM_PATH}/.info/version`` when unset
     - ROCm version string for path logic and compatibility checks.
   * - ``CMAKE_INSTALL_PREFIX``
     - ``${ROCM_PATH}`` when left at the CMake default
     - Install root for libraries, headers, and tools.
   * - ``BUILD_SHARED_LIBS``
     - ``ON``
     - Builds shared libraries when ``ON`` and static libraries when ``OFF``.
   * - ``AIS_CXX_STANDARD``
     - ``20``
     - C++ dialect. Allowed values are ``20``, ``23``, and ``26``.
   * - ``AIS_INSTALL_EXAMPLES``
     - ``ON``
     - Installs sample binaries such as ``aiscp`` when enabled.
   * - ``AIS_INSTALL_TOOLS``
     - ``ON``
     - Installs host tools such as ``ais-stats`` when enabled on AMD builds.
   * - ``AIS_BUILD_DOCS``
     - ``OFF``
     - Enables the ``doc`` target when ``ON``. Requires Doxygen in your environment.
   * - ``AIS_USE_CODE_COVERAGE``
     - ``OFF``
     - Adds LLVM coverage instrumentation for Clang builds.
   * - ``BUILD_TESTING``
     - ``ON`` unless you pass ``-DBUILD_TESTING=OFF``
     - Controls whether CTest registers the hipFile unit and system tests.
   * - ``CMAKE_BUILD_TYPE``
     - ``RelWithDebInfo``
     - CMake build flavor. Other common values are ``Debug``, ``Release``,
       ``MinSizeRel``, and ``None``.
