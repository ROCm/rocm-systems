.. meta::
   :description: Complete reference of CMake build options for configuring hipfile builds, including cache variables, defaults, and valid values.
   :keywords: hipfile, CMake, build options, ROCm, configuration, cache variables, build from source

=============================
CMake build options reference
=============================

This page documents all CMake cache variables available when building hipfile from source. For step-by-step build instructions, see :doc:`/install/build-from-source`.

.. note::

   The source material available does not include the full ``CMakeLists.txt`` contents or the Python ``CMakeLists.txt`` contents. The following table is constructed from the documentation plan and file tree. Some defaults and valid values could not be confirmed from source and are noted accordingly.

Platform and path variables
***************************

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``CMAKE_HIP_PLATFORM``
     - Auto-detected
     - Selects the target GPU platform. Set to ``amd`` for AMD GPUs (uses the fastpath backend with automatic fallback to POSIX I/O) or ``nvidia`` for NVIDIA GPUs (wraps cuFile). The build system uses this variable to choose between the AMD and NVIDIA backend source trees located under ``src/amd_detail/`` and ``src/nvidia_detail/``, respectively.
   * - ``ROCM_PATH``
     - System default (typically ``/opt/rocm``)
     - Path to the ROCm installation. Used to locate the HIP runtime headers and libraries required for building hipfile.
   * - ``ROCM_VERSION``
     - Auto-detected from the ROCm installation
     - Overrides the detected ROCm version string. Useful when cross-compiling or when the auto-detection does not match the target environment.

Library build variables
***********************

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``BUILD_SHARED_LIBS``
     - ``ON``
     - When ``ON``, builds hipfile as a shared library (``libhipfile.so``). Set to ``OFF`` to produce a static library.
   * - ``AIS_CXX_STANDARD``
     - (see note)
     - Sets the C++ standard used to compile hipfile. The library uses C++17 features at minimum. Higher values (for example, ``20``) enable additional ``requires`` constraints via the ``HIPFILE_REQUIRES`` macro defined in the internal header ``shared/hipfile-cpp20.h``.

.. note::

   The exact default value for ``AIS_CXX_STANDARD`` could not be confirmed from the available source material. Consult the top-level ``CMakeLists.txt`` for the authoritative default.

Installation variables
**********************

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``AIS_INSTALL_EXAMPLES``
     - ``OFF``
     - When ``ON``, installs the example programs (such as ``aiscp`` and the API examples under ``examples/``) alongside the library.
   * - ``AIS_INSTALL_TOOLS``
     - ``OFF``
     - When ``ON``, installs the accompanying tools (``ais-check`` and ``ais-stats``) alongside the library.

Documentation and testing variables
***********************************

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``AIS_BUILD_DOCS``
     - ``OFF``
     - When ``ON``, builds the documentation during the CMake build. Requires Sphinx and related documentation tooling.
   * - ``AIS_USE_CODE_COVERAGE``
     - ``OFF``
     - When ``ON``, instruments the build with code coverage flags (for example, ``--coverage`` on GCC and Clang). The utility script ``util/llvm-coverage.sh`` can process the resulting data.
   * - ``AIS_USE_SANITIZERS``
     - ``OFF``
     - When ``ON``, enables one or more compiler sanitizers (such as AddressSanitizer or UndefinedBehaviorSanitizer) for detecting memory errors and undefined behavior during testing.

Internal path overrides
***********************

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``AIS_CAPABLE_DIR``
     - (auto)
     - Overrides the directory path used to locate platform-capability detection files. Typically set automatically by the build system.

Python binding variables
************************

The following variables are used when building or locating the hipfile Python bindings (Cython-based). They are primarily relevant to the Python ``CMakeLists.txt`` under ``python/``.

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Variable
     - Default
     - Description
   * - ``HIPFILE_INCLUDE_DIR``
     - Auto-detected
     - Path to the directory containing ``hipfile.h``. The Python binding build uses this to find the hipfile C header during Cython compilation.
   * - ``HIPFILE_LIBRARY``
     - Auto-detected
     - Path to the hipfile shared library (``libhipfile.so``). The Python bindings link against this library.
   * - ``HIP_INCLUDE_DIR``
     - Auto-detected
     - Path to the HIP runtime include directory (containing ``hip/hip_runtime_api.h``). Required by both the hipfile C header and the Python bindings.

Usage example
*************

The following example shows a typical CMake configuration command using several of these variables:

.. code-block:: bash

   cmake -B build \
       -DCMAKE_HIP_PLATFORM=amd \
       -DROCM_PATH=/opt/rocm \
       -DBUILD_SHARED_LIBS=ON \
       -DAIS_INSTALL_EXAMPLES=ON \
       -DAIS_INSTALL_TOOLS=ON \
       ..

For the complete build workflow, including prerequisites and installation steps, see :doc:`/install/build-from-source`.
