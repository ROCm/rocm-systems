.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

*************************************
Build and install rocJPEG from source
*************************************

To build rocJPEG as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocJPEG standalone using the following
instructions.

Prerequisites
=============

rocJPEG requires a supported AMD GPU. For more information, see :ref:`ROCm Core
SDK components <rocm:release-components>`.

Linux prerequisites
-------------------

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config

Windows prerequisites
---------------------

rocJPEG supports Windows 11 through the VA-API on D3D12 (vaon12) backend from Mesa.

* HIP runtime from `TheRock <https://github.com/ROCm/TheRock>`__ for Windows
* vaon12 — VA-API on D3D12 libraries (Mesa's VA-API driver plus libva), provided by
  TheRock for Windows under ``%ROCM_PATH%\lib\rocm_sysdeps``. No separate download is required
* Visual Studio 2022 version 17.13 or later with the C++ desktop workload (MSVC 19.43+, C++17).
  Earlier toolsets fail at link time because ROCm's prebuilt libraries reference C++ standard
  library internals introduced in MSVC 19.43. See `TheRock Windows support
  <https://github.com/ROCm/TheRock/blob/main/docs/development/windows_support.md>`__.
* CMake 3.21 or later (the ``Visual Studio 17 2022`` generator requires 3.21)
* Windows SDK

.. note::

   Hardware JPEG decoding is not yet available on Windows. The vaon12 driver does not
   currently expose the JPEG baseline decode profile, so ``rocJpegCreate`` with
   ``ROCJPEG_BACKEND_HARDWARE`` returns ``ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED``.
   JPEG stream parsing (``rocJpegStreamCreate``, ``rocJpegStreamParse``, ``rocJpegStreamDestroy``)
   is available; the decode APIs are not.

Build and install
=================

Clone the repository
--------------------

1. The rocJPEG source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg>`__. Use sparse checkout when cloning the rocJPEG project.

   .. code-block:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/rocjpeg

2. Then use ``git checkout`` to check out the branch you need.

   .. code-block:: bash

      git checkout develop
      cd projects/rocjpeg

Build on Linux
--------------

Build and install rocJPEG using the following commands:

.. code-block:: bash

   mkdir build && cd build
   cmake ../
   make -j8
   sudo make install

After installation, the rocJPEG libraries will be copied to ``/opt/rocm/lib`` and the rocJPEG header files will be copied to ``/opt/rocm/include/rocjpeg``.

To run the installed CTest-based verification:

.. code-block:: bash

   mkdir rocjpeg-test && cd rocjpeg-test
   cmake /opt/rocm/share/rocjpeg/test/
   ctest -VV

To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS="-VV"``.

Build on Windows
----------------

Build and install rocJPEG using the following commands:

.. code-block:: bat

   mkdir build && cd build
   set ROCM_PATH=<path-to-TheRock-build>
   cmake ..
   cmake --build . --config Release
   cmake --install . --config Release

.. note::

   * Set ``ROCM_PATH`` as an environment variable and use the same command prompt for the steps
     below. CMake reads it from the environment, so no ``-DROCM_PATH`` is needed. Keep it set:
     libva reads it at run time to locate the VA-API driver, and the verification commands below
     expand ``%ROCM_PATH%``.
   * The ``Visual Studio 17 2022`` generator is the default. The library is compiled with MSVC,
     and the HIP color-conversion kernels are compiled separately with ``amdclang++`` for each
     GPU in ``GPU_TARGETS`` and linked into ``rocjpeg.dll``. Pass ``-DGPU_TARGETS=<gfx...>``
     (for example, ``-DGPU_TARGETS=gfx1035``) to limit the targets and speed up the build.
   * rocprofiler-register is not available on Windows; ``ROCJPEG_ENABLE_ROCPROFILER_REGISTER``
     is ignored. ``BUILD_TESTS=ON`` is also ignored; build the samples from the installed
     location instead.

After installation, the files are installed under ``%ROCM_PATH%``:

* ``bin\rocjpeg.dll`` - the rocJPEG runtime library
* ``lib\rocjpeg.lib`` - the import library
* ``include\rocjpeg`` - the rocJPEG header files
* ``lib\cmake\rocjpeg`` - the CMake package files
* ``share\rocjpeg\samples``, ``share\rocjpeg\images``, and ``share\rocjpeg\test`` - samples,
  test images, and CTest definitions

At run time, add ``%ROCM_PATH%\bin`` to ``PATH`` so that ``rocjpeg.dll`` and the HIP runtime
are found. rocJPEG loads the libva libraries from ``%ROCM_PATH%\lib\rocm_sysdeps\bin`` itself,
so that directory does not need to be on ``PATH``.

To run the installed CTest-based verification:

.. code-block:: bat

   set PATH=%ROCM_PATH%\bin;%PATH%
   mkdir rocjpeg-test && cd rocjpeg-test
   cmake "%ROCM_PATH%\share\rocjpeg\test"
   ctest -C Release -VV

``-C Release`` is required with the multi-config Visual Studio generator. Running
``ctest -C Release`` from the build directory also works. Because hardware JPEG decoding is
not yet available on Windows, the decode tests are reported as skipped when ``rocJpegCreate``
returns ``ROCJPEG_STATUS_HW_JPEG_DECODER_NOT_SUPPORTED``; any other failure still fails.

To build and run a sample from the installed location:

.. code-block:: bat

   set PATH=%ROCM_PATH%\bin;%PATH%
   mkdir rocjpeg-sample && cd rocjpeg-sample
   cmake "%ROCM_PATH%\share\rocjpeg\samples\jpegDecode"
   cmake --build . --config Release
   Release\jpegdecode.exe -i "%ROCM_PATH%\share\rocjpeg\images"
