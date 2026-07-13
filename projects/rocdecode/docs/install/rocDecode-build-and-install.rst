.. meta::
  :description: Build and install rocDecode with the source code
  :keywords: install, building, rocDecode, AMD, ROCm, source code, developer

***************************************
Build and install rocDecode from source
***************************************

To build rocDecode as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocDecode standalone using the following
instructions.

Prerequisites
=============

rocDecode requires a supported AMD GPU. For more information, see :ref:`ROCm
Core SDK components <rocm:release-components>`.

Linux prerequisites
-------------------

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config

To build and run samples and extended tests, FFmpeg development libraries must be installed separately.
For example, on Ubuntu:

.. code-block:: shell

   sudo apt install libavcodec-dev libavformat-dev libavutil-dev

Windows prerequisites
---------------------

* HIP runtime from `TheRock <https://github.com/ROCm/TheRock>`__ for Windows
* vaon12 — VA-API on D3D12 libraries (``va.dll``, ``va_win32.dll``, ``vaon12_drv_video.dll``), available via the `Microsoft.Direct3D.VideoAccelerationCompatibilityPack NuGet package <https://www.nuget.org/packages/Microsoft.Direct3D.VideoAccelerationCompatibilityPack>`__
* Visual Studio 2022 with C++ desktop workload (MSVC compiler, C++17)
* CMake 3.10 or later
* Windows SDK (provides D3D12 and DXGI headers/libraries)
* FFmpeg (optional) — pre-built libraries or built from source, required for samples and the host decoder library

Build and install
=================

rocDecode is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_ on both Linux and Windows. For TheRock installation details, refer to the `TheRock documentation <https://github.com/ROCm/TheRock#readme>`_.

To build standalone from source, follow the instructions for your platform below.

Clone the repository
--------------------

1. The rocDecode source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode>`__. Use sparse checkout when cloning the rocDecode project.

   .. code-block:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/rocdecode

2. Then use ``git checkout`` to check out the branch you need.

   .. code-block:: bash

      git checkout develop
      cd projects/rocdecode

Build on Linux
--------------

3. Build and install rocDecode using the following commands:

   .. code-block:: bash

      mkdir build && cd build
      cmake ../
      make -j8
      sudo make install

   After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

4. To run the installed CTest-based verification:

   .. code-block:: bash

      mkdir rocdecode-test && cd rocdecode-test
      cmake /opt/rocm/share/rocdecode/test/
      ctest -VV

   Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``.

Build on Windows
----------------

3. Build and install rocDecode using the following commands:

   .. code-block:: bat

      mkdir build && cd build
      cmake .. -DVAON12_ROOT=<path-to-vaon12> -DROCM_PATH=<path-to-TheRock-build>
      cmake --build . --config Release
      cmake --install . --config Release

   .. note::

      * Set ``VAON12_ROOT`` to the vaon12 NuGet package or custom build directory.
      * Set ``ROCM_PATH`` to the TheRock build output directory.
      * To include FFmpeg support for samples and the host decoder, add ``-DFFMPEG_ROOT=<path-to-ffmpeg>``.

4. To verify the build, run a sample:

   .. code-block:: bat

      cd samples\videoDecodeRaw
      mkdir build && cd build
      cmake .. -DROCM_PATH=<path-to-install>
      cmake --build . --config Release
      cd Release
      videodecoderaw.exe -i <input_stream> -f 5

