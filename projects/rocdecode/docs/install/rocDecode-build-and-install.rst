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

.. note::
   
   To use the rocDecode samples and tutorials, the ``ROCM_PATH`` environment variable needs to point to the location of your ROCm installation:

   .. code:: shell

      export ROCM_PATH=path_to_your_ROCm_installation

   Set this variable after installation.


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
* vaon12 — VA-API on D3D12 libraries (Mesa's ``vaon12_drv_video.dll`` plus libva), provided by
  TheRock for Windows under ``%ROCM_PATH%\lib\rocm_sysdeps``; no separate download is required
* Visual Studio 2022 with C++ desktop workload (MSVC compiler, C++17)
* CMake 3.21 or later (the ``Visual Studio 17 2022`` generator requires 3.21)
* Windows SDK (provides D3D12 and DXGI headers/libraries)
* FFmpeg (optional) — pre-built libraries or built from source, required for FFmpeg-based samples and the host decoder library

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

Build and install rocDecode using the following commands:

.. code-block:: bash

   mkdir build && cd build
   cmake ../
   make -j8
   sudo make install

After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

To run the installed CTest-based verification:

.. code-block:: bash

   mkdir rocdecode-test && cd rocdecode-test
   cmake /opt/rocm/share/rocdecode/test/
   ctest -VV

Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``.

Build on Windows
----------------

Build and install rocDecode using the following commands:

.. code-block:: bat

   mkdir build && cd build
   set ROCM_PATH=<path-to-TheRock-build>
   cmake .. -DROCM_PATH="%ROCM_PATH%"
   cmake --build . --config Release
   cmake --install . --config Release

.. note::

   * Set ``ROCM_PATH`` as an environment variable, not only on the ``cmake`` command line, and
     use the same command prompt for the steps below. The VA-API headers and import libraries
     are found there at build time, libva reads it at run time to locate the VA-API driver, and
     the verification commands below expand ``%ROCM_PATH%``.
   * FFmpeg, needed for the samples and the host decoder, is detected automatically when it is
     installed in a common location: on ``PATH``, under Chocolatey or scoop, in
     ``%ProgramFiles%\ffmpeg``, or in ``C:\ffmpeg``. Set ``FFMPEG_ROOT=<path-to-ffmpeg>`` the same
     way as ``ROCM_PATH`` and add ``-DFFMPEG_ROOT="%FFMPEG_ROOT%"`` only if it lives elsewhere, or to
     pin a specific build.

To verify the build, build and run a sample from the installed location:

.. code-block:: bat

   mkdir rocdecode-sample && cd rocdecode-sample
   cmake "%ROCM_PATH%\share\rocdecode\samples\videoDecodeRaw" -DROCM_PATH="%ROCM_PATH%"
   cmake --build . --config Release
   set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\rocm_sysdeps\bin;%PATH%
   Release\videodecoderaw.exe -i "%ROCM_PATH%\share\rocdecode\video\AMD_driving_virtual_20-H265.265" -f 5

