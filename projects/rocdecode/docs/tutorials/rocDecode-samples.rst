.. meta::
  :description: rocDecode Sample Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, samples, prerequisites, dependencies, requirements

********************************************************************
rocDecode samples
********************************************************************

rocDecode samples are available in the `rocDecode GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples>`_.

Linux setup
-----------

The ``ROCM_PATH`` environment variable must point to the location of your ROCm installation to use the samples.

.. code:: shell

  export ROCM_PATH=path_to_rocm_installation

FFmpeg development libraries must be installed to build and run samples that use FFmpeg for either demultiplexing or decoding:

.. code::

  sudo apt install libavcodec-dev libavformat-dev libavutil-dev

Windows setup
-------------

To build and run samples on Windows:

.. code:: bat

  set ROCM_PATH=<path-to-rocm-installation>
  set FFMPEG_ROOT=<path-to-ffmpeg>
  set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\rocm_sysdeps\bin;%FFMPEG_ROOT%\bin;%PATH%
  mkdir rocdecode-sample && cd rocdecode-sample
  cmake %ROCM_PATH%\share\rocdecode\samples\videoDecode -DROCM_PATH=%ROCM_PATH% -DFFMPEG_ROOT=%FFMPEG_ROOT%
  cmake --build . --config Release
  Release\videodecode.exe -i %ROCM_PATH%\share\rocdecode\video\AMD_driving_virtual_20-H265.mp4

.. note::

  ``PATH`` must include the rocDecode, VA-API, and FFmpeg DLL directories so the executable can load
  them at run time. ``ROCM_PATH`` must stay set at run time as well: libva uses it to locate the VA-API
  driver in ``%ROCM_PATH%\lib\rocm_sysdeps\bin``.

  ``videoDecode`` demultiplexes the container with FFmpeg, so FFmpeg must be present when CMake
  configures the sample. ``FFMPEG_ROOT`` and ``-DFFMPEG_ROOT`` are only needed if FFmpeg is not in one
  of the locations CMake probes automatically; the recipe sets it because the ``PATH`` line uses it to
  name the DLL directory. For a sample with no FFmpeg dependency at all, use ``videoDecodeRaw``.

You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.



