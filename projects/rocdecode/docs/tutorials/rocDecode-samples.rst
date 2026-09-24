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

  mkdir rocdecode-sample && cd rocdecode-sample
  cmake %ROCM_PATH%\share\rocdecode\samples\videoDecode -DROCM_PATH=%ROCM_PATH% -DFFMPEG_ROOT=%FFMPEG_ROOT%
  cmake --build . --config Release
  set PATH=%ROCM_PATH%\bin;%ROCM_PATH%\lib\rocm_sysdeps\bin;%FFMPEG_ROOT%\bin;%PATH%
  Release\videodecode.exe -i %ROCM_PATH%\share\rocdecode\video\AMD_driving_virtual_20-H265.mp4

.. note::

  ``PATH`` must include the rocDecode, VA-API, and FFmpeg DLL directories so the executable can load
  them at run time. ``ROCM_PATH`` must stay set at run time as well: libva uses it to locate the VA-API
  driver in ``%ROCM_PATH%\lib\rocm_sysdeps\bin``.

You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.



