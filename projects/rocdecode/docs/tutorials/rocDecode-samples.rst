.. meta::
  :description: rocDecode Sample Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, samples, prerequisites, dependencies, requirements

********************************************************************
rocDecode samples
********************************************************************

rocDecode samples are available in the `rocDecode GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples>`_.

Linux setup
-----------

To run the samples on Linux, set the ``ROCM_PATH`` to point to the location of your ROCm installation, and set ``LD_PRELOAD`` and ``LIBVA_DRIVERS_PATH`` to point to the ROCm systems libraries and drivers:

.. code:: shell

  export LD_PRELOAD=$ROCM_PATH/lib/rocm_sysdeps/lib/librocm_sysdeps_va.so.2:$ROCM_PATH/lib/rocm_sysdeps/lib/librocm_sysdeps_va-drm.so.2

  export LIBVA_DRIVERS_PATH=$ROCM_PATH/lib/rocm_sysdeps/lib/

FFmpeg development libraries must be installed to build and run samples that use FFmpeg for either demultiplexing or decoding:

.. code::

  sudo apt install libavcodec-dev libavformat-dev libavutil-dev

Windows setup
-------------

To build and run samples on Windows:

.. code:: bat

  cd samples\videoDecode
  mkdir build && cd build
  cmake .. -DROCM_PATH=<path-to-install> -DFFMPEG_ROOT=<path-to-ffmpeg>
  cmake --build . --config Release
  cd Release
  videodecode.exe -i <input_video>

.. note::

  On Windows, ``LIBVA_DRIVERS_PATH`` is auto-detected from the directory containing ``rocdecode.dll``. No manual environment variable setup is needed.


You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.



