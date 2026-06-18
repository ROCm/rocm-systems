.. meta::
  :description: rocDecode Sample Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, samples, prerequisites, dependencies, requirements

********************************************************************
rocDecode samples
********************************************************************

rocDecode samples are available in the `rocDecode GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples>`_.

You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.

FFmpeg development libraries must be installed to build and run samples that use FFmpeg for either demultiplexing or decoding:

.. code:: 

  sudo apt install libavcodec-dev libavformat-dev libavutil-dev



