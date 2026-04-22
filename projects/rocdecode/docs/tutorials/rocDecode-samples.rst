.. meta::
  :description: rocDecode Sample Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, samples, prerequisites, dependencies, requirements

********************************************************************
rocDecode samples
********************************************************************

rocDecode samples are available in the `rocDecode GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/samples>`_.

You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.

rocDecode is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_. It can also be :doc:`built and installed from source <../install/rocDecode-build-and-install>`.

FFmpeg development libraries must be installed to build and run samples. To run the installed CTest-based verification:

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV



