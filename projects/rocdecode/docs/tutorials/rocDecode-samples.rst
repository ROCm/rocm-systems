.. meta::
  :description: rocDecode Sample Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, samples, prerequisites, dependencies, requirements

********************************************************************
rocDecode samples
********************************************************************

rocDecode samples are available in the `rocDecode GitHub repository <https://github.com/ROCm/rocDecode/tree/develop/samples>`_.

You can find a walkthrough of the ``videodecode.cpp`` sample at :doc:`Understanding the videodecode.cpp sample <../how-to/using-rocDecode-videodecode-sample>`.

All rocDecode packages, ``rocDecode``, ``rocdecode-dev``, ``rocdecode-host``, and ``rocdecode-test``, must be installed to use the rocDecode samples.

If you're using a :doc:`package installer <../install/rocDecode-package-install>`, install ``rocdecode``, ``rocdecode-dev``, ``rocdecode-host``, and ``rocdecode-test``.

If you're building and installing rocDecode from its :doc:`source code <../install/rocDecode-build-and-install>`, ``rocDecode-setup.py`` needs to be run with ``--developer`` set to ``ON``:

.. code:: cpp

   python3 rocDecode-setup.py --developer ON

The ``rocDecode-test`` package needs to be built and installed as well:

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV



