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

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config

To build and run samples and extended tests, FFmpeg development libraries must be installed separately.
For example, on Ubuntu:

.. code-block:: shell

   sudo apt install libavcodec-dev libavformat-dev libavutil-dev

Build and install
=================

rocDecode is delivered as part of `TheRock <https://github.com/ROCm/TheRock>`_. For TheRock installation details, refer to the `TheRock documentation <https://github.com/ROCm/TheRock#readme>`_.

1. The rocDecode source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode>`__. Use sparse checkout when cloning the rocDecode project. Clone the repo using `sparse-checkout`.

.. code:: shell

  cd rocm-systems/projects/rocdecode

Build and install rocDecode using the following commands:

.. code:: shell

  mkdir build && cd build
  cmake ../
  make -j8
  sudo make install

After installation, the rocDecode libraries will be copied to ``/opt/rocm/lib`` and the rocDecode header files will be copied to ``/opt/rocm/include/rocdecode``.

To run the installed CTest-based verification:

.. code:: shell

  mkdir rocdecode-test && cd rocdecode-test
  cmake /opt/rocm/share/rocdecode/test/
  ctest -VV

Run ``make test`` to test your build. To run the test with the verbose option, run ``make test ARGS="-VV"``.

