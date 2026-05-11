.. meta::
  :description: Build and install rocDecode with the source code
  :keywords: install, building, rocDecode, AMD, ROCm, source code, developer

******************************************
Build and installing rocDecode from source
******************************************

Prerequisites
=============

rocDecode requires a supported `AMD GPU <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_. See the ROCm system requirements for the current list of supported GPUs and gfx targets.

rocDecode is built and installed as part of `TheRock <https://github.com/ROCm/TheRock>`_. All core dependencies are provided by the TheRock build, including:

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

1. The rocDecode source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode>`_. Use sparse checkout when cloning the rocDecode project. Clone the repo using `sparse-checkout`.

   .. code-block:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/rocdecode

2. Then use ``git checkout`` to check out the branch you need.

   .. code-block:: bash

      git checkout develop
      cd rocm-systems/projects/rocdecode

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

