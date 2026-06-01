.. meta::
  :description: Install rocJPEG with the source code
  :keywords: install, building, rocJPEG, AMD, ROCm, source code, developer

*************************************
Build and install rocJPEG from source
*************************************

To build rocJPEG as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build rocJPEG standalone using the following
instructions.

Prerequisites
=============

rocJPEG requires a supported AMD GPU. For more information, see :ref:`ROCm Core
SDK components <rocm:release-components>`.

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config

Build and install
=================

1. The rocJPEG source code is available from the `ROCm systems GitHub repository <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg>`__. Use sparse checkout when cloning the rocJPEG project. Clone the repo using `sparse-checkout`.

   .. code-block:: bash

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/rocjpeg

2. Then use ``git checkout`` to check out the branch you need.

   .. code-block:: bash

      git checkout develop
      cd projects/rocjpeg

3. Build and install rocJPEG using the following commands:

   .. code-block:: bash

      mkdir build && cd build
      cmake ../
      make -j8
      sudo make install

   After installation, the rocJPEG libraries will be copied to ``/opt/rocm/lib`` and the rocJPEG header files will be copied to ``/opt/rocm/include/rocjpeg``.

4. To run the installed CTest-based verification:

   .. code-block:: bash

     mkdir rocjpeg-test && cd rocjpeg-test
     cmake /opt/rocm/share/rocjpeg/test/
     ctest -VV

   To test your build, run ``make test``. To run the test with the verbose option, run ``make test ARGS="-VV"``.
