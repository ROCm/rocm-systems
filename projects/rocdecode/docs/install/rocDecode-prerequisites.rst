.. meta::
  :description: rocDecode Installation Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocDecode prerequisites
********************************************************************

rocDecode requires a supported `AMD GPU <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_. See the ROCm system requirements for the current list of supported GPUs and gfx targets.

rocDecode is built and installed as part of `TheRock <https://github.com/ROCm/TheRock>`_. All core dependencies are provided by the TheRock build, including:

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config

To build and run samples and extended tests, FFmpeg development libraries must be installed separately:

.. code:: shell

  sudo apt install libavcodec-dev libavformat-dev libavutil-dev

