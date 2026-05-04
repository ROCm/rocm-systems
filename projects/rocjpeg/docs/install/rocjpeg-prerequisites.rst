.. meta::
  :description: rocJPEG Installation Prerequisites
  :keywords: install, rocJPEG, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocJPEG prerequisites
********************************************************************

rocJPEG requires an `AMD GPU <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ with ``gfx908`` or higher.

rocJPEG is built and installed as part of `TheRock <https://github.com/ROCm/TheRock>`_. All core dependencies are provided by the TheRock build, including:

* HIP runtime and development libraries
* AMD Clang++ compiler (C++17 required)
* Libva and VA-API drivers
* Libdrm (amdgpu)
* CMake and pkg-config
