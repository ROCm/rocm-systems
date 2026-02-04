.. meta::
  :description: rocDecode Installation Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocDecode prerequisites
********************************************************************

rocDecode requires ROCm running on `GPUs based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm must be installed using the AMDGPU installer with the ``rocm`` usecase:

.. code:: shell

  sudo amdgpu-install --usecase=rocm

rocDecode has been tested on the following Linux environments:
  
* Ubuntu 22.04 and 24.04
* RHEL 8 and 9
* SLES 15 SP7

See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

The following prerequisites are installed by the package installer. If you are building and installing using the source code, use the `rocDecode-setup.py <https://github.com/ROCm/rocDecode/blob/develop/rocDecode-setup.py>`_ to install these prerequisites. 

.. note:: 

  To use the rocDecode samples, the ``rocdecode``, ``rocdecode-dev``, ``rocdecode-host``, and ``rocdecode-test`` packages need to be installed.
  
  If you're installing using the rocDecode source code, the ``rocDecode-setup.py`` script must be run with ``--developer`` set to ``ON``.

* Libva-amdgpu-dev, an AMD implementation for Video Acceleration API (VA-API)
* AMD VA Drivers
* CMake version 3.10 or later
* AMD Clang++ Version 18.0.0 or later
* pkg-config
* FFmpeg runtime and headers
* libstdc++-12-dev for installations on Ubuntu 22.04 
* HIP, specifically the ``hip-dev`` package

