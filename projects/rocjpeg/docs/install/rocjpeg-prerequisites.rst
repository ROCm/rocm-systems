.. meta::
  :description: rocJPEG Installation Prerequisites
  :keywords: install, rocJPEG, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocJPEG prerequisites
********************************************************************

rocJPEG requires ROCm running on `GPUs based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

ROCm must be installed before building rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions.

See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

CMake version 3.10 or later is required to build rocJPEG.

Use the `rocJPEG-setup.py <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocjpeg/rocJPEG-setup.py>`_ setup script available in the rocJPEG GitHub repository to install prerequisites:

* AMD VA Drivers
* ``libva-devel`` on RHEL and SLES
* ``libva-dev`` on Ubuntu 24.04 and later
* ``libva-amdgpu-dev`` on Ubuntu 22.04 only
* ``libstdc++-12-dev`` on Ubuntu 22.04 only
* ``hip-runtime-amd``
* ``hip-dev`` on Ubuntu
* ``hip-devel`` on RHEL and SLES