.. meta::
  :description: rocDecode Installation Prerequisites
  :keywords: install, rocDecode, AMD, ROCm, prerequisites, dependencies, requirements

********************************************************************
rocDecode prerequisites
********************************************************************

rocDecode requires ROCm running on `GPUs based on the CDNA architecture <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_.

See `Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_ for the complete list of ROCm supported Linux environments.

CMake version 3.10 or later is required to build rocDecode.

Use the `rocDecode-setup.py <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocdecode/rocDecode-setup.py>`_ to install these prerequisites.

* VA API runtime
* FFmpeg runtime and headers when the script is run with ``--developer ON`` only 
* ``libva-amdgpu-dev`` on Ubuntu 22.04 
* ``libva-dev`` on Ubuntu 24.04
*  ``libstdc++-12-dev`` on Ubuntu 22.04 
* ``hip-runtime-amd``
* ``hip-dev`` on Ubuntu
* ``hip-devel`` on RHEL and SLES
* ``libva-devel`` on RHEL and SLES
* ``pkg-config``

.. note:: 

  To use the rocDecode samples, the ``rocDecode-setup.py`` script must be run with ``--developer ON``.