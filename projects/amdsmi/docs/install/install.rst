.. meta::
  :description: Install AMD SMI
  :keywords: install, SMI, AMD, ROCm

********************************************************************
Installation
********************************************************************

AMD System Management Interface (AMD SMI) library
-------------------------------------------------

The AMD System Management Interface Library (AMD SMI library) is a C library for Linux that provides a user space interface for applications to monitor and control AMD devices.

.. Note::

This project is a successor to `rocm_smi_lib. <https://github.com/RadeonOpenCompute/rocm_smi_lib>`_ 

Supported platforms
=====================
In its initial release, the AMD SMI library supports Linux bare metal and Linux virtual machine guest for AMD GPUs. In a future release, the library will extend to support AMD EPYC™ CPUs.

The AMD SMI library can run on AMD ROCm-supported platforms. Refer to `System requirements - Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html>`_ for more information.

To run the AMD SMI library, the `amdgpu` driver and the `hsmp` driver must be installed. Optionally, `libdrm` can be installed to query firmware information and hardware IPs.


CLI tool and libraries installation
------------------------------------

Requirements
=============

* Python 3.6.8+ 64-bit
* amdgpu driver must be loaded for `amdsmi_init()` to pass

Installation steps
-------------------

1. Install amdgpu using ROCm.

2. Install amdgpu driver. See the following example. Note that your release and link may differ. The `amdgpu-install --usecase=rocm` triggers both the amdgpu driver update and AMD SMI packages to be installed on your device.

.. code-block:: shell

              sudo apt update
      
              wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
      
              sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
      
              sudo amdgpu-install --usecase=rocm
      
              amd-smi --help

3. Install an example for Ubuntu 22.04 (without ROCm).

.. code-block:: bash

  apt install amd-smi-lib

  # if installed with rocm ignore the export

  export PATH="${PATH:+${PATH}:}~/opt/rocm/bin"

  amd-smi --help


Optional autocompletion
------------------------

The `amd-smi` cli application supports autocompletion. The package should attempt to install it, if argcomplete is not installed, you can enable it by using the following commands:

.. code:: bash

  python3 -m pip install argcomplete

  activate-global-python-argcomplete --user

  # restart shell to enable


Manual/Multiple ROCm instance Python library install
------------------------------------------------------

In the event there are multiple ROCm installations and `pyenv` is not being used to use the correct amdsmi version, you must uninstall previous versions of AMD SMI and install the latest version you want directly from your ROCm instance.

Python library install example for Ubuntu 22.04
=================================================

1. Remove any existing AMD SMI installation:

.. code-block:: bash

    python3 -m pip list | grep amd
    
    python3 -m pip uninstall amdsmi


2. Install Python library from your target ROCm instance:

.. code:: bash

        apt install amd-smi-lib
            
        cd /opt/rocm/share/amd_smi
            
        python3 -m pip install --upgrade pip
            
        python3 -m pip install --user 

  
Now you have the AMD SMI Python library in your Python path:


.. code:: bash

    ~$ python3
  
    Python 3.8.10 (default, May 26 2023, 14:05:08)
  
    [GCC 9.4.0] on linux

3. Type "help", "copyright", "credits" or "license" for more information

.. code:: bash

    import amdsmi


Sphinx documentation
=====================

Run the following commands to build the documentation locally:

.. code-block:: bash

  cd docs
  
  python3 -m pip install -r sphinx/requirements.txt
  
  python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html


The output is available in `docs/_build/html`.

For additional details, see `Contribute to ROCm documentation <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

