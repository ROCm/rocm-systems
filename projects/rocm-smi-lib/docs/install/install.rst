.. meta::
  :description: Install ROCm SMI
  :keywords: install, SMI, library, api, AMD, ROCm


*********************
Installing ROCm SMI
*********************

Planned deprecation notice
----------------------------

ROCm System Management Interface (ROCm SMI) Library is planned to be ***deprecated***, and the release date will be announced soon. We recommend migration to AMD SMI.

Install amdgpu using ROCm
--------------------------
Use the following instructions to install AMDGPU using ROCm:

1. Install amdgpu driver. Refer to the following example, your release and link may differ. The `amdgpu-install --usecase=rocm` triggers both an amdgpu driver update and ROCm SMI packages to be installed on your device.

.. code-block:: shell

    sudo apt update
    wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
    sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
    sudo amdgpu-install --usecase=rocm

* `rocm-smi --help`

Building ROCm SMI
******************

Addtional required software
============================

To build the ROCm SMI library, the following components are required.

The following software versions are what was used in development. Earlier versions are not guaranteed to work:

* CMake (v3.14.0)
* g++ (5.4.0)

To build the latest documentation, the following are required:

* Python 3.8+
* NPM (sass)

The source code for ROCm SMI is available on `Github <https://github.com/RadeonOpenCompute/rocm_smi_lib>`_.

After the ROCm SMI library git repository is cloned to a local Linux machine, use the following CMake build sequence to build the library. Specifically,

.. code-block:: shell

    mkdir -p build
    cd build
    cmake ..
    make -j $(nproc)
    # Install library file and header; default location is /opt/rocm
    make install


The built library will appear in the `build` folder.

To build the rpm and deb packages follow the above steps with:

.. code-block:: shell

    make package


Building documentation
=======================

The following is an example of how to build the docs:

.. code-block:: shell

    python3 -m venv .venv
    .venv/bin/python3 -m pip install -r docs/sphinx/requirements.txt
    .venv/bin/python3 -m sphinx -T -E -b html -d docs/_build/doctrees -D language=en docs docs/_build/html


Building tests
=================

To verify the build and capability of ROCm SMI on your system and to see an example of how ROCm SMI can be used, you may build and run the tests that are available in the repo. To build the tests, follow these steps:

.. code-block:: bash

    mkdir build
    cd build
    cmake -DBUILD_TESTS=ON ..
    make -j $(nproc)

To run the test, execute the program `rsmitst` that is built from the preceding steps.

