.. meta::
  :description: The Component Unified ID (CUID) generates a deterministic unique ID for various devices such as GPUs, CPUs, NICs, and platforms in a data center environment.
  :keywords: CUID installation, Build CUID, Install CUID, Installing CUID, Building CUID

.. _Building-cuid:

*****************************
Building and installing CUID
*****************************

This topic explains how to build and install the CUID library from source.

System requirements
====================

To build CUID from source, the following dependencies are required:

- CMake v3.20 or later
- G++ v7.0 or later (C++17)
- For Microsoft Windows: `Bcrypt <https://www.npmjs.com/package/bcrypt?activeTab=code>`_ (Windows Native crypto library)

Building and installing CUID library
=====================================

To build and install the CUID library from source, follow these steps:

1. Download the latest version of CUID from the GitHub repository.

   .. code-block:: shell

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
    cd rocm-systems
    git sparse-checkout init --cone
    git sparse-checkout set shared/cuid
    git checkout develop
    cd rocm-systems/shared/cuid

2. Build and install the project using CMake:

   .. code-block:: shell

    mkdir build
    cd build
    cmake ..
    make -j $(nproc)
    sudo make install

   .. note::

      The default install directory is ``/opt/rocm/core``. However, you can choose a different directory using the ``-DCMAKE_INSTALL_PREFIX`` option.

Installation ships the static library ``libamdcuid_static.a``, its header
``amd_cuid.h`` (under ``include/amdcuid``) and a CMake package
(``find_package(amdcuid)``, target ``amdcuid::amdcuid``).

Setting the node key
====================

amdgpu starts without a node key and keeps one only in memory, so set it after
every boot or amdgpu reload:

.. code-block:: shell

   head -c 32 /dev/urandom | sudo amd-smi set --cuid-seed -
   # or, to share one key across a fleet:
   sudo amd-smi set --cuid-seed /path/to/fleet-key.bin

Without a key, amdgpu publishes no derived CUID: a whole GPU gets a temporary
CUID and a partition none. CPU, NIC, NPU and platform CUIDs are temporary
without a key and for every non-root caller. See
:ref:`manage-node-key` for the refusal rules and failure modes.
