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

- CMake v3.14 or later
- G++ v5.0 or later
- For Ubuntu or Debian: OpenSSL v1.1 or later
- For Microsoft Windows: `Bcrypt <https://www.npmjs.com/package/bcrypt?activeTab=code>`_ (Windows Native crypto library)

Building and installing CUID library
=====================================

To build and install the CUID library from source, follow these steps:

1. Download the latest version of CUID from the GitHub repository.

   .. code-block:: shell

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
    cd rocm-systems
    git sparse-checkout init --cone
    git sparse-checkout set projects/cuid
    git checkout develop
    cd rocm-systems/projects/cuid

2. Build the project using CMake. Run as root or use ``sudo`` before running ``make install``.

   .. code-block:: shell

    mkdir build
    cd build
    cmake ..
    make -j $(nproc)

   .. note::

      The default install directory is ``/opt/rocm/core``. However, you can choose a different directory using the ``-DCMAKE_INSTALL_PREFIX`` option.

3. Configure the Daemon mode by setting the ``daemonize`` variable in the ``amdcuid_daemon.conf`` file in the ``daemon`` directory. Setting the ``daemonize`` variable to ``true`` installs a ``systemd`` service and a set of ``udev`` rules to detect devices and generate CUIDs for them automatically. Whereas, setting ``daemonize`` to ``false`` installs a cron job, which detects devices only during system boot and generates CUIDs for the devices found during system boot. The default setting is ``false``.

4. Perform the install and post-install tasks by running the post-install script located at ``<install prefix>/share/amdcuid/amdcuid_postinst.sh``. This script needs to be run as root or with ``sudo`` as it restarts the ``systemd`` and ``udev`` services.

   .. code-block:: shell

    make install
    <install prefix>/share/amdcuid/amdcuid_postinst.sh
