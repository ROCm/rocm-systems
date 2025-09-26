.. meta::
   :description: "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
   :keywords: "Installing ROCprofiler-SDK, Install ROCprofiler-SDK, Build ROCprofiler-SDK"

.. _installing-rocprofiler-sdk:

Installing ROCprofiler-SDK
=============================

This document provides information required to install ROCprofiler-SDK from source.

Supported systems
-----------------

ROCprofiler-SDK is supported on the Linux distributions specified in the `system requirements <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_.

Identifying the operating system
--------------------------------

To identify the Linux distribution and version, see the ``/etc/os-release`` and ``/usr/lib/os-release`` files:

.. code-block:: bash

    $ cat /etc/os-release
    NAME="Ubuntu"
    VERSION="20.04.4 LTS (Focal Fossa)"
    ID=ubuntu
    ...
    VERSION_ID="20.04"
    ...

The relevant fields are ``ID`` and the ``VERSION_ID``.

Build requirements
------------------

Install the following dependencies:

- Debian/Ubuntu
.. code-block:: bash

    sudo apt install -y libdw-dev libsqlite3-dev

- Red Hat Enterprise Linux/Alma Linux/Rocky Linux/Fedora

.. code-block:: bash

    sudo dnf install elfutils elfutils-devel libsqlite3x-devel clang-tools-extra gcc gcc-c++ cmake make openssl-devel
    python3 -m pip install --upgrade pip
    python3 -m pip install scikit-build

- SUSE Linux Enterprise Server

.. code-block:: bash

    sudo zypper install gcc12 gcc12-c++ cmake make python3-devel elfutils sqlite3-devel libelf-devel libdw-devel
    export CXX=/usr/bin/g++-12
    export CC=/usr/bin/gcc-12

To build ROCprofiler-SDK, install ``CMake`` as explained in the following section.

Install CMake
++++++++++++++

Install `CMake <https://cmake.org/>`_ version 3.21 (or later).

.. note::
    If the ``CMake`` installed on the system is too old, you can install a new version using various methods. One of the easiest options is to use PyPi (Python's pip).

.. code-block:: bash

    /usr/local/bin/python -m pip install --user 'cmake==3.22.0'
    export PATH=${HOME}/.local/bin:${PATH}

Building ROCprofiler-SDK from source
-------------------------------------

.. code-block:: bash

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
    cd rocm-systems
    git sparse-checkout init --cone
    git sparse-checkout set projects/rocprofiler-sdk
    git checkout develop
    python3 -m pip install -r projects/rocprofiler-sdk/requirements.txt
    cmake                                         \
        -B rocprofiler-sdk-build                \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm        \
        -DCMAKE_PREFIX_PATH=/opt/rocm           \
        projects/rocprofiler-sdk

    cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)

Installing ROCprofiler-SDK
---------------------------

To install ROCprofiler-SDK from the ``rocprofiler-sdk-build`` directory, run:

.. code-block:: bash

    cmake --build rocprofiler-sdk-build --target install

Testing ROCprofiler-SDK
------------------------

To run the built tests, ``cd`` into the ``rocprofiler-sdk-build`` directory and run:

.. code-block:: bash

    ctest --output-on-failure -O ctest.all.log


.. note::
    Running a few of these tests require you to install `pandas <https://pandas.pydata.org/>`_ and `pytest <https://docs.pytest.org/en/stable/>`_ first.

.. code-block:: bash

    /usr/local/bin/python -m pip install -r requirements.txt

Install using package manager
------------------------------

If you have ROCm version 6.2 or later installed, you can use the package manager to install a prebuilt copy of ROCprofiler-SDK.

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         $ sudo apt install rocprofiler-sdk

   .. tab-item:: Red Hat Enterprise Linux

      .. code-block:: shell

         $ sudo dnf install rocprofiler-sdk
