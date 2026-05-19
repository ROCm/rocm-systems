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

System requirements
--------------------

- `CMake <https://cmake.org/>`_ 3.21 or later. To install CMake, use:

  .. code-block:: bash

    /usr/local/bin/python -m pip install --user 'cmake==3.22.0'
    export PATH=${HOME}/.local/bin:${PATH}

  .. note::

    If the ``CMake`` installed on the system is too old, you can install a new version using various methods. One of the easiest options is to use PyPi (Python's pip).

- ROCm 6.2+ (for installation using package manager).

- Python3 with pip.

- Dependencies according to your OS:

  .. tab-set::

    .. tab-item:: Ubuntu/Debian

        .. code-block:: bash

            sudo apt install -y libdw-dev libsqlite3-dev

    .. tab-item:: RHEL/Rocky Linux

        .. code-block:: shell

            sudo dnf install elfutils elfutils-devel sqlite-devel clang-tools-extra gcc gcc-c++ cmake make openssl-devel
            python3 -m pip install --upgrade pip
            python3 -m pip install scikit-build

    .. tab-item:: SUSE

        .. code-block:: shell

            sudo zypper install gcc12 gcc12-c++ cmake make python3-devel elfutils sqlite3-devel libelf-devel libdw-devel
            export CXX=/usr/bin/g++-12
            export CC=/usr/bin/gcc-12

        .. note::

            The above ``export`` statements set the compiler environment variables only for the current terminal session. Opening a new terminal or logging out unsets these variables. To make these settings permanent, add the following lines to your ``~/.bashrc`` file:

            .. code-block:: bash

                export CXX=/usr/bin/g++-12
                export CC=/usr/bin/gcc-12

            Alternatively, ensure these variables are set before building ROCprofiler-SDK.

Building from source
---------------------

To build ROCprofiler-SDK from source, follow these steps:

1. Clone repository:

   .. code-block:: bash

    git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
    cd rocm-systems
    git sparse-checkout init --cone
    git sparse-checkout set projects/rocprofiler-sdk
    git checkout develop
    python3 -m pip install -r projects/rocprofiler-sdk/requirements.txt

2. Configure build:

   .. code-block:: bash

    cmake                                       \
        -B rocprofiler-sdk-build                \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm        \
        -DCMAKE_PREFIX_PATH=/opt/rocm           \
        projects/rocprofiler-sdk

3. Build:

   .. code-block:: bash

    cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)

4. Install ROCprofiler-SDK from the ``rocprofiler-sdk-build`` directory:

   .. code-block:: bash

    cmake --build rocprofiler-sdk-build --target install

5. Setup environment:

   .. code-block:: bash

    source /opt/rocm/share/rocprofiler-sdk/setup-env.sh

Testing ROCprofiler-SDK
------------------------

To run the built tests, ``cd`` into the ``rocprofiler-sdk-build`` directory and run:

.. code-block:: bash

    ctest --output-on-failure -O ctest.all.log

.. note::
    Running a few of these tests require you to install `pandas <https://pandas.pydata.org/>`_ and `pytest <https://docs.pytest.org/en/stable/>`_ first.

.. code-block:: bash

    /usr/local/bin/python -m pip install -r requirements.txt

Installing using package manager
---------------------------------

If you have ROCm version 6.2 or later installed, you can use the package manager to install a prebuilt copy of ROCprofiler-SDK.

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         $ sudo apt install rocprofiler-sdk

   .. tab-item:: RHEL

      .. code-block:: shell

         $ sudo dnf install rocprofiler-sdk

Verifying installation
-----------------------

To verify if ROCprofiler-SDK is successfully installed on your system, run:

.. code-block:: shell

    $ rocprofv3 --version

    $ rocprofv3 --help

Default install locations
--------------------------

By default the ROCprofiler-SDK files are installed under ``/opt/rocm/``. Under ``/opt/rocm/``, here are the default install locations for the following files:

- ``bin/``: Command-line tools such as ``rocprofv3``, ``rocprofv3-avail``, and ``rocpd2``.

- ``lib/rocprofiler-sdk``: Libraries.

- ``include/rocprofiler-sdk``: Header files.

- ``share/rocprofiler-sdk``: Environment setup script as ``setup-env.sh``.
