.. meta::
   :description: "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
   :keywords: "Installing ROCprofiler-SDK, Install ROCprofiler-SDK, Build ROCprofiler-SDK"

.. _installing-rocprofiler-sdk:

*********************************
Build ROCprofiler-SDK from source
*********************************

To build ROCprofiler-SDK as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build ROCprofiler-SDK standalone using the following
instructions.

Supported systems
=================

ROCprofiler-SDK is supported on Linux. For more information, see :ref:`ROCm
Core SDK components <rocm:release-components>`.

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
==================

To build on Linux, install the following dependencies:

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

          sudo apt install -y libdw-dev libsqlite3-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

          sudo dnf install elfutils elfutils-devel sqlite-devel clang-tools-extra gcc gcc-c++ cmake make openssl-devel
          python3 -m pip install --upgrade pip
          python3 -m pip install scikit-build

   .. tab-item:: SLES

      .. code-block:: bash

          sudo zypper install gcc12 gcc12-c++ cmake make python3-devel elfutils sqlite3-devel libelf-devel libdw-devel
          export CXX=/usr/bin/g++-12
          export CC=/usr/bin/gcc-12

      .. note::

         The above ``export`` statements set the compiler environment variables only for the current terminal session. If you open a new terminal or log out, these variables will be unset. To make these settings permanent, add the following lines to your ``~/.bashrc`` file:

         .. code-block:: bash

            export CXX=/usr/bin/g++-12
            export CC=/usr/bin/gcc-12

         Alternatively, ensure these variables are set before building ROCprofiler-SDK.

To build ROCprofiler-SDK, install ``CMake`` as explained in the following section.

Install CMake
-------------

Install `CMake <https://cmake.org/>`_ version 3.21 (or later).

.. note::

   If the ``CMake`` installed on the system is too old, you can install a new
   version using various methods. One of the easiest options is to use PyPi
   (Python's pip).

.. code-block:: bash

    /usr/local/bin/python -m pip install --user 'cmake==3.22.0'
    export PATH=${HOME}/.local/bin:${PATH}

Build ROCprofiler-SDK
=====================

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
