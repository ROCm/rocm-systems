.. meta::
   :description: "ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software."
   :keywords: "Installing ROCprofiler-SDK, Install ROCprofiler-SDK, Build ROCprofiler-SDK"

.. _installing-rocprofiler-sdk:

Installing ROCprofiler-SDK
=============================

This document provides information required to install ROCprofiler-SDK from source.

Supported systems
-----------------

ROCprofiler-SDK is supported only on Linux. The following distributions are tested:

- Ubuntu 20.04
- Ubuntu 22.04
- Ubuntu 24.04
- OpenSUSE 15.5
- OpenSUSE 15.6
- Red Hat 8.8
- Red Hat 8.9
- Red Hat 8.10
- Red Hat 9.2
- Red Hat 9.3
- Red Hat 9.4

ROCprofiler-SDK might operate as expected on other `Linux distributions <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_, but has not been tested.

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

    git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
    cmake                                         \
        -B rocprofiler-sdk-build                \
        -D ROCPROFILER_BUILD_TESTS=ON           \
        -D ROCPROFILER_BUILD_SAMPLES=ON         \
        -D CMAKE_INSTALL_PREFIX=/opt/rocm       \
        rocprofiler-sdk-source

    cmake --build rocprofiler-sdk-build --target all --parallel 8

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

   .. tab-item:: SUSE Linux Enterprise Server

      .. code-block:: shell

         $ sudo zypper install rocprofiler-sdk

