.. meta::
   :description: A library that can be loaded by ROCr to print the AMDGPU wavefront states
   :keywords: ROCdebug-agent installation, ROCR Debug Agent installation, install ROCdebug-agent, install ROCR Debug Agent,
    build ROCdebug-agent, build ROCR Debug Agent


.. _debug-agent-installation:

==============================
ROCR Debug Agent installation
==============================

This topic provides information required to build and install ROCR Debug Agent (ROCdebug-agent) library.

System requirements
-------------------

- A system supporting ROCm. See the `supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_.

- A C++ 17 compiler such as GCC 7 or Clang 5.

- The AMD ROCm software stack. See the :doc:`ROCm installation instructions <rocm-install-on-linux:index>`.

- Install the required packages according to the OS:

.. tab-set::

  .. tab-item:: Ubuntu
    :sync: ubuntu

    .. code-block:: shell

      apt install gcc g++ make cmake libelf-dev libdw-dev

  .. tab-item:: RHEL
    :sync: rhel

    .. code-block:: shell

      yum install gcc gcc-c++ make cmake elfutils-libelf-devel elfutils-devel

  .. tab-item:: SLES
    :sync: sles

    .. code-block:: shell

      zypper install gcc gcc-c++ make cmake libelf-devel libdw-devel

- Python 3.6 or later to run the tests.

- :doc:`ROCdbgapi library <rocdbgapi:index>`. This can be installed using the ROCdbgapi package as part of the ROCm release. See the instructions to install :doc:`ROCdbgapi library <rocdbgapi:install/build>`.

.. note::

  ROCdebug-agent might become unresponsive in SELinux-enabled distributions. To learn more about this issue, see `installation troubleshooting <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/install-faq.html#issue-10-rocm-debugging-tools-might-become-unresponsive-in-selinux-enabled-distributions>`_.

Build and install
-------------------

An example command line to build and install the ROCdebug-agent library on Linux:

.. code-block:: shell

    cd rocm-debug-agent
    mkdir build && cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install ..
    make

To specify the location for the installation, use ``CMAKE_INSTALL_PREFIX``. The default location is ``/usr``.

To specify a list of paths (separated by semicolons) that are used to locate the ``cmake`` modules, use ``CMAKE_MODULE_PATH``. It is used to locate the HIP ``cmake`` modules required to build the tests. The default is location is ``/opt/rocm/hip/cmake``.

The built ROCdebug-agent library is placed in ``build/librocm-debug-agent.so.2*``.

To install the ROCdebug-agent library, use:

.. code:: shell

    make install

The installed ROCdebug-agent library and tests are placed in:

- <install-prefix>/lib/librocm-debug-agent.so.2*
- <install-prefix>/share/rocm-debug-agent/LICENSE.txt
- <install-prefix>/share/rocm-debug-agent/README.md
- <install-prefix>/src/rocm-debug-agent-test/*

Test
-----

To test the ROCdebug-agent library, use:

.. code:: shell

    make test

Output:

.. code-block:: shell

    Running tests...
    Test project /rocm-debug-agent/build
    Start 1: rocm-debug-agent-test
    1/1 Test #1: rocm-debug-agent-test ............   Passed    1.59 sec

    100% tests passed, 0 tests failed out of 1
    Total Test time (real) =   1.59 sec

You can run the tests individually outside of the ``CTest`` harness as shown:

.. code-block:: shell

    HSA_TOOLS_LIB=librocm-debug-agent.so.2 test/rocm-debug-agent-test 0
    HSA_TOOLS_LIB=librocm-debug-agent.so.2 test/rocm-debug-agent-test 1
    HSA_TOOLS_LIB=librocm-debug-agent.so.2 test/rocm-debug-agent-test 2
