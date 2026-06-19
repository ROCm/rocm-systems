.. meta::
  :description: ROCdbgapi provides support for a debugger and other tools to perform low-level control of running code and inspection of GPU architectures
  :keywords: Install ROCdbgapi, build ROCdbgapi, Install AMD Debugger API, Build AMD debugger API

.. _build-rocdbgapi:

Building ROCdbgapi
-------------------

This topic provides information required to build and install ROCdbgapi.

System requirements
====================

* AMD ROCm-supported platform. See the list of `supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_.

* A C++17 compiler such as GCC 7 or Clang 5.

* AMD Code Object Manager Library (ROCcomgr). Install this library (``libamd_comgr.so.1``) using the ``comgr`` package included in the AMD ROCm release.

* To enable AMD GPU debugging, load the ROCr library. Install this library (``libhsa-runtime64.so.1``) as part of the AMD ROCm release using the ``hsa-rocr-dev`` package.

* :doc:`ROCm CMake <rocmcmakebuildtools:index>` module. Install this module using the ``rocm-cmake`` package included in AMD ROCm.

* Add the given packages according to the OS:

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         apt install gcc g++ make cmake doxygen graphviz texlive-full

   .. tab-item:: RHEL

      .. code-block:: shell

         yum install -y gcc gcc-g++ make cmake doxygen graphviz texlive \
         texlive-xtab texlive-multirow texlive-sectsty texlive-tocloft \
         texlive-tabu texlive-adjustbox

   .. tab-item:: SLES

      .. code-block:: shell

         zypper in gcc gcc-g++ make cmake doxygen graphviz texlive-scheme-medium \
         texlive-hanging texlive-stackengine texlive-tocloft texlive-etoc \
         texlive-tabu

.. note::

   For OS-specific issues, see :ref:`troubleshooting`.

An example command line to build the ROCdbgapi library on Linux:

.. code-block:: shell

    cd rocdbgapi
    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install ..
    make

You can specify the path using the ``CMAKE_INSTALL_PREFIX`` parameter.

The built ROCdbgapi library is placed in:

- ``build/include/amd-dbgapi.h``
- ``build/librocm-dbgapi.so*``

Building documentation
=======================

An example command line to generate the HTML and PDF library documentation:

.. code-block:: shell

    make doc

The generated ROCdbgapi library documentation is placed in:

- ``doc/html/index.html``
- ``doc/latex/refman.pdf``

Installing ROCdbgapi
=====================

To install the ROCdbgapi library and documentation, use:

.. code-block:: shell

    make install

The installed ROCdbgapi library and documentation are placed in:

- ``../install/include/amd-dbgapi.h``
- ``../install/lib/librocm-dbgapi.so*``
- ``../install/share/amd-dbgapi/LICENSE.txt``
- ``../install/share/amd-dbgapi/README.md``
- ``../install/share/html/amd-dbgapi/index.html``
- ``../install/share/doc/amd-dbgapi/amd-dbgapi.pdf``

Running ROCdbgapi
==================

The ROCdbgapi library has an optional runtime dependency on the ``amdgpu.ids`` database file, located in ``/opt/amdgpu/share/libdrm/amdgpu.ids`` or ``/usr/share/libdrm/amdgpu.ids``.

The ``libdrm-amdgpu-common`` ROCm package provides the ``/opt/amdgpu/share/libdrm/amdgpu.ids`` database on all distributions.

The following packages provide ``/usr/share/libdrm/amdgpu.ids``:

- SLES: ``libdrm-amdgpu``

- RHEL: ``libdrm``

- Debian and Ubuntu: ``libdrm-common``

.. _troubleshooting:

Troubleshooting
================

- ROCdbgapi might become unresponsive in SELinux-enabled distributions. To learn more about this issue, see `installation troubleshooting <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/install-faq.html#issue-10-rocm-debugging-tools-might-become-unresponsive-in-selinux-enabled-distributions>`_.

- The ``doxygen`` 1.8.14 installed by RHEL 8.1 has a bug that prevents PDF creation. To avoid this issue, build ``doxygen`` 1.8.11 from source.
