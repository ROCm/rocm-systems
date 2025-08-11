.. meta::
   :description: HSA runtime implementation
   :keywords: ROCR, ROCm, library, tool, runtime

.. _installation:

====================
Installation
====================

This document provides information required to build and install ROCR using prebuilt binaries or from source.

Build and install using prebuilt binaries
-------------------------------------------

Here is how you can install ROCR using prebuilt binaries.

Prerequisites
*******************

- A system supporting ROCm. See the `supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_.

- Install ROCm. See `how to install ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.

- Install ``libdrm`` package.

.. code-block:: shell
    
    sudo apt install libdrm-dev

The ROCR prebuilt binaries include:

**Core runtime package:**

- HSA include files to support application development on the HSA runtime for the ROCR runtime

- A 64-bit version of AMD’s HSA core runtime for the ROCR runtime

**Runtime extension package:**

- A 64-bit version of AMD’s runtime tools library

- A 64-bit version of AMD’s runtime image library

The contents of these packages are installed in ``/opt/rocm/hsa`` and ``/opt/rocm`` by default. The core runtime package depends on the ``hsakmt-roct-dev`` package.

Build and install from source
--------------------------------

Here is how you can build ROCR from source.

Prerequisites
***************

- CMake 3.7 or later. Export CMake bin into your PATH.

- Support packages ``libelf-dev`` and ``g++``.

.. code-block:: shell

    sudo apt install libelf-dev g++

- A compatible version of the ``libhsakmt`` library and the ``hsakmt.h`` header file. Obtain the latest version of these files from the `ROCT-Thunk-Interface repository <https://github.com/ROCm/ROCT-Thunk-Interface>`_.

- Install ``xxd``.

.. code-block:: shell

    sudo apt install xxd
    
Building the runtime
----------------------

The ``libhsakmt`` development packages include a CMake package config file. The runtime locates ``libhsakmt`` via ``find_package`` if ``libhsakmt`` is installed in a standard location. For installations that don't use standard ROCm paths, set CMake variables ``CMAKE_PREFIX_PATH`` or ``hsakmt_DIR`` to override ``find_package`` search paths.
The runtime includes an optional image support module (previously ``hsa-ext-rocr-dev``). By default this module is included in the runtime builds. To exclude the image module from the runtime, set the CMake variable ``IMAGE_SUPPORT`` to OFF.
To build the optional image module, install AMDGCN-compatible clang and device library. You can find the latest version of these additional build dependencies in the `ROCm package repository <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/how-to/native-install/package-manager-integration.html#packages-in-rocm-programming-models>`_.
The latest source for these projects are available in the `llvm project <https://github.com/ROCm/llvm-project>`_ and `ROCm device libs <https://github.com/ROCm/ROCm-Device-Libs>`_ repositories.

The runtime optionally supports use of the CMake user package registry. By default the registry is not modified. Set CMake variable ``EXPORT_TO_USER_PACKAGE_REGISTRY`` to ON to enable updating the package registry.

To build, install, and produce packages on a system with standard ROCm packages installed, clone your copy of ROCR and run the following from ``src/``:

.. code-block:: shell

    mkdir build
    cd build
    cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
    make
    make install
    make package

Example with a custom installation path, build dependency path, and options:

.. code-block:: shell

    cmake -DIMAGE_SUPPORT=OFF \
          -DEXPORT_TO_USER_PACKAGE_REGISTRY=ON \
          -DCMAKE_VERBOSE_MAKEFILE=1 \
          -DCMAKE_PREFIX_PATH=<alternate path(s) to build dependencies> \
          -DCMAKE_INSTALL_PATH=<custom install path for this build> \
          ..

Alternatively, use ``ccmake`` and ``cmake-gui``:

.. code-block:: shell

    mkdir build
    cd build
    ccmake ..
    press c to configure
    populate variables as desired
    press c again
    press g to generate and exit
    make

Building against the runtime
---------------------------------

The runtime provides a CMake package config file, installed by default to ``/opt/rocm/lib/cmake/hsa-runtime64``. The runtime exports CMake target ``hsa-runtime64`` in namespace ``hsa-runtime64``. A CMake project (``Foo``) using the runtime may locate, include, and link the runtime using the following template:

.. code-block:: shell

    # Add /opt/rocm to CMAKE_PREFIX_PATH.

    find_package(hsa-runtime64 1.0 REQUIRED)
    ...
    add_library(Foo ...)
    ...
    target_link_libraries(Foo PRIVATE hsa-runtime64::hsa-runtime64)
