AMD Debugger API (ROCdbgapi)
============================

> [!NOTE]
> The published documentation is available at [ROCdbgapi documentation](https://rocm.docs.amd.com/projects/ROCdbgapi/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents.

Introduction
------------

The AMD Debugger API is a library that provides all the support necessary for a
debugger and other tools to perform low level control of the execution and
inspection of execution state of AMD's commercially available GPU architectures.

For more information about the AMD ROCm ecosystem, see:

- https://rocmdocs.amd.com/

Build the AMD Debugger API Library
----------------------------------

The ROCdbgapi library can be built on Ubuntu 18.04, Ubuntu 20.04, Centos 8.1,
RHEL 8.1, SLES 15 Service Pack 1, and Windows 11.

Building the ROCdbgapi library has the following prerequisites:

1. A C++17 compiler such as GCC 7 or Clang 5.

2. AMD Code Object Manager Library (ROCcomgr).  On Linux, this can be installed
   as part of the AMD ROCm release by the ``comgr`` package.  On Windows, this
   is installed as part of the HIP SDK release.

3. The HSA runtime headers.  On Linux, this is installed as part of the
   ``hsa-runtime64`` package.  The headers can also be found in the sources of
   the ROCr (ROCm Runtime) project.

4. ROCm CMake modules, which can be installed using the ``rocm-cmake`` package
   available as part of the AMD ROCm release.

5. For Ubuntu 18.04 and Ubuntu 20.04, the following adds the needed packages:

   ````shell
   apt install gcc g++ make cmake doxygen graphviz texlive-full
   ````

   NOTE: The ``doxygen`` 1.8.13 that is installed by Ubuntu 18.04 has a bug
   that prevents the PDF from being created.  ``doxygen`` 1.8.11 can be built
   from source to avoid the issue.

6. For CentOS 8.1 and RHEL 8.1, the following adds the needed packages:

   ````shell
   yum install -y gcc gcc-g++ make cmake doxygen graphviz texlive \
     texlive-xtab texlive-multirow texlive-sectsty texlive-tocloft \
     texlive-tabu texlive-adjustbox
   ````

   NOTE: The ``doxygen`` 1.8.14 that is installed by CentOS 8.1 and RHEL 8.1,
   has a bug that prevents the PDF from being created. ``doxygen`` 1.8.11 can be
   built from source to avoid the issue.

7. For SLES 15 Service Pack 15, the following adds the needed packages:

   ````shell
   zypper in gcc gcc-g++ make cmake doxygen graphviz texlive-scheme-medium \
     texlive-hanging texlive-stackengine texlive-tocloft texlive-etoc \
     texlive-tabu
   ````

8. For Windows using MSYS2, the following adds the needed packages:

   ````shell
   pacman -S make mingw-w64-ucrt-x86_64-gcc \
     texinfo texinfo-tex \
     mingw-w64-x86_64-texlive-plain-generic \
     mingw-w64-x86_64-texlive-latex-recommended \
     mingw-w64-x86_64-texlive-latex-extra \
     doxygen mingw-w64-x86_64-graphviz
   ````

Here is an example command line to build the ROCdbgapi library:

````shell
cd rocdbgapi
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install ..
make
````

You may substitute a path of your own choosing for ``CMAKE_INSTALL_PREFIX``.

By default, ROCdbapi is built as a shared library.  You may build it
instead as a static library by setting the ``BUILD_SHARED_LIBS`` cmake
variable to ``OFF``.

To specify a custom path to the HSA runtime headers, set the
``HSA_RUNTIME_INCLUDE_DIRECTORIES`` cmake variable.

The built ROCdbgapi library will be placed in:

- ``build/include/amd-dbgapi.h``
- ``build/librocm-dbgapi.(a|so*|dll|dll.a|lib)``

An example command-line to generate the HTML and PDF library documentation is:

````shell
make doc
````

The generated ROCdbgapi library documentation is put in:

- ``doc/html/index.html``
- ``doc/latex/refman.pdf``

To install the ROCdbgapi library and documentation:

````shell
make install
````

The installed ROCdbgapi library and documentation will be placed in:

- ``../install/include/amd-dbgapi.h``
- ``../install/lib/librocm-dbgapi.*``
- ``../install/share/amd-dbgapi/LICENSE.txt``
- ``../install/share/amd-dbgapi/README.md``
- ``../install/share/html/amd-dbgapi/index.html``
- ``../install/share/doc/amd-dbgapi/amd-dbgapi.pdf``

Running the AMD Debugger API Library
------------------------------------

To use the ROCdbgapi library, the ROCcomgr library must be installed.  On Linux,
this can be installed as part of the AMD ROCm release by the ``comgr`` package:

- ``libamd_comgr.so``

On Windows, ROCcomgr is installed as part of the HIP SDK release:

- ``C:\Windows\System32\amd_comgr_*.dll``

To enable AMD GPU debugging, the ROCdbgapi library requires that an AMD GPU
runtime library is loaded in the inferior:

- On Linux, the required runtime library is AMD's implementation of the HSA
  runtime, maintained by the ROCr (ROCm Runtime) project.  This can be installed
  using the ``hsa-rocr-dev`` package available as part of the AMD ROCm release:

  ``libhsa-runtime64.so``

- On Windows, the required runtime library is AMD's HIP runtime library,
  maintained by the CLR project.  This is installed as part of the HIP SDK
  release:

  ``C:\Windows\System32\amdhip64_*.dll``

The AMD Debugger API library has an optional runtime dependency to the
`amdgpu.ids` database file, located in `/opt/amdgpu/share/libdrm/amdgpu.ids`
or `/usr/share/libdrm/amdgpu.ids`.

The `/opt/amdgpu/share/libdrm/amdgpu.ids` database is provided by the
`libdrm-amdgpu-common` ROCm package on all Linux distributions.

The `/usr/share/libdrm/amdgpu.ids` is provided by the following packages:
- SLES: `libdrm-amdgpu`
- RHEL: `libdrm`
- Debian and Ubuntu: `libdrm-common`

Known Limitations and Restrictions
----------------------------------

You can refer to the following sections in the *AMD Debugger API Specification*
documentation for:

- *Supported AMD GPU Architectures* provides the list of supported AMD GPU
  architectures.
- *Known Limitations and Restrictions* provides information about known
  limitations and restrictions.

The ROCdbgapi library is compatible with the following interface versions:

- *AMD GPU Driver Version*
  - See ``KFD_IOCTL_MAJOR_VERSION`` and ``KFD_IOCTL_MINOR_VERSION`` in
    ``src/linux/kfd_ioctl.h`` which conform to [semver](http://semver.org/).
- *AMD GPU Driver Debug ioctl Version*
  - See ``KFD_IOCTL_DBG_MAJOR_VERSION`` and ``KFD_IOCTL_DBG_MINOR_VERSION`` in
    ``src/linux/kfd_ioctl.h`` which conform to [semver](http://semver.org/).
- *ROCm Runtime r_debug ABI Version*
  - See ``RUNTIME_RDEBUG_VERSION_*`` in ``src/runtime_rdebug.h``.
- *Architectures and Firmware Versions*
  - See ``s_gfxip_lookup_table`` in ``src/os_driver.cpp``.

Disclaimer
----------

The information contained herein is for informational purposes only and is
subject to change without notice.  While every precaution has been taken in the
preparation of this document, it may contain technical inaccuracies, omissions
and typographical errors, and AMD is under no obligation to update or otherwise
correct this information.  Advanced Micro Devices, Inc. makes no
representations or warranties with respect to the accuracy or completeness of
the contents of this document, and assumes no liability of any kind, including
the implied warranties of noninfringement, merchantability or fitness for
particular purposes, with respect to the operation or use of AMD hardware,
software or other products described herein.  No license, including implied or
arising by estoppel, to any intellectual property rights is granted by this
document.  Terms and limitations applicable to the purchase or use of AMD’s
products are as set forth in a signed agreement between the parties or in AMD’s
Standard Terms and Conditions of Sale.

AMD®, the AMD Arrow logo, ROCm® and combinations thereof are trademarks of
Advanced Micro Devices, Inc.  Linux® is the registered trademark of Linus
Torvalds in the U.S. and other countries.  PCIe® is a registered trademark of
PCI-SIG Corporation.  RedHat® and the Shadowman logo are registered trademarks
of Red Hat, Inc. www.redhat.com in the U.S. and other countries.  SUSE® is a
registered trademark of SUSE LLC in the United Stated and other countries.
Ubuntu® and the Ubuntu logo are registered trademarks of Canonical Ltd.  Other
product names used in this publication are for identification purposes only and
may be trademarks of their respective companies.

Copyright (c) 2019-2024 Advanced Micro Devices, Inc.  All rights reserved.
