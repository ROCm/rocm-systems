.. meta::
   :description: Installation instructions for the rocDecode library
   :keywords: install, rocdecode, video, rocm, lib, library,

.. _installation:

*****************
Install rocDecode
*****************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./rocDecode-build-and-install`.

.. note::
   
   To use the rocDecode samples and tutorials, the ``ROCM_PATH`` environment variable needs to point to the location of your ROCm installation:
   
   .. code:: shell

      export ROCM_PATH=path_to_your_ROCm_installation

   Set this variable after installation.

.. _install-rocm:

Install the ROCm Core SDK
=========================

rocDecode is included with the ROCm Core SDK on Linux. For the most complete
installation, we recommend that developers use the ``amdrocm-core-sdk`` meta
package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install rocDecode on Linux
==========================

Alternatively, if you want to install rocDecode as part of the ROCm
video decode package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-decode`` package.
This includes the ROCm runtime and system dependencies.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm video decode package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-decode-<dev/devel><rocm_version>

   Where:

   * ``<dev/devel>`` specifies whether to install library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   For example, to install the latest ROCm video decode package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-decode-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-decode-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-decode-devel

.. _install-build-utilities:

Locating build utilities and samples
====================================

Building applications against rocDecode, or building and running the samples and
CTest-based tests, requires the CMake helper modules, utility sources, and sample
sources that are installed to ``$ROCM_PATH/share/rocdecode/``:

* ``share/rocdecode/cmake/`` - ``FindFFmpeg``, ``FindLibva``, and ``FindLibdrm_amdgpu``
  helper modules (ships with the development package)
* ``share/rocdecode/utils/`` - utility classes and HIP kernels shared by the samples
  (for example, ``RocVideoDecoder`` and ``VideoDemuxer``)
* ``share/rocdecode/samples/`` - sample application sources
* ``share/rocdecode/video/`` and ``share/rocdecode/frames/`` - test media and reference frames
* ``share/rocdecode/test/`` - CTest definitions and test scripts

The rocDecode runtime library, development headers, and the CMake helper modules
(``share/rocdecode/cmake/``) come with a standard ROCm Core SDK install (the
``amdrocm-core-sdk`` meta package) and the development package. However, the
remaining ``share/rocdecode/`` files listed above — the ``utils/`` sources, the
``samples/`` sources, and the test media — do **not** ship with the Core SDK or
the development package. When installing with a package manager, they are provided
by the ``amdrocm-decode-test`` package. Install it if you need the utility sources
or samples to build against rocDecode (for example, when building rocAL). The
``amdrocm-decode`` package names apply to ROCm 7.13 and later; earlier ROCm
releases use different package naming.

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt install amdrocm-decode-test

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf install amdrocm-decode-test

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper install amdrocm-decode-test

.. note::

   In the generic (``.tar.zst``) artifacts published by TheRock, the ``utils/``,
   ``samples/``, and test media under ``share/rocdecode/`` are packaged in the
   ``rocdecode-test`` artifact (for example, ``rocdecode_test_generic``) rather
   than the ``rocdecode-dev`` artifact. If you obtain rocDecode from those
   artifacts, install or extract the ``rocdecode-test`` artifact in addition to
   ``rocdecode-dev``.

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including rocDecode.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
