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

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including rocDecode.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
