.. meta::
   :description: Instruction on how to install the RCCL library for collective communication primitives using the quick start install script
   :keywords: RCCL, ROCm, library, API, install

.. _install:

************
Install RCCL
************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./building-installing`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

RCCL is included with the ROCm Core SDK on Linux. For the most complete
installation, we recommend that developers use the ``amdrocm-core-sdk`` meta
package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install RCCL on Linux
=====================

Alternatively, if you want to install RCCL without additional ROCm libraries
and tools, install the ``amdrocm-rccl`` package. This package also includes
base ROCm packages.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the RCCL package that matches your ROCm version and development
   package needs. Package names use the following format:

   .. code-block:: shell-session

      amdrocm-rccl<-dev/devel><rocm_version>

   Where:

   * ``<-dev/devel>`` specifies whether to install library files and
     headers. Omit this suffix to only install runtime packages.

     * ``-dev`` is used on Debian-based distributions, including Ubuntu.

     * ``-devel`` is used on RPM-based distributions, including RHEL and SLES.

   * ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
     suffix to install the latest available version.

   For example: ``amdrocm-rccl-dev7.13-gfx950``

   Use the following command to install the latest RCCL development package
   release for supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-rccl-dev

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-rccl-devel

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-rccl-devel

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including RCCL.
See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
