.. meta::
  :description: The ROCm Data Center tool (RDC) addresses key infrastructure challenges regarding AMD GPUs in cluster and data center environments and simplifies their administration
  :keywords: RDC installation, Install RDC, Install ROCm Data Center tool, Building ROCm Data Center, Building RDC

.. _rdc-install:

***********************************
Install ROCm Data Center Tool (RDC)
***********************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./handbook`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

RDC is included with the ROCm Core SDK on Linux. For the most complete
installation, we recommend that developers use the ``amdrocm-core-sdk`` meta
package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install RDC on Linux
====================

Alternatively, if you want to install RDC without additional ROCm libraries and
tools, install the ``amdrocm-rdc`` package. This includes the RDC, base ROCm
runtime components, and system dependencies.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the RDC package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-rdc<rocm_version>

   Where ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest RDC package release for supported GPU
   architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-rdc

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-rdc

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-rdc

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including RDC. See
`Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
