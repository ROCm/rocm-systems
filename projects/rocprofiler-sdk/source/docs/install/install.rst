.. meta::
   :description: Installation instructions for ROCprofiler-SDK
   :keywords: rocprof, sys, rocprofiler, systems, sdk, rocm, tool, profiler, install

.. _installation:

*************************************
Install ROCprofiler-SDK and rocprofv3
*************************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

.. _install-rocm:

Install ROCm Core SDK
=========================

ROCprofiler-SDK and the rocprofv3 CLI are included with the ROCm Core SDK on
Linux. For the most complete installation, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm profiler base package on Linux
===============================================

Alternatively, if you want to install ROCprofiler-SDK as part of the ROCm Profiler
base package without additional ROCm libraries and tools, install the
``amdrocm-profiler-base`` package. If you want :doc:`rocprofiler-systems
<rocprofiler-systems:index>` and :doc:`rocprofiler-compute
<rocprofiler-compute:index>`, install ``amdrocm-profiler``.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm Profiler base package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-profiler-base<rocm_version>

   Where ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest ROCm Profiler package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-profiler-base

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-profiler-base

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-profiler-base

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including
ROCprofiler-SDK. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
