.. meta::
   :description: Installation instructions for ROCm Compute Profiler (rocprofiler-compute)
   :keywords: rocprof, sys, rocprofiler, compute, rocm, tool, profiler, install, comp, perf

.. _installation:

*****************************
Install ROCm Compute Profiler
*****************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./source-install`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

ROCm Compute Profiler (rocprofiler-compute) is included with the ROCm Core SDK
on Linux. For the most complete installation, we recommend that developers use
the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm profilers on Linux
===============================

Alternatively, if you want to install ROCm Compute Profiler as part of the ROCm
Profiler package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-profiler`` package.
This includes the ROCm profilers, dependencies, and base packages.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm Profiler package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-profiler<rocm_version>

   Where ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest ROCm Profiler package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-profiler

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-profiler

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-profiler

Configuration and modulefile usage
==================================

For runtime configuration options and modulefile-based environment setup, see
:doc:`./source-install`:

* :ref:`core-install-rocprof-var` — configure the ``ROCPROF`` environment
  variable to select a profiling backend.
* :ref:`core-install-modulefiles` — load ROCm Compute Profiler via Lmod
  modulefiles for shared multi-user installations.
* :ref:`core-install-cmake-vars` — CMake variables for custom install paths,
  Python dependency locations, and build options.

.. _tarball-install:

Install from the tarball
========================

#. Download the rocprofiler-compute specific tarball for the latest release from `<https://github.com/ROCm/rocm-systems/releases>`_.
#. Untar the downloaded tarball and navigate to the ``rocprofiler-compute`` directory.
#. Follow the installation steps under :ref:`source-install`.

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including ROCm Compute
Profiler. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
