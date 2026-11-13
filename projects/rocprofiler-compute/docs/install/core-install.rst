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

   .. tab-set::

      .. tab-item:: Package manager

         On Linux, package names use the following format:

         .. code-block:: shell-session

            amdrocm-profiler<rocm_version>

         ``<rocm_version>`` represents the ROCm Core SDK version to install. Omit
         this suffix to install the latest available version.

         For example, to install the latest ROCm Profiler package release for
         supported GPU architectures:

         .. tab-set::

            .. tab-item:: Debian-based distros

               Use the following command on Ubuntu and other Debian-based Linux
               distributions to install ROCm profilers:

               .. code-block:: bash

                  sudo apt install amdrocm-profiler

            .. tab-item:: RHEL-based distros

               Use the following command on RHEL, Oracle Linux, and other RHEL-based
               Linux distributions to install ROCm profilers:

               .. code-block:: bash

                  sudo dnf install amdrocm-profiler

            .. tab-item:: SLES

               Use the following command on SLES to install ROCm profilers:

               .. code-block:: bash

                  sudo zypper install amdrocm-profiler

      .. tab-item:: pip

         Use the following commands to create and activate a Python virtual
         environment, then install ROCm with the ``[profiler]`` extra:

         .. code-block:: bash

            # Create and activate a Python virtual environment.
            python3 -m venv .venv
            source .venv/bin/activate

            # Install ROCm and the profilers from the AMD package repository.
            python -m pip install --index-url https://stable.repo.amd.com/rocm/whl-next/ "rocm[profiler]"

         .. note::

            This pip installation option installs only the components required to run ROCm profiling tools (ROCm Compute Profiler and :doc:`ROCm Systems Profiler <rocprofiler-systems:index>`). It doesn't include other components for developing or building ROCm applications. To install the full development environment, use ``pip install "rocm[devel]"`` or refer to the appropriate pip installation instruction under :doc:`Install AMD ROCm <rocm:install/rocm>` for your system environment.

            This environment runs profile mode. Analyze mode needs Python
            packages that profile mode does not, so install those separately as
            described in :ref:`analyze-deps`.

.. _analyze-deps:

Install the analyze mode dependencies
=====================================

Profile mode uses only the standard library. Analyze mode needs packages such as
``numpy``, ``pandas``, ``plotly``, and ``rich``, at the versions it pins.

Install them into a virtual environment of their own, separate from the one your
profiled application uses. Those pins can conflict with a workload that brings
its own ``torch`` or ``numpy``, and a shared environment breaks either side.

.. code-block:: bash

   python3 -m venv ~/.venvs/rocprof-compute-analyze
   source ~/.venvs/rocprof-compute-analyze/bin/activate
   python -m pip install --extra-index-url https://<stable/nightly>.repo.amd.com/rocm/whl-next/ "rocm-profiler[compute-analyze]"

This command uses ``--extra-index-url`` rather than the ``--index-url`` used
above because the analyze packages come from PyPI while ``rocm-profiler`` comes
from the AMD package repository. ``--index-url`` would replace PyPI and leave
the analyze packages unresolvable.

To confirm the environment has everything analyze mode needs:

.. code-block:: bash

   rocprof-compute analyze --verify-deps

If you installed ROCm Compute Profiler from a package manager or the tarball
rather than pip, install the dependencies from the requirements file in that
installation:

.. code-block:: bash

   pip install -r <ROCM_PATH>/libexec/rocprofiler-compute/requirements.txt

``<ROCM_PATH>`` is your ROCm installation path, for example ``/opt/rocm``.

Install from source
===================

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
