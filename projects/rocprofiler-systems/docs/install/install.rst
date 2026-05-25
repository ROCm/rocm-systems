.. meta::
   :description: Installation instructions for ROCm Systems Profiler (rocprofiler-systems)
   :keywords: rocprof, sys, rocprofiler, systems, rocm, tool, profiler, install

.. _installation:

*****************************
Install ROCm Systems Profiler
*****************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./build`.

.. _install-rocm:

Install the ROCm Core SDK
=========================

ROCm Systems Profiler (rocprofiler-systems) is included with the ROCm Core SDK
on Linux. For the most complete installation, we recommend that developers use
the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm profilers on Linux
===============================

Alternatively, if you want to install ROCm Systems Profiler as part of the ROCm
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
            python -m pip install --index-url https://repo.amd.com/rocm/whl-multi-arch/ "rocm[profiler]"

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including ROCm Systems
Profiler. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.

.. _post-installation-steps:

Post-installation steps
=======================

After installation, you can optionally configure the ROCm Systems Profiler environment.
You should also test the executables to confirm ROCm Systems Profiler is correctly installed.

Configure the environment
-------------------------

If environment modules are available and preferred, then add them using these commands:

* Replacing ``1.0.0`` with the desired version number to load:

.. code-block:: shell

   module use /opt/rocprofiler-systems/share/modulefiles
   module load rocprofiler-systems/1.0.0

* Alternatively, you can directly source the ``setup-env.sh`` script:

.. code-block:: shell

   source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

Test the executables
-----------------------------------

Successful execution of these commands confirms that the installation does not have any
issues locating the installed libraries:

.. code-block:: shell

   rocprof-sys-instrument --help
   rocprof-sys-avail --help

.. note::

   If ROCm support is enabled, you might have to add the path to the ROCm libraries to ``LD_LIBRARY_PATH``,
   for example, ``export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}``.

.. _post-installation-troubleshooting:

Troubleshooting
===============

This section explains how to resolve certain issues that might happen when you first use ROCm Systems Profiler.

Issues with RHEL and SELinux
----------------------------

RHEL (Red Hat Enterprise Linux) and related distributions of Linux automatically enable a security feature
named SELinux (Security-Enhanced Linux) that prevents ROCm Systems Profiler from running.
This issue applies to any Linux distribution with SELinux installed, including RHEL and Rocky Linux.
The problem can happen with any GPU, or even without a GPU.

The problem occurs after you instrument a program and try to
run ``rocprof-sys-run`` with the instrumented program.

.. code-block:: shell

   g++ hello.cpp -o hello
   rocprof-sys-instrument -M sampling -o hello.instr -- ./hello
   rocprof-sys-run -- ./hello.instr

Instead of successfully running the binary with call-stack sampling,
ROCm Systems Profiler crashes with a segmentation fault.

.. note::

   If you are physically logged in on the system (not using SSH or a remote connection),
   the operating system might display an SELinux pop-up warning in the notifications.

To workaround this problem, either disable SELinux or configure it to use a more
permissive setting.

To avoid this problem for the duration of the current session, run this command
from the shell:

.. code-block:: shell

   sudo setenforce 0

For a permanent workaround, edit the SELinux configuration file using the command
``sudo vim /etc/sysconfig/selinux`` and change the ``SELINUX`` setting to
either ``Permissive`` or ``Disabled``.

.. note::

   Permanently changing the SELinux settings can have security implications.
   Ensure you review your system security settings before making any changes.

Modifying RPATH details
-----------------------

If you're experiencing problems loading your application with an instrumented library,
then you might have to check and modify the RPATH specified in your application.
See the section on `troubleshooting RPATHs <../how-to/instrumenting-rewriting-binary-application.html#rpath-troubleshooting>`_
for further details.

Configuring PAPI to collect hardware counters
---------------------------------------------

To use PAPI to collect the majority of hardware counters, ensure
the ``/proc/sys/kernel/perf_event_paranoid`` setting has a value less than or equal to ``2``.
For more information, see the :ref:`rocprof-sys_papi_events` section.
