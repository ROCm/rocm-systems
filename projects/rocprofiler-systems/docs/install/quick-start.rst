.. meta::
   :description: ROCm Systems Profiler quick start documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, profiler, quick start, getting started, quick install, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
ROCm Systems Profiler quick start
*************************************

To install ROCm Systems Profiler, download the
`ROCm Systems Profiler installer <https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprofiler-systems-install.py>`_
and specify ``--prefix <install-directory>``. The script attempts to auto-detect
the appropriate OS distribution and version. To include AMD ROCm Software support,
specify ``--rocm X.Y``, where ``X`` is the ROCm major
version and ``Y`` is the ROCm minor version, for example, ``--rocm 6.3``.

.. code-block:: shell

   wget https://github.com/ROCm/rocprofiler-systems/releases/latest/download/rocprofiler-systems-install.py
   python3 ./rocprofiler-systems-install.py --prefix /opt/rocprofiler-systems --rocm 6.3

This script supports installation on Ubuntu, OpenSUSE, Red Hat, Debian, CentOS, and Fedora.
If the target OS is compatible with one of the operating system versions listed in
the comprehensive :doc:`Installation guidelines <./install>`,
specify ``-d <DISTRO> -v <VERSION>``. For example, if the OS is compatible with Ubuntu 22.04, pass
``-d ubuntu -v 22.04`` to the script.

Install via package manager
============================

   If you have ROCm version 6.3 or higher installed, you can use the
   package manager to install a pre-built copy of ROCm Systems Profiler.

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         $ sudo apt install rocprofiler-systems

   .. tab-item:: Red Hat Enterprise Linux

      .. code-block:: shell

         $ sudo dnf install rocprofiler-systems

   .. tab-item:: SUSE Linux Enterprise Server

      .. code-block:: shell

         $ sudo zypper install rocprofiler-systems
