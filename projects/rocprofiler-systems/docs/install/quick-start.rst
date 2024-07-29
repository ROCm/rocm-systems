.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
Omnitrace quick start
*************************************

To install Omnitrace, download the `Omnitrace installer <https://github.com/ROCm/omnitrace/releases/latest/download/omnitrace-install.py>`_ 
and specify ``--prefix <install-directory>``. The script attempts to auto-detect 
the appropriate OS distribution and version. To include AMD ROCm Software support, 
specify ``--rocm X.Y``, where ``X`` is the ROCm major
version and ``Y`` is the ROCm minor version, for example, ``--rocm 6.2``.

.. code-block:: shell

   wget https://github.com/ROCm/omnitrace/releases/latest/download/omnitrace-install.py
   python3 ./omnitrace-install.py --prefix /opt/omnitrace --rocm 6.2

This script supports installation on Ubuntu, OpenSUSE, Red Hat, Debian, CentOS, and Fedora.
If the target OS is compatible with one of the operating system versions listed in
the comprehensive :doc:`Installation guidelines <./install>`,
specify ``-d <DISTRO> -v <VERSION>``. For example, if the OS is compatible with Ubuntu 22.04, pass
``-d ubuntu -v 22.04`` to the script.

.. note::

   If you have ROCm version 6.2 or higher installed, you can use the
   package manager to install a pre-built copy of Omnitrace using 
   ``apt install omnitrace`` or ``dnf install omnitrace``.
