.. meta::
   :description: Install hipFile
   :keywords: hipFile, install, ROCm, build, CMake, GPU I/O, AMD, direct storage, P2PDMA, CONFIG_PCI_P2PDMA, kernel

.. _hipfile-installation:

****************
Install hipFile
****************

Before you begin, verify that your system is supported. For more information,
see :doc:`Install AMD ROCm <rocm:install/rocm>`.

For source builds, CMake options, and sparse-checkout layout from ``rocm-systems``,
see :doc:`./build-from-source`. For the Python bindings after the C library is on
the machine, see :doc:`./python-bindings`.

.. _hipfile-kernel-p2pdma:

Linux kernel requirements
=========================

hipFile's fastpath moves data directly between the GPU and the storage device
using PCIe peer-to-peer DMA. This requires a Linux kernel built with
``CONFIG_PCI_P2PDMA=y``. Without it, the fastpath backend cannot run at all.

.. warning::

   A distribution being supported by ROCm does not mean its default kernel
   enables ``CONFIG_PCI_P2PDMA``. Ubuntu 22.04 LTS is a supported ROCm
   distribution, but its default 5.15 kernel is built without the option, so the
   hipFile fastpath cannot run on it.

When the kernel lacks P2PDMA support, hipFile does not fail. Every I/O request
silently uses the fallback path instead, which copies through a host bounce
buffer. The only symptom is lower than expected throughput.

Check for kernel P2PDMA support
-------------------------------

Read the configuration of the running kernel:

.. code:: shell

   grep CONFIG_PCI_P2PDMA /boot/config-$(uname -r)

``CONFIG_PCI_P2PDMA=y`` means the kernel supports peer-to-peer DMA. Output of
``# CONFIG_PCI_P2PDMA is not set``, or no output at all, means it does not.

Some distributions do not install a config file under ``/boot``. Try these
locations instead:

.. code:: shell

   zgrep CONFIG_PCI_P2PDMA /proc/config.gz
   grep CONFIG_PCI_P2PDMA /lib/modules/$(uname -r)/build/.config

.. note::

   ``CONFIG_PCI_P2PDMA`` is a boolean kernel option. It is either compiled into
   the kernel image or absent from it, and there is no loadable module to enable
   after boot. Checking ``lsmod`` won't tell you anything, and turning the option
   on means running a different kernel.

Verified distributions
----------------------

The following table records kernels that AMD has checked. Distributions update
their kernels between point releases, so treat this as a snapshot rather than a
guarantee and confirm support on the target machine.

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - Distribution
     - Kernel tested
     - ``CONFIG_PCI_P2PDMA``
   * - Ubuntu 24.10
     - ``6.11.0-19-generic``
     - Yes
   * - Ubuntu 24.04.2 LTS
     - ``6.8.0-52-generic``
     - Yes
   * - Ubuntu 22.04.5 LTS (default kernel)
     - ``5.15.0-134-generic``
     - No
   * - Ubuntu 22.04.5 LTS (HWE kernel)
     - ``6.8.0-generic``
     - Yes
   * - Ubuntu 20.04 LTS
     - ``5.4.0-150-generic``
     - No
   * - RHEL 9.4
     - ``5.14.0-503.29.1.el9_5``
     - Yes
   * - RHEL 9.2
     - ``5.14.0-284.30.1.el9_2`` or ``5.14.0-503.29.1.el9_5``
     - Yes
   * - RHEL 9.0
     - ``5.14.0-70.30.1.el9_0`` or ``5.14.0-503.29.1.el9_5``
     - Yes
   * - openSUSE Tumbleweed (March 2025)
     - ``6.13.0-1-default``
     - Yes

Enable kernel P2PDMA support
----------------------------

If the running kernel is built without ``CONFIG_PCI_P2PDMA``, boot a kernel that
has it.

* On Ubuntu 22.04 LTS, install the hardware enablement (HWE) kernel, which is
  based on 6.8 and enables the option:

  .. code:: shell

     sudo apt install linux-generic-hwe-22.04
     sudo systemctl reboot

  After rebooting, confirm the running kernel with ``uname -r`` and re-check the
  configuration.

* On Ubuntu 20.04 LTS, no available kernel enables the option. Upgrade to a newer
  release.

* On other distributions, either move to a release whose kernel enables the
  option or build a custom kernel with ``CONFIG_PCI_P2PDMA=y``.

After the kernel is in place, run ``ais-check`` to confirm that hipFile sees
P2PDMA support along with the rest of the fastpath prerequisites. See
:doc:`/how-to/checking-system-compatibility`.

.. _hipfile-install-rocm:

Install the ROCm Core SDK
=========================

hipFile ships with the ROCm Core SDK on Linux. For the broadest set of
components in one step, install the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to match your distribution and hardware.

.. _hipfile-install-linux:

Install hipFile on Linux
========================

If you want hipFile as a smaller ``amdrocm-*`` group instead of installing the full
``amdrocm-core-sdk`` stack, install the group that carries the hipFile runtime and
headers for your ROCm release. Package names follow this pattern:

.. code-block:: shell-session

   amdrocm-<group><-dev/-devel><rocm_version><-llvm_target>

Where:

* ``<-dev/-devel>`` selects library files and headers. Omit the suffix for
  runtime-only packages.

  * ``-dev`` applies on Debian-based distributions, including Ubuntu.

  * ``-devel`` applies on RPM-based distributions, including RHEL and SLES.

* ``<rocm_version>`` pins the ROCm Core SDK version. Omit it to track the latest
  release your repository publishes.

* ``<-llvm_target>`` starting with ``gfx`` limits the install to one AMD GPU
  architecture. Omit it to pull every supported architecture at higher disk
  cost.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` so
   dependencies and GPU access permissions are in place.

2. Install the ``amdrocm-*`` group that matches your ROCm version, development
   package needs, and GPU architecture. The exact ``<group>`` string for hipFile
   can change between ROCm releases. Confirm the name in the release notes for
   your target version before you run the package manager.

3. Run the install command for your distribution. Replace ``<group>`` with the
   value from step 2.

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code:: shell

            sudo apt update
            sudo apt install amdrocm-<group>-dev

      .. tab-item:: RHEL-based distros

         .. code:: shell

            sudo dnf install amdrocm-<group>-devel

      .. tab-item:: SLES

         .. code:: shell

            sudo zypper install amdrocm-<group>-devel


.. note::

   hipFile has experimental support for NVMeoF and NFSoRDMA on Linux. 

   ``amdgpu-dkms`` version 31.40 or later must be installed to use NVMeoF and NFSoRDMA.
   
   NFSoRDMA requires ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true``.

   See :doc:`/reference/hipFile-io-backends` for information on file type rules.


.. _hipfile-install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system publishes nightly
builds for the ROCm Core SDK and its components. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for download links and
support notes.
