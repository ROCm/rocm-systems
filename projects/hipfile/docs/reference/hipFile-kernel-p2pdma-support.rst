.. meta::
   :description: Check a kernel for P2PDMA support and see the kernels and distributions tested for hipFile fastpath support
   :keywords: hipFile, ROCm, P2PDMA, CONFIG_PCI_P2PDMA, kernel, distribution, support matrix, AMD

.. _hipfile-kernel-p2pdma-support:

**********************************
hipFile kernel P2PDMA support
**********************************

hipFile's fastpath requires a Linux kernel PCIe peer-to-peer DMA (P2PDMA) support. hipFile's fastpath moves data directly between the GPU and the storage device
using PCIe peer-to-peer DMA. This requires a Linux kernel built with
``CONFIG_PCI_P2PDMA=y``. Without it, the fastpath backend cannot run at all.

.. warning::

   Not all ROCm-supported distributions have a default kernel that
   enables ``CONFIG_PCI_P2PDMA``. See :ref:`hipfile-os-vm-testing` for a list of tested distribution kernels and their P2PDMA support.

.. _hipfile-check-p2pdma:

Check for kernel P2PDMA support
===============================

Use the following command to read the configuration of the running kernel and determine whether it supports P2PDMA:

.. code:: shell

   grep CONFIG_PCI_P2PDMA /boot/config-$(uname -r)

If ``CONFIG_PCI_P2PDMA=y`` is returned, the kernel supports peer-to-peer DMA.
If no string is returned or if ``# CONFIG_PCI_P2PDMA is not set`` is returned,
peer-to-peer DMA isn't supported.

If there are no configuration files present under ``/boot/``, search for the
configuration files in these directories instead:

.. code:: shell

   zgrep CONFIG_PCI_P2PDMA /proc/config.gz
   grep CONFIG_PCI_P2PDMA /lib/modules/$(uname -r)/build/.config

.. note::

   ``CONFIG_PCI_P2PDMA`` is a boolean kernel option. It's either compiled into
   the kernel image or it isn't. It can't be changed.

.. _hipfile-os-vm-testing:

OS VM testing
=============

A set of tests was run in virtual machines using operating systems from the
`ROCm operating system compatibility matrix
<https://rocm.docs.amd.com/en/latest/about/release-notes.html#operating-system-support>`_.
Equivalent or closest open-source releases were used for RHEL and SLES.

In the table, ``TARGET`` refers to the operating system from the ROCm
compatibility matrix and ``OS`` refers to the actual operating system that was
used for testing. ``KERNEL`` shows the kernel version tested. ``P2PDMA``
indicates whether the kernel has ``CONFIG_PCI_P2PDMA`` set in its config.

.. list-table::
   :header-rows: 1
   :widths: 18 30 36 16

   * - TARGET
     - OS
     - KERNEL
     - P2PDMA
   * - debian-12
     - Debian GNU/Linux 12 (bookworm)
     - ``6.1.0-53-amd64``
     - No
   * - debian-13
     - Debian GNU/Linux 13 (trixie)
     - ``6.12.107+deb13-amd64``
     - No
   * - oracle-10
     - Oracle Linux Server 10.1
     - ``6.12.0-109.67.6.el10uek.x86_64``
     - No
   * - oracle-9
     - Oracle Linux Server 9.8
     - ``6.12.0-1.23.3.2.el9uek.x86_64``
     - No
   * - rhel-10.0
     - Rocky Linux 10.0 (Red Quartz)
     - ``6.12.0-55.41.1.el10_0.x86_64``
     - Yes
   * - rhel-10.2
     - Rocky Linux 10.2 (Red Quartz)
     - ``6.12.0-211.58.1.el10_2.x86_64``
     - Yes
   * - rhel-8.10
     - Rocky Linux 8.10 (Green Obsidian)
     - ``4.18.0-553.166.1.el8_10.x86_64``
     - No
   * - rhel-9.4
     - Rocky Linux 9.4 (Blue Onyx)
     - ``5.14.0-427.42.1.el9_4.x86_64``
     - Yes
   * - rhel-9.6
     - Rocky Linux 9.6 (Blue Onyx)
     - ``5.14.0-570.58.1.el9_6.x86_64``
     - Yes
   * - rhel-9.8
     - Rocky Linux 9.8 (Blue Onyx)
     - ``5.14.0-687.49.1.el9_8.x86_64``
     - Yes
   * - sles-15.7-native
     - SLES 15 SP7
     - ``6.4.0-150700.53.81-default``
     - Yes
   * - sles-16.0
     - openSUSE Leap 16.0
     - ``6.12.0-160000.37-default``
     - Yes
   * - ubuntu-2204
     - Ubuntu 22.04.5 LTS
     - ``5.15.0-194-generic``
     - No
   * - ubuntu-2204-hwe
     - Ubuntu 22.04.5 LTS
     - ``6.8.0-138-generic``
     - Yes
   * - ubuntu-2404
     - Ubuntu 24.04.4 LTS
     - ``6.8.0-142-generic``
     - Yes
   * - ubuntu-2404-hwe
     - Ubuntu 24.04.4 LTS
     - ``7.0.0-34-generic``
     - Yes
   * - ubuntu-2604
     - Ubuntu 26.04 LTS
     - ``7.0.0-34-generic``
     - Yes
   * - ubuntu-2604-hwe
     - Ubuntu 26.04 LTS
     - ``7.0.0-34-generic``
     - Yes

All OS kernels that support P2PDMA pass the fastpath I/O tests except for RHEL
9.8 and RHEL 10.2. These two kernels report ``CONFIG_PCI_P2PDMA=y``, but using
P2PDMA on them can cause memory corruption, so don't use the fastpath on these
kernels.
