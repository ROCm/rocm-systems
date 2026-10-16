.. meta::
   :description: Kernels and distributions tested for hipFile fastpath support
   :keywords: hipFile, ROCm, P2PDMA, CONFIG_PCI_P2PDMA, kernel, distribution, support matrix, AMD

.. _hipfile-kernel-support-matrix:

**********************************
hipFile kernel support matrix
**********************************

hipFile's fastpath requires a Linux kernel built with ``CONFIG_PCI_P2PDMA=y``.
This page records the kernels and distributions that AMD has tested. For the
requirement itself and for instructions on booting a kernel that enables the
option, see :ref:`hipfile-kernel-p2pdma`.

.. _hipfile-verified-distributions:

Verified distributions
======================

The following table lists kernels that AMD has tested for peer-to-peer DMA support.
Distributions update their kernels between point releases. This table is a
snapshot and doesn't guarantee support.

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
``PASS`` indicates whether a set of I/O tests using the fastpath succeeded.

.. list-table::
   :header-rows: 1
   :widths: 16 27 32 12 13

   * - TARGET
     - OS
     - KERNEL
     - P2PDMA
     - PASS
   * - debian-12
     - Debian GNU/Linux 12 (bookworm)
     - ``6.1.0-53-amd64``
     - No
     - No
   * - debian-13
     - Debian GNU/Linux 13 (trixie)
     - ``6.12.107+deb13-amd64``
     - No
     - No
   * - oracle-10
     - Oracle Linux Server 10.1
     - ``6.12.0-109.67.6.el10uek.x86_64``
     - No
     - No
   * - oracle-9
     - Oracle Linux Server 9.8
     - ``6.12.0-1.23.3.2.el9uek.x86_64``
     - No
     - No
   * - rhel-10.0
     - Rocky Linux 10.0 (Red Quartz)
     - ``6.12.0-55.41.1.el10_0.x86_64``
     - Yes
     - Yes
   * - rhel-10.2
     - Rocky Linux 10.2 (Red Quartz)
     - ``6.12.0-211.58.1.el10_2.x86_64``
     - Yes
     - No
   * - rhel-8.10
     - Rocky Linux 8.10 (Green Obsidian)
     - ``4.18.0-553.166.1.el8_10.x86_64``
     - No
     - No
   * - rhel-9.4
     - Rocky Linux 9.4 (Blue Onyx)
     - ``5.14.0-427.42.1.el9_4.x86_64``
     - Yes
     - Yes
   * - rhel-9.6
     - Rocky Linux 9.6 (Blue Onyx)
     - ``5.14.0-570.58.1.el9_6.x86_64``
     - Yes
     - Yes
   * - rhel-9.8
     - Rocky Linux 9.8 (Blue Onyx)
     - ``5.14.0-687.49.1.el9_8.x86_64``
     - Yes
     - No
   * - sles-15.7
     - openSUSE Leap 15.6
     - ``6.4.0-150600.23.103-default``
     - Yes
     - Yes
   * - sles-15.7-native
     - SLES 15 SP7
     - ``6.4.0-150700.53.81-default``
     - Yes
     - Yes
   * - sles-16.0
     - openSUSE Leap 16.0
     - ``6.12.0-160000.37-default``
     - Yes
     - Yes
   * - ubuntu-2204
     - Ubuntu 22.04.5 LTS
     - ``5.15.0-194-generic``
     - No
     - No
   * - ubuntu-2204-hwe
     - Ubuntu 22.04.5 LTS
     - ``6.8.0-138-generic``
     - Yes
     - Yes
   * - ubuntu-2404
     - Ubuntu 24.04.4 LTS
     - ``6.8.0-142-generic``
     - Yes
     - Yes
   * - ubuntu-2404-hwe
     - Ubuntu 24.04.4 LTS
     - ``7.0.0-34-generic``
     - Yes
     - Yes
   * - ubuntu-2604
     - Ubuntu 26.04 LTS
     - ``7.0.0-34-generic``
     - Yes
     - Yes
   * - ubuntu-2604-hwe
     - Ubuntu 26.04 LTS
     - ``7.0.0-34-generic``
     - Yes
     - Yes

All OS kernels that support P2PDMA pass the fastpath I/O tests except for RHEL
9.8 and RHEL 10.2. They backported a change that broke DMA mapping for P2PDMA
pages that are transferred through the host bridge. RHEL 9.9 and RHEL 10.3,
which are in development, do contain the fix: `dma-mapping: direct: fix missing
mapping for THRU_HOST_BRIDGE segments
<https://github.com/torvalds/linux/commit/560000d619ef162568746ce287f0c725e24ea967>`_.