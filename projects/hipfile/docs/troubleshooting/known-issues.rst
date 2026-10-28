.. meta::
  :description: Known issues with hipFile
  :keywords: hipFile, known issues, limitations

************
Known issues
************

This page documents known issues with hipFile.

.. _hipfile-greatest-priority-streams:

Fallback I/O can hang behind an application's blocked greatest-priority stream
==============================================================================

The fallback path copies data through the GPU on streams that hipFile creates
at the device's greatest priority, which gives them hardware queues that
streams of other priorities don't use. If an application blocks a stream of its
own at that priority, for example on a stream wait value, a hipFile call that
uses the fallback path can hang: its copy can be queued behind the blocked
stream on a shared hardware queue and never run.

hipFile rejects asynchronous I/O on streams at the greatest priority (see
:doc:`/reference/api-async`), so this cannot happen through the asynchronous
API. It can still happen when an application blocks its own greatest-priority
stream and then issues synchronous fallback I/O.

To work around this issue, don't block a greatest-priority stream while hipFile
I/O that may use the fallback path is in progress.

Poor performance with GPU Virtual Functions
===========================================

If hipFile I/O is only able to achieve half of the expected bandwidth of
the storage, check if the GPU is a virtual function.

Some AMD GPUs can be partitioned into multiple virtual functions (VFs) using
Single Root I/O Virtualization (SR-IOV). hipFile's fastpath is disabled on GPU
VFs and will instead use the fallback path resulting in reduced performance.

Use ``amd-smi`` to determine if your GPU device is a virtual function. The GPU's
name will end with "VF" if the device is a virtual function.

Example:

.. code-block:: none

  $ amd-smi
   +------------------------------------------------------------------------------+
   | AMD-SMI 26.2.2+1d06e2956f    amdgpu version: 6.16.13  ROCm version: 7.2.1    |
   | VBIOS version: 00143754                                                      |
   | Platform: Linux Guest                                                        |
   |-------------------------------------+----------------------------------------|
   | BDF                        GPU-Name | Mem-Uti   Temp   UEC       Power-Usage |
   | GPU  HIP-ID  OAM-ID  Partition-Mode | GFX-Uti    Fan               Mem-Usage |
   |=====================================+========================================|
   | 0000:05:00.0 AMD Instinct MI300X VF | 0 %      39 °C   0           142/750 W |
   |   0       0       2        SPX/NPS1 | 0 %        N/A           285/196288 MB |
   +-------------------------------------+----------------------------------------+

hipFile's fastpath is only supported on GPU physical functions (PFs).

``ais-check`` also detects virtual functions via ``amd-smi`` and prints a warning
when one is present. This warning won't change the exit code.

Memory corruption on RHEL 9.8 and RHEL 10.2 kernels
===================================================

A change was backported to RHEL 9.8 and 10.2 that broke DMA mapping for P2PDMA pages that are transferred through the host bridge. These two kernels report ``CONFIG_PCI_P2PDMA=y``, but using P2PDMA on them can cause memory corruption, so don't use the fastpath on these kernels.

RHEL 9.9 and RHEL 10.3, which are in development, contain a fix. See `dma-mapping: direct: fix missing mapping for THRU_HOST_BRIDGE segments <https://github.com/torvalds/linux/commit/560000d619ef162568746ce287f0c725e24ea967>`_.

.. _hipfile-ubuntu-2204-fastpath:

The Ubuntu 22.04 default kernel does not support the fastpath
=============================================================

On Ubuntu 22.04 LTS, hipFile I/O runs entirely on the fallback path and never
uses the fastpath even when ``ais-check`` reports fastpath-capable volumes.

Ubuntu 22.04 LTS is a supported ROCm distribution, but its default 5.15 kernel is
built without ``CONFIG_PCI_P2PDMA``. hipFile's fastpath requires that option in
order to perform peer-to-peer DMA between the GPU and the storage device.

``ais-check`` reports this condition as ``Kernel P2PDMA support : False``. To
confirm it from the kernel configuration, see :ref:`hipfile-check-p2pdma`.

To work around this issue, install the Ubuntu 22.04 hardware enablement (HWE)
kernel, which is based on 6.8 and enables the option:

.. code-block:: none

  $ sudo apt install linux-generic-hwe-22.04
  $ sudo systemctl reboot

See :ref:`hipfile-check-p2pdma` for how to check the running kernel on other
distributions, and :doc:`/reference/hipFile-kernel-p2pdma-support` for the
kernels AMD has tested.

High memory utilization with hipFile
====================================

Processes using child processes for I/O parallelism may observe each child
process consuming ~200MiB of memory.

For example, when fio uses the hipFile engine with four job processes, ``htop``
shows each child process consuming ~200MiB of memory.

.. code-block:: none

    PID  VIRT   RES   SHR Command
   7851 4646M  197M 79364 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7853 4646M  199M 81516 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7852 4646M  197M 79804 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0
   7850 4646M  199M 81504 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=libhipfile --rocm_io=hipfile --gpu_dev_ids=0

When fio uses the psync engine with four job processes, ``htop`` shows each child process consuming ~18MiB of memory.

.. code-block:: none

    PID VIRT   RES   SHR Command
   8021 237M 18004  1640 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8022 237M 18004  1640 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8023 237M 18044  1680 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync
   8024 237M 18040  1676 fio --time_based --runtime=120s --filename=/dev/nvme0n1 --direct=1 --rw=read --bs=8M --name=bw --numjobs=4 --ioengine=psync

The increased memory usage can be attributed to each child process independently
loading the HIP runtime and its dependencies. Using threads instead reduces
overall memory usage as the HIP runtime and its dependencies are loaded once and
shared among the process' threads.

GPU Reset on RDNA4 GPUs
=======================

hipFile's fallback path may trigger GPU resets on RDNA4 GPUs. This issue has not
been observed on other GPU architectures. The fallback path is used when I/O does
not meet the requirements to use the fastpath. When using the fallback path on
RDNA4 GPUs, the GPU may enter an unrecoverable state which triggers a GPU reset.
When a GPU reset occurs, messages similar to the following may be observed in
the system log:

.. code-block:: none

  kernel: amdgpu 0000:f3:00.0: amdgpu: MES might be in unrecoverable state, issue a GPU reset
  kernel: amdgpu 0000:f3:00.0: amdgpu: Suspending all queues failed
  kernel: amdgpu 0000:f3:00.0: amdgpu: Failed to evict process queues
  kernel: amdgpu: Failed to quiesce KFD
  kernel: amdgpu 0000:f3:00.0: amdgpu: GPU reset begin!. Source:  3

To work around this issue, ensure I/O requests meet the alignment, size,
filesystem, and device requirements of the fastpath. The fallback path can also
be disabled by setting the ``HIPFILE_ALLOW_COMPAT_MODE`` environment variable to
``false``. When the fallback path is disabled, I/O requests will be rejected if
they do not meet the requirements of the fastpath.

Poor IOPS with small I/O sizes and many threads/processes
===========================================================

Every I/O operation requires the GPU buffer to be pinned (locked in place) at the
start of the operation and unpinned at the end of the operation. Pin and unpin
operations are protected by a lock. With small I/O sizes and many
threads/processes, GPU buffers are pinned and unpinned rapidly resulting in high
lock contention and lower than expected IOPS. With I/O sizes >= 64KiB, the GPU
buffer is pinned and unpinned less frequently which reduces contention on the
lock and results in IOPS that are in line with expectations.

Lock contention can impact IOPS with I/O sizes < 64KiB and process/thread counts
>= 16. To work around this issue, use I/O sizes >= 64KiB or use fewer
threads/processes.
