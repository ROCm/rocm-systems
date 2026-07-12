.. meta::
  :description: Known issues with hipFile
  :keywords: hipFile, known issues, limitations

************
Known issues
************

This page documents known issues with hipFile.

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
========================================================

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
