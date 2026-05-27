.. meta::
  :description: Troubleshooting hipFile
  :keywords: hipFile, troubleshooting

***************
Troubleshooting
***************

This page provides guidance on troubleshooting common hipFile issues.

``ais-check``
=============

Use the ``ais-check`` utility provided by hipFile to verify that the system
meets the prerequisites for the fastpath. If any check fails, hipFile uses the
fallback path instead.

hipFile's fastpath requires the following:

* A Linux kernel with P2PDMA support enabled
* ROCm 7.4 or later, including:

  * the HIP runtime
  * ``amdgpu``

.. code-block:: none

  $ ais-check
  AIS support in:
          Kernel P2PDMA support   : True
          HIP runtime             : True
          amdgpu                  : True

For hipFile to use the fastpath, each reported requirement should be ``True``.
If any check fails, resolve that issue first before investigating further.

Backing Storage
===============

hipFile's fastpath is currently supported only when the target file or device
uses one of the following storage configurations:

* raw NVMe block device (without multipathing)
* ext4 on an NVMe block device mounted with ``data=ordered``
* xfs on an NVMe block device

If the target file or device is backed by any other storage configuration,
hipFile uses the fallback path.

.. note::

   Setting ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true`` bypasses the file system
   check and allows the fastpath to attempt I/O on unsupported file systems.
   This is not recommended for production use, as it may result in data
   corruption or data loss.

NVMe multipath
--------------

hipFile does not support NVMe multipath devices for fastpath I/O.

To disable NVMe multipath on Ubuntu 24.04:

.. code-block:: bash

  sudo bash -c 'echo "options nvme_core multipath=N" > /etc/modprobe.d/nvme_core.conf'
  sudo update-initramfs -c -k all
  sudo systemctl reboot

Alignment and I/O size
=====================

Each I/O request must satisfy alignment and size requirements of the underlying
file system and storage device to use hipFile's fastpath. Clients can use
``statx`` to determine file alignment (``stx_dio_offset_align``), I/O size
(``stx_dio_offset_align``), and memory alignment (``stx_dio_mem_align``)
requirements.

If an I/O request does not satisfy alignment and size requirements, hipFile will
use the fallback path. ``ais-stats`` can be used to determine if I/O requests
are using the fallback path.

I/O statistics
=============

Use ``ais-stats`` to display runtime hipFile I/O statistics. These stats help
determine whether requests are using the fastpath or fallback path.

Set ``HIPFILE_STATS_LEVEL`` to ``1`` or ``2`` before running ``ais-stats``.
For usage details and examples, see :doc:`/reference/hipFile-ais-stats-tool`.

Performance baseline
====================

Use ``fio`` with the ``psync`` engine to establish a performance baseline for
hipFile. When using the same storage device and similar I/O settings (for
example block size, queue depth, and read/write pattern), hipFile should be
able to achieve similar performance to ``fio`` ``psync``.

For example, to estimate the expected baseline on the file system mounted at
``/mnt/storage``, run:

.. code-block:: none

    $ fio \
          --name=test \
          --directory /mnt/storage \
          --ioengine=psync \
          --rw=randread \
          --direct=1 \
          --size=128M \
          --bs=1M \
          --time_based \
          --ramp_time=5 \
          --runtime=10 \
          --numjobs=1 \
          --group_reporting

PCIe Topology
=============

Use the ``lstopo`` command to inspect the link speeds and NUMA topology of the
system's PCIe devices. On some systems, bandwidth between the GPU and storage
device may be limited by the PCIe topology.

Run the following command to display the system PCIe topology:

.. code-block:: none

  lstopo --filter core:none --filter group:none --no-caches --no-smt -.ascii

System Log
==========

Inspect the system log for any hipFile/AIS-related errors.

For example, the following log messages from ``amdgpu`` indicate that hipFile
I/O could not be mapped to a supported I/O device.

.. code-block:: none

  Jun 01 13:11:39 kernel: amdgpu: Invalid file path or mount point
  Jun 01 13:11:39 kernel: amdgpu: Failed to read AIS file: -19

Disable the Fallback Path
=========================

When hipFile is unable to use its fastpath, it routes I/O through the fallback
path. When investigating performance issues, it can be useful to disable the
fallback path. With fallback disabled, hipFile returns an error when an I/O
request cannot be completed using the fastpath.

Disable the fallback path by adding ``HIPFILE_ALLOW_COMPAT_MODE=false`` to the
environment.

For example:

.. code-block:: none

  $ HIPFILE_ALLOW_COMPAT_MODE=false <command>

Poor performance within QEMU virtual machines
=============================================

Within a QEMU virtual machine, if hipFile I/O throughput is lower than expected,
PCIe devices may be incorrectly attached to the virtual machine root bus.
Ensure that each passthrough PCIe device (GPU, NVMe, ...) is attached to its
own root port. Devices attached directly to the root bus may be unable to fully
utilize available PCIe bandwidth.

When PCIe devices are attached directly to the root bus, ``lspci -tv`` will
look similar to the following:

.. code-block:: none

  $ lspci -tv
  -[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
             +-01.0  Device 1234:1111
             +-02.0  Red Hat, Inc. Virtio block device
             +-03.0  Red Hat, Inc. Virtio network device
             +-04.0  KIOXIA Corporation NVMe SSD Controller XG8
             +-11.0  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]
             +-11.1  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI/DP Audio
             +-11.2  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 USB
             +-11.3  Advanced Micro Devices, Inc. [AMD/ATI] Device 7444
             +-1f.0  Intel Corporation 82801IB (ICH9) LPC Interface Controller
             +-1f.2  Intel Corporation 82801IR/IO/IH (ICH9R/DO/DH) 6 port SATA Controller [AHCI mode]
             \-1f.3  Intel Corporation 82801I (ICH9 Family) SMBus Controller


In this configuration, TransferBench CPU -> GPU and GPU -> CPU averages are
typically much lower than bare-metal measurements:

.. code-block:: none

  $ TransferBench p2p

                             CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
  Averages (During UniDir):       N/A      5.48      7.17       N/A


When PCIe devices are attached to their own root ports, ``lspci -tv`` output
will be similar to the following:

.. code-block:: none

  $ lspci -tv
  -[0000:00]-+-00.0  Intel Corporation 82G33/G31/P35/P31 Express DRAM Controller
             +-01.0  Device 1234:1111
             +-02.0  Red Hat, Inc. Virtio block device
             +-03.0  Red Hat, Inc. Virtio network device
             +-04.0-[01]----00.0  KIOXIA Corporation NVMe SSD Controller XG8
             +-05.0-[02]--+-00.0  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 [Radeon RX 7900 XT/7900 XTX/7900M]
             |            +-00.1  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 HDMI/DP Audio
             |            +-00.2  Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 USB
             |            \-00.3  Advanced Micro Devices, Inc. [AMD/ATI] Device 7444
             +-1f.0  Intel Corporation 82801IB (ICH9) LPC Interface Controller
             +-1f.2  Intel Corporation 82801IR/IO/IH (ICH9R/DO/DH) 6 port SATA Controller [AHCI mode]
             \-1f.3  Intel Corporation 82801I (ICH9 Family) SMBus Controller

In this configuration, TransferBench CPU -> GPU and GPU -> CPU averages are
typically in line with bare-metal measurements:

.. code-block:: none

  $ TransferBench p2p

                             CPU->CPU  CPU->GPU  GPU->CPU  GPU->GPU
  Averages (During UniDir):       N/A     21.24     28.00       N/A

Below is an example QEMU command where GPU and NVMe devices are attached to
their own root ports:

.. code-block:: none

  ./qemu-system-x86_64 \
      -machine q35,accel=kvm,kernel_irqchip=on \
      -cpu host,topoext=on,migratable=off \
      -smp 32,sockets=1,dies=2,cores=8,threads=2 \
      -m 96G \
      -drive file=disk.qcow2,if=none,id=disk0,format=qcow2,cache=none,aio=io_uring,discard=unmap \
      -device virtio-blk-pci,drive=disk0,id=virtio-disk0 \
      -netdev user,id=net0 \
      -device virtio-net-pci,netdev=net0,id=nic0 \
      -device pcie-root-port,id=pcie.1,bus=pcie.0,chassis=1,slot=1 \
      -device vfio-pci,bus=pcie.1,host=0000:01:00.0 \
      -device pcie-root-port,id=pcie.2,bus=pcie.0,chassis=2,slot=2,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.0,addr=0.0,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.1,addr=0.1 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.2,addr=0.2 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.3,addr=0.3

From the QEMU command above, the NVMe device is passed through with:

.. code-block:: none

      -device pcie-root-port,id=pcie.1,bus=pcie.0,chassis=1,slot=1 \
      -device vfio-pci,bus=pcie.1,host=0000:01:00.0 \

From the QEMU command above, the GPU device is passed through with:

.. code-block:: none

      -device pcie-root-port,id=pcie.2,bus=pcie.0,chassis=2,slot=2,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.0,addr=0.0,multifunction=on \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.1,addr=0.1 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.2,addr=0.2 \
      -device vfio-pci,bus=pcie.2,host=0000:83:00.3,addr=0.3

See QEMU's documentation for more information about PCIe passthrough.
