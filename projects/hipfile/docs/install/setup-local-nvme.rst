.. meta::
   :description: Set up a local NVMe drive for use with hipFile.
   :keywords: hipFile, NVMe, install, ROCm, direct storage, GPU I/O

**********************************
Set up a local NVMe drive
**********************************

This page walks through a simple example of preparing a local NVMe drive for
use with hipFile. This guide will use the gdisk utility, but there are other
utilities that can be used instead. Before you begin, make sure hipFile is
installed. See :doc:`./install` for package options or :doc:`./build-from-source`
to build from source.

Prerequisites
=============

- An NVMe drive attached to the system
- ``nvme-cli`` or another tool for inspecting NVMe devices
- Root or ``sudo`` access to partition, format, and mount the drive

Identify the NVMe device
=========================

List the NVMe devices on the system to find the one you want to use.

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Ideally, pick a drive that does not have the OS installed on it. For this
guide, we will use ``/dev/nvme1n1``.

Partition and format the drive
================================

.. warning::

   The following commands erase all data on the target drive. Double-check
   the device path before proceeding.

Create a single partition on the whole drive.

.. code:: shell

   sudo sgdisk -n 1:0:0 /dev/nvme1n1

Format the partition with the filesystem hipFile supports.

.. code:: shell

   sudo mkfs.ext4 /dev/nvme1n1p1

Mount the drive
================

Create a mount point and mount the partition.

.. code:: shell

   sudo mkdir /mnt/ext4
   sudo mount /dev/nvme1n1p1 /mnt/ext4 -o data=ordered

Alternatively, add an entry into ``/etc/fstab`` to automatically mount the
partition at boot time. It is highly recommended to reference the partition
by either UUID or PARTUUID as the ``/dev/nvme#n#p#`` scheme is not guaranteed to
be consistent across reboots.

Verify the setup
=================

Create a user-accessible directory and then confirm if hipFile can see and use
this mounted drive. For example, running one of the hipFile example programs.

.. code:: shell

   sudo mkdir /mnt/ext4/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/ext4/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/ext4/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest
   md5sum /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest
