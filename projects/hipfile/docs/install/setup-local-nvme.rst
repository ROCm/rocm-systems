.. meta::
   :description: Set up a local NVMe drive for use with hipFile.
   :keywords: hipFile, NVMe, install, ROCm, direct storage, GPU I/O

**********************************
Set up a local NVMe drive
**********************************

To use hipFile with a locally attached NVMe drive, partition and format the
drive, mount it with a supported file system, and verify I/O on the mounted
path. Before you begin, make sure hipFile is installed. See
:doc:`./install` for package options or :doc:`./build-from-source` to build
from source.

Prerequisites
=============

- Version 31.40 or newer of the ``amdgpu-dkms`` driver, with ROCm 7.4 or later
  and the HIP runtime
- An NVMe drive attached to the system
- ``nvme-cli`` or another tool for inspecting NVMe devices
- Root or ``sudo`` access to partition, format, and mount the drive

Prepare and mount the drive
===========================

List the NVMe devices on the system to find the one you want to use.

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Pick a drive that does not have the OS installed on it. The example device
path below is ``/dev/nvme1n1``.

Partitioning uses ``sgdisk`` from the GPT fdisk package. Other partitioning
tools work as well.

.. warning::

   The commands below erase all data on the target drive. Double-check the
   device path before proceeding.

Create a single partition on the whole drive.

.. code:: shell

   sudo sgdisk -n 1:0:0 /dev/nvme1n1

Format the partition with a file system hipFile supports.

.. code:: shell

   sudo mkfs.ext4 /dev/nvme1n1p1

Create a mount point and mount the partition with ``data=ordered``.

.. code:: shell

   sudo mkdir /mnt/ext4
   sudo mount /dev/nvme1n1p1 /mnt/ext4 -o data=ordered

Mount with ``data=ordered`` so hipFile's fastpath accepts the ext4 file system.

Alternatively, add an entry into ``/etc/fstab`` to automatically mount the
partition at boot time. It is highly recommended to reference the partition
by either UUID or PARTUUID as the ``/dev/nvme#n#p#`` scheme is not guaranteed to
be consistent across reboots.

Build or install ``aiscp`` before running the verification. See
:doc:`../tutorials/copy-a-file` for build steps. Create a user-accessible
directory, then verify that hipFile can access the mounted path.

.. code:: shell

   sudo mkdir /mnt/ext4/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/ext4/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/ext4/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest
   md5sum /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest

Unmount the drive
=================

Unmount when finished, if applicable.

.. code:: shell

   sudo umount /mnt/ext4
