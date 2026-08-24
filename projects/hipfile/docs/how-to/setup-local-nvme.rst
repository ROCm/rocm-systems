.. meta::
   :description: Set up a local NVMe drive for use with hipFile.
   :keywords: hipFile, NVMe, install, ROCm, direct storage, GPU I/O

.. _hipfile-setup-local-nvme:

*******************************************************
Setting up a local NVMe drive for use with hipFile
*******************************************************

To configure a local NVMe drive for use with hipFile, it must be partitioned with a filesystem hipFile supports. ``amdgpu-dkms`` version 31.40 or newer is required to use an NVMe drive.

.. note:: 
 
   hipFile does not support NVMe multipath devices with the fastpath backend.

   To disable NVMe multipath on Ubuntu 24.04:

   .. code-block:: bash

      sudo bash -c 'echo "options nvme_core multipath=N" > /etc/modprobe.d/nvme_core.conf'
      sudo update-initramfs -c -k all
      sudo systemctl reboot

Use ``lsblk`` to see the list of drives:

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Choose a drive without an OS installation. Partition it using ``sgdisk`` or any other partitioning tool.

.. warning::

   Partitioning the drive will permanently erase all the data stored on it.

Create a single partition on the drive. For example, to partition the ``/dev/nvme1n1`` drive:

.. code:: shell

   sudo sgdisk -n 1:0:0 /dev/nvme1n1

Format the partition with a file system supported by hipFile:

.. note::
   The filesystem must sit directly on the drive's partition, as shown. For
   block layers that break hipFile's direct path, see :doc:`../troubleshooting/limitations`.

.. code:: shell

   sudo mkfs.ext4 /dev/nvme1n1p1

Create a mount point and mount the partition with ``data=ordered``:

.. code:: shell

   sudo mkdir /mnt/ext4
   sudo mount /dev/nvme1n1p1 /mnt/ext4 -o data=ordered

The mount point can also be added to ``/etc/fstab`` to automatically mount the partition during the boot process.

Verify that hipFile has access to the mounted path by copying a file using 
`aiscp.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_:

.. code:: shell

   sudo mkdir /mnt/ext4/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/ext4/"${USER}"
   # Create a file
   dd if=/dev/urandom of=/mnt/ext4/"${USER}"/source bs=4K count=16
   # Copy the file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest
   md5sum /mnt/ext4/"${USER}"/source /mnt/ext4/"${USER}"/dest


Use the Linux ``umount`` command to unmount the partition when it's no longer needed:

.. code:: shell

   sudo umount /mnt/ext4