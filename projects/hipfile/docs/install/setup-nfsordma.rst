.. meta::
   :description: Set up an NFSoRDMA share for use with hipFile.
   :keywords: hipFile, NFSoRDMA, NFS, RDMA, install, ROCm, direct storage, GPU I/O

**********************************
Set up an NFSoRDMA share
**********************************

To use hipFile with an NFS share mounted over RDMA, configure an RDMA-capable
NFS server, mount the share on the client over InfiniBand or RoCE, and verify
I/O on the mounted path. Before you begin, make sure hipFile is installed. See
:doc:`./install` for package options or :doc:`./build-from-source` to build
from source.

Prerequisites
=============

- Version 31.40 or newer of the ``amdgpu-dkms`` driver, with ROCm 7.4 or later
  and the HIP runtime
- An InfiniBand or RoCE fabric connecting the NFS server and client
- ``nfs-kernel-server`` installed on the server, with the ``svcrdma`` kernel
  module available
- ``nfs-common`` installed on the client, with the ``xprtrdma`` kernel module
  available
- Root or ``sudo`` access on both the server and client

Set up the server and client
============================

Configure a static IP address on the InfiniBand or RoCE interface on the
server. This example uses ``netplan``. Substitute your distribution's network
configuration tool as needed. Create ``/etc/netplan/<config>.yaml``, replacing
``<ib_interface>`` and ``<server_ip>/<prefix>`` with values that match your
fabric:

.. code-block:: yaml

   network:
     version: 2
     renderer: networkd
     ethernets:
       <ib_interface>:
         dhcp4: false
         dhcp6: false
         accept-ra: false
         addresses:
           - <server_ip>/<prefix>

.. code:: shell

   sudo netplan apply

Load the RDMA-capable NFS server module.

.. code:: shell

   sudo modprobe svcrdma

Add the directories you want to export to ``/etc/exports``.

.. code-block:: none

   <export_path> *(rw,async,insecure,no_root_squash)

Reload the NFS server so it picks up the new exports.

.. code:: shell

   sudo systemctl reload nfs-server

Add an RDMA port for NFS to listen on.

.. code:: shell

   echo rdma 20049 | sudo tee /proc/fs/nfsd/portlist

Confirm the RDMA port is active.

.. code-block:: none

   $ cat /proc/fs/nfsd/portlist
   rdma 20049
   tcp 2049

On the client, load the RDMA-capable NFS client module.

.. code:: shell

   sudo modprobe xprtrdma

Create a mount point and mount the share using RDMA.

.. code:: shell

   sudo mkdir /mnt/nfs
   sudo mount -o rdma,port=20049 <server_ip>:<export_path> /mnt/nfs

Set ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true`` before hipFile I/O on the mounted
share. The client sees an NFS file system, not ext4 or XFS. Zero-copy over the
full path depends on the NFS server's RDMA support for its storage backend.

Build or install ``aiscp`` before running the verification. See
:doc:`../tutorials/copy-a-file` for build steps. Create a user-accessible
directory, then verify that hipFile can access the mounted share.

.. code:: shell

   sudo mkdir /mnt/nfs/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/nfs/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/nfs/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true ./aiscp /mnt/nfs/"${USER}"/source /mnt/nfs/"${USER}"/dest
   md5sum /mnt/nfs/"${USER}"/source /mnt/nfs/"${USER}"/dest

Unmount the share
=================

Unmount when finished, if applicable.

.. code:: shell

   sudo umount /mnt/nfs
