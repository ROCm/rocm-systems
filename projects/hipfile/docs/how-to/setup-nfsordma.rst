.. meta::
   :description: Set up an NFSoRDMA share for use with hipFile.
   :keywords: hipFile, NFSoRDMA, NFS, RDMA, install, ROCm, direct storage, GPU I/O

***************************************************
Setting up an NFSoRDMA share for use with hipFile
***************************************************

The following prerequisites are required to use an RDMA-capable NFS server over RDMA or RoCE with hipFile:

* ``amdgpu-dkms`` version 31.40 or later.
* An RDMA or RoCE fabric connecting the NFS server and client.
* ``nfs-kernel-server`` installed on the server, with the ``svcrdma`` kernel
  module available.
- ``nfs-common`` installed on the client, with the ``xprtrdma`` kernel module
  available.

Configure the NFS server
==========================

Configure a static IP address on the RDMA or RoCE interface. 

.. note::

   The example commands use ``netplan``. Use the appropriate distribution network configuration tools.

Create ``/etc/netplan/config.yaml``, replacing ``<rdma_interface>`` and ``<server_ip>/<prefix>`` with values that match the fabric, then run ``netplan apply``:

.. code-block:: yaml

   network:
     version: 2
     renderer: networkd
     ethernets:
       <rdma_interface>:
         dhcp4: false
         dhcp6: false
         accept-ra: false
         addresses:
           - <server_ip>/<prefix>

.. code:: shell

   sudo netplan apply

Load the RDMA-capable NFS server module:

.. code:: shell

   sudo modprobe svcrdma

Add the directories to export to ``/etc/exports``, where ``EXPORT_PATH`` is the path to each directory to export:

.. code-block:: none

   EXPORT_PATH *(rw,async,insecure,no_root_squash)

Reload the NFS server to pick up the new exported files:

.. code:: shell

   sudo systemctl reload nfs-server

Add an RDMA port for NFS to listen on:

.. code:: shell

   echo rdma 20049 | sudo tee /proc/fs/nfsd/portlist

Confirm that the RDMA port is active:

.. code-block:: none

   $ cat /proc/fs/nfsd/portlist
   rdma 20049
   tcp 2049

Configure the client
======================

Load the RDMA-capable NFS client module:

.. code:: shell

   sudo modprobe xprtrdma

Create a mount point and mount the share using RDMA:

.. code:: shell

   sudo mkdir /mnt/nfs
   sudo mount -o rdma,port=20049 <server_ip>:<export_path> /mnt/nfs

Set ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true`` before running hipFile I/O. 

Verify that hipFile has access to the mounted share by copying a file using 
`aiscp.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_:

.. code:: shell

   sudo mkdir /mnt/nfs/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/nfs/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/nfs/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true ./aiscp /mnt/nfs/"${USER}"/source /mnt/nfs/"${USER}"/dest
   md5sum /mnt/nfs/"${USER}"/source /mnt/nfs/"${USER}"/dest

Use the Linux ``umount`` command to unmount the share when it's no longer needed:

.. code:: shell

   sudo umount /mnt/nfs
