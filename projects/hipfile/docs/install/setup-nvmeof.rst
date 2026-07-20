.. meta::
   :description: Set up an NVMeoF disk for use with hipFile.
   :keywords: hipFile, NVMeoF, install, ROCm, direct storage, GPU I/O

**********************************
Set up an NVMeoF disk
**********************************

This page walks through a simple example of connecting to an NVMeoF target
and preparing it for use with hipFile. Before you begin, make sure hipFile is
installed. See :doc:`./install` for package options or :doc:`./build-from-source`
to build from source.

Prerequisites
=============

- Version 31.40 or newer of the ``amdgpu-dkms`` driver
- ``nvme-cli`` installed on the initiator (client) system
- Network or fabric connectivity to an NVMeoF target. This guide's examples
  use RDMA; TCP is also supported.
- The ``nvmet`` and ``nvmet-rdma`` (or ``nvmet-tcp``) kernel modules loaded on
  the target
- The ``nvme-rdma`` (or ``nvme-tcp``) kernel module loaded on the initiator
- Root or ``sudo`` access to discover, connect, and mount the device
- A partition on the target with a filesystem hipFile supports already set up.
  See :doc:`./setup-local-nvme` for an example of partitioning and formatting
  a drive.

Create the NVMeoF target
==========================

This section configures the target (server) side using the Linux kernel's
NVMe target subsystem (``nvmet``) through ``configfs``. Run these commands on
the target system, not the initiator.

Create a subsystem directory and allow any host to connect. Replace
``<subsystem_name>`` with a name of your choosing; it becomes part of the
subsystem's NQN when the initiator discovers it.

.. code:: shell

   cd /sys/kernel/config/nvmet/subsystems
   sudo mkdir <subsystem_name>
   cd <subsystem_name>
   echo 1 | sudo tee ./attr_allow_any_host

.. note::

   Restricting ``attr_allow_any_host`` to specific host NQNs is more secure
   for production deployments but is out of scope for this guide.

Create a namespace and point it at the partition you prepared in the
prerequisites. Prefer a stable identifier such as a PCI BDF or drive ID over
a ``/dev/nvme<N>n<N>`` path if one is available. Use ``lsblk`` and ``blkid``
to identify the correct block device.

.. code:: shell

   cd namespaces
   sudo mkdir 1
   cd 1
   echo -n /dev/disk/by-path/pci-0000:01:00.0-nvme-1 | sudo tee ./device_path
   echo 1 | sudo tee ./enable

Create a port and configure its transport address. This example uses RDMA
over IPv4; substitute values that match your fabric.

.. code:: shell

   cd /sys/kernel/config/nvmet/ports
   sudo mkdir 1
   cd 1
   echo <target_ip> | sudo tee ./addr_traddr
   echo ipv4 | sudo tee ./addr_adrfam
   echo rdma | sudo tee ./addr_trtype
   echo <target_port> | sudo tee ./addr_trsvcid

Link the subsystem to the port to expose it to initiators.

.. code:: shell

   sudo ln -s /sys/kernel/config/nvmet/subsystems/<subsystem_name> \
     /sys/kernel/config/nvmet/ports/1/subsystems/<subsystem_name>

Confirm that the kernel has accepted the configuration and exposed the target
to the network. The kernel logs this as a syslog message.

.. code-block:: none

   $ sudo journalctl -k | grep nvmet
   nvmet_rdma: enabling port 1 (<target_ip>:<target_port>)

.. warning::

   ext4 and xfs cannot safely be mounted by more than one host at the same
   time, including the host the drive resides on; doing so makes filesystem
   corruption a near certainty. Consider enabling MMP
   (Multiple Mount Protection) on filesystems that support it to help prevent
   accidental corruption.

Discover the NVMeoF target
============================

Discover the subsystems exposed by the target.

.. code:: shell

   sudo nvme discover -t rdma -a <target_ip> -s <target_port>

Connect to the target
=======================

Connect to the discovered subsystem using its NQN.

.. code:: shell

   sudo nvme connect -t rdma -a <target_ip> -s <target_port> -n <target_nqn>

Confirm the new NVMe device appears on the initiator.

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Mount the device
=================

Create a mount point and mount the device. This example assumes the target
partition uses ext4.

.. code:: shell

   sudo mkdir /mnt/nvmeof
   sudo mount /dev/nvme<N>n1 /mnt/nvmeof -o data=ordered

Verify the setup
=================

Create a user-accessible directory and then confirm if hipFile can see and use
this mounted drive. For example, running one of the hipFile example programs.

.. code:: shell

   sudo mkdir /mnt/nvmeof/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/nvmeof/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/nvmeof/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest
   md5sum /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest

Disconnect from the target
============================

Disconnect when finished, if applicable.

.. code:: shell

   sudo nvme disconnect -n <target_nqn>
