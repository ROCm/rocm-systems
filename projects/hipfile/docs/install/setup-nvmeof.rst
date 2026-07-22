.. meta::
   :description: Set up an NVMeoF disk for use with hipFile.
   :keywords: hipFile, NVMeoF, install, ROCm, direct storage, GPU I/O

**********************************
Set up an NVMeoF disk
**********************************

To use hipFile with storage exported over NVMeoF, configure the NVMe target
on the server, connect from the initiator, mount the block device, and verify
I/O on the mounted path. Before you begin, make sure hipFile is installed. See
:doc:`./install` for package options or :doc:`./build-from-source` to build
from source.

Prerequisites
=============

- Version 31.40 or newer of the ``amdgpu-dkms`` driver, with ROCm 7.4 or later
  and the HIP runtime
- ``nvme-cli`` installed on the initiator, or client, system
- Network or fabric connectivity to an NVMeoF target over RDMA or TCP
- The ``nvmet`` and ``nvmet-rdma``, or ``nvmet-tcp``, kernel modules loaded on
  the target
- The ``nvme-rdma``, or ``nvme-tcp``, kernel module loaded on the initiator
- Root or ``sudo`` access to discover, connect, and mount the device
- A partition on the target with a file system hipFile supports already set up.
  See :doc:`./setup-local-nvme` for an example of partitioning and formatting
  a drive.

Set up the target and initiator
===============================

On the target, configure the NVMe target subsystem, ``nvmet``, through
``configfs``. The commands below use RDMA over IPv4. Substitute transport
values that match your fabric. Run target-side commands on the target system,
not the initiator.

Create a subsystem directory and allow any host to connect. Replace
``<subsystem_name>`` with a name of your choosing. It becomes part of the
subsystem's NQN when the initiator discovers it.

.. code:: shell

   cd /sys/kernel/config/nvmet/subsystems
   sudo mkdir <subsystem_name>
   cd <subsystem_name>
   echo 1 | sudo tee ./attr_allow_any_host

.. note::

   Restricting ``attr_allow_any_host`` to specific host NQNs is more secure
   for production deployments but is out of scope here.

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

Create a port and configure its transport address.

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
   time, including the host the drive resides on. Doing so makes file system
   corruption a near certainty. Consider enabling Multiple Mount Protection on
   file systems that support it to help prevent accidental corruption.

On the initiator, discover the subsystems exposed by the target.

.. code:: shell

   sudo nvme discover -t rdma -a <target_ip> -s <target_port>

Connect to the discovered subsystem using its NQN.

.. code:: shell

   sudo nvme connect -t rdma -a <target_ip> -s <target_port> -n <target_nqn>

Confirm the new NVMe device appears on the initiator.

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Create a mount point and mount the device with ``data=ordered``. The example
below assumes the target partition uses ext4.

.. code:: shell

   sudo mkdir /mnt/nvmeof
   sudo mount /dev/nvme<N>n1 /mnt/nvmeof -o data=ordered

Mount with ``data=ordered`` so hipFile's fastpath accepts the ext4 file system.

Build or install ``aiscp`` before running the verification. See
:doc:`../tutorials/copy-a-file` for build steps. Create a user-accessible
directory, then verify that hipFile can access the mounted path.

.. code:: shell

   sudo mkdir /mnt/nvmeof/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/nvmeof/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/nvmeof/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest
   md5sum /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest

Disconnect from the target
==========================

Disconnect when finished, if applicable.

.. code:: shell

   sudo nvme disconnect -n <target_nqn>
