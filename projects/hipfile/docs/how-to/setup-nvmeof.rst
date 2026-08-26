.. meta::
   :description: Set up an NVMe-oF disk for use with hipFile.
   :keywords: hipFile, NVMe-oF, install, ROCm, direct storage, GPU I/O

****************************************************
Setting up a local NVMe-oF disk for use with hipFile
****************************************************

To use hipFile with storage exported over NVMe-oF, first :ref:`set up a local NVMe drive <hipfile-setup-local-nvme>`. 

NVMe-oF also requires:

* Network or fabric connectivity over RDMA. hipFile's optimized I/O requires an RDMA transport.
* The ``nvmet`` and ``nvmet-rdma`` kernel modules loaded on the target.
- The ``nvme-rdma`` kernel module loaded on the initiator.

Configure the target 
=====================

Configure the NVMe target subsystem, ``nvmet``. 

.. note::

   The example commands use RDMA over IPv4. Use the appropriate transport values for the target being configured.

Create a subsystem directory and allow any host to connect. 

.. code:: shell

   cd /sys/kernel/config/nvmet/subsystems
   sudo mkdir SUBSYSTEM_NAME
   cd SUBSYSTEM_NAME
   echo 1 | sudo tee ./attr_allow_any_host

Where ``SUBSYSTEM_NAME`` name is the name given to the subsystem. This name will become part of the subsystem's NQN when the initiator discovers it.

.. note::
   
   Specific host NQNs can be specified for a more secure environment.

Create a namespace and point it at the NVMe partition. Use a stable identifier such as a PCI BDF or drive ID. Use ``lsblk`` and ``blkid``
to identify the block device. For example:

.. code:: shell

   cd namespaces
   sudo mkdir 1
   cd 1
   echo -n /dev/disk/by-path/pci-0000:01:00.0-nvme-1 | sudo tee ./device_path
   echo 1 | sudo tee ./enable

Create a port and configure its transport address:

.. code:: shell

   cd /sys/kernel/config/nvmet/ports
   sudo mkdir 1
   cd 1
   echo <target_ip> | sudo tee ./addr_traddr
   echo ipv4 | sudo tee ./addr_adrfam
   echo rdma | sudo tee ./addr_trtype
   echo <target_port> | sudo tee ./addr_trsvcid

Link the subsystem to the port to expose it to initiators:

.. code:: shell

   sudo ln -s /sys/kernel/config/nvmet/subsystems/<subsystem_name> \
     /sys/kernel/config/nvmet/ports/1/subsystems/<subsystem_name>

Confirm that the kernel has accepted the configuration and exposed the target
to the network. 

Check the kernel logs for the syslog message:

.. code-block:: none

   $ sudo journalctl -k | grep nvmet
   nvmet_rdma: enabling port 1 (<target_ip>:<target_port>)

.. warning::

   ext4 and xfs cannot safely be mounted by more than one host at the same time, including the host the drive resides on. Enabling Multiple Mount Protection helps prevent accidental corruption due to the devices being mounted by multiple hosts.


Configure the initiator
========================

Discover the subsystems exposed by the target:

.. code:: shell

   sudo nvme discover -t rdma -a <target_ip> -s <target_port>

Connect to the discovered subsystem using its NQN:

.. code:: shell

   sudo nvme connect -t rdma -a <target_ip> -s <target_port> -n <target_nqn>

Confirm the new NVMe device appears on the initiator:

.. code:: shell

   lsblk -d -o NAME,SIZE,MODEL | grep nvme

Create a mount point and mount the device with ``data=ordered``. Mounting with ``data=ordered`` ensures that the fastpath can be used by hipFile.  

For example, with ext4:

.. code:: shell

   sudo mkdir /mnt/nvmeof
   sudo mount /dev/nvme<N>n1 /mnt/nvmeof -o data=ordered

Verify that hipFile has access to the mounted path by copying a file using 
`aiscp.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_:

.. code:: shell

   sudo mkdir /mnt/nvmeof/"${USER}"
   sudo chown "${USER}":"${USER}" /mnt/nvmeof/"${USER}"
   # Create a random input file
   dd if=/dev/urandom of=/mnt/nvmeof/"${USER}"/source bs=4K count=16
   # Copy file
   HIPFILE_ALLOW_COMPAT_MODE=false ./aiscp /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest
   md5sum /mnt/nvmeof/"${USER}"/source /mnt/nvmeof/"${USER}"/dest


Disconnect from the target once the connection is no longer needed:

.. code:: shell

   sudo nvme disconnect -n <target_nqn>
