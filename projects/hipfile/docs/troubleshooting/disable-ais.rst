.. meta::
   :description: Disable the AIS module from the AMD GPU driver.
   :keywords: hipFile, AIS, disable, AMDGPU

*************
Disabling AIS
*************

When the AMD GPU driver initializes AIS, it registers the GPU's VRAM with the
P2PDMA kernel subsystem. This generates about 1GB of metadata for every 64GB
of VRAM. This metadata is stored in the host's DRAM.

If you will not use AIS, you can reclaim this memory by disabling AIS from
being initialized by the GPU driver.

Persist across reboots
=======================

Setting ``ais_disabled`` with ``modprobe`` only takes effect until the next
reboot. To disable AIS persistently, add the module option to a
``modprobe.d`` configuration file, then rebuild the initramfs so the setting
takes effect the next time ``amdgpu`` is loaded during boot:

.. code:: shell

   sudo bash -c 'echo "options amdgpu ais_disabled=1" > /etc/modprobe.d/amdgpu-ais.conf'
   sudo update-initramfs -c -k all
   sudo systemctl reboot

To re-enable AIS persistently, remove the configuration file and rebuild the
initramfs again:

.. code:: shell

   sudo rm /etc/modprobe.d/amdgpu-ais.conf
   sudo update-initramfs -c -k all
   sudo systemctl reboot

Disable AIS until the next reboot
==================================

Unload the ``amdgpu`` driver:

.. code:: shell

   sudo modprobe -r amdgpu

Reload the ``amdgpu`` driver with ``ais_disabled=1`` set:

.. code:: shell

   sudo modprobe amdgpu ais_disabled=1

Re-enable AIS
=============

Unload the ``amdgpu`` driver:

.. code:: shell

   sudo modprobe -r amdgpu

Reload the ``amdgpu`` driver with ``ais_disabled=0`` set:

.. code:: shell

   sudo modprobe amdgpu ais_disabled=0
