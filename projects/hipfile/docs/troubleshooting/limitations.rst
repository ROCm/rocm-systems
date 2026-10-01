.. meta::
   :description: Limitations of hipFile's direct GPU-to-storage I/O path.
   :keywords: hipFile, limitations, NVMe, direct storage, GPU I/O, compat mode

**********************************
Limitations
**********************************

hipFile requires a direct path from the GPU to the storage device. The
filesystem must be backed by a local NVMe device that either resides on the
device's partition or is stacked on it through LVM.
Any other interposing block layer between the filesystem and the device breaks
the direct path and forces a fallback to compatibility mode. This
includes, but is not limited to:

- multipath
- dm-crypt (encrypted volumes)
- MD software RAID (``mdadm``, ``/dev/md*``)
- loopback devices

LVM logical volumes
===================

LVM (Logical Volume Manager) volumes are supported for the fastpath when their
underlying physical volumes are all local NVMe devices. A volume whose physical
volumes include any non-NVMe or multipath device falls back to compatibility
mode.

This includes mirrored logical volumes created with ``lvcreate --type raid1``.
LVM implements a mirror as a set of sub-volumes that are themselves LVM
volumes, so a mirror whose every leg resides on a local NVMe physical volume
keeps the direct path and qualifies for the fastpath. If any leg resides on a
non-NVMe physical volume, the volume falls back to compatibility mode.

Note that this applies to LVM's own RAID1 implementation only. MD software RAID
is a separate block layer and is not supported, even when every member device
is a local NVMe.
