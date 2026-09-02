.. meta::
   :description: Limitations of hipFile's direct GPU-to-storage I/O path.
   :keywords: hipFile, limitations, NVMe, direct storage, GPU I/O, compat mode

**********************************
Limitations
**********************************

hipFile requires a direct path from the GPU to the storage device. The
filesystem must sit directly on the device's partition. With the exception of
LVM (see below), any interposing block layer between the filesystem and the
device breaks the direct path and forces a fallback to compatibility (POSIX)
mode. This includes, but is not limited to:

- multipath
- dm-crypt (encrypted volumes)
- MD software RAID
- loopback devices

LVM logical volumes
===================

LVM (Logical Volume Manager) volumes are supported for the fastpath when their
underlying physical volumes are all local NVMe devices. A volume whose physical
volumes include any non-NVMe or multipath device falls back to compatibility
mode.
