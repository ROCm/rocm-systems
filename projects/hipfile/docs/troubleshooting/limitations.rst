.. meta::
   :description: Limitations of hipFile's direct GPU-to-storage I/O path.
   :keywords: hipFile, limitations, NVMe, direct storage, GPU I/O, compat mode

**********************************
Limitations
**********************************

hipFile requires a direct path from the GPU to the storage device. The
filesystem must sit directly on the device's partition. Any interposing block
layer between the filesystem and the device breaks the direct path and forces a
fallback to compatibility (POSIX) mode. This includes, but is not limited to:

- LVM / device-mapper (dm)
- multipath
- dm-crypt (encrypted volumes)
- MD software RAID
- loopback devices
