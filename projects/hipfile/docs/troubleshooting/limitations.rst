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

This includes RAID logical volumes. LVM builds a RAID LV out of sub-volumes
that are themselves LVM volumes, so a RAID LV whose every member resides on a
local NVMe physical volume keeps the direct path and qualifies for the
fastpath. If any member resides on a non-NVMe physical volume, the volume falls
back to compatibility mode.

This applies to LVM's own RAID implementation only. MD software RAID
is a separate block layer and is not supported, even when every member device
is a local NVMe.

File descriptor limits
======================

hipFile opens file descriptors of its own, which count toward the process's
open file limit (``RLIMIT_NOFILE``):

- ``hipFileHandleRegister()`` opens a second file descriptor for each
  registered file, one with ``O_DIRECT`` and one without. It is closed by
  ``hipFileHandleDeregister()``. For more information, see
  :doc:`/reference/hipFile-file-registration`.
- hipFile also opens file descriptors internally, some for the lifetime of
  the process and some only temporarily.

If the process reaches its open file limit, or the system-wide limit is
reached, hipFile API calls that need a new file descriptor, such as
``hipFileHandleRegister()``, return ``hipFileGetNewFDFailed``.

To register a large number of files, allow about two file descriptors per
registered file plus the descriptors the application uses itself. Check the
current limits with:

.. code-block:: none

  $ ulimit -Sn
  $ ulimit -Hn

The soft limit can be raised up to the hard limit, for example with
``ulimit -n <limit>`` before starting the application.
