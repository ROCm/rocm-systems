.. meta::
   :description: Limitations of hipFile's direct GPU-to-storage I/O path.
   :keywords: hipFile, limitations, NVMe, direct storage, GPU I/O, compat mode

**********************************
Limitations
**********************************

hipFile requires a direct path from the GPU to the storage device. The
filesystem must be backed by a local NVMe device that either resides on the
device's partition or is stacked on it through LVM or MD software RAID. Any
other interposing block layer between the filesystem and the device breaks the
direct path and forces a fallback to compatibility mode. This includes, but is
not limited to:

- multipath
- dm-crypt (encrypted volumes)
- parity RAID (RAID 4, RAID 5, RAID 6), whether MD or LVM RAID
- loopback devices

LVM logical volumes
===================

LVM (Logical Volume Manager) volumes are supported for the fastpath when their
underlying physical volumes are all local NVMe devices. A volume whose physical
volumes include any non-NVMe or multipath device falls back to compatibility
mode.

This includes RAID logical volumes, with the same restriction that applies to
MD arrays below. LVM implements a RAID LV with the ``dm-raid`` target, which
runs the MD personality code, so a RAID LV qualifies for the fastpath only when
it uses a pass-through RAID level, ``raid0``, ``raid1``, or ``raid10``, and
every member resides on a local NVMe physical volume. A parity RAID LV
(``raid4``, ``raid5``, ``raid6``) falls back to compatibility mode, as does a
volume with any member on a non-NVMe physical volume.

Check the level of a RAID LV with::

   lvs -o lv_name,segtype vg

``ais-check`` reads the same information from the device-mapper table, which
needs the privileges ``dmsetup`` requires. Without them it reports a RAID LV as
``unverified`` rather than guessing; plain linear and striped volumes are
unaffected.

MD software RAID
================

MD software RAID arrays (``mdadm``, ``/dev/md*``) are supported for the
fastpath when both of the following hold:

- the array runs one of the pass-through personalities, ``linear``, ``raid0``,
  ``raid1``, or ``raid10``, and
- every member device of the array is a local NVMe device.

Those personalities remap the original request onto a member device without
interposing a buffer, so the direct path survives. An array with any non-NVMe
or multipath member falls back to compatibility mode. Because reads can be
served from any member of a mirrored or striped array, a single unsupported
member disqualifies the whole array.

The parity personalities, ``raid4``, ``raid5``, and ``raid6``, are **not**
supported and always fall back to compatibility mode, even when every member is
a local NVMe device. They route the transfer through the MD stripe cache to
compute parity, copying it into host memory instead of handing the original
request to a member device, so there is no direct path left to use. This is the
same code an LVM parity RAID LV runs, which is why both are excluded. The same
applies to any other personality, such as ``faulty`` or ``multipath``.

Check the personality of an array with::

   cat /sys/block/md0/md/level

Stacking LVM and MD RAID
========================

LVM and MD RAID can be combined in either order, to any depth, as long as every
RAID layer uses a pass-through level and every device at the bottom of the
stack is a local NVMe. An LVM volume group built on MD arrays, or an MD array
assembled from logical volumes, both keep the direct path. Inserting any other
layer anywhere in the stack, such as multipath, dm-crypt, or a parity array,
forces the fallback to compatibility mode.

Run ``ais-check`` to see how a given mount is classified. For more information,
see :doc:`/how-to/checking-system-compatibility`.

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
