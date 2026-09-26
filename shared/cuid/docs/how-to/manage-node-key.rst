.. meta::
  :description: How to set the CUID node key and read CUIDs with amd-smi or the CUID library.
  :keywords: CUID, node key, AmdCuidKey, amd-smi, libamdcuid

.. _manage-node-key:

*****************************
Managing the node key
*****************************

The CUID library is a static library, ``libamdcuid_static.a``. It has no
command-line tool of its own. Administrators use amd-smi, and programs call the
library's API directly.

The node key
============

A derived CUID for a component the driver does not answer for (CPU, NIC, NPU,
Platform) is an HMAC under one node-wide 256-bit key. Root reads the key once
per enumeration or device lookup from:

1. ``cuid_seed`` (0600, ``CAP_SYS_ADMIN``) on any amdgpu device.
2. Failing that, the ``AmdCuidKey`` efivarfs variable
   (``/sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d``):
   a 4-octet attribute header, then version 1, flags (bit 0 = set by an
   administrator), two reserved zero octets and the 32-octet key. A variable
   of any other shape is malformed and ignored, never repaired automatically.

Without root, or without either source, such a component's CUID is a
temporary CUID (see :ref:`temporary-cuid`). GPUs and partitions are
unaffected: their identity comes from the driver, or, for a whole GPU the driver does not
answer for, the same temporary construction regardless of the node key.

.. note::

  Temporary CUIDs are keyed with the machine-id, so they are node-local, and a
  component with no serial number on a host with no machine-id gets an error
  rather than a CUID. See :ref:`cuid-machine-id` for the effect on containers.

Setting the key
===============

amdgpu creates a random key on its first load. To replace it, for example with
one key shared across a fleet, use amd-smi as root:

.. code-block:: shell

   sudo amd-smi set --cuid-seed /path/to/fleet-key.bin
   # or a fresh random key:
   head -c 32 /dev/urandom | sudo amd-smi set --cuid-seed -

A program does the same with ``amdcuid_set_hash_key()``. amd-smi itself runs
only when one of its drivers (amdgpu, amd_hsmp, ionic, bnxt_en) is loaded; the
library call has no such requirement. Either way:

* the key must be exactly 32 bytes; a key whose bytes are all equal, or that
  equals a published constant zero-padded to 32 bytes, is refused;
* with amdgpu loaded, the key is written to ``cuid_seed``, and the driver
  stores it in ``AmdCuidKey`` before using it; without amdgpu, it is written
  to ``AmdCuidKey`` through efivarfs;
* the variable is marked as set by an administrator, so ``cuid_seed_state``
  reads ``provisioned``;
* every derived CUID on the host changes; primary CUIDs do not.

efivarfs shows the variable as mode 0644. The installed
``tmpfiles.d/amdcuid.conf`` makes it 0600 at every boot, and setting the key
also makes it 0600 where efivarfs already lists it. A variable amdgpu created
during the current boot is not listed until the next one.

Reading CUIDs
=============

``sudo amd-smi node --cuid`` lists every component on the node (platform, CPU
packages, GPUs and GPU partitions, NICs) with its CUID, source (``DRIVER`` or
``LIBRARY``), whether it is temporary, its BDF and sysfs path, and the node
key's state; ``--cuid-primary`` adds the primary CUIDs. ``amd-smi list`` and
``amd-smi static --cuid`` report the same for each GPU amd-smi manages. A
program calls ``amdcuid_get_all_handles()`` and ``amdcuid_query_device_property()``,
or amd-smi's ``amdsmi_get_cuid_components()``;
the `sample program <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_
shows both.
