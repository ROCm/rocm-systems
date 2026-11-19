.. meta::
  :description: How to set the CUID node key and read CUIDs with amd-smi or the CUID library.
  :keywords: CUID, node key, cuid_seed, amd-smi, libamdcuid

.. _manage-node-key:

*****************************
Managing the node key
*****************************

The CUID library is a static library, ``libamdcuid_static.a``. It has no
command-line tool of its own. Administrators use amd-smi, and programs call the
library's API directly.

The node key
============

A derived CUID is an HMAC under one node-wide 256-bit key. amdgpu holds the key
in memory for all its devices. It never generates a key and never stores one:
it starts without a key at every module load, keeps the key across unbind and
rebind, and wipes it when the module is unloaded. A reboot or an amdgpu reload
therefore starts with no key.

Root reads the key from ``cuid_seed`` (0600, ``CAP_SYS_ADMIN``) on any amdgpu
device on every enumeration, device lookup and property query, and the library
zeroes it before that call returns, so it keeps no copy between calls. While
amdgpu holds no key the read fails with ``ENODATA``. There is no other key
source.

AMD GPUs and their partitions, AMD NPUs, network-class PCI functions of any
vendor, AMD and Intel CPU packages, and the platform have a CUID (see
:ref:`temporary-cuid` for the exact rules). Without a key, or without root, a
CPU, NIC, NPU or Platform CUID is a temporary CUID. A GPU or partition takes its
derived CUID from the driver's ``cuid_derived``, which fails with ``ENODATA``
until a key is set. Until then whole GPUs and partitions get temporary CUIDs;
a partition's UnitID comes from ``cuid_unit_id`` or the KFD topology (see
:ref:`cuid-gpu-partitions`).

.. note::

  Temporary CUIDs are keyed with the machine-id, so they are node-local, and on
  a host with no machine-id every component that would get one gets an error
  instead. Without a key that is every component; with a key it is a CPU, NIC,
  NPU or Platform read without root, and any component with no serial number.
  See :ref:`cuid-machine-id` for the effect on containers.

Setting the key
===============

Set the key as root after amdgpu loads, at every boot:

.. code-block:: shell

   sudo amd-smi set --cuid-seed /path/to/fleet-key.bin
   # or a fresh random key:
   head -c 32 /dev/urandom | sudo amd-smi set --cuid-seed -

A program does the same with ``amdcuid_set_hash_key()``. Either way:

* the key must be exactly 32 bytes; a key whose bytes are all equal, or that
  equals a published constant zero-padded to 32 bytes, is refused;
* the key is written to one amdgpu device's ``cuid_seed``, which keys every
  GPU and partition; without amdgpu the call fails with
  ``AMDCUID_STATUS_UNSUPPORTED`` and nothing changes;
* every derived CUID on the host changes; primary and temporary CUIDs do not.

The key passes through user space at every boot, so where it is kept between
boots is the administrator's choice. Keep it readable only by root: anyone
holding it can confirm a guessed serial number, and a fleet key read on one
host exposes every host that shares it. Hosts that should report the same
derived CUIDs need the same key.

The node key is not persisted; set it again after a reboot or after amdgpu reloads.

Reading CUIDs
=============

* ``sudo amd-smi node --cuid`` lists every component on the node (platform,
  CPU packages, GPUs and GPU partitions, NPUs, NIC functions) with its CUID, source
  (``DRIVER`` or ``LIBRARY``), whether it is temporary, its BDF and sysfs path,
  and the node key's state. A GPU and each of its partitions are separate
  entries; a partition's BDF is the address amd-smi uses for it, as
  ``amd-smi list`` shows it. amd-smi prints no primary CUID.
* ``amd-smi list`` and ``amd-smi static --cuid`` report each managed GPU's
  CUID, source and whether it is temporary, without the node key's state.
* A program calls amd-smi's ``amdsmi_get_cuid_components()``, or libamdcuid's
  ``amdcuid_get_all_handles()`` and ``amdcuid_query_device_property()``, which
  the `sample program <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_
  shows.

amd-smi's C, Python and Rust APIs keep the names of their first release: the
``derived`` field holds the component's CUID, and ``auxiliary`` is the
temporary marker. They are the CLI's ``cuid`` and ``temporary``. libamdcuid's
``AMDCUID_QUERY_DERIVED_CUID`` likewise returns the component's CUID: its
DerivedID when it has a permanent CUID, otherwise its temporary CUID.
