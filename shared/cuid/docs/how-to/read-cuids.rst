.. meta::
  :description: How to read CUIDs with amd-smi or the CUID library.
  :keywords: CUID, amd-smi, libamdcuid

.. _read-cuids:

*****************************
Reading CUIDs
*****************************

The CUID library is a static library, ``libamdcuid_static.a``. It has no
command-line tool of its own. Administrators use amd-smi, and programs call the
library's API directly.

Where CUIDs come from
=====================

This library holds no node key and reads no CUID from a driver, so every CUID
it returns is a temporary CUID (see :ref:`temporary-cuid`), the same for root
and every other caller. Every whole GPU, GPU partition, NPU, NIC function, CPU
package and the platform has one. A partition's UnitID comes from amdgpu's
``cuid_unit_id`` or, on a kernel without it, from the KFD topology (see
:ref:`cuid-gpu-partitions`). amd-smi reports the whole GPU's CUID for a GPU in
SPX, and the partition's for each partition in DPX and above.

.. note::

  Temporary CUIDs are keyed with the machine-id, so they are node-local. On a
  host with no machine-id every component gets an error rather than a CUID.
  See :ref:`cuid-machine-id` for the effect on containers.

Reading CUIDs
=============

``amd-smi node --cuid`` lists every component on the node (platform, CPU
packages, GPUs and their partitions, NPUs, NICs) with its CUID, source,
whether it is temporary, and its BDF and sysfs path. ``amd-smi list`` and ``amd-smi static --cuid`` report the
same for each GPU amd-smi manages. A program calls amd-smi's
``amdsmi_get_cuid_components()``, or libamdcuid's ``amdcuid_get_all_handles()``
and ``amdcuid_query_device_property()``, which the
`sample program <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_
shows.
