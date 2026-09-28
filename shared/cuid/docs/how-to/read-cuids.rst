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

A GPU or compute partition is published by the driver when amdgpu exposes
``cuid_unit_id`` (0444) for it. Its primary CUID is then the driver's
``cuid_primary`` (0400, ``CAP_SYS_ADMIN``), which only root can read.

A derived CUID that is not temporary needs a node-wide 256-bit key that other
local users cannot read. This library holds no node key, so every derived CUID
it returns is a temporary CUID (see :ref:`temporary-cuid`), for root and every
other caller. A compute partition has no temporary CUID, so the library returns
no handle for it.

.. note::

  Temporary CUIDs are keyed with the machine-id, so they are node-local, and a
  component with no serial number on a host with no machine-id gets an error
  rather than a CUID. See :ref:`cuid-machine-id` for the effect on containers.

Reading CUIDs
=============

``sudo amd-smi node --cuid`` lists every component on the node (platform, CPU
packages, GPUs, NICs) with its CUID, source (``DRIVER`` or ``LIBRARY``),
whether it is temporary, and its BDF and sysfs path; ``--cuid-primary`` adds
the primary CUIDs. ``amd-smi list`` and ``amd-smi static --cuid`` report the
same for each GPU amd-smi manages. A program calls
``amdcuid_get_all_handles()`` and ``amdcuid_query_device_property()``, or
amd-smi's ``amdsmi_get_cuid_components()``;
the `sample program <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_
shows both.
