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
no handle for it. amd-smi reports the whole GPU's CUID for a GPU in SPX, whose
one partition covers every XCC, and none for a partition in DPX and above.

.. note::

  Temporary CUIDs are keyed with the machine-id, so they are node-local. Every
  derived CUID this library returns is temporary, so on a host with no
  machine-id every component gets an error rather than a CUID. See
  :ref:`cuid-machine-id` for the effect on containers.

Reading CUIDs
=============

``sudo amd-smi node --cuid`` lists every component on the node (platform, CPU
packages, GPUs, NPUs, NICs) with its CUID, source (``DRIVER`` or ``LIBRARY``),
whether it is temporary, and its BDF and sysfs path; ``--cuid-primary`` adds
the primary CUIDs. ``amd-smi list`` and ``amd-smi static --cuid`` report the
same for each GPU amd-smi manages. A program calls amd-smi's
``amdsmi_get_cuid_components()``, or libamdcuid's ``amdcuid_get_all_handles()``
and ``amdcuid_query_device_property()``, which the
`sample program <https://github.com/ROCm/rocm-systems/blob/develop/shared/cuid/example/main.cc>`_
shows.
