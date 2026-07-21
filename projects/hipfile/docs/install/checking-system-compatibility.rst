.. meta::
   :description: How to run the ais-check tool to verify that a system supports hipFile's fast path, and how to read its output.
   :keywords: hipFile, ais-check, compatibility, fast path, O_DIRECT, P2PDMA, ROCm, GPU I/O, AMD

*****************************
Checking system compatibility
*****************************

Before you run a hipFile workload, use the ``ais-check`` tool to confirm that
your system has the components hipFile needs and that at least one mounted
volume can use hipFile's fast (direct-to-GPU) path. If any required component is
missing, or no volume qualifies, hipFile silently falls back to a slower
compatibility path.

``ais-check`` ships with hipFile under ``tools/ais-check`` and is installed
alongside the other host tools when ``AIS_INSTALL_TOOLS`` is enabled. After
installation it lands in the ``bin`` directory beneath your ROCm prefix, for
example ``/opt/rocm/bin/ais-check`` or, on versioned core installs,
``/opt/rocm/core-<version>/bin/ais-check``. It is a self-contained Python 3 script
that reads kernel state, probes the HIP runtime, and scans mounted filesystems.
It makes no changes to your system.

Run ais-check
=============

.. code:: shell

   ais-check

For an accurate verdict, run it as root. The ``O_DIRECT`` and ``HIPFILE``
columns are probed by opening a temporary file with ``O_DIRECT`` on each volume,
which needs write access to the mountpoint. Without it, those columns show
``unverified`` instead of a yes/no result.

.. code:: shell

   sudo ais-check

Options:

- ``-v``, ``--verbose`` also lists the HIP runtime libraries that were
  discovered and whether each one exports the AIS symbols.
- ``-q``, ``--quiet`` silences the report and returns only an exit code, which
  is useful in scripts.

``ais-check`` exits ``0`` when every required component is present and at least
one volume is fast-path capable, and non-zero otherwise.

Read the output
===============

The report has three parts: a system identification line, a table of mounted
volumes, and a summary of AIS support by component.

.. code:: shell-session

   $ sudo ais-check

   Linux myhost 6.16.13-2278356.24.04 #1 SMP ... x86_64

   Mounted volumes:
   MOUNTPOINT  FSTYPE          DEVICE  BACKING  O_DIRECT  HIPFILE
   /           ext4 (ordered)  nvme0n1  nvme     yes       yes
   /home       xfs             nvme1n1  nvme     yes       yes
   /data       ext4 (ordered)  dm-0     lvm      yes       no

   AIS support in:
           Kernel P2PDMA support   : True
           HIP runtime             : True
           amdgpu                  : True
           hipFile-capable volume  : True

Mounted volumes table
---------------------

Each row is a block-backed mount that hipFile could potentially use. Pseudo and
network filesystems are omitted.

- **MOUNTPOINT** — where the filesystem is mounted.
- **FSTYPE** — the filesystem type. hipFile's fast path accepts only ``xfs`` or
  ``ext4`` mounted with ``data=ordered`` (the ext4 default), shown here as
  ``ext4 (ordered)``. Any other type or journal mode is rejected.
- **DEVICE** — the underlying block device.
- **BACKING** — what the device sits on (for example ``nvme``, ``lvm``, ``md``,
  ``mpath``). The fast path requires a local, non-multipath NVMe device. Any
  interposing layer such as LVM, MD RAID, dm-crypt, or multipath breaks the
  direct path to the device, so those volumes are not capable.
- **O_DIRECT** — whether an ``O_DIRECT`` open succeeded on the volume. Shows
  ``unverified`` when the probe could not run (for example without write
  permission).
- **HIPFILE** — the overall verdict for this volume: ``yes`` only when the
  filesystem qualifies, the backing is a direct local NVMe, and ``O_DIRECT``
  works.

In the example above ``/`` and ``/home`` are fast-path capable, while ``/data``
is not because it sits on LVM.

AIS support summary
-------------------

The final block reports the required components. All four must be ``True`` for
hipFile to use its fast path:

- **Kernel P2PDMA support** — the kernel supports peer-to-peer DMA between the
  GPU and the NVMe device. On recent amdgpu drivers this reflects the KFD
  topology's AIS capability bit; on older drivers it reflects
  ``CONFIG_PCI_P2PDMA=y`` in the kernel config.
- **HIP runtime** — a discovered HIP runtime library exports the AIS file I/O
  entry points (``hipAmdFileRead`` / ``hipAmdFileWrite``).
- **amdgpu** — the loaded amdgpu kernel driver exposes the AIS file I/O symbol.
- **hipFile-capable volume** — at least one volume in the table has ``HIPFILE``
  set to ``yes``.

If any component is ``False``, resolve it before expecting fast-path
performance. For component-specific guidance, see :doc:`/troubleshooting/troubleshooting`.
