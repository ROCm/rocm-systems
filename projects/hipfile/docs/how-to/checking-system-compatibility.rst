.. meta::
   :description: Step-by-step instructions for running ais-check to verify that a system supports hipFile's direct-to-GPU fastpath.
   :keywords: hipFile, ais-check, compatibility, fastpath, O_DIRECT, P2PDMA, ROCm, GPU I/O, AMD

******************************************
Check system compatibility with ais-check
******************************************

The ``ais-check`` tool reports whether the kernel, driver, runtime, and mounted volumes
meet hipFile fastpath requirements. If any component is missing or if no volume qualifies, hipFile uses the fallback path
instead.

``ais-check`` is located under ``tools/ais-check`` in the hipFile source tree and is installed in the ``bin`` directory of your ROCm installation unless hipFile was built with ``AIS_INSTALL_TOOLS=OFF``.

``ais-check`` must be run as root to produce accurate information. 

.. code:: shell

   sudo ais-check [-v|--verbose] [-q|--quiet]

Optional flags:

| ``-v`` or ``--verbose``: Lists discovered HIP runtime libraries and whether each exports AIS symbols.
| ``-q`` or ``--quiet``: Suppresses output and returns only an exit code.

``ais-check`` exits with ``0`` when at least one volume is fastpath capable. It exits with a non-zero code otherwise. It will also output a short report that includes a table of mounted volumes and a summary block. 

When run with the verbose option, a list of discovered HIP runtime libraries is also included in the output.

The mounted volumes table indicates whether a hipFile can use fastpath with each mount point. Each row in the table describes one block-backed mount. Pseudo file systems are omitted. Only mount points with "yes" in the``HIPFILE`` column are eligible for fastpath.

For example, this table indicates that only ``/`` and ``/home`` are eligible for fastpath:

.. code-block:: shell-session

   Mounted volumes:
   MOUNTPOINT  FSTYPE          DEVICE  BACKING  O_DIRECT  HIPFILE
   /           ext4 (ordered)  nvme0n1  nvme     yes       yes
   /home       xfs             nvme1n1  nvme     yes       yes
   /data       ext4 (ordered)  dm-0     lvm      yes       no

| ``MOUNTPOINT``: Location where the file system is mounted.
| ``FSTYPE``: The file system type. hipFile's fastpath accepts only ``xfs`` or ``ext4`` with ``data=ordered`` journaling. When ext4 uses the default mode, ``ais-check`` labels it ``ext4 (ordered)``.
| ``DEVICE``: The underlying block device.
| ``BACKING``: The type of block device that backs the mount. For example ``nvme``, ``lvm``, ``md``, or ``mpath``. Direct local NVMe backing is required.
| ``O_DIRECT``: Whether an ``O_DIRECT`` open succeeded on the volume.
| ``HIPFILE``: Will be "yes" when the file system qualifies, the backing is direct local NVMe, and ``O_DIRECT`` works. Will be "no" otherwise.

The ``AIS support in`` table at the end of the report is a final pass or fail summary. It shows whether the four parts of the fastpath stack are
present: kernel P2PDMA, a HIP runtime with AIS symbols, the amdgpu driver hook, and at least one qualifying mounted volume. Each line reads ``True`` or
``False``. hipFile uses fastpath only when all four read ``True``.

.. code-block:: shell-session

   AIS support in:
           Kernel P2PDMA support   : True
           HIP runtime             : True
           amdgpu                  : True
           hipFile-capable volume  : True

| ``Kernel P2PDMA support``: Peer-to-peer DMA between the GPU and storage is available.
| ``HIP runtime``: A discovered HIP runtime library exports  ``hipAmdFileRead()`` and ``hipAmdFileWrite()``.
| ``amdgpu``: The loaded amdgpu kernel driver exposes ``kfd_ais_rw_file`` in ``/proc/kallsyms``.
| ``hipFile-capable volume``: At least one row in the mounted volumes table has ``HIPFILE`` set to ``yes``.


.. note::

   hipFile's fastpath is only supported on GPU physical functions (PFs). When a
   GPU is an SR-IOV virtual function (VF), the fastpath is disabled and I/O falls
   back to the compatibility path. ``ais-check`` uses ``amd-smi`` to detect VFs.
   If ``amd-smi`` is not installed, the check is skipped. If a VF is present,
   ``amd-smi`` will print a warning to stderr. This warning won't affect the exit
   code. VFs can be identified by a device name that ends in ``VF``, such as
   ``AMD Instinct MI300X VF``.

