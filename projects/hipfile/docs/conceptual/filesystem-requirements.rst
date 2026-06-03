.. meta::
   :description: Filesystem and file type requirements for the hipfile fastpath backend, including supported filesystems, mount info discovery, and direct I/O alignment.
   :keywords: hipfile, filesystem, ext4, xfs, fastpath, direct I/O, DIO alignment, libmount, mountinfo, ROCm

=====================================
Filesystem and file type requirements
=====================================

The hipfile fastpath backend performs direct-to-GPU I/O through the HIP runtime. Before it accepts an I/O request, it validates that the target file meets specific filesystem and file type criteria. If validation fails, the request either falls back to the POSIX backend or is rejected, depending on configuration. For an overview of how backends are selected, see :doc:`/conceptual/io-backends`.

Supported file types
********************

The fastpath backend requires that the target file is one of the following:

- A **block device**
- A **regular file**

These properties are determined at file registration time using the Linux ``statx`` system call. The ``IFile`` interface exposes the results through ``isBlockDevice()`` and ``isRegularFile()``. If ``statx`` does not return type information, both methods return ``false``, which prevents the fastpath backend from accepting the file.

Supported filesystems
*********************

For regular files, the fastpath backend validates the underlying filesystem. Only two filesystem types are supported:

ext4 with ordered journaling
----------------------------

The file must reside on an **ext4** filesystem configured with the **ordered** journaling mode. The ordered mode ensures that file data is flushed to disk before metadata is committed to the journal, which provides the consistency guarantees that the direct I/O path requires.

hipfile checks the journaling mode by inspecting the mount options obtained from the system. The ``IFile`` interface exposes this check through ``onExt4Ordered()``, which returns ``true`` only when the filesystem type is ext4 and the journaling mode is ``ordered``. Other ext4 journaling modes (``journal`` and ``writeback``) are not supported by default.

xfs
---

The file must reside on an **xfs** filesystem. No additional mount option validation is performed for xfs. The ``IFile`` interface exposes this check through ``onXfs()``.

Mount information discovery
***************************

hipfile discovers filesystem type and mount options by reading ``/proc/self/mountinfo`` through the **libmount** library. The ``LibMountHelper`` class wraps the libmount API and provides a ``getMountInfo()`` method that accepts a device number (``dev_t``) and returns a ``MountInfo`` structure containing:

- ``type`` — the filesystem type, represented as a ``FilesystemType`` enum with values ``ext4``, ``xfs``, or ``other``
- ``options`` — filesystem-specific mount options; for ext4, this includes the journaling mode (``journal``, ``ordered``, ``writeback``, or ``unknown``)

The device number is obtained from the ``statx`` call performed during file registration. The ``LibMount`` class provides a thin wrapper over the following libmount functions:

- ``mnt_new_context()`` and ``mnt_free_context()`` for context lifecycle management
- ``mnt_context_get_mtab()`` to read the mount table
- ``mnt_table_find_devno()`` to locate the mount entry for a specific device
- ``mnt_fs_get_fstype()`` to retrieve the filesystem type string
- ``mnt_fs_get_option()`` to retrieve specific mount options

If mount information is not available for a file (for example, if ``/proc/self/mountinfo`` is inaccessible), the ``mountinfo`` field on the file object is empty, and both ``onExt4Ordered()`` and ``onXfs()`` return ``false``. In this case, the fastpath backend refuses the request unless the unsupported filesystem escape hatch is enabled.

Allowing unsupported filesystems
********************************

You can bypass filesystem validation by setting the ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` environment variable to ``true``. When enabled, the fastpath backend allows I/O on filesystems other than ext4 (with ordered journaling) and xfs.

.. warning::

   Using unsupported filesystems with the fastpath backend may lead to data corruption or unexpected behavior. Use this option only for testing or when you have confirmed that your filesystem is compatible with direct I/O.

The ``Configuration`` class exposes this setting through its ``unsupportedFileSystems()`` method, which reads the value of the ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` environment variable. The default value is ``false``.

For the full list of environment variables, see :doc:`/reference/environment-variables`.

Direct I/O alignment requirements
*********************************

The fastpath backend uses direct I/O (``O_DIRECT``), which imposes alignment requirements on memory buffers, file offsets, and I/O segment lengths. hipfile determines these requirements at file registration time using information from ``statx``:

``dioMemAlign``
   The memory alignment (in bytes) required for buffers used with direct I/O. If the file does not support direct I/O, this value is ``0``. If ``statx`` does not provide alignment information, hipfile defaults to ``4096`` bytes.

``dioOffsetAlign``
   The alignment (in bytes) required for file offsets and I/O segment lengths when using direct I/O. If the file does not support direct I/O, this value is ``0``. If ``statx`` does not provide alignment information, hipfile defaults to ``4096`` bytes.

During file registration, hipfile attempts to open an additional file descriptor with the ``O_DIRECT`` flag (the "unbuffered" file descriptor). If the file or filesystem does not support ``O_DIRECT``, the unbuffered file descriptor is not available, and the fastpath backend cannot service the request.

The ``aiscp`` example demonstrates proper alignment handling when performing I/O:

.. code-block:: cpp

   /// @brief Round value to the next multiple of align. Align _must_ be a power of 2.
   static inline size_t
   align_up(size_t value, size_t align)
   {
       return (value + align - 1) & ~(align - 1);
   }

   // When writing, align the size to the block size
   nbytes = hipFileWrite(dst_handle, devbuf,
                         align_up(static_cast<size_t>(nread - nwrite), block_size),
                         file_offset + nwrite, nwrite);

Fastpath eligibility summary
****************************

The following diagram summarizes the decision process for fastpath eligibility based on filesystem and file type checks:

.. mermaid::

   flowchart TD
       A[File registered] --> B{Block device?}
       B -- Yes --> G[Fastpath eligible]
       B -- No --> C{Regular file?}
       C -- No --> H[Fastpath ineligible]
       C -- Yes --> D{ext4 ordered or xfs?}
       D -- Yes --> G
       D -- No --> E{HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true?}
       E -- Yes --> G
       E -- No --> H

When the fastpath backend is ineligible, the request is routed to the fallback (POSIX) backend if it is enabled. For details on how backend selection works, see :doc:`/conceptual/io-backends`.
