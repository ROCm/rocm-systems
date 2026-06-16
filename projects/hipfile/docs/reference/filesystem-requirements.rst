.. meta::
  :description: File system and file type requirements for the hipFile fastpath backend, including supported layouts and direct I/O alignment.
  :keywords: hipFile, file system, ext4, XFS, fastpath, direct I/O, DIO alignment, ROCm

*************************************
Filesystem and file type requirements
*************************************

The hipFile fastpath backend performs direct-to-GPU I/O through the HIP runtime. Before it accepts an I/O request, it validates that the target file meets specific file system and file type criteria. If validation fails, the request either falls back to the fallback backend or is rejected, depending on configuration. For backend selection and scoring, see :doc:`/reference/hipFile-io-backends`.

Supported file types
====================

The fastpath backend accepts only:

- A block device
- A regular file

hipFile checks file type at registration time. Files that don't meet either criterion can't use the fastpath.

Supported file systems
======================

For regular files, the fastpath backend validates the underlying file system. Only two file system types are supported by default:

ext4 with ordered journaling
----------------------------

The file must reside on an ext4 file system configured with ordered journaling mode. Ordered mode flushes file data to disk before metadata is committed to the journal, which matches what the direct I/O path requires.

hipFile reads journaling mode from mount options. Only ext4 with ``ordered`` journaling is accepted. Other ext4 modes (``journal`` and ``writeback``) are rejected unless you enable the unsupported file system escape hatch.

XFS
---

The file must reside on an XFS file system. hipFile doesn't perform additional mount option checks for XFS.

If hipFile can't read mount information (for example when ``/proc/self/mountinfo`` is inaccessible), the fastpath treats the file system as unsupported and refuses the request unless you enable the escape hatch.

Allowing unsupported file systems
=================================

You can bypass file system validation by setting ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` to ``true``. When enabled, the fastpath accepts I/O on file systems other than ext4 (ordered) and XFS.

.. warning::

   Using unsupported file systems with the fastpath may lead to data corruption or unexpected behavior. Data integrity is validated only on ext4 (ordered) and XFS. Use this option for testing or when you've confirmed your file system works with direct I/O.

For export examples and when to use this setting, see :doc:`/reference/hipFile-io-backends`. For all environment variables, see :doc:`/reference/hipFile-environment-variables`.

Direct I/O alignment requirements
=================================

The fastpath uses direct I/O (``O_DIRECT``), which imposes alignment on buffers, file offsets, and I/O segment lengths. hipFile reads alignment requirements from ``statx`` at file registration:

Memory alignment
   Byte alignment required for buffers used with direct I/O. Defaults to ``4096`` when ``statx`` doesn't report a value.

Offset and length alignment
   Byte alignment required for file offsets and I/O lengths. Defaults to ``4096`` when ``statx`` doesn't report a value.

During registration, hipFile tries to open an additional file descriptor with ``O_DIRECT``. If the file or file system doesn't support ``O_DIRECT``, the fastpath can't service the request.

For alignment examples in application code, see :doc:`/tutorials/copy-a-file`.
