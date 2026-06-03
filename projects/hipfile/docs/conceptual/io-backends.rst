.. meta::
   :description: Describes the hipfile IO backend architecture, including fastpath and fallback backend selection, scoring, automatic retry, and environment variable configuration.
   :keywords: hipfile, backend, fastpath, fallback, score, POSIX, direct-to-GPU, ROCm, IO, HIPFILE_ALLOW_COMPAT_MODE, HIPFILE_FORCE_COMPAT_MODE

=======================
IO backend architecture
=======================

hipfile routes each IO request through a multi-backend stack on AMD platforms. The library selects the backend with the highest score for the request and can retry a failed operation on another backend.

Backend model
=============

hipfile defines a ``Backend`` base class that every IO path implements. Two concrete backends exist for the AMD platform:

Fastpath
   Uses the HIP runtime's direct-IO functions, ``hipAmdFileRead`` and ``hipAmdFileWrite``, to transfer data directly between storage and GPU memory without a CPU bounce buffer.

Fallback
   Uses POSIX ``pread`` and ``pwrite`` with a host-memory bounce buffer to move data between storage and GPU memory.

When you submit an IO request through ``hipFileRead`` or ``hipFileWrite``, hipfile evaluates every registered backend and selects the one best suited to the request.

Backend scoring
===============

Each backend implements a ``score()`` method that returns an integer. The integer reflects how eager the backend is to handle a given IO request. The scoring rules are:

- A non-negative return value means the backend accepts the request. A higher value means greater eagerness.
- A negative return value means the backend refuses the request. If hipfile routed the request to that backend, the operation would likely fail.

hipfile iterates over all available backends, calls ``score()`` on each, and selects the backend with the highest non-negative score. You can add or reorder backends without changing the dispatch logic.

.. code-block:: cpp

   /// Indicates the willingness of the backend to service an IO
   /// request. The higher the number the more eager the backend is
   /// to service the request. If a negative number is returned,
   /// the backend is refusing to perform the operation.
   virtual int score(std::shared_ptr<IFile> file,
                     std::shared_ptr<IBuffer> buffer,
                     size_t size,
                     hoff_t file_offset,
                     hoff_t buffer_offset) const = 0;

Fastpath eligibility
====================

The fastpath backend returns a positive score and is selected when all of the following conditions are met:

1. HIP runtime symbols are present. At initialization, hipfile looks up ``hipAmdFileRead`` and ``hipAmdFileWrite`` in the HIP runtime library. If either symbol is missing, the fastpath backend is disabled.

2. Force-compat mode isn't set. If you set ``HIPFILE_FORCE_COMPAT_MODE`` to ``true``, the configuration layer disables the fastpath backend regardless of symbol availability.

3. The target file type is eligible. The file must be a block device or a regular file, as reported by ``statx``.

4. The file system is supported. The file must reside on one of the following file systems:

   - ext4 with ordered journaling mode
   - xfs

   hipfile reads ``/proc/self/mountinfo`` through ``libmount`` to determine the file system type and mount options. You can skip this check by setting ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true``. That setting lets the fastpath operate on file systems that haven't been validated.

For file system requirements, see :doc:`/conceptual/filesystem-requirements`.

Fallback path
=============

The fallback backend uses POSIX ``pread`` and ``pwrite`` to transfer data between storage and a host-memory bounce buffer. It then copies data to or from GPU memory with HIP memory copy operations. The fallback backend accepts most requests and returns a lower score than the fastpath. hipfile selects it only when the fastpath is unavailable or refuses the request.

You can disable the fallback backend by setting ``HIPFILE_ALLOW_COMPAT_MODE=false``. When the fallback is disabled, IO requests that the fastpath can't handle fail instead of falling back to POSIX IO.

Automatic retry with BackendWithFallback
========================================

The fastpath backend extends ``BackendWithFallback``, a ``Backend`` specialization that adds automatic retry logic. When a fastpath IO operation fails, ``BackendWithFallback`` checks whether the request can retry on a registered fallback backend.

.. mermaid::

   flowchart TD
       A[IO request submitted] --> B[Score all backends]
       B --> C{Highest score >= 0?}
       C -- Yes --> D[Execute IO on selected backend]
       D --> E{IO succeeded?}
       E -- Yes --> F[Return bytes transferred]
       E -- No --> G{Fallback eligible?}
       G -- Yes --> H[Retry IO on fallback backend]
       H --> F
       G -- No --> I[Return error]
       C -- No --> I

The retry flow works as follows:

1. The fastpath backend's ``_io_impl`` attempts the IO operation on the HIP runtime direct-IO path.
2. If the operation throws an exception or otherwise fails, ``BackendWithFallback::io()`` calls ``is_fallback_eligible()`` to check whether the fallback path can recover from the error.
3. If the request is eligible and a fallback backend is registered through ``register_fallback_backend()``, hipfile reissues the same IO operation to the fallback backend.
4. If the request isn't eligible for fallback, the original error propagates to the caller.

The retry is transparent to your application. You receive either the bytes transferred or an error. The retry is an internal implementation detail.

.. note::

   ``register_fallback_backend()`` rejects ``nullptr`` and self-references to prevent infinite retry loops.

Environment variable controls
=============================

Two environment variables control backend selection:

``HIPFILE_ALLOW_COMPAT_MODE``
   Controls whether the POSIX fallback backend is available. When set to ``false``, the fallback backend is disabled and IO requests that the fastpath can't handle fail. The default is ``true``, so the fallback is allowed.

``HIPFILE_FORCE_COMPAT_MODE``
   Forces all IO through the POSIX fallback path and bypasses the fastpath. When set to ``true``, the fastpath backend is disabled regardless of whether HIP runtime direct-IO symbols are available. The default is ``false``.

The two variables interact as follows:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - ``HIPFILE_FORCE_COMPAT_MODE``
     - ``HIPFILE_ALLOW_COMPAT_MODE``
     - Behavior
   * - ``false`` (default)
     - ``true`` (default)
     - Fastpath preferred, fallback available on failure
   * - ``false``
     - ``false``
     - Fastpath only, no fallback on failure
   * - ``true``
     - any
     - Fallback only, fastpath disabled

For a complete list of environment variables and their default values, see :doc:`/reference/environment-variables`.
