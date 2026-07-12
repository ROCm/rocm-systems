.. meta::
   :description: hipFile environment variables
   :keywords: hipFile, environment variables, HIPFILE_ALLOW_COMPAT_MODE, HIPFILE_FORCE_COMPAT_MODE, HIPFILE_STATS_LEVEL, HIPFILE_UNSUPPORTED_FILE_SYSTEMS, ROCm, AMD

******************************
hipFile environment variables
******************************

The following environment variables affect the hipFile runtime.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Environment variable
     - Values
   * - ``HIPFILE_ALLOW_COMPAT_MODE``
     - | Controls whether device-memory I/O that can't use the direct-to-GPU fastpath uses fallback I/O.
       | When ``HIPFILE_FORCE_COMPAT_MODE`` is ``true`` and ``HIPFILE_ALLOW_COMPAT_MODE`` is ``false``, I/O fails.
       | ``true``: Use fallback I/O when fastpath can't be used. (default)
       | ``false``: I/O that can't use fastpath fails.
       |
   * - ``HIPFILE_FORCE_COMPAT_MODE``
     - | Forces the use of fallback I/O without attempting to use fastpath.
       | ``true``: Always use fallback I/O.
       | ``false``: Use fastpath when possible. (default)
   * - ``HIPFILE_STATS_LEVEL``
     - | Controls how much information hipFile collects for the ``ais-stats`` tool.
       | ``0``: Histogram recording is disabled.
       | ``1``: Collect per-GPU and per-backend bandwidth, latency, and error histograms. (default)
   * - ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS``
     - | Controls whether fastpath is used with unsupported file systems.
       | ``true``: Fastpath is used with any file system without validation.
       | ``false``: Fastpath is only used with supported file systems. (default)
