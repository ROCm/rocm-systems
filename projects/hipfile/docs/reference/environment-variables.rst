.. meta::
   :description: Reference table of runtime environment variables for hipfile, including defaults and effects on I/O backend selection and statistics collection.
   :keywords: hipfile, environment variables, HIPFILE_ALLOW_COMPAT_MODE, HIPFILE_FORCE_COMPAT_MODE, HIPFILE_STATS_LEVEL, HIPFILE_UNSUPPORTED_FILE_SYSTEMS, ROCm, configuration

===============================
Environment variables reference
===============================

hipfile reads several environment variables at runtime to control I/O backend selection, statistics collection, and filesystem restrictions. Each variable is read from the process environment when the library initializes. If a variable is unset, the documented default takes effect.

.. list-table:: hipfile environment variables
   :header-rows: 1
   :widths: 30 10 10 50

   * - Variable
     - Type
     - Default
     - Effect

   * - ``HIPFILE_ALLOW_COMPAT_MODE``
     - Boolean
     - ``true``
     - When enabled, I/O operations that cannot be completed through the fast path (direct-to-GPU) are allowed to fall back to POSIX I/O APIs. When disabled, I/O operations that cannot use the fast path are rejected. See :doc:`/conceptual/io-backends` for details on how the fast path and fallback path interact.

   * - ``HIPFILE_FORCE_COMPAT_MODE``
     - Boolean
     - ``false``
     - When enabled, all I/O operations are forced to use the POSIX I/O (compatibility) path, bypassing the fast path entirely. When disabled, I/O operations use the fast path when conditions are satisfied. See :doc:`/conceptual/io-backends` for more information on backend selection.

   * - ``HIPFILE_STATS_LEVEL``
     - Unsigned integer
     - ``1``
     - Controls how much information is collected for the ``ais-stats`` tool. A value of ``0`` disables statistics recording. A value of ``1`` or higher enables basic statistics collection, including per-GPU and per-backend bandwidth, latency, and error histograms. See :doc:`/conceptual/statistics-collection` for guidance on interpreting statistics output.

   * - ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS``
     - Boolean
     - ``false``
     - When enabled, the fast path backend permits I/O on file systems other than ext4 (with ordered journaling) and XFS. When disabled, only supported file systems are allowed in the fast path. See :doc:`/conceptual/filesystem-requirements` for the list of supported file systems and their requirements.

Variable details
****************

``HIPFILE_ALLOW_COMPAT_MODE``
-----------------------------

This variable accepts the string values ``true`` or ``false`` (case-sensitive). If the variable is set to a value other than ``true`` or ``false``, it is treated as unset and the default (``true``) applies.

When both ``HIPFILE_ALLOW_COMPAT_MODE`` and ``HIPFILE_FORCE_COMPAT_MODE`` are set, the force setting takes precedence: all I/O is routed through the POSIX compatibility path regardless of the allow setting.

``HIPFILE_FORCE_COMPAT_MODE``
-----------------------------

This variable accepts the string values ``true`` or ``false`` (case-sensitive). If the variable is set to a value other than ``true`` or ``false``, it is treated as unset and the default (``false``) applies.

Setting this variable to ``true`` is useful for debugging or benchmarking scenarios where you want to compare POSIX I/O performance against the direct-to-GPU fast path.

``HIPFILE_STATS_LEVEL``
-----------------------

This variable accepts an unsigned integer value. If the variable is set to a non-numeric value, it is treated as unset and the default (``1``) applies.

.. note::

   Setting ``HIPFILE_STATS_LEVEL`` to ``0`` completely disables statistics collection, which eliminates any overhead associated with recording I/O metrics.

``HIPFILE_UNSUPPORTED_FILE_SYSTEMS``
------------------------------------

This variable accepts the string values ``true`` or ``false`` (case-sensitive). If the variable is set to a value other than ``true`` or ``false``, it is treated as unset and the default (``false``) applies.

.. warning::

   Enabling unsupported file systems may lead to data corruption or unexpected behavior. Use this setting only for testing or when you have verified that your file system works correctly with direct I/O from GPU memory.

Usage examples
**************

Disable the fallback to POSIX I/O so that only the fast path is used:

.. code-block:: bash

   export HIPFILE_ALLOW_COMPAT_MODE=false

Force all I/O through the POSIX compatibility path:

.. code-block:: bash

   export HIPFILE_FORCE_COMPAT_MODE=true

Disable statistics collection:

.. code-block:: bash

   export HIPFILE_STATS_LEVEL=0

Allow I/O on file systems beyond ext4 (ordered) and XFS:

.. code-block:: bash

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true
