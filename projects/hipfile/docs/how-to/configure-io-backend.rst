.. meta::
  :description: How to configure the hipFile I/O backend using environment variables to control fastpath and fallback behavior.
  :keywords: hipFile, I/O backend, environment variables, POSIX, fallback, fastpath, ROCm, configuration

*************************
Configure the I/O backend
*************************

hipFile uses environment variables to control which I/O backend handles data transfers. By default, hipFile attempts to use the fastpath backend (direct GPU I/O via HIP runtime extensions) and automatically falls back to POSIX I/O when the fastpath cannot service a request. You can override this behavior with three environment variables.

For background on how the fastpath and fallback backends work, see :doc:`/conceptual/io-backends`. For a complete list of environment variables, see :doc:`/reference/environment-variables`.

Force POSIX-only I/O
*******************

Set ``HIPFILE_FORCE_COMPAT_MODE`` to ``true`` to force all I/O operations to use the POSIX I/O path, bypassing the fastpath backend entirely.

.. code:: shell

   export HIPFILE_FORCE_COMPAT_MODE=true

When to use this setting
------------------------

- You are troubleshooting I/O issues and want to rule out the fastpath backend as the cause.
- Your system does not meet the requirements for the fastpath backend and you want to avoid the overhead of attempting it before falling back.
- You need deterministic POSIX I/O behavior for testing or benchmarking comparisons.

To restore default behavior (fastpath with automatic fallback), unset the variable or set it to ``false``:

.. code:: shell

   export HIPFILE_FORCE_COMPAT_MODE=false

Disable automatic fallback
**************************

Set ``HIPFILE_ALLOW_COMPAT_MODE`` to ``false`` to prevent hipFile from falling back to POSIX I/O when the fastpath backend cannot complete a request. With this setting, I/O operations that cannot be completed by the fastpath will be rejected rather than retried through POSIX.

.. code:: shell

   export HIPFILE_ALLOW_COMPAT_MODE=false

When to use this setting
------------------------

- You want to make sure that all I/O operations use the direct GPU I/O path and you prefer an error over silently degraded performance.
- You are validating that your system configuration fully supports the fastpath backend.

By default, automatic fallback is enabled. To explicitly re-enable it:

.. code:: shell

   export HIPFILE_ALLOW_COMPAT_MODE=true

.. warning::

   Disabling fallback means that I/O operations will fail if the fastpath backend cannot service them. Make sure your system, file system, and file configuration support the fastpath before using this setting in production.

Allow non-validated file systems
*******************************

Set ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` to ``true`` to allow the fastpath backend to operate on file systems other than those that have been validated (ext4 with ordered journaling and XFS).

.. code:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true

When to use this setting
------------------------

- Your files reside on a file system that hipFile does not explicitly validate, but you want to attempt fastpath I/O regardless.
- You are experimenting with fastpath I/O on a new file system type.

By default, the fastpath backend only permits I/O on supported file systems. To restore this default:

.. code:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=false

.. note::

   Enabling unsupported file systems does not guarantee correct behavior. Data integrity has only been validated on ext4 (with ordered journaling) and XFS. Use this setting with caution.

Combine settings
****************

You can combine these environment variables to fine-tune backend behavior. For example, to allow non-validated file systems while keeping automatic fallback enabled:

.. code:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true
   export HIPFILE_ALLOW_COMPAT_MODE=true

All three variables accept the values ``true`` or ``false`` (case-insensitive). If a variable is unset or contains an invalid value, hipFile uses its default behavior.

.. note::

   Setting ``HIPFILE_FORCE_COMPAT_MODE=true`` disables the fastpath backend. I/O uses the fallback path only when ``HIPFILE_ALLOW_COMPAT_MODE`` is also ``true``. If fallback is disabled while force is enabled, both backends refuse I/O.
