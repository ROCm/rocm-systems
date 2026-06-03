.. meta::
   :description: How to configure the hipfile IO backend using environment variables to control fastpath and fallback behavior.
   :keywords: hipfile, IO backend, environment variables, POSIX, fallback, fastpath, ROCm, configuration

========================
Configure the IO backend
========================

hipfile uses environment variables to control which IO backend handles data transfers. By default, hipfile attempts to use the fastpath backend (direct GPU IO via HIP runtime extensions) and automatically falls back to POSIX IO when the fastpath cannot service a request. You can override this behavior with three environment variables.

For background on how the fastpath and fallback backends work, see :doc:`/conceptual/io-backends`. For a complete list of environment variables, see :doc:`/reference/environment-variables`.

Force POSIX-only IO
*******************

Set ``HIPFILE_FORCE_COMPAT_MODE`` to ``true`` to force all IO operations to use the POSIX IO path, bypassing the fastpath backend entirely.

.. code-block:: shell

   export HIPFILE_FORCE_COMPAT_MODE=true

**When to use this setting:**

- You are troubleshooting IO issues and want to rule out the fastpath backend as the cause.
- Your system does not meet the requirements for the fastpath backend and you want to avoid the overhead of attempting it before falling back.
- You need deterministic POSIX IO behavior for testing or benchmarking comparisons.

To restore default behavior (fastpath with automatic fallback), unset the variable or set it to ``false``:

.. code-block:: shell

   export HIPFILE_FORCE_COMPAT_MODE=false

Disable automatic fallback
**************************

Set ``HIPFILE_ALLOW_COMPAT_MODE`` to ``false`` to prevent hipfile from falling back to POSIX IO when the fastpath backend cannot complete a request. With this setting, IO operations that cannot be completed by the fastpath will be rejected rather than retried through POSIX.

.. code-block:: shell

   export HIPFILE_ALLOW_COMPAT_MODE=false

**When to use this setting:**

- You want to ensure that all IO operations use the direct GPU IO path and you prefer an error over silently degraded performance.
- You are validating that your system configuration fully supports the fastpath backend.

By default, automatic fallback is enabled. To explicitly re-enable it:

.. code-block:: shell

   export HIPFILE_ALLOW_COMPAT_MODE=true

.. warning::

   Disabling fallback means that IO operations will fail if the fastpath backend cannot service them. Ensure your system, filesystem, and file configuration support the fastpath before using this setting in production.

Allow non-validated filesystems
*******************************

Set ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` to ``true`` to allow the fastpath backend to operate on filesystems other than those that have been validated (ext4 with ordered journaling and XFS).

.. code-block:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true

**When to use this setting:**

- Your files reside on a filesystem that hipfile does not explicitly validate, but you want to attempt fastpath IO regardless.
- You are experimenting with fastpath IO on a new filesystem type.

By default, the fastpath backend only permits IO on supported filesystems. To restore this default:

.. code-block:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=false

.. note::

   Enabling unsupported filesystems does not guarantee correct behavior. Data integrity has only been validated on ext4 (with ordered journaling) and XFS. Use this setting with caution.

Combine settings
****************

You can combine these environment variables to fine-tune backend behavior. For example, to allow non-validated filesystems while keeping automatic fallback enabled:

.. code-block:: shell

   export HIPFILE_UNSUPPORTED_FILE_SYSTEMS=true
   export HIPFILE_ALLOW_COMPAT_MODE=true

All three variables accept the values ``true`` or ``false`` (case-sensitive). If a variable is unset or contains an invalid value, hipfile uses its default behavior.

.. note::

   Setting ``HIPFILE_FORCE_COMPAT_MODE=true`` takes precedence over other backend settings. When compatibility mode is forced, the fastpath backend is not used regardless of other variable values.
