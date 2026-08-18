.. meta::
   :description: How to use the RCCL environment plugin API
   :keywords: RCCL, ROCm, library, API, environment, plugin, NCCL_ENV_PLUGIN, JSON

.. _using-rccl-env-plugin:

*************************************
Using the RCCL environment plugin API
*************************************

The RCCL environment plugin API lets deployments supply NCCL/RCCL configuration
parameters from a custom source — such as a centralized configuration database,
a secrets manager, or a JSON file — without relying on process-level environment
variables. The active plugin intercepts every internal ``ncclGetEnv()`` call, so
all RCCL parameters that are normally read from the environment are transparently
routed through the plugin.

Overview
========

By default RCCL reads parameters directly from the process environment via
``getenv()``. When ``NCCL_ENV_PLUGIN`` is set to a shared library path, RCCL
loads that library at initialization time and delegates all environment lookups
to the plugin's ``getEnv`` function.

This is useful in the following scenarios:

* **Centralized configuration** — a cluster operator wants to push RCCL parameters
  to all jobs from a single source of truth without requiring users to set environment
  variables in their job scripts.
* **Secret management** — sensitive values (for example, authentication tokens used
  by a custom net plugin) can be fetched from a secrets manager at runtime instead
  of being stored in plain-text environment variables.
* **Per-job configuration files** — a JSON file packaged with the job can override
  specific parameters without polluting the process environment.

Environment variables
=====================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - **Environment variable**
     - **Description**

   * - | ``NCCL_ENV_PLUGIN``
       | Path to the env plugin shared library, or ``none`` to disable
         external plugin loading.
     - | String: a path to a ``.so`` file, a bare name that RCCL expands to
         ``librccl-env-<name>.so`` and looks up on the loader path, or the literal
         string ``none``.
       | Default: unset. RCCL first tries to load ``librccl-env.so``; if that is not
         found it reads directly from the process environment.
       | Example: ``export NCCL_ENV_PLUGIN=/opt/rccl-plugins/librccl-env-json.so``
       | Example: ``export NCCL_ENV_PLUGIN=json`` (loads ``librccl-env-json.so``)

Lookup precedence
-----------------

When a plugin is loaded, RCCL routes all ``ncclGetEnv()`` calls to the plugin's
``getEnv`` function. The plugin is responsible for deciding the precedence between
its own source and the process environment. The reference JSON plugin (described
below) uses the following order:

1. Value from the JSON file (if the key is present).
2. Value from the process environment (``getenv()``).
3. ``NULL`` if the key is not found in either source.

API description
===============

To build a custom env plugin, implement and export the ``ncclEnvPlugin_v2``
symbol of type ``ncclEnv_v2_t``.

RCCL looks for ``ncclEnvPlugin_v2`` first and falls back to the older
``ncclEnvPlugin_v1``, so plugins written against v1 keep working. New plugins
should use v2, which additionally receives RCCL's debug logger.

Structure: ``ncclEnv_v2_t``
---------------------------

.. code-block:: c

   typedef struct {
     const char* name;
     ncclResult_t (*init)(uint8_t ncclMajor, uint8_t ncclMinor,
                          uint8_t ncclPatch, const char* suffix,
                          ncclDebugLogger_t logFunction);
     ncclResult_t (*finalize)(void);
     const char*  (*getEnv)(const char* name);
   } ncclEnv_v2_t;

**Fields**

* ``name``

  * **Type**: ``const char*``
  * **Description**: Human-readable plugin name, used in RCCL log output when
    ``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=ENV`` are set.

**Functions**

* ``init`` (called once during RCCL initialization)

  Sets up any plugin state, opens configuration sources, and prepares for
  subsequent ``getEnv`` calls.

  * **Parameters**:

    * ``ncclMajor``, ``ncclMinor``, ``ncclPatch`` (``uint8_t``): RCCL version
      numbers, allowing version-specific behavior in the plugin.
    * ``suffix`` (``const char*``): RCCL version suffix string (for example,
      ``"rc1"``).
    * ``logFunction`` (``ncclDebugLogger_t``): RCCL's debug logger. Messages
      logged through it honor ``NCCL_DEBUG`` and ``NCCL_DEBUG_SUBSYS`` and are
      interleaved with RCCL's own output, which makes it the recommended way to
      report plugin diagnostics.

  * **Return**: ``ncclResult_t`` — return ``ncclSuccess`` on success.

* ``finalize`` (called during RCCL teardown)

  Releases all resources allocated by the plugin.

  * **Return**: ``ncclResult_t`` — return ``ncclSuccess`` on success.

* ``getEnv`` (called for every RCCL parameter lookup)

  Returns the value of the named variable, or ``NULL`` if the variable is not
  found. The returned pointer must remain valid until RCCL calls ``finalize``
  or calls ``getEnv`` again for the same variable name. Modifying the variable
  via ``setenv``/``putenv`` after returning a pointer to it is undefined behavior.

  * **Parameters**:

    * ``name`` (``const char*``): The name of the variable to look up, for example
      ``"NCCL_DEBUG"`` or ``"NCCL_ALGO"``.

  * **Return**: ``const char*`` — pointer to the value string, or ``NULL``.

Plugin symbol
-------------

The plugin shared library must export the following symbol at the top level
(not ``static``):

.. code-block:: c

   const ncclEnv_v2_t ncclEnvPlugin_v2 = {
     .name     = "myPlugin",
     .init     = myInit,
     .finalize = myFinalize,
     .getEnv   = myGetEnv,
   };

The required headers are available under
``plugins/env/example/nccl/`` in the RCCL source tree.

.. note::

   The example plugins and the reference JSON plugin are C translation units,
   and the exported symbol has C linkage. If you write your plugin in C++, wrap
   the definition in ``extern "C"`` — a ``const`` object at namespace scope has
   internal linkage in C++, so without it ``dlsym`` cannot find the symbol and
   RCCL reports the plugin as unsupported.

Reference JSON plugin
=====================

RCCL ships a reference implementation at ``plugins/env/json/plugin.c`` that
reads parameters from a flat JSON file. It can be used directly in deployments
or as a starting point for a custom plugin.

JSON file format
----------------

The JSON file must be a single flat object mapping variable names to string
values:

.. code-block:: json

   {
     "NCCL_DEBUG": "INFO",
     "NCCL_ALGO": "Ring",
     "NCCL_DEBUG_SUBSYS": "ENV,INIT"
   }

.. note::

   Only flat string values are supported. Nested objects, arrays, numbers,
   and booleans are not valid. Keys and values must not exceed 255 and 4095
   characters respectively. A file holds at most 256 entries and must not be
   larger than 1 MB.

   Within a string, only ``\\`` and ``\"`` are recognized. Any other escape is
   stored with the backslash dropped, so ``\n`` becomes the letter ``n`` rather
   than a newline. Write values literally instead of relying on escapes.

   The whole file is rejected if it breaks any of these rules, including
   exceeding the entry limit; RCCL then logs a warning and reads every variable
   from the process environment instead. It is never applied in part.

Usage
-----

#. Build the JSON plugin by configuring RCCL with ``-DBUILD_TESTS=ON -DBUILD_PLUGIN_EXAMPLES=ON``:

   .. code-block:: shell

      cmake /path/to/rccl -DBUILD_TESTS=ON -DBUILD_PLUGIN_EXAMPLES=ON
      make rccl-env-json

   The plugin is built as ``librccl-env-json.so`` in the build output under
   ``test/unit/plugins/``.

#. Create a JSON configuration file:

   .. code-block:: shell

      cat > /etc/rccl/config.json << 'EOF'
      {
        "NCCL_DEBUG": "WARN",
        "NCCL_SOCKET_IFNAME": "eth0"
      }
      EOF

#. Set the two environment variables before launching your workload:

   .. code-block:: shell

      export NCCL_ENV_PLUGIN=/path/to/librccl-env-json.so
      export NCCL_ENV_JSON_FILE=/etc/rccl/config.json

   .. list-table::
      :header-rows: 1
      :widths: 40 60

      * - **Environment variable**
        - **Description**

      * - | ``NCCL_ENV_JSON_FILE``
          | Path to the JSON configuration file used by ``librccl-env-json.so``.
        - | String: absolute path to a JSON file.
          | Default: unset (plugin falls back to ``getenv()`` for all lookups).

.. note::

   ``NCCL_ENV_PLUGIN`` and ``NCCL_ENV_JSON_FILE`` must be set in the process
   environment before RCCL is initialized. They are read during plugin load
   and are not re-read after initialization.

Building a custom plugin
========================

#. Copy the reference headers from ``plugins/env/example/nccl/`` into your
   project, keeping the ``nccl/`` directory next to your source file. They are
   self-contained, so no other RCCL headers are needed. If you put them
   somewhere else, add ``-I<parent-of-nccl>`` to the compile command below.

#. Implement the ``ncclEnv_v2_t`` interface and export ``ncclEnvPlugin_v2``:

   .. code-block:: c

      #include <stdlib.h>   /* getenv */
      #include "nccl/env.h"

      static ncclResult_t myInit(uint8_t major, uint8_t minor,
                                 uint8_t patch, const char* suffix,
                                 ncclDebugLogger_t logFunction) {
        /* open your configuration source here */
        return ncclSuccess;
      }

      static ncclResult_t myFinalize(void) {
        /* release resources here */
        return ncclSuccess;
      }

      static const char* myGetEnv(const char* name) {
        /* look up name in your configuration source;
           return NULL if not found */
        return getenv(name); /* minimal fallback */
      }

      const ncclEnv_v2_t ncclEnvPlugin_v2 = {
        .name     = "myPlugin",
        .init     = myInit,
        .finalize = myFinalize,
        .getEnv   = myGetEnv,
      };

#. Build as a shared library:

   .. code-block:: shell

      cc -shared -fPIC -o librccl-env-myplugin.so plugin.c

#. Set ``NCCL_ENV_PLUGIN`` to the absolute path of the resulting ``.so`` file.

Fallback behavior
=================

If ``NCCL_ENV_PLUGIN`` is unset, set to ``none``, or points to a library that
cannot be loaded (for example, wrong path or neither ``ncclEnvPlugin_v2`` nor
``ncclEnvPlugin_v1`` exported),
RCCL silently falls back to reading parameters directly from the process
environment via ``getenv()``. No error is raised and the application continues
to run normally.

Setting ``NCCL_ENV_PLUGIN=none`` is always a safe rollback path.

Troubleshooting
===============

Set ``NCCL_DEBUG=INFO`` and ``NCCL_DEBUG_SUBSYS=ENV,INIT`` to see plugin load
messages:

.. code-block:: shell

   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ENV,INIT
   export NCCL_ENV_PLUGIN=/path/to/librccl-env-json.so

Expected output on successful load:

.. code-block:: text

   NCCL INFO NCCL_ENV_PLUGIN set by environment to /path/to/librccl-env-json.so
   NCCL INFO ENV/Plugin: Using ncclEnvJson (v2)
   NCCL INFO Successfully loaded external env plugin /path/to/librccl-env-json.so
   NCCL INFO ENV/Plugin: ncclEnvJson loaded 2 entries from /etc/rccl/config.json

If the plugin fails to load (wrong path, or neither plugin symbol exported):

.. code-block:: text

   NCCL INFO NCCL_ENV_PLUGIN set by environment to /path/to/librccl-env-json.so
   NCCL INFO External env plugin /path/to/librccl-env-json.so is unsupported

If the plugin loads but its JSON file is missing, unreadable, or malformed, the
plugin stays active and serves every lookup from ``getenv()``:

.. code-block:: text

   NCCL WARN ENV/Plugin: ncclEnvJson could not read /etc/rccl/config.json, using getenv()
