
.. meta::
    :description: Environment variables relevant to authors of custom ROCprofiler-SDK tools
    :keywords: ROCprofiler-SDK, custom tool, environment variables, ROCP_TOOL_LIBRARIES, ROCPROFILER_REGISTER_LIBRARY, logging

.. _tool-library-environment:

Environment variables for custom tools
======================================

This page documents the environment variables that affect a **custom tool** built
against ROCprofiler-SDK — that is, a shared library that implements
``rocprofiler_configure`` (see :ref:`tool-library`). It does **not** cover the
``ROCPROF_*`` variables consumed by ``rocprofv3``; those are documented separately
in the rocprofv3 how-to guides.

Variables are grouped by the role they play in a custom tool's lifecycle.

Tool discovery and loading
--------------------------

These variables control how ROCprofiler-SDK locates and loads your tool library.
At least one of the following mechanisms must place your library into the target
process's address space *before* the first ROCm runtime call:

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Variable
      - Default
      - Description
    * - ``ROCP_TOOL_LIBRARIES``
      - (unset)
      - Colon-separated list of absolute paths to shared libraries that export
        ``rocprofiler_configure``. ROCprofiler-SDK ``dlopen``\ s each entry and
        invokes its ``rocprofiler_configure`` symbol during registration. This is
        the recommended way to load a custom tool that is not already in the
        process's link map.
    * - ``LD_PRELOAD``
      - (unset)
      - Standard dynamic-loader mechanism. If your tool library is listed in
        ``LD_PRELOAD``, it is loaded before any runtime, and its
        ``rocprofiler_configure`` symbol is discovered automatically (without
        needing ``ROCP_TOOL_LIBRARIES``). Use this when the tool must intercept
        symbols or perform work before the runtime initializes.
    * - ``ROCPROFILER_OPTIMIZE_FIND_CLIENTS``
      - ``true``
      - When ``true``, ROCprofiler-SDK uses ELF parsing to quickly determine
        whether a library exports ``rocprofiler_configure`` before ``dlopen``\ ing
        it. Set to ``false`` only if the optimized scan produces incorrect results
        for an unusual library layout.

.. note::

    ``ROCP_TOOL_LIBRARIES`` entries that do not exist on disk or do not export
    ``rocprofiler_configure`` are skipped with a warning logged via
    ``ROCPROFILER_LOG_LEVEL=warning``.

rocprofiler-register integration
--------------------------------

The ``rocprofiler-register`` library is the indirection layer that the ROCm
runtimes consult on initialization to decide whether ROCprofiler-SDK services
must be enabled. The SDK sets these variables in its constructor so that all
runtimes loaded in the same process use the same SDK instance.

.. list-table::
    :header-rows: 1
    :widths: 35 15 50

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_SET_ROCPROFILER_REGISTER_LIBRARY``
      - ``true``
      - When ``true``, ROCprofiler-SDK sets ``ROCPROFILER_REGISTER_LIBRARY`` to
        its own resolved library path during initialization. Set to ``false`` to
        opt out (for example, when an external launcher already provides the
        correct value).
    * - ``ROCPROFILER_REGISTER_LIBRARY``
      - (auto-set)
      - Absolute path of the ``librocprofiler-sdk.so`` instance that
        ``rocprofiler-register`` should bind against. Setting this manually is
        only required in multi-install environments where the SDK cannot be
        located via the standard library search path.
    * - ``ROCPROFILER_FORCE_ROCPROFILER_REGISTER_LIBRARY``
      - ``false``
      - When ``true``, ROCprofiler-SDK overwrites any pre-existing
        ``ROCPROFILER_REGISTER_LIBRARY`` value with its own path. By default an
        existing value is preserved and a warning is logged on mismatch.

Logging and diagnostics
-----------------------

These variables are the first thing to reach for when debugging a custom tool
that is not being loaded, not being initialized, or not receiving callbacks.

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_LOG_LEVEL``
      - ``warning``
      - Severity threshold for SDK log output. Accepted values:
        ``trace``, ``info``, ``warning``, ``error``, ``fatal``. A negative integer
        (for example, ``-1``) enables verbose ``vmodule`` output at that level.
    * - ``ROCPROFILER_LOG_DIR``
      - (stderr)
      - Directory to write log files into. When unset, log messages are written
        to ``stderr``.
    * - ``ROCPROFILER_vmodule``
      - (unset)
      - glog-style per-module verbose specification, for example
        ``registration=2,agent=1``. Requires ``ROCPROFILER_LOG_LEVEL`` to be set
        to a negative integer.
    * - ``ROCPROFILER_LIBRARY_CTOR``
      - ``false``
      - When ``true``, prints a diagnostic message from the SDK shared-library
        constructor. Useful for verifying that ``librocprofiler-sdk.so`` is being
        loaded into the target process at all.
    * - ``ROCPROFILER_LIBRARY_DTOR``
      - ``false``
      - When ``true``, prints a diagnostic message from the SDK shared-library
        destructor. Useful for confirming that the SDK is being unloaded cleanly
        at process exit.

Beta-feature opt-in
-------------------

Tools that use beta services must enable them explicitly. Without these
variables, the corresponding configuration calls return an error.

.. list-table::
    :header-rows: 1
    :widths: 35 15 50

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_PC_SAMPLING_BETA_ENABLED``
      - ``false``
      - Enables the PC sampling service. Required to call
        ``rocprofiler_configure_pc_sampling_service`` from a custom tool.
    * - ``ROCPROFILER_SPM_BETA_ENABLED``
      - ``false``
      - Enables Streaming Performance Monitor (SPM) counter collection.

Topology and agent visibility
-----------------------------

These variables influence which agents the SDK exposes to your tool through the
agent-information API.

.. list-table::
    :header-rows: 1
    :widths: 28 15 57

    * - Variable
      - Default
      - Description
    * - ``ROCR_VISIBLE_DEVICES``
      - (all)
      - Standard ROCm runtime selector. Restricts the set of agents the runtime
        — and therefore the SDK — sees. Affects iteration through
        ``rocprofiler_query_available_agents``.
    * - ``HIP_VISIBLE_DEVICES`` / ``CUDA_VISIBLE_DEVICES`` / ``GPU_DEVICE_ORDINAL``
      - (all)
      - HIP-level device selectors. These affect which agents HIP exposes but
        do not directly hide agents from the SDK; tools that correlate HIP
        device ordinals with SDK agent IDs must account for the mapping.
    * - ``ROCPROFILER_FORCE_PLATFORM``
      - (auto-detect)
      - Forces a specific platform back-end (for example, ``gnulinux`` or
        ``wsl``). Intended for development and testing; production tools should
        leave this unset.
    * - ``ROCPROFILER_KFD_TOPOLOGY``
      - (auto-detect)
      - Overrides the path used to read KFD topology data. The SDK also honors
        ``AMD_KFD_TOPOLOGY`` and ``HSA_MODEL_TOPOLOGY`` for compatibility with
        other AMD tooling.
    * - ``ROCPROFILER_ONDEMAND_QUEUE``
      - ``false``
      - When ``true``, internal HSA queues used by counter and trace services
        are created on first use instead of at SDK startup. Reduces startup
        overhead for tools that may never collect on every agent.

Counter and metric configuration
--------------------------------

Relevant only to tools that use the counter-collection service.

.. list-table::
    :header-rows: 1
    :widths: 28 15 57

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_METRICS_PATH``
      - (install dir)
      - Directory containing the metric definition YAML files
        (``basic_counters.yaml``, ``derived_counters.yaml``,
        ``counter_defs.yaml``). Override to use a custom metric set or a
        development build of the metric definitions.

Minimal worked example
----------------------

Launching an application with a custom tool ``libmy_tool.so`` and verbose SDK
logging:

.. code-block:: bash

    export ROCP_TOOL_LIBRARIES=/path/to/libmy_tool.so
    export ROCPROFILER_LOG_LEVEL=info
    ./my_application

Equivalent invocation using ``LD_PRELOAD`` (useful when the tool must also
intercept symbols from the target):

.. code-block:: bash

    LD_PRELOAD=/path/to/libmy_tool.so \
        ROCPROFILER_LOG_LEVEL=info \
        ./my_application

To verify that ``librocprofiler-sdk.so`` is reaching the target process at all:

.. code-block:: bash

    ROCPROFILER_LIBRARY_CTOR=true ./my_application

See also
--------

- :ref:`tool-library` — overview of the custom tool API and ``rocprofiler_configure``.
- :ref:`process_attachment_implementation` — environment considerations when attaching to an already-running process.
