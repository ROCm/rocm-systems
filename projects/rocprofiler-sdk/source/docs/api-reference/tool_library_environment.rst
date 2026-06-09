
.. meta::
    :description: Environment variables relevant to authors of custom ROCprofiler-SDK tools
    :keywords: ROCprofiler-SDK, custom tool, environment variables, ROCP_TOOL_LIBRARIES, logging

.. _tool-library-environment:

Environment variables for custom tools
======================================

This page documents the environment variables that affect a **custom tool** built
against ROCprofiler-SDK — that is, a shared library that implements
``rocprofiler_configure`` (see :ref:`tool-library`). It does **not** cover the
``ROCPROF_*`` variables consumed by ``rocprofv3``; those are documented separately
in the rocprofv3 how-to guides.

Only variables that are part of the public custom-tool contract are listed here.
Internal SDK tuning knobs and development/test overrides are intentionally
omitted: they are subject to change without notice and should not be relied on
by external tools.

Variables are grouped by the role they play in a custom tool's lifecycle.

Tool discovery and loading
--------------------------

These variables control how ROCprofiler-SDK locates and loads your tool library.
At least one of the following mechanisms must place your library into the target
process's address space *before* the first ROCm runtime call.

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

.. note::

    ``ROCP_TOOL_LIBRARIES`` entries that do not exist on disk or do not export
    ``rocprofiler_configure`` are skipped with a warning logged at
    ``ROCPROFILER_LOG_LEVEL=warning`` or higher.

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
        ``trace``, ``info``, ``warning``, ``error``, ``fatal``. Integer values
        ``4``..``0`` are also accepted (``4``=trace, ``3``=info, ``2``=warning, ``1``=error, ``0``=fatal).
    * - ``ROCPROFILER_LOG_DIR``
      - (stderr)
      - Directory to write log files into. When unset, log messages are written
        to ``stderr``.
    * - ``ROCPROFILER_vmodule``
      - (unset)
      - glog-style per-module verbose specification, for example
        ``registration=2,agent=1``. Requires ``ROCPROFILER_LOG_LEVEL`` to be set
        to a negative integer.

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

Agent visibility
----------------

These standard ROCm runtime variables influence which agents the SDK exposes to
your tool through the agent-information API. They are not owned by
ROCprofiler-SDK, but custom tools that enumerate agents need to be aware of
them.

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

See also
--------

- :ref:`tool-library` — overview of the custom tool API and ``rocprofiler_configure``.
- :ref:`process_attachment_implementation` — environment considerations when attaching to an already-running process.
