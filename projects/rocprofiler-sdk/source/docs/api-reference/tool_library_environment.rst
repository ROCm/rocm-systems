
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
At least one of the following mechanisms must be configured *before* the first
ROCm runtime call so that registration can discover the tool. With
``ROCP_TOOL_LIBRARIES`` the library is ``dlopen``\ed by the SDK during
registration (triggered by the first ROCm runtime call); with ``LD_PRELOAD`` the
library is mapped by the dynamic loader at process start.

.. list-table::
    :header-rows: 1
    :widths: 25 15 60

    * - Variable
      - Default
      - Description
    * - ``ROCP_TOOL_LIBRARIES``
      - (unset)
      - Colon-separated list of absolute paths to shared libraries that export
        ``rocprofiler_configure``. ROCprofiler-SDK ``dlopen``\s each entry and
        invokes its ``rocprofiler_configure`` symbol during registration. This
        is the recommended way to load a custom tool that is not already in the
        process's link map.

        .. warning::

            The current parser splits on ``:`` but drops the token after the
            **last** ``:`` separator. Until this is fixed, prefer a single path,
            or append a trailing ``:`` (for example
            ``/path/libA.so:/path/libB.so:``) to ensure every entry is loaded.
    * - ``LD_PRELOAD``
      - (unset)
      - Standard dynamic-loader mechanism. If your tool library is listed in
        ``LD_PRELOAD``, it is loaded before any runtime, and its
        ``rocprofiler_configure`` symbol is discovered automatically (without
        needing ``ROCP_TOOL_LIBRARIES``). Use this when the tool must intercept
        symbols or perform work before the runtime initializes.

.. note::

    ``ROCP_TOOL_LIBRARIES`` failure modes differ:

    * If an entry cannot be ``dlopen``\ed (for example, the file does not exist
      or has unresolved dependencies), the SDK calls ``ROCP_FATAL`` and the
      process terminates.
    * If an entry is loaded but does not export ``rocprofiler_configure``, a
      warning is logged and that library is simply not registered as a tool;
      the process continues.

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

        * Named levels: ``trace``, ``info``, ``warning``, ``error``, ``fatal``.
        * Integer levels ``0``–``4`` mapping to the named levels above
          (``0`` = ``fatal``, ``4`` = ``trace``).

        Log messages are emitted to ``stderr``.

Queue interception behavior
---------------------------

This variable controls how ROCprofiler-SDK intercepts HSA queue operations. It
is an advanced knob: most tools do not need to set it, and the default is chosen
automatically based on the services your tool enables.

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_QUEUE_INTERPOSITION``
      - (auto)
      - Boolean (``true``/``false``). Selects between *inline* queue interposition
        (a shadow write-pointer that intercepts queue operations without the
        legacy intercept queue) and the legacy
        ``hsa_amd_queue_intercept_create`` path.

        When unset, the SDK chooses automatically: inline interposition is
        enabled only when **no** registered context requires dispatch counter
        collection, thread trace (ATT), or PC sampling; otherwise the legacy
        path is used.

        If you set this to ``true`` while a context that requires the legacy
        path (counter collection, ATT, or PC sampling) is registered, the SDK
        logs a warning and falls back to the legacy path anyway.

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
