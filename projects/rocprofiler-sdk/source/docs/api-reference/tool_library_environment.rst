
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

Kernel dispatch timestamp source
--------------------------------

.. list-table::
    :header-rows: 1
    :widths: 30 15 55

    * - Variable
      - Default
      - Description
    * - ``ROCPROFILER_KFD_DISPATCH_LOG_SIZE_KB``
      - ``10240``
      - Size in KiB of the firmware dispatch-log ring, as an integer from ``1`` to
        ``4194303``. Defaults to 10 MiB. A non-integer, empty, zero, or
        out-of-range value is ignored with a warning and the default is used. If
        the firmware laps the reader the SDK logs an overrun warning and the
        affected signal-less dispatches emit no record (their firmware records were
        overwritten before the reader copied them); raising this value gives the
        reader more headroom.

        The driver only accepts ring sizes of ``80 * 2^k`` bytes (80 KiB, 160 KiB,
        320 KiB, ... up to the 640 MiB maximum), so the requested size is rounded
        DOWN to the nearest accepted size, and a size below 80 KiB or above
        640 MiB is clamped. The effective size is logged whenever it differs from
        the requested one.
    * - ``ROCPROFILER_KFD_DISPATCH_LOG_POLL_TIMEOUT_MS``
      - ``10``
      - Timeout in milliseconds, as an integer from ``1`` to ``2147483647``, that
        the dispatch-log reader passes to ``poll()``. The reader wakes on a
        firmware notification and drains; if none arrives it wakes on this timeout
        and does a full scan anyway, so this value bounds how long a sparse or
        lost-interrupt tail can sit undrained. It is the only timer -- there is no
        separate watchdog thread. A non-integer, empty, zero, or out-of-range
        value is ignored with a warning and the default is used. Lowering it
        shortens sparse-tail latency at the cost of more idle wakeups; raising it
        does the reverse.

    * - ``ROCPROFILER_KFD_DISPATCH_LOG_CLOSE_DRAIN_MS``
      - ``250``
      - Internal/advanced. Per-queue close-drain budget in milliseconds: how long
        ``destroy_queue`` waits for the hardware to finish a signal-less queue's
        in-flight dispatches before it stops draining and closes the queue. A
        negative value is clamped to ``0``. This is a per-queue budget only; there
        is no aggregate teardown ceiling. Most tools should leave this at the
        default.

    * - ``ROCPROFILER_KFD_DISPATCH_LOG_SIGNAL_LESS``
      - ``false``
      - Boolean. The master switch for the entire KFD dispatch-log feature. When
        unset (the default) the SDK does not probe the KFD dispatch-log interface
        and every dispatch uses ``hsa_amd_profiling_get_dispatch_time``. When set,
        the SDK probes the interface and, on GPUs that support it, both reports
        kernel dispatch ``start_timestamp``/``end_timestamp`` from the firmware
        dispatch log (taken at the true hardware dispatch boundaries, so tighter
        than the HSA signal-based interval) and opts in to signal-less
        completion.

        Under signal-less completion, a dispatch batch that qualifies is published
        with its AQL packet **untouched** -- the SDK allocates no completion signal
        and does not modify the application's -- and the dispatch completes from the
        firmware dispatch-log record instead of a signal. A batch that does not
        qualify keeps the signal path, so the two coexist. Firmware timestamps
        require inline queue interposition, so enabling a service that only the
        legacy interception path supports (counter collection, advanced thread
        trace, or PC sampling) puts every dispatch on the HSA timestamps.

        A batch qualifies only when the dispatch log is live for that GPU, the
        reader is healthy, and every packet's doorbell slot has exactly one live
        owning queue; a doorbell collision (two live queues sharing a slot) retires
        that slot to the signal path. A queue destroy does **not** retire the slot:
        a later queue that reuses the doorbell opens a fresh owner window and its
        records are attributed by dispatch time, so a queue-churning workload (for
        example a HIP stream pool) keeps using signal-less. If the firmware ring
        overruns, an end-of-pipe record observed under the overrun cannot be trusted
        to belong to any specific dispatch, so it is not attributed: those
        dispatches emit no firmware record, and a warning names the counts.
        Signal-less is **not** disabled -- later, loss-free dispatches keep using
        it. As a current limitation, such un-attributed dispatches keep their
        correlation-id references until process teardown rather than being retired
        eagerly.

        One limitation is inherent to the doorbell-slot design: attribution
        requires that no two live queues on the same GPU share a page-relative
        doorbell slot at the same time. The doorbell allocator makes this rare, but
        if it does occur the slot is treated as ambiguous and its dispatches fall
        back to the signal path rather than risk a misattribution.

        **Experimental, and gfx950-only today.** This feature and its
        ``ROCPROFILER_KFD_DISPATCH_LOG_*`` variables are experimental and may change
        or be replaced by context/service API options. Signal-less completion is
        validated and enabled only on gfx950 (MI350), the one SKU whose GPU clock
        domain has passed the per-SKU T-CLK screen; on every other GPU those
        dispatches use the HSA signal path regardless of this setting.

        Disabled by default. Enabling it changes when dispatch records are
        delivered relative to the application observing its own completion
        signal, so it is opt-in.

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
