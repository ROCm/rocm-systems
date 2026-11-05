.. meta::
   :description: How to profile RCCL collectives and point-to-point operations with the RCCL Inspector plugin
   :keywords: RCCL, ROCm, library, Inspector, profiler, plugin, Prometheus, Grafana, telemetry

.. _using-rccl-inspector-plugin:

********************************
Using the RCCL Inspector plugin
********************************

The RCCL Inspector is a profiler plugin that records per-communicator,
per-operation performance data for collectives and point-to-point (P2P)
operations. Prometheus textfile mode (``NCCL_INSPECTOR_PROM_DUMP=1``) shipped
with that same port. It attaches to RCCL through the profiler plugin
interface, so it requires no application changes: load the plugin, set a few
environment variables, and run the workload as usual.

The Inspector writes either of the following:

*  Structured JSON, one record per operation per communicator, for offline
   analysis.
*  Prometheus textfile metrics, a bounded snapshot suitable for scraping with
   the node exporter textfile collector and plotting in Grafana.

The plugin source lives in ``plugins/profiler/inspector``. For the complete
list of environment variables, see the "Inspector profiling" section of
:doc:`../api-reference/env-variables`.

Building the plugin
===================

The plugin builds as part of the RCCL CMake build. The
``BUILD_PROFILER_INSPECTOR`` option controls it and defaults to the value of
``BUILD_TESTS``:

.. code-block:: shell

    cmake -B build -DBUILD_PROFILER_INSPECTOR=ON .
    cmake --build build

The result is ``build/lib/librccl-profiler-inspector.so``.

The plugin can also be built on its own, outside of an RCCL build tree:

.. code-block:: shell

    cd plugins/profiler/inspector
    cmake -B build . && cmake --build build

Pass ``-DNCCL_INSPECTOR_ENABLE_WARN=ON`` at configuration time to raise the
plugin's own log messages from ``INFO`` to ``WARN`` level.

Enabling the Inspector
======================

Point ``NCCL_PROFILER_PLUGIN`` at the shared library and set
``NCCL_INSPECTOR_ENABLE=1``. Both variables are required. The plugin is inert
unless ``NCCL_INSPECTOR_ENABLE=1``, so a job script can keep the plugin path
set permanently and turn profiling on per run.

.. code-block:: shell

    export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-inspector.so
    export NCCL_INSPECTOR_ENABLE=1
    export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=500
    ./your_rccl_application

Point-to-point tracking is enabled by default
(``NCCL_INSPECTOR_ENABLE_P2P=1``) and is required for the ``nccl_p2p_*``
metrics and the P2P panels of the Grafana dashboard. Set it to ``0`` to record
collectives only.

By default only operations of at least 8192 bytes are recorded. Lower
``NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES`` to capture smaller messages, at the cost
of more output volume.

Only events with GPU-based kernel timing are kept by default. If a workload
reports no events at all, the timing source is likely falling back to
CPU measurement; set ``NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0`` to retain those
events.

On 8-rank intra-node AllReduce, DDA can claim the collective so the Inspector
records nothing. Set ``RCCL_DDA_ENABLE=0`` when you need Inspector coverage of
that path.

Choosing the output directory
-----------------------------

Set ``NCCL_INSPECTOR_DUMP_DIR`` to control where output lands. If it is unset,
the Inspector creates ``nccl-inspector-<jobid>`` from ``SLURM_JOB_ID``,
``SLURM_JOBID``, ``PBS_JOBID``, or ``LSB_JOBID``, or
``nccl-inspector-unknown-jobid`` when none of those are set.

.. code-block:: shell

    export NCCL_INSPECTOR_DUMP_DIR=/path/to/logs/${SLURM_JOB_ID}/
    srun your_rccl_application

Controlling when output is written
----------------------------------

``NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS`` sets the period of the
internal dump thread:

*  ``-1`` (default) writes only at communicator teardown.
*  ``0`` writes continuously.
*  A positive value writes every *N* microseconds.

In Prometheus mode a minimum of 30 seconds (``30000000``) is enforced to match
the node exporter polling interval, so a workload must run for longer than
30 seconds for a periodic dump to land.

JSON output
===========

JSON is the default format. Each record covers one operation on one
communicator:

.. code-block:: json

    {
      "header": {
        "id": "0x7f8c496ae9f661",
        "comm_name": "DP Group 0",
        "rank": 2,
        "n_ranks": 8,
        "nnodes": 1
      },
      "metadata": {
        "inspector_output_format_version": "v4.0",
        "git_rev": "9019a1912-dirty",
        "rec_mechanism": "nccl_profiler_interface",
        "dump_timestamp_us": 1748030377748202,
        "hostname": "example-hostname",
        "pid": 1639453
      },
      "coll_perf": {
        "coll": "AllReduce",
        "coll_sn": 1407,
        "coll_msg_size_bytes": 17179869184,
        "coll_exec_time_us": 61974,
        "coll_timing_source": "kernel_gpu",
        "coll_algobw_gbs": 277.210914,
        "coll_busbw_gbs": 485.119099
      }
    }

The ``header`` object identifies the communicator the record belongs to, and
the ``metadata`` object identifies the process and the moment the record was
written:

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Field**
      - **Description**

    * - ``id``
      - Communicator hash, unique per communicator.

    * - ``comm_name``
      - Application-assigned communicator name, or ``unknown`` if unset.

    * - ``rank``, ``n_ranks``
      - This process's rank within the communicator, and the communicator size.

    * - ``nnodes``
      - Number of nodes spanned by the communicator. ``1`` means the
        communicator is intra-node.

    * - ``inspector_output_format_version``
      - JSON schema version of the record (currently ``v4.0``). Distinct from
        the Prometheus ``version="v5.1"`` label.

    * - ``git_rev``
      - Git revision the Inspector plugin was built from.

    * - ``rec_mechanism``
      - How the data was recorded. Always the NCCL ABI name
        ``nccl_profiler_interface`` (the plugin still profiles RCCL).

    * - ``dump_timestamp_us``
      - Wall-clock time the record was written, in microseconds since the
        epoch.

    * - ``hostname``, ``pid``
      - Host and process that produced the record, for correlating records
        across ranks.

A ``coll_perf`` object describes one completed collective. Point-to-point
records use a ``p2p_perf`` object with the same fields under a ``p2p_``
prefix, plus ``p2p_peer``:

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Field**
      - **Description**

    * - ``coll`` / ``p2p``
      - Operation name, for example ``AllReduce``, ``Send``, or ``Recv``.

    * - ``coll_sn`` / ``p2p_sn``
      - Sequence number of the operation on this communicator, counting from
        the first operation recorded. Use it to order records and to spot
        gaps where operations were filtered out.

    * - ``p2p_peer``
      - Rank this operation exchanged data with. Point-to-point records only.

    * - ``coll_msg_size_bytes`` / ``p2p_msg_size_bytes``
      - Message size of the operation, in bytes.

    * - ``coll_exec_time_us`` / ``p2p_exec_time_us``
      - Execution time of the operation, in microseconds.

    * - ``coll_timing_source`` / ``p2p_timing_source``
      - Where the time came from: ``kernel_gpu`` for GPU kernel timing, or
        ``kernel_cpu`` / ``collective_cpu`` for CPU-measured fallbacks. Only
        ``kernel_gpu`` events are kept unless
        ``NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0``.

    * - ``coll_algobw_gbs`` / ``p2p_algobw_gbs``
      - Algorithm bandwidth in GB/s: message size divided by execution time.

    * - ``coll_busbw_gbs`` / ``p2p_busbw_gbs``
      - Bus bandwidth in GB/s: algorithm bandwidth scaled by the operation's
        communication pattern, so it can be compared against the hardware
        peak.

Setting ``NCCL_INSPECTOR_DUMP_VERBOSE=1`` adds ``event_trace_sn`` and
``event_trace_ts`` blocks with the per-channel kernel sequence numbers and
timestamps behind each measurement.

JSON files grow for the lifetime of the application, at roughly 200 to 500
bytes per recorded operation per communicator. Budget accordingly for
long-running jobs, or raise ``NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES``.

Prometheus output
=================

Set ``NCCL_INSPECTOR_PROM_DUMP=1`` to emit node exporter textfile metrics
(``nccl_inspector_metrics_<device_uuid>.prom``) instead of JSON. Write them
into the directory watched by the node exporter textfile collector:

.. code-block:: shell

    export NCCL_PROFILER_PLUGIN=/path/to/librccl-profiler-inspector.so
    export NCCL_INSPECTOR_ENABLE=1
    export NCCL_INSPECTOR_PROM_DUMP=1
    export NCCL_INSPECTOR_DUMP_DIR=/var/lib/node_exporter/nccl_inspector/
    export NCCL_INSPECTOR_DUMP_THREAD_INTERVAL_MICROSECONDS=30000000

Unlike the JSON logs, ``.prom`` files are rewritten in place at each dump, so
their size is bounded by the number of communicators sharing a device (roughly
500 to 1000 bytes per communicator per metric). The files are removed when the
communicator is destroyed.

Each file starts with ``# HELP`` / ``# TYPE gauge`` for these four series.

.. list-table::
    :header-rows: 1
    :widths: 45,55

    * - **Metric**
      - **Description**

    * - ``nccl_bus_bandwidth_gbs``
      - Collective bus bandwidth, in GB/s.

    * - ``nccl_collective_exec_time_microseconds``
      - Collective execution time, in microseconds.

    * - ``nccl_p2p_bus_bandwidth_gbs``
      - Point-to-point bus bandwidth, in GB/s. Requires
        ``NCCL_INSPECTOR_ENABLE_P2P=1``.

    * - ``nccl_p2p_exec_time_microseconds``
      - Point-to-point execution time, in microseconds. Requires
        ``NCCL_INSPECTOR_ENABLE_P2P=1``.

All samples carry the ``version``, ``cluster``, ``slurm_job_id``, ``node``,
``gpu``, ``comm_name``, ``n_nodes``, ``nranks``, and ``message_size`` labels,
where ``message_size`` is a bucketed range string such as ``8-9MB``. Collective
samples add ``collective`` and ``algo_proto``; P2P samples add
``p2p_operation``, which is either ``Send`` or ``Recv``. ``cluster`` comes from
``NCCL_INSPECTOR_CLUSTER`` or ``SLURM_CLUSTER_NAME`` (else ``unknown``).
``slurm_job_id`` comes from ``SLURM_JOB_ID``, then ``SLURM_JOBID``,
``PBS_JOBID``, ``LSB_JOBID``, else ``unknown``.

NCCL / RCCL mapping for Prometheus
----------------------------------

.. list-table::
    :header-rows: 1
    :widths: 28,28,44

    * - **NCCL header**
      - **RCCL equivalent**
      - **How to read it**

    * - ``nccl_bus_bandwidth_gbs``
      - RCCL collective bus bandwidth
      - GPU-kernel bus BW in GB/s. Can differ from rccl-tests host-timer bus BW.

    * - ``nccl_collective_exec_time_microseconds``
      - RCCL collective kernel time
      - Microseconds of GPU kernel time.

    * - ``nccl_p2p_*``
      - RCCL Send/Recv
      - HELP/TYPE are always written. Samples appear only when P2P ops were recorded.

    * - ``version``
      - Inspector Prometheus format
      - ``v5.1`` is profiler-interface major + Inspector format minor, **not** the RCCL version.

    * - ``gpu``
      - HIP device index
      - ``GPU3`` means ``hipGetDevice() == 3``, remapped by ``HIP_VISIBLE_DEVICES``.

    * - ``cluster``
      - Site name
      - ``NCCL_INSPECTOR_CLUSTER``, else ``SLURM_CLUSTER_NAME``, else ``unknown``.
        Set the override when Slurm's name is the unhelpful literal ``cluster``.

    * - ``slurm_job_id``
      - Scheduler job id
      - ``unknown`` on standalone ``mpirun`` (no SLURM/PBS/LSF). Grafana **All** still matches.

    * - ``algo_proto``
      - RCCL algorithm/protocol
      - For example ``RING_LL``, ``TREE_LL128``. Intra-node views are xGMI, not NVLink.

.. code-block:: text

    # HELP nccl_bus_bandwidth_gbs RCCL collective bus bandwidth in GB/s
    # TYPE nccl_bus_bandwidth_gbs gauge
    # HELP nccl_collective_exec_time_microseconds RCCL collective execution time in microseconds
    # TYPE nccl_collective_exec_time_microseconds gauge
    # HELP nccl_p2p_bus_bandwidth_gbs RCCL point-to-point bus bandwidth in GB/s
    # TYPE nccl_p2p_bus_bandwidth_gbs gauge
    # HELP nccl_p2p_exec_time_microseconds RCCL point-to-point execution time in microseconds
    # TYPE nccl_p2p_exec_time_microseconds gauge
    nccl_bus_bandwidth_gbs{version="v5.1",cluster="alola",slurm_job_id="67941748",node="node001",gpu="GPU0",comm_name="unknown",n_nodes="2",nranks="16",collective="AllReduce",message_size="1-2MB",algo_proto="TREE_LL128"} 34.2887
    nccl_collective_exec_time_microseconds{version="v5.1",cluster="alola",slurm_job_id="67941748",node="node001",gpu="GPU0",comm_name="unknown",n_nodes="2",nranks="16",collective="AllReduce",message_size="1-2MB",algo_proto="TREE_LL128"} 58.8652

Visualizing the metrics in Grafana
==================================

``plugins/profiler/inspector/grafana`` ships a dashboard template,
``nccl-inspector-job-performance-template.json``, with rows for P2P ``Send``
and ``Recv`` and for ReduceScatter, AllReduce, and AllGather. Each row splits
into an intra-node view (``n_nodes="1"``) and a network view
(``n_nodes!="1"``).

Import the JSON from **Dashboards**, **New**, **Import**, then set
**Prometheus**, **Cluster**, and **Job ID** from top to bottom. Those
variables come from Inspector labels (``cluster``, ``slurm_job_id``);
there is no MySQL or placeholder job metric. The ``README.md`` file in
the same directory describes each variable and the panel queries.

Debugging the plugin
====================

The Inspector logs through the RCCL debug subsystem ``PROFILE``, which reports
plugin initialization, the resolved configuration, the dump thread state, and
the paths of the files it creates:

.. code-block:: shell

    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=PROFILE

If no output appears at all, check the following:

*  ``NCCL_INSPECTOR_ENABLE`` is set to ``1``, and ``NCCL_PROFILER_PLUGIN``
   resolves to a readable ``librccl-profiler-inspector.so``.
*  The operations in the workload are at least
   ``NCCL_INSPECTOR_DUMP_MIN_SIZE_BYTES`` in size.
*  DDA did not claim the collective. On 8-rank intra-node AllReduce set
   ``RCCL_DDA_ENABLE=0``.
*  Events have GPU kernel timing, or ``NCCL_INSPECTOR_REQUIRE_KERNEL_TIMING=0``.
*  In Prometheus mode, the workload ran for longer than the enforced 30 second
   dump interval, or the process reached communicator teardown.
