.. meta::
   :description: ROCm Systems Profiler hipFile GPU-direct storage I/O telemetry
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, how to, profiler, hipFile, GPU-direct storage, GDS, I/O, telemetry, AMD

********************************************
hipFile GPU-direct storage I/O telemetry
********************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_
can collect GPU-direct storage I/O statistics from applications that use
`hipFile <https://github.com/ROCm/hipFile>`_. A background sampler periodically
queries hipFile's in-process statistics API and reports the results as per-GPU
counter tracks in both the Perfetto trace and the RocPD database.

Metrics collected
==================

All hipFile telemetry is per GPU, reported under tracks named
``hipFile GPU<N> <metric>``:

* **Read Bytes** / **Write Bytes** -- cumulative bytes transferred (``bytes``)
* **Read Ops** / **Write Ops** -- cumulative read/write operations (``count``)
* **Fastpath Reads** / **Fastpath Writes** -- operations that used hipFile's
  GPU-direct fast path
* **Fallback Reads** / **Fallback Writes** -- operations that used the POSIX
  fallback path
* **Unaligned Reads** / **Unaligned Writes** -- operations that were not aligned
* **Read Errors** / **Write Errors** -- failed operations
* **Read Bandwidth** / **Write Bandwidth** -- transfer rate over the sampling
  interval (``bytes/s``)

Interpreting the values
-----------------------

The counters are **cumulative** totals over the process lifetime, matching every
other byte and operation counter the profiler reports (the AMD SMI PCIe bandwidth
accumulator, the XGMI accumulators, and the AI NIC counters). To see activity per
sampling window rather than the running total, switch the counter track to its
delta view in the Perfetto UI.

The bandwidth metrics are normalized to **wall-clock time** over the sampling
interval, so they are directly comparable to the AMD SMI instantaneous PCIe
bandwidth track on the same GPU. They are not derived from the time spent inside
hipFile I/O calls, which would produce a rate that does not correspond to the
timeline it is drawn against. The first sample of a run reports zero bandwidth,
because no interval has elapsed yet.

Note that hipFile also maintains process-scoped counters, such as file handle and
buffer registrations. These are not GPU measurements, so they are not collected;
attributing them to a specific GPU track would misreport them on multi-GPU runs.

Requirements
============

* A ROCm release that includes hipFile (ROCm 7.15 or later), with the hipFile
  runtime and development packages installed.
* ROCm Systems Profiler built with hipFile support (see `Build support`_).
* A target application that links and uses hipFile. The statistics API is
  in-process, so telemetry is only produced when the profiled application
  actually performs hipFile I/O.
* A Linux kernel version 5.3 or later (required by hipFile's statistics server).

Build support
=============

hipFile support is optional and controlled at build time by the
``ROCPROFSYS_USE_HIPFILE`` CMake option. Configure the build with:

.. code-block:: shell

   cmake -D ROCPROFSYS_USE_HIPFILE=ON <other options> <path/to/source>

If the hipFile package cannot be found, the option is disabled automatically and
the rest of the profiler builds normally. To point CMake at a specific hipFile
installation, pass ``-Dhipfile_DIR=<prefix>/lib/cmake/hipfile`` or add the
installation prefix to ``CMAKE_PREFIX_PATH``.

Enabling collection at run time
===============================

hipFile telemetry is disabled by default at run time. Enable it by setting
``ROCPROFSYS_USE_HIPFILE=ON``. Collection runs on the process-sampling thread, so
process sampling must also be enabled (it is on by default).

.. code-block:: shell

   ROCPROFSYS_USE_HIPFILE=ON
   ROCPROFSYS_USE_PROCESS_SAMPLING=ON
   ROCPROFSYS_PROCESS_SAMPLING_FREQ=100

Details of the settings:

* **ROCPROFSYS_USE_HIPFILE**: Enables the hipFile telemetry sampler.
* **ROCPROFSYS_USE_PROCESS_SAMPLING**: Enables the background sampling thread
  that drives the hipFile sampler (default ``ON``).
* **ROCPROFSYS_PROCESS_SAMPLING_FREQ**: Samples per second. A higher frequency
  captures short-lived I/O bursts more precisely.
* **ROCPROFSYS_HIPFILE_METRICS**: Which hipFile metrics to collect. Accepts
  ``all`` (the default), ``none``, or a comma-separated list of metric keys such
  as ``read_bytes,write_bytes,read_bandwidth``.
* **ROCPROFSYS_SAMPLING_GPUS**: Which GPUs to collect from. hipFile telemetry
  honors the same GPU selection as the AMD SMI GPU metrics.

When hipFile telemetry is enabled, the profiler sets ``HIPFILE_STATS_LEVEL=1``
during configuration so that hipFile starts its statistics server. If you set
``HIPFILE_STATS_LEVEL`` explicitly, your value is preserved.

Running the profiler
====================

Run the target application under ``rocprof-sys-run`` (or ``rocprof-sys-sample``)
with the settings above. For example:

.. code-block:: shell

   ROCPROFSYS_USE_HIPFILE=ON ROCPROFSYS_USE_PROCESS_SAMPLING=ON \
     rocprof-sys-run -- ./my_hipfile_app --input data.bin

You can also place the settings in a configuration file and point to it with
``ROCPROFSYS_CONFIG_FILE``:

.. code-block:: shell

   ROCPROFSYS_USE_HIPFILE=ON
   ROCPROFSYS_USE_PROCESS_SAMPLING=ON
   ROCPROFSYS_PROCESS_SAMPLING_FREQ=100
   ROCPROFSYS_TRACE=ON

Visualize the results in Perfetto
=================================

To view the ``.proto`` file generated by the profiler:

#. Open the `Perfetto UI page <https://ui.perfetto.dev/>`_.
#. Click ``Open trace file`` and select the ``.proto`` file. The hipFile
   counter tracks appear as ``hipFile GPU<N> <metric>``.

Save the profiling output to RocPD
==================================

To save the output to RocPD, set ``ROCPROFSYS_USE_ROCPD=ON``:

.. code-block:: shell

   export ROCPROFSYS_USE_ROCPD=ON

Running the profiler then produces a ``.db`` file. The hipFile metrics are stored
as PMC descriptions (``rocpd_info_pmc``) and counter events
(``rocpd_pmc_event``). You can view the generated file in
`ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_.

Troubleshooting
===============

* **No hipFile tracks in the output**: Confirm that the profiler was built with
  ``ROCPROFSYS_USE_HIPFILE=ON``, that ``ROCPROFSYS_USE_HIPFILE=ON`` is set at run
  time, and that the target application actually performs hipFile I/O. If the
  application never calls into hipFile, no statistics are produced.
* **Counters are zero**: Verify that process sampling is enabled and that the
  workload runs long enough to be sampled at least once during active I/O.
* **A counter track looks flat**: The counters are cumulative, so a track that
  stops climbing means I/O stopped, not that collection failed. Use the delta
  view to see per-window activity.
* **All operations are on the fallback path**: hipFile's fast path requires a
  supported filesystem (ext4 with ordered journaling, or xfs) and ``O_DIRECT``.
  When those conditions are not met, hipFile transparently uses POSIX I/O and
  the **Fallback** counters climb while the **Fastpath** counters stay at zero.
