.. meta::
   :description: ROCm Systems Profiler hipFile GPU-direct storage I/O telemetry
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, how to, profiler, hipFile, GPU-direct storage, GDS, I/O, telemetry, AMD

********************************************
hipFile GPU-direct storage I/O telemetry
********************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_
can collect GPU-direct storage I/O telemetry from applications that use
`hipFile <https://github.com/ROCm/hipFile>`_. A background sampler periodically
queries hipFile's in-process telemetry API and reports the results as per-GPU
counter tracks in both the Perfetto trace and the RocPD database.

Metrics collected
==================

All hipFile telemetry is per GPU, reported under tracks named
``hipFile GPU<N> <metric>``. Metrics are selected in groups, and each group covers
both the read and the write track. The group name is the token accepted by
``ROCPROFSYS_HIPFILE_METRICS``.

Collected by default:

* ``fastpath`` -- **Fastpath Reads** / **Fastpath Writes**, operations that took
  hipFile's GPU-direct fast path (``count``)
* ``fallback`` -- **Fallback Reads** / **Fallback Writes**, operations that fell back
  to POSIX I/O (``count``)
* ``bandwidth`` -- **Read Bandwidth** / **Write Bandwidth**, transfer rate over the
  sampling interval (``bytes/s``)

Available, but off by default:

* ``bytes`` -- **Read Bytes** / **Write Bytes**, cumulative bytes transferred
  (``bytes``)
* ``ops`` -- **Read Ops** / **Write Ops**, cumulative operation counts (``count``)
* ``unaligned`` -- **Unaligned Reads** / **Unaligned Writes**, operations that were
  not aligned (``count``)
* ``errors`` -- **Read Errors** / **Write Errors**, failed operations (``count``)

The default set answers the two questions that usually come first: whether GPU-direct
I/O is actually being used (``fastpath`` against ``fallback``) and how fast it is
(``bandwidth``). Add the remaining groups, or ``all``, when you need the full picture.

Interpreting the values
-----------------------

The counters are **cumulative** totals over the process lifetime, matching every
other byte and operation counter the profiler reports. To see activity per
sampling window rather than the running total, switch the counter track to its
delta view in the Perfetto UI.

The bandwidth metrics are normalized to **wall-clock time** over the sampling
interval, so they are directly comparable to the AMD SMI instantaneous PCIe
bandwidth track on the same GPU. They are not derived from the time spent inside
hipFile I/O calls, which would produce a rate that does not correspond to the
timeline it is drawn against. The first sample of a run reports zero bandwidth,
because no interval has elapsed yet.

When hipFile cannot be queried, usually because the application has not
initialized it yet, the sample is dropped rather than recorded as zero. A gap in
a track therefore means "not measured", while a zero means either "measured, and
no I/O occurred" or "collection was paused".

hipFile telemetry honors the same pause and resume controls as the other process
sampling metrics, including ``roctxProfilerPause`` / ``roctxProfilerResume`` and
the ROCTX region filters. Pausing writes one zeroed sample to every track so the
counters visibly drop. Because these counters are cumulative, that appears as a
dip to zero followed by a jump back to the running total on resume, rather than
the flat line an instantaneous metric such as GPU power would show. The totals
themselves are not reset, so values continue from where they left off, and the
first bandwidth sample after resuming is averaged over the whole paused span.

Requirements
============

* A ROCm release that includes **hipFile 0.5.0 or later**, with the hipFile runtime
  and development packages installed. The per-GPU statistics API this collector
  reads (``hipFileGetStatsL3``) is not present in earlier versions.
* ROCm Systems Profiler built with hipFile support (see `Build support`_).
* A target application that links and uses hipFile. The statistics API is
  in-process, so telemetry is only produced when the profiled application
  actually performs hipFile I/O.
* A Linux kernel version 5.3 or later (required by hipFile's statistics server).

Build support
=============

There are two independent switches, one for each question:

* ``ROCPROFSYS_BUILD_HIPFILE`` (CMake option) decides whether hipFile support is
  compiled into the profiler at all.
* ``ROCPROFSYS_USE_HIPFILE`` (environment variable or configuration file setting)
  decides whether a given run collects hipFile samples. See
  `Enabling collection at run time`_.

``ROCPROFSYS_BUILD_HIPFILE`` defaults to ``ON``, so hipFile support is built
automatically whenever a suitable hipFile package is present. Building it in costs
nothing at run time unless collection is also enabled. To exclude the feature from a
package entirely, configure with:

.. code-block:: shell

   cmake -D ROCPROFSYS_BUILD_HIPFILE=OFF <other options> <path/to/source>

The build requires hipFile 0.5.0 or later. A package older than that is treated the
same as no package at all: the option is disabled and the rest of the profiler builds
normally, because the per-GPU statistics API does not exist to build against. Because
the option demotes itself rather than failing, a missing or too-old hipFile never
breaks the build.

The version is also re-checked at run time, against the ``libhipfile`` actually loaded
into the process. When the loaded runtime is too old, the profiler logs a warning
naming both versions and emits no hipFile tracks, leaving the rest of the profile
unaffected.

To point CMake at a specific hipFile installation, pass
``-Dhipfile_DIR=<prefix>/lib/cmake/hipfile`` or add the installation prefix to
``CMAKE_PREFIX_PATH``. The configure output reports which way it resolved, either
``hipFile stats support enabled`` with the version it found, or
``hipFile stats support disabled`` with the reason.

Enabling collection at run time
===============================

Even in a build that includes hipFile support, collection is off until you ask for
it: ``ROCPROFSYS_USE_HIPFILE`` defaults to ``OFF``. Enable it by setting
``ROCPROFSYS_USE_HIPFILE=ON``. Collection runs on the process-sampling thread, so
process sampling must also be enabled (it is on by default). Setting this in a build
made with ``ROCPROFSYS_BUILD_HIPFILE=OFF`` has no effect, because the collector is
not present in the binary.

.. code-block:: shell

   ROCPROFSYS_USE_HIPFILE=ON
   ROCPROFSYS_USE_PROCESS_SAMPLING=ON
   ROCPROFSYS_PROCESS_SAMPLING_FREQ=100
   # Optional; the default is "fastpath, fallback, bandwidth"
   ROCPROFSYS_HIPFILE_METRICS=all

Details of the settings:

* **ROCPROFSYS_USE_HIPFILE**: Enables the hipFile telemetry sampler.
* **ROCPROFSYS_USE_PROCESS_SAMPLING**: Enables the background sampling thread
  that drives the hipFile sampler (default ``ON``).
* **ROCPROFSYS_PROCESS_SAMPLING_FREQ**: Samples per second. A higher frequency
  captures short-lived I/O bursts more precisely.
* **ROCPROFSYS_HIPFILE_METRICS**: Which hipFile metrics to collect. Defaults to
  ``fastpath, fallback, bandwidth``. Accepts ``all`` or ``on``, ``none`` or ``off``,
  or a comma or semicolon separated list of the group names listed in
  `Metrics collected`_: ``bytes``, ``ops``, ``fastpath``, ``fallback``,
  ``unaligned``, ``errors``, and ``bandwidth``. Each name selects both the read and
  the write track, in the same way ``ROCPROFSYS_AMD_SMI_METRICS`` treats ``power``
  and ``temp``. An unrecognized name is ignored with a warning.
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
Visualize the results in ROCm Optiq or Perfetto
==================================

To view the ``.db`` file generated by the profiler in
`ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_,
set ``ROCPROFSYS_USE_ROCPD=ON``:

.. code-block:: shell

   ROCPROFSYS_USE_ROCPD=ON
#. Open the `ROCm Optiq UI page <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_ and click ``Open trace file`` and select the ``.db`` file. The hipFile
   counter tracks appear as ``hipFile GPU<N> <metric>``.
To view the ``.proto`` file generated by the profiler in Perfetto, open the
`Perfetto UI page <https://ui.perfetto.dev/>`_ and click ``Open trace file`` and select the ``.proto`` file. The hipFile
counter tracks appear as ``hipFile GPU<N> <metric>``.

#. Open the `Perfetto UI page <https://ui.perfetto.dev/>`_.
#. Click ``Open trace file`` and select the ``.proto`` file. The hipFile
   counter tracks appear as ``hipFile GPU<N> <metric>``.

Troubleshooting
===============

* **No hipFile tracks in the output**: Confirm that the profiler was built with
  ``ROCPROFSYS_BUILD_HIPFILE=ON`` (check the configure output for
  ``hipFile stats support enabled``), that ``ROCPROFSYS_USE_HIPFILE=ON`` is set at
  run time, and that the target application actually performs hipFile I/O. If the
  application never calls into hipFile, no statistics are produced.
* **Counters are zero**: Verify that process sampling is enabled and that the
  workload runs long enough to be sampled at least once during active I/O.
* **A track has gaps**: Samples taken while hipFile could not be queried are
  omitted rather than written as zero, so gaps early in a run are expected while
  the application is still starting up.
* **A counter track looks flat**: The counters are cumulative, so a track that
  stops climbing means I/O stopped, not that collection failed. Use the delta
  view to see per-window activity.
* **All operations are on the fallback path**: hipFile's fast path requires a
  supported filesystem (ext4 with ordered journaling, or xfs) and ``O_DIRECT``.
  When those conditions are not met, hipFile transparently uses POSIX I/O and
  the **Fallback** counters climb while the **Fastpath** counters stay at zero.
