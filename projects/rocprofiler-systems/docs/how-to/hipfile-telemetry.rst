.. meta::
   :description: ROCm Systems Profiler hipFile GPU-direct storage I/O telemetry
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, how to, profiler, hipFile, GPU-direct storage, GDS, I/O, telemetry, AMD

********************************************
hipFile GPU-direct storage I/O telemetry
********************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_
can collect GPU-direct storage I/O telemetry from applications that use
`hipFile <https://github.com/ROCm/rocm-systems/tree/develop/projects/hipfile>`_.
A background sampler periodically queries hipFile's in-process telemetry API and
reports the results as per-GPU counter tracks in both the Perfetto trace and the
RocPD database.

Metrics collected
==================

All hipFile telemetry is per GPU, reported under tracks named
``GPU [<N>] Storage <metric> (S)``, matching the naming other sampled GPU metrics
use. Metrics are selected in groups, and each group covers
both the read and the write track. The group name is the token accepted by
``ROCPROFSYS_HIPFILE_METRICS``.

Collected by default:

* ``fastpath`` -- **Fastpath Reads** / **Fastpath Writes**, operations that took
  hipFile's GPU-direct fast path (``count``)
* ``fallback`` -- **Fallback Reads** / **Fallback Writes**, operations that fell back
  to POSIX I/O (``count``)
* ``bandwidth`` -- **Read Bandwidth** / **Write Bandwidth**, transfer rate over the
  sampling interval (``bytes/s``)
* ``bytes`` -- **Read Bytes** / **Write Bytes**, cumulative bytes transferred
  (``bytes``)
* ``errors`` -- **Read Errors** / **Write Errors**, failed operations (``count``)

Available, but off by default:

* ``ops`` -- **Read Ops** / **Write Ops**, cumulative operation counts (``count``)
* ``unaligned`` -- **Unaligned Reads** / **Unaligned Writes**, operations that were
  not aligned (``count``)

The default set covers path usage (``fastpath`` against ``fallback``), transfer
volume and rate (``bytes`` and ``bandwidth``), and I/O failures (``errors``). Add
``ops``, ``unaligned``, or ``all`` when you need the full picture.

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

Build support
=============

There are two independent switches, one for each question:

* ``ROCPROFSYS_BUILD_HIPFILE`` (CMake option) decides whether hipFile support is
  compiled into the profiler at all. It is tri-state: ``AUTO`` (the default),
  ``ON``, or ``OFF``.
* ``ROCPROFSYS_USE_HIPFILE`` (environment variable or configuration file setting)
  decides whether a given run collects hipFile samples. See
  `Enabling collection at run time`_.

``ROCPROFSYS_BUILD_HIPFILE`` defaults to ``AUTO``, so hipFile support is built
whenever a suitable hipFile package is present and skipped (with a status
message) when it is not. To require the feature (failing configure if hipFile
is missing or older than ``ROCPROFSYS_HIPFILE_MIN_VERSION``, 0.5.0), pass
``ON``. To exclude the feature from a package entirely, configure with
``OFF``:

.. code-block:: shell

   cmake -D ROCPROFSYS_BUILD_HIPFILE=OFF <other options> <path/to/source>

The build requires hipFile 0.5.0 or later. With the default ``AUTO``, a package
older than that is treated the same as no package at all: support is left off
and the rest of the profiler builds normally, because the per-GPU statistics
API does not exist to build against. ``ON`` does not demote; CMake stops and
names the version it found (if any), the version required, and how to point it
at a different prefix.

When the feature is compiled in, the profiler links ``libhipfile`` the same way it
links AMD SMI: ``libhipfile.so.0`` must be present when the profiler starts, even if
``ROCPROFSYS_USE_HIPFILE`` is off. Official ROCm 10.1 and later packages ship a
matching hipFile with the profiler. Source builds must run against the same ROCm
prefix they were configured with. A missing ``libhipfile``, or an older hipFile that
still uses SONAME ``libhipfile.so.0`` but does not export the stats API (for example
0.4), can prevent the profiler from loading at all. Build with
``ROCPROFSYS_BUILD_HIPFILE=OFF`` if that dependency is not acceptable.

If a new enough ``libhipfile`` does load, the version is re-checked at run time.
When it is too old, the profiler logs a warning and emits no hipFile tracks.

To point CMake at a specific hipFile installation, pass
``-Dhipfile_DIR=<prefix>/lib/cmake/hipfile`` or add the installation prefix to
``CMAKE_PREFIX_PATH``. The configure output reports which way it resolved, either
``hipFile stats support enabled`` with the version it found, or
``hipFile stats support disabled`` with the reason. The derived result is the
CMake variable ``ROCPROFSYS_HIPFILE_SUPPORT``; ``ROCPROFSYS_BUILD_HIPFILE`` keeps
the value you passed (``AUTO``, ``ON``, or ``OFF``).

Enabling collection at run time
===============================

Even in a build that includes hipFile support, collection is off until you ask for
it: ``ROCPROFSYS_USE_HIPFILE`` defaults to ``OFF``. Enable it by setting
``ROCPROFSYS_USE_HIPFILE=ON``. Collection runs on the process-sampling thread, so
process sampling must also be enabled (it is on by default). If it is off, the
profiler logs a warning and records no hipFile telemetry. When hipFile support
is not compiled in (``ROCPROFSYS_BUILD_HIPFILE=OFF``, or ``AUTO`` with no new
enough package) the collector is not present and the settings are not
registered. Their presence in ``rocprof-sys-avail --settings`` is a direct
indicator of compile-time support:

.. code-block:: shell

   rocprof-sys-avail --settings | grep HIPFILE

To enable collection:

1. Set ``ROCPROFSYS_USE_HIPFILE=ON``. Collection runs on the process-sampling
   thread, which is on by default.

   .. code-block:: shell

      export ROCPROFSYS_USE_HIPFILE=ON

2. Optionally set ``ROCPROFSYS_HIPFILE_METRICS``. The default is:

   .. code-block:: shell

      ROCPROFSYS_HIPFILE_METRICS=fastpath,fallback,bandwidth,bytes,errors

   To include the groups that are off by default (``ops`` and ``unaligned``),
   list them with the rest, or collect every group:

   .. code-block:: shell

      ROCPROFSYS_HIPFILE_METRICS=fastpath,fallback,bandwidth,bytes,errors,ops,unaligned
      ROCPROFSYS_HIPFILE_METRICS=all

   The setting accepts ``all`` or ``on``, ``none`` or ``off``, or a comma or
   semicolon separated list of the group names in `Metrics collected`_. Each
   name selects both the read and the write track. An unrecognized name is
   ignored with a warning.

Details of the settings:

* **ROCPROFSYS_USE_HIPFILE**: Enables the hipFile telemetry sampler.
* **ROCPROFSYS_USE_PROCESS_SAMPLING**: Enables the background sampling thread
  that drives the hipFile sampler (default ``ON``).
* **ROCPROFSYS_PROCESS_SAMPLING_FREQ**: Samples per second. A higher frequency
  captures short-lived I/O bursts more precisely.
* **ROCPROFSYS_HIPFILE_METRICS**: Which hipFile metrics to collect. Defaults to
  ``fastpath, fallback, bandwidth, bytes, errors``. Accepts ``all`` or ``on``,
  ``none`` or ``off``, or a comma or semicolon separated list of the group names listed in
  `Metrics collected`_: ``bytes``, ``ops``, ``fastpath``, ``fallback``,
  ``unaligned``, ``errors``, and ``bandwidth``. Each name selects both the read and
  the write track, in the same way ``ROCPROFSYS_AMD_SMI_METRICS`` treats ``power``
  and ``temp``. An unrecognized name is ignored with a warning.
* **ROCPROFSYS_SAMPLING_GPUS**: Which GPUs to collect from. hipFile telemetry
  honors the same GPU selection as the AMD SMI GPU metrics. ``HIP_VISIBLE_DEVICES``
  / ``ROCR_VISIBLE_DEVICES`` must be an increasing integer list (for example
  ``4,5``). A permutation (``5,4``) or UUID list cannot be mapped from hipFile's
  HIP ordinals to profiler GPU indices, so hipFile telemetry is disabled with a
  warning. AMD SMI and rocprofiler-sdk GPU metrics are unaffected.

hipFile's statistics collection is on by default (``HIPFILE_STATS_LEVEL``
defaults to ``1``). The profiler does not overwrite that variable. If you set
``HIPFILE_STATS_LEVEL=0`` while also enabling ``ROCPROFSYS_USE_HIPFILE``, hipFile
produces no statistics and the profiler logs a warning rather than overriding
your setting. Unset it, or set it to ``1`` or higher, to collect telemetry.

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
===============================================

To view the ``.db`` file generated by the profiler in
`ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_,
set ``ROCPROFSYS_USE_ROCPD=ON``:

.. code-block:: shell

   ROCPROFSYS_USE_ROCPD=ON

Then load the trace:

#. Open the `ROCm Optiq UI <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`__.
#. Click ``Open trace file`` and select the ``.db`` file. The hipFile counter
   tracks appear as ``GPU [<N>] Storage <metric> (S)``.

To view the ``.proto`` file generated by the profiler in Perfetto:

#. Open the `Perfetto UI page <https://ui.perfetto.dev/>`_.
#. Click ``Open trace file`` and select the ``.proto`` file. The hipFile
   counter tracks appear as ``GPU [<N>] Storage <metric> (S)``.

Troubleshooting
===============

* **No hipFile tracks in the output**: Confirm that the profiler was built with
  hipFile support (check the configure output for ``hipFile stats support enabled``,
  or that ``ROCPROFSYS_HIPFILE_SUPPORT`` is ON), that ``ROCPROFSYS_USE_HIPFILE=ON``
  is set at run time, and that the target application actually performs hipFile
  I/O. If the application never calls into hipFile, no statistics are produced.
  ``rocprof-sys-avail --settings | grep HIPFILE`` is empty in a build that left
  the collector out.
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
