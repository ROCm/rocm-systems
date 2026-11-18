.. meta::
   :description: ROCm Systems Profiler hipFile Infinity Storage I/O telemetry
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, how to, profiler, hipFile, Infinity Storage, direct-to-GPU, fastpath, fallback, AIS, I/O, telemetry, AMD

**************************************
hipFile Infinity Storage I/O telemetry
**************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_
can collect direct-to-GPU I/O telemetry from applications that use
`hipFile <https://rocm.docs.amd.com/projects/hipFile/en/latest/>`_,
AMD's Infinity Storage library. hipFile provides direct-to-GPU I/O without a
host-side buffer, and falls back to POSIX I/O when an operation cannot use the
direct-to-GPU path.
A background sampler periodically queries hipFile's in-process I/O statistics
and reports the results as per-GPU counter tracks in the RocPD database.

Metrics collected
==================

All hipFile telemetry is per GPU, reported under tracks named
``GPU [<N>] Storage <metric> (S)``, matching the naming other sampled GPU metrics
use. Metrics are selected in groups, and each group covers
both the read and the write track. The group name is the token accepted by
``ROCPROFSYS_HIPFILE_METRICS``.

Collected by default:

* ``fastpath`` -- **Fastpath Reads** / **Fastpath Writes**, I/O completed by the
  fastpath backend, which transfers data between storage and the GPU without a
  host-side buffer (``count``)
* ``fallback`` -- **Fallback Reads** / **Fallback Writes**, I/O completed by the
  fallback backend, which routes the operation through a host-side buffer before
  transferring data to the GPU (``count``)
* ``bandwidth`` -- **Read Bandwidth** / **Write Bandwidth**, read and write
  bandwidth over the sampling interval (``bytes/s``)
* ``bytes`` -- **Read Bytes** / **Write Bytes**, cumulative read size and write
  size (``bytes``)
* ``errors`` -- **Read Errors** / **Write Errors**, read and write error counts
  (``count``)

Available, but off by default:

* ``ops`` -- **Read Ops** / **Write Ops**, cumulative I/O operation counts
  (``count``)
* ``unaligned`` -- **Unaligned Reads** / **Unaligned Writes**, unaligned read and
  write counts. The fastpath does not support unaligned I/O (``count``)

The default set covers fastpath and fallback backend usage, read size and write
size (``bytes``), read and write bandwidth (``bandwidth``), and error counts
(``errors``). Add ``ops``, ``unaligned``, or ``all`` when you need the full
picture.

Requirements
============

* **ROCm 10.1 or later**, with hipFile installed. The per-GPU I/O statistics API
  this collector reads is not present in earlier releases.

  .. code-block:: shell

     apt install amdrocm-hipfile     # or: apt install amd-rocmcore
     dnf install amdrocm-hipfile     # or: dnf install amd-rocmcore

* ROCm Systems Profiler built with hipFile support (see `Build support`_).
* A target application that links and uses hipFile. I/O statistics are
  in-process, so telemetry is only produced when the profiled application
  actually performs hipFile I/O.

Build support
=============

``ROCPROFSYS_USE_HIPFILE`` answers both questions, at two different stages:

* As a **CMake option** it decides whether hipFile support is compiled into the
  profiler at all. It is tri-state: ``AUTO`` (the default), ``ON``, or ``OFF``.
* As an **environment variable or configuration file setting** it decides whether
  a given run collects hipFile samples. It is a plain boolean and defaults to
  ``OFF``. See `Enabling collection at run time`_.

The two stages are independent: a profiler that was built with hipFile support
still collects nothing until the run-time setting is turned on.

As a CMake option, ``ROCPROFSYS_USE_HIPFILE`` defaults to ``AUTO``, so support is built
whenever a suitable hipFile package is present and skipped (with a status
message) when it is not. To require the feature (failing configure if the required
hipFile is absent, or ROCm is older than 10.1), pass ``ON``. To exclude the
feature from a package entirely, configure with ``OFF``:

.. code-block:: shell

   cmake -D ROCPROFSYS_USE_HIPFILE=OFF <other options> <path/to/source>

The build requires ROCm 10.1 or later. With the default ``AUTO``, an older
release is treated the same as when the required hipFile is absent: support is left off
and the rest of the profiler builds normally, because the per-GPU statistics
API does not exist to build against. ``ON`` does not demote; CMake stops and
names the version it found (if any), the version required, and how to point it
at a different prefix.

To point CMake at a specific hipFile installation, pass
``-Dhipfile_DIR=<prefix>/lib/cmake/hipfile`` or add the installation prefix to
``CMAKE_PREFIX_PATH``. The configure output reports which way it resolved, either
``hipFile stats support enabled`` with the version it found, or
``hipFile stats support disabled`` with the reason. The derived result is the
CMake variable ``ROCPROFSYS_HIPFILE_SUPPORT``; the ``ROCPROFSYS_USE_HIPFILE`` cache
entry keeps the value you passed (``AUTO``, ``ON``, or ``OFF``).

Enabling collection at run time
===============================

Even in a build that includes hipFile support, collection is off until you ask for
it: as a run-time setting, ``ROCPROFSYS_USE_HIPFILE`` defaults to ``OFF`` (unlike
the CMake option of the same name, which defaults to ``AUTO``). Enable it by setting
``ROCPROFSYS_USE_HIPFILE=ON``. When hipFile support
is not compiled in (``-D ROCPROFSYS_USE_HIPFILE=OFF``, or ``AUTO`` with no new
enough package) the collector is not present and the settings are not
registered. Their presence in ``rocprof-sys-avail --settings`` is a direct
indicator of compile-time support:

.. code-block:: shell

   rocprof-sys-avail --settings | grep HIPFILE

To enable collection:

1. Set ``ROCPROFSYS_USE_HIPFILE=ON``.

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
* **ROCPROFSYS_PROCESS_SAMPLING_FREQ**: Samples per second. A higher frequency
  captures short-lived I/O bursts more precisely.
* **ROCPROFSYS_HIPFILE_METRICS**: Which hipFile metrics to collect. Defaults to
  ``fastpath, fallback, bandwidth, bytes, errors``. Accepts ``all`` or ``on``,
  ``none`` or ``off``, or a comma or semicolon separated list of the group names listed in
  `Metrics collected`_: ``bytes``, ``ops``, ``fastpath``, ``fallback``,
  ``unaligned``, ``errors``, and ``bandwidth``. Each name selects both the read and
  the write track. An unrecognized name is ignored with a warning.
* **ROCPROFSYS_SAMPLING_GPUS**: Which GPUs to collect hipFile telemetry from.
  ``HIP_VISIBLE_DEVICES`` / ``ROCR_VISIBLE_DEVICES`` must be an increasing integer
  list (for example ``4,5``). A permutation (``5,4``) or UUID list cannot be mapped
  from hipFile's HIP ordinals to profiler GPU indices, so hipFile telemetry is
  disabled with a warning.

hipFile statistics collection is on by default (``HIPFILE_STATS_LEVEL``
defaults to ``1``, which collects per-GPU and per-backend statistics). The
profiler does not overwrite that variable. If you set ``HIPFILE_STATS_LEVEL=0``
while also enabling ``ROCPROFSYS_USE_HIPFILE``, statistics collection is disabled
and the profiler logs a warning rather than overriding your setting. Unset it,
or set it to ``1`` or higher, to collect I/O statistics.

Running the profiler
====================

Run the target application under ``rocprof-sys-run`` (or ``rocprof-sys-sample``)
with the settings above. For example:

.. code-block:: shell

   ROCPROFSYS_USE_HIPFILE=ON rocprof-sys-run -- ./my_hipfile_app --input data.bin

Visualize the results in ROCm Optiq
===================================

To view the ``.db`` file generated by the profiler in
`ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_:

#. Open the `ROCm Optiq UI <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`__.
#. Click ``Open trace file`` and select the ``.db`` file. The hipFile counter
   tracks appear as ``GPU [<N>] Storage <metric> (S)``.

Troubleshooting
===============

* **No hipFile tracks in the output**: Confirm that the profiler was built with
  hipFile support (check the configure output for ``hipFile stats support enabled``,
  or that ``ROCPROFSYS_HIPFILE_SUPPORT`` is ON), that ``ROCPROFSYS_USE_HIPFILE=ON``
  is set at run time, and that the target application actually performs hipFile
  I/O. If the application never calls into hipFile, no statistics are produced.
  ``rocprof-sys-avail --settings | grep HIPFILE`` is empty in a build that left
  the collector out.
* **"hipFile telemetry unavailable" in the log**: The collector is compiled in but
  could not use the hipFile runtime, and the rest of the profile is unaffected. The
  message says which case it is. ``could not be loaded`` means the required hipFile
  is absent. ``does not export the per-GPU stats API`` means the installed hipFile
  is from a ROCm release older than 10.1. A message naming the loaded version
  means the runtime is older than ROCm 10.1.
* **Counters are zero**: Verify that the workload runs long enough to be sampled
  at least once during active I/O.
* **A track has gaps**: Samples taken while hipFile could not be queried are
  omitted rather than written as zero, so gaps early in a run are expected while
  the application is still starting up.
* **A counter track looks flat**: The counters are cumulative, so a track that
  stops climbing means I/O stopped, not that collection failed. Use the delta
  view to see per-window activity.
* **All I/O uses the fallback path**: The fastpath requires a file opened with
  ``O_DIRECT`` and a supported storage configuration: a raw NVMe block device,
  ext4 on NVMe mounted with ``data=ordered``, or xfs on NVMe. File offsets,
  buffer offsets, and I/O sizes must be aligned to the file system's direct I/O
  alignment. When those conditions are not met, hipFile uses the fallback path
  and the **Fallback** counters climb while the **Fastpath** counters stay at
  zero.
