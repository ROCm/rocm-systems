.. meta::
  :description: Application replay in rocprofv3, the default way to collect several hardware-counter groups
  :keywords: rocprofv3, application replay, replay-mode application, multi-pass, pmc, counter groups

.. _using-application-replay:

==========================
Using application replay
==========================

Application replay is the default way ``rocprofv3`` collects more hardware counters than fit in one
pass. Each ``--pmc`` flag is one counter group. The whole application runs once per group, in a new
process, and that run collects only that group.

Nothing is snapshotted and nothing is restored. Device memory, host memory, and files start from
whatever the application does at startup. The GPU executes each kernel once per process, with the
same stream overlap the application asked for.

.. code-block:: text

   pass 1, one process                         pass 2, a new process
   host: startup, load data, ...               host: startup, load data, ...
   GPU:  kernel A ---- kernel B ---- kernel C  GPU:  kernel A ---- kernel B ---- kernel C
         counters: group 1                           counters: group 2
   device memory lives for this process only   device memory is the new process's own
   output: pass_1/                             output: pass_2/

Hardware limits how many performance counters a GPU can collect from one kernel execution. A second
``--pmc`` flag is a second execution of the application, which is what makes the groups fit.
:ref:`using-rocprofv3-avail` and ``rocprofv3-avail pmc-check`` tell you whether one group fits in a
single hardware pass. Application replay does not split a group that does not fit.

This page is the user guide. The same behavior is summarized under multi-pass collection in
:ref:`kernel-counter-collection`.

Collecting several groups
=========================

``--replay-mode application`` is the default. Naming it keeps a script stable when the default
changes.

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES SQ_WAVE_CYCLES --pmc GRBM_COUNT GRBM_GUI_ACTIVE --replay-mode application -- <application_path>

The command above launches ``<application_path>`` twice:

* Pass 1 collects ``SQ_WAVES`` and ``SQ_WAVE_CYCLES``.
* Pass 2 collects ``GRBM_COUNT`` and ``GRBM_GUI_ACTIVE``.

Counters inside one ``--pmc`` flag are one group and are collected together. A space or a comma
separates counters in that group. A new ``--pmc`` flag starts another group and another run.

One group needs no second run:

.. code-block:: bash

   rocprofv3 --pmc SQ_WAVES GRBM_COUNT -- <application_path>

An input file's ``pmc:`` lines are further groups. With both a file and ``--pmc`` flags, each source
contributes its own passes. See :ref:`kernel-counter-collection` for the combined ordering and for
JSON and YAML jobs, which each get their own ``pass_n`` directory.

Output
======

Each pass writes a ``pass_n`` directory. ``n`` starts at 1.

.. code-block:: text

   output_directory/
   ├── pass_1/
   │   └── counter_collection.csv
   ├── pass_2/
   │   └── counter_collection.csv

There is no combined record and no ``replay_pass`` field. Each directory is one process. Dispatch
IDs start over in every pass, so a row in ``pass_2`` is a different launch from the row with the
same ``Dispatch_Id`` in ``pass_1``.

Correlate rows yourself, by kernel name and by order within the file. ``rocprofv3`` does not match
launches across passes. The application has to launch the same kernels, on the same devices, in the
same order, in every pass. A run whose kernel sequence depends on a clock, a random source, or an
unfixed work size will put different work in ``pass_1`` and ``pass_2``.

The default output format is ``rocpd``. Convert it as described in :ref:`using-rocpd-output-format`.

What each pass contains
=======================

A pass is a full process:

* Startup, file reads, and host setup run again.
* Every kernel runs again, including kernels you did not mean to compare.
* GPU caches start cold, then warm up the way they do in an ordinary run. Earlier kernels in the
  same pass leave cache contents for later kernels.
* Host and device allocations are the application's. The tool does not copy them.
* Inter-process collectives run again, because every rank's process runs again. Profile all ranks
  with the same groups.

Counter collection still inserts its own packets around a dispatch so the hardware counters can
start and stop. That is true of a single-pass run as well. Application replay adds no extra replay
barrier and does not serialize kernels beyond what that counter bracketing already does.

The tool does not change GPU clocks between passes. A later pass can run at a higher clock if the
GPU has warmed up. For numbers you want to compare, pin a performance level before the run. See
:ref:`gpu-performance-level`.

When to use it
==============

Use application replay when:

* The kernels call back to the host while they run, or otherwise depend on host progress during
  the launch. Each pass runs the host code again, so those responses happen again.
* The result you care about includes cache contents left by earlier kernels in the same process.
* The device footprint is large enough that a snapshot copy would cost more than running the
  application again. See :ref:`using-kernel-replay-rocprofv3`.
* The buffers of interest are unified memory, managed memory, or ``hipMallocAsync`` allocations.
  In-process replay restores coarse-grained ``hipMalloc`` device memory and module-scope device
  variables, and leaves those other pools alone. See
  `HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_.

Prefer :ref:`kernel replay <using-kernel-replay-rocprofv3>` when startup dominates, or when the
process is not deterministic and you still need several groups on the same dispatch.

Prefer counter-group rotation (``pmc_groups`` / ``pmc_group_interval``, described in
:ref:`kernel-counter-collection`) when one run is enough and different dispatches may sample
different groups.

Limitations
===========

* **One process per group.** Setup cost is paid every time.
* **The kernel sequence must match** across passes if you want to compare rows. There is no
  launch-matching option.
* **Incompatible with attach mode** (``--pid``) and with ``--collection-period``.
* **Each group must fit in one hardware pass.**
* **Clocks are not pinned** by the tool.
* **Caches are not flushed** by the tool. A kernel's counters include the cache state the rest of
  that pass created.

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * - Mode
     - Application replay
     - Kernel replay
     - Counter-group rotation
   * - Process
     - One new process per group
     - One process
     - One process
   * - What the GPU repeats
     - The whole run
     - One dispatch
     - Nothing; groups rotate across dispatches
   * - Device memory
     - The application's own
     - Snapshot and restore of tracked device memory
     - Untouched
   * - Output
     - ``pass_n/`` per process
     - One directory, ``replay_pass`` in JSON
     - One directory

See also
========

* :ref:`kernel-counter-collection` — command lines, input files, and ``pass_n`` layout
* :ref:`using-kernel-replay-rocprofv3` — replay one dispatch in process
* :ref:`using-kernel-replay` — the same service from a custom tool
* :ref:`gpu-performance-level` — pinning clocks for a comparison
* :ref:`using-rocprofv3-avail` — which counters exist, and whether a group fits
* :ref:`using-rocpd-output-format` — the default output
* `HIP memory management <https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/memory_management.html>`_
