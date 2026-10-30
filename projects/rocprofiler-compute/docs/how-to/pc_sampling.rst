.. meta::
   :description: ROCm Compute Profiler: using PC sampling
   :keywords: ROCm Compute Profiler, PC sampling

********************************************
Using PC sampling in ROCm Compute Profiler
********************************************

.. warning::

   PC sampling is an experimental feature. Enable it in ``profile``
   mode by passing ``--experimental --pc-sampling``. The ``analyze``
   command detects PC sampling automatically from the profiling
   configuration and needs no extra pc-sampling flag. Behavior and
   command-line surface may change in future releases.

Program Counter (PC) sampling service for GPU profiling is a profiling technique that periodically samples the program counter during the GPU kernel execution to understand code execution patterns and hotspots.

ROCm Compute Profiler supports Host Trap PC sampling and Stochastic (Hardware-Based) PC sampling.
Host Trap PC sampling is enabled for AMD Instinct MI200 Series and later
GPUs. Stochastic (hardware-based) PC sampling is enabled for
AMD Instinct MI300 Series and later GPUs. Stochastic PC sampling provides additional information that tells whether a sampled wave issued an instruction for a particular PC. It also provides the reason
for not issuing the instruction (stall reason). This type of information is
particularly useful for understanding stalls during the kernel execution. The PC sampling can be used with profiling and analysis options.

Profiling options
=================
For using profiling options for PC sampling the configuration needed are:

* ``--pc-sampling-method``: Should be either ``stochastic`` or ``host_trap``, (DEFAULT: stochastic)
* ``--pc-sampling-interval``: The accepted range is read from the device; see ``rocprofv3-avail info --pc-sampling``. When the device cannot be queried, 1 to 1048576 is accepted. For ``stochastic`` sampling, the interval is in cycles and must be a power of 2 (DEFAULT: 1048576). For ``host_trap`` sampling, the interval is in microseconds (DEFAULT: 512). When omitted, the method-appropriate default is used.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute profile -n pc_test --no-roof --experimental --pc-sampling --pc-sampling-method stochastic -VVV -- target_app

Profile multi-process workloads
-------------------------------

The same profile command supports applications that create multiple processes.
Every process that runs GPU kernels is sampled, and no additional option is
required.

Stall reasons
=============

The ``stall_reason`` column of a stochastic PC sampling table holds the raw
reason names reported by ROCprofiler-SDK for samples where a wave did not
issue an instruction. Those names are defined in the ROCprofiler-SDK
documentation, and the analysis output prints a link to it above the table:

.. code-block:: shell-session

   --------------------------------------------------------------------------------
   21. PC Sampling
   21.1 PC Sampling
   Stall reason definitions: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/cdna3-cdna4-pc-sampling.html#stall-reasons

Analysis options
================
For using analysis options for PC sampling the configuration needed are:

* ``--pc-sampling-rows``: Maximum number of kernels shown in the PC sampling table (Default: 10). Must be a non-negative integer; use ``0`` to show all kernels.

The PC sampling table shows one row per kernel, sorted by sample ``count`` with
the most-sampled kernel first. A kernel's counts are summed over all of its
instructions.

Two columns describe how well each kernel used the machine:

* ``active_thread_percent``: percent of the wave's lanes that were active. A
  low value means control flow masked lanes off. Shown for both sampling
  methods.
* ``wave_occupancy_percent``: percent of the machine's wave slots that held a
  wave. A low value means there were too few waves to hide latency. Shown for
  ``stochastic`` only, because ``host_trap`` records carry no wave count.

Both are averaged over the kernel's samples. A kernel built for wave64 on a
device that reports a wave size of 32 sets more lanes than that size, so
``active_thread_percent`` is capped at 100 and analyze warns once with how many
lines were capped.

To find the hot instructions and source lines of a kernel, use the
:ref:`per-kernel ISA and source files <pc-sampling-per-kernel-csv>` of CSV
output.

**Sample output:**

``host_trap`` PC sampling:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir>

   ╒═════════╤═════════════════════════╤═════════╤═════════════════════════╕
   │   index │ Kernel_Name             │   count │   active_thread_percent │
   ╞═════════╪═════════════════════════╪═════════╪═════════════════════════╡
   │       0 │ sgprbound(int, double*) │  210836 │                  100.00 │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┤
   │       1 │ vgprbound(int, double*) │  122973 │                   99.45 │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┤
   │       2 │ ldsbound(int, double*)  │   18445 │                   98.80 │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┤
   │       3 │ empty(int, double*)     │       4 │                  100.00 │
   ╘═════════╧═════════════════════════╧═════════╧═════════════════════════╛

``stochastic`` PC sampling, which adds the ``issue_count``, ``stall_count``,
``wave_occupancy_percent``, and ``stall_reason`` columns:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir>

   ╒═════════╤═════════════════════════╤═════════╤═════════════════════════╤═══════════════╤═══════════════╤══════════════════════════╤═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
   │   index │ Kernel_Name             │   count │   active_thread_percent │   issue_count │   stall_count │   wave_occupancy_percent │ stall_reason                                                                                                                                                                                        │
   ╞═════════╪═════════════════════════╪═════════╪═════════════════════════╪═══════════════╪═══════════════╪══════════════════════════╪═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
   │       0 │ vgprbound(int, double*) │  253159 │                   99.41 │         77966 │        175193 │                    49.51 │ [('ARBITER_NOT_WIN', 164088), ('ALU_DEPENDENCY', 5487), ('NO_INSTRUCTION_AVAILABLE', 3477), ('INTERNAL_INSTRUCTION', 1163), ('WAITCNT', 921), ('ARBITER_WIN_EX_STALL', 57)]                         │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┼───────────────┼───────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
   │       1 │ sgprbound(int, double*) │  141176 │                  100.00 │         30958 │        110218 │                    98.98 │ [('ARBITER_NOT_WIN', 102816), ('ALU_DEPENDENCY', 4283), ('NO_INSTRUCTION_AVAILABLE', 2434), ('INTERNAL_INSTRUCTION', 641), ('WAITCNT', 44)]                                                         │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┼───────────────┼───────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
   │       2 │ ldsbound(int, double*)  │   79579 │                   98.73 │         42296 │         37283 │                    24.69 │ [('ARBITER_NOT_WIN', 25600), ('ALU_DEPENDENCY', 3464), ('NO_INSTRUCTION_AVAILABLE', 2886), ('WAITCNT', 1869), ('ARBITER_WIN_EX_STALL', 1716), ('INTERNAL_INSTRUCTION', 936), ('BARRIER_WAIT', 812)] │
   ├─────────┼─────────────────────────┼─────────┼─────────────────────────┼───────────────┼───────────────┼──────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
   │       3 │ empty(int, double*)     │      31 │                  100.00 │             0 │            31 │                     8.97 │ [('NO_INSTRUCTION_AVAILABLE', 31)]                                                                                                                                                                  │
   ╘═════════╧═════════════════════════╧═════════╧═════════════════════════╧═══════════════╧═══════════════╧══════════════════════════╧═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

Analyze multi-process workloads
-------------------------------

Pass the workload directory to ``analyze`` as usual. No additional
multi-process option is required:

.. code-block:: shell

   $ rocprof-compute analyze -p <workload_dir>

Analyze mode loads the PC sampling data for every process in the workload
directory and reports it in a single table. A kernel that ran in several
processes is shown once, with the samples of all processes combined.

.. _pc-sampling-per-kernel-csv:

Per-kernel ISA and source in CSV output
=======================================

``--output-format csv`` writes, alongside the analysis tables, one file per
kernel per code object per process holding that kernel's instruction lines with
the samples collected on them, and exports the source those instructions were
compiled from beside it:

.. code-block:: none

   <output_name>/
       kernel.csv, pc_sampling_summary.csv, ...
       per_kernel_pc_sampling/
           <workload_name>/<workload_sub_name>/
               source/<source path with the leading separator dropped>
               source/<source path with the leading separator dropped>_pc_sampling.csv
               <short_name>_uuid_<kernel_uuid>/
                   isa_code_object_id_<code_object_id>_pid_<pid>.csv

A folder leads with the kernel's short name, the identifier its C++ signature
demangles down to, so ``vecCopy_2(double*, double*, double*, int, int)`` is
filed under ``vecCopy_2_uuid_7``. Overloads and template instantiations share a
short name, so the ``kernel_uuid`` suffix keeps each folder unique. ``kernel.csv``
carries ``kernel_uuid`` alongside ``short_name`` and ``kernel_name``, so it maps
a folder back to the kernel it holds. One kernel has more than one file when it
was compiled into several code objects, or when several processes ran it.

Each file holds one row per instruction line, ordered by code object offset:

.. list-table::
   :header-rows: 1

   * - Column
     - Description
   * - ``Instruction line number``
     - Position of the line within this file, counting from 1.
   * - ``Code object offset``
     - Offset of the instruction from the code object's load address.
   * - ``Instruction line``
     - Disassembled instruction.
   * - ``Instruction type``
     - Hardware pipeline that runs the instruction, such as ``VALU`` or
       ``MATRIX``. Empty when the instruction is not classified.
   * - ``Total count``
     - Samples that landed on this instruction.
   * - ``Active count``
     - Samples where the wave issued. Empty for ``host_trap``.
   * - ``Stall count``
     - Samples where the wave was stalled. Empty for ``host_trap``.
   * - ``Wave occupancy percent``
     - Percent of the machine's wave slots that held a wave at this
       instruction. Empty for ``host_trap``, whose records carry no wave count.
   * - ``Active thread percent``
     - Percent of the wave's lanes that were active at this instruction. A low
       value means control flow masked lanes off.
   * - ``Stall <REASON>``
     - Samples stalled at this instruction for that reason.
   * - ``Source``
     - Inline stack of ``path:line`` frames, innermost first.
   * - ``Code object id``
     - Code object the instruction belongs to, local to its process.
   * - ``Pid``
     - Process the code object was loaded in.

The ``Stall <REASON>`` columns are the reasons the workload's samples actually
carry, so they vary between runs, and a ``host_trap`` workload has none of them.
Every file of one workload has the same columns. A cell is empty where that
reason was not seen at that offset.

An instruction that no sample landed on keeps its row, with empty counts. It
still carries an ``Instruction type``.

The ``Source`` column and the ``source/`` folder are populated only when the
target app was built with debug info; see the :ref:`note <pc-sampling-note>`.

.. _pc-sampling-per-source-line-csv:

Per-source-line counts in CSV output
====================================

Beside each exported source file, ``--output-format csv`` writes
``<source file name>_pc_sampling.csv``. It has one row for each line of that
file, with the samples of every kernel and process that ran code compiled from
that line:

.. list-table::
   :header-rows: 1

   * - Column
     - Description
   * - ``Source line number``
     - Source line number in the source file.
   * - ``Source line``
     - Text of the line.
   * - ``Total count``, ``Active count``, ``Stall count``, ``Stall <REASON>``
     - Sums over the instructions compiled from this line.
   * - ``Wave occupancy percent``, ``Active thread percent``
     - Averages over the samples of those instructions.
   * - ``ISA references``
     - Rows of those instructions in the per-kernel ISA files, as
       ``<kernel folder>/<ISA file>:<instruction line numbers>``, with one ISA
       file per line of the cell.

An instruction from inlined code counts in full toward every source line of its
inline stack. For example, an instruction that reads ``blockDim.x`` counts toward
both the HIP header lines that implement it and the application line that reads
it.

Some instructions have no source line number, shown as ``<file>:?`` in the
per-kernel ISA files. A file with such instructions starts with an
``Unattributable instructions`` row. Its ``Total count`` is the number of samples
lost because those instructions have no source line number, and its
``ISA references`` lists the instructions. Its other columns are empty. See the
:ref:`note <pc-sampling-note>` for why the compiler leaves out the source line
number.

.. _pc-sampling-note:

.. note::

  * PC sampling now only shows assembly instructions collected in our record of pc samples and not all instructions of compiled code are represented.
  * Source information requires the target app to be built with debug info (for example ``hipcc -g``). Without it, samples map to assembly only: profile mode captures no source files, and the ``Source`` column of the per-kernel CSV is empty.
  * The ``Source`` column can show ``?`` instead of a line number, such as ``kernel.cpp:?``. The compiler does this when it builds one instruction from several source lines. To find the source of such an instruction, look at the source lines of the instructions it depends on. For example, a wait instruction such as ``s_waitcnt`` depends on earlier memory and LDS operations, such as ``global_*``, ``buffer_*``, ``s_load*`` or ``ds_*`` instructions.
