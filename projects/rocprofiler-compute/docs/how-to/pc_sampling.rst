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
documentation, and section 21 of ``report.txt`` prints a link to it above the
table:

.. code-block:: shell-session

   --------------------------------------------------------------------------------
   21. PC Sampling
   21.1 PC Sampling
   Stall reason definitions: https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/cdna3-cdna4-pc-sampling.html#stall-reasons

Analysis options
================
The PC sampling table is section 21 of ``report.txt`` in the run folder. To
also print it on the terminal, add ``-b 21``. The sample outputs below show
section 21. The table is built from the analysis database. ``-k`` narrows it to the
selected kernels; the operator options do not.

For using analysis options for PC sampling the configuration needed are:

* ``--pc-sampling-sorting-type``: ``offset`` or ``count``. The default option is ``count``, which surfaces the most-sampled instructions (hotspots) first. ``offset`` is an assembly instruction offset in the code object.
* ``--pc-sampling-rows``: Maximum number of rows shown in the PC sampling table (DEFAULT: 10). Must be a non-negative integer; use ``0`` to show all rows.

Two columns describe how well each instruction used the machine:

* ``active_thread_percent``: percent of the wave's lanes that were active. A
  low value means control flow masked lanes off. Shown for both sampling
  methods.
* ``wave_occupancy_percent``: percent of the machine's wave slots that held a
  wave. A low value means there were too few waves to hide latency. Shown for
  ``stochastic`` only, because ``host_trap`` records carry no wave count.

Both are averaged over the samples that landed on the instruction. A kernel
built for wave64 on a device that reports a wave size of 32 sets more lanes
than that size, so ``active_thread_percent`` is capped at 100 and analyze warns
once with how many lines were capped.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute analyze -p <workload_dir> -k 0 --pc-sampling-sorting-type offset

**Sample output:**

``source_line`` shows ``N/A`` because the example binary was built without
``-g`` (See the :ref:`note <pc-sampling-note>` at the end of this page).

``instruction_type`` names the hardware pipeline that runs the instruction,
such as ``VALU`` for vector math or ``MATRIX`` for matrix math.

Selecting a single kernel with ``host_trap`` PC sampling:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0

   ╒═════════╤═════════╤═══════════════╤════════════════════════════════╤════════════════════╤══════════════════╤══════════╤═════════╤═════════════════════════╤══════════════════════════════════════╕
   │   index │     pid │ source_line   │ instruction                    │ instruction_type   │   code_object_id │ offset   │   count │   active_thread_percent │ Kernel_Name                          │
   ╞═════════╪═════════╪═══════════════╪════════════════════════════════╪════════════════════╪══════════════════╪══════════╪═════════╪═════════════════════════╪══════════════════════════════════════╡
   │      83 │ 1429079 │ N/A           │ v_add_u32_e32 v16, s2, v0      │ VALU               │                2 │ 0x3f30   │   42959 │                  100.00 │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                │                    │                  │          │         │                         │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼──────────────────────────────────────┤
   │      84 │ 1429079 │ N/A           │ v_ashrrev_i32_e32 v17, 31, v16 │ VALU               │                2 │ 0x3f38   │   10908 │                   96.88 │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                │                    │                  │          │         │                         │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼──────────────────────────────────────┤
   │      85 │ 1429079 │ N/A           │ s_load_dword s0, s[0:1], 0x0   │ SCALAR             │                2 │ 0x3f40   │       1 │                   87.50 │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                │                    │                  │          │         │                         │ __vector(4)*, int)                   │
   ╘═════════╧═════════╧═══════════════╧════════════════════════════════╧════════════════════╧══════════════════╧══════════╧═════════╧═════════════════════════╧══════════════════════════════════════╛

Selecting a single kernel with ``stochastic`` PC sampling, which adds the
``issue_count``, ``stall_count``, ``wave_occupancy_percent``, and
``stall_reason`` columns:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0

   ╒═════════╤═════════╤═══════════════╤════════════════════════════════════╤════════════════════╤══════════════════╤══════════╤═════════╤═════════════════════════╤════════════════╤═════════════════╤══════════════════════════╤═══════════════════════════════════════════════════════╤══════════════════════════════════════╕
   │   index │     pid │ source_line   │ instruction                        │ instruction_type   │   code_object_id │ offset   │   count │   active_thread_percent │   issue_count  │   stall_count   │   wave_occupancy_percent │ stall_reason                                          │ Kernel_Name                          │
   ╞═════════╪═════════╪═══════════════╪════════════════════════════════════╪════════════════════╪══════════════════╪══════════╪═════════╪═════════════════════════╪════════════════╪═════════════════╪══════════════════════════╪═══════════════════════════════════════════════════════╪══════════════════════════════════════╡
   │      90 │ 1429079 │ N/A           │ s_load_dword s8, s[0:1], 0x24      │ SCALAR             │                2 │ 0x3f00   │       3 │                  100.00 │              0 │               3 │                    78.12 │ [('ARBITER_NOT_WIN', 3)]                              │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                    │                    │                  │          │         │                         │                │                 │                          │                                                       │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │      91 │ 1429079 │ N/A           │ s_load_dwordx4 s[4:7], s[0:1], 0x0 │ SCALAR             │                2 │ 0x3f08   │       2 │                  100.00 │              0 │               2 │                    71.88 │ [('ARBITER_NOT_WIN', 1), ('ARBITER_WIN_EX_STALL', 1)] │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                    │                    │                  │          │         │                         │                │                 │                          │                                                       │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼───────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │      92 │ 1429079 │ N/A           │ s_load_dword s3, s[0:1], 0x10      │ SCALAR             │                2 │ 0x3f10   │       2 │                   96.88 │              0 │               2 │                    71.88 │ [('ARBITER_WIN_EX_STALL', 1), ('ARBITER_NOT_WIN', 1)] │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                    │                    │                  │          │         │                         │                │                 │                          │                                                       │ __vector(4)*, int)                   │
   ╘═════════╧═════════╧═══════════════╧════════════════════════════════════╧════════════════════╧══════════════════╧══════════╧═════════╧═════════════════════════╧════════════════╧═════════════════╧══════════════════════════╧═══════════════════════════════════════════════════════╧══════════════════════════════════════╛

Without a kernel filter, the same per-instruction table is shown across all
kernels, with a ``Kernel_Name`` column identifying each row's kernel:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir>

   ╒═════════╤═════════╤═══════════════╤══════════════════════════════════════════════════════════════╤════════════════════╤══════════════════╤══════════╤═════════╤═════════════════════════╤════════════════╤═════════════════╤══════════════════════════╤══════════════════════════╤══════════════════════════════════════════╕
   │   index │     pid │ source_line   │ instruction                                                  │ instruction_type   │   code_object_id │ offset   │   count │   active_thread_percent │   issue_count  │   stall_count   │   wave_occupancy_percent │ stall_reason             │ Kernel_Name                              │
   ╞═════════╪═════════╪═══════════════╪══════════════════════════════════════════════════════════════╪════════════════════╪══════════════════╪══════════╪═════════╪═════════════════════════╪════════════════╪═════════════════╪══════════════════════════╪══════════════════════════╪══════════════════════════════════════════╡
   │      12 │ 1429079 │ N/A           │ s_cmp_eq_u32 s1, 0                                           │ SCALAR             │                2 │ 0x3bb8   │     199 │                  100.00 │            197 │               2 │                    84.38 │ [('ARBITER_NOT_WIN', 2)] │ _Z22matmul_fp16_throughputPDv4_DF16_PDv4 │
   │         │         │               │                                                              │                    │                  │          │         │                         │                │                 │                          │                          │ _fi                                      │
   ├─────────┼─────────┼───────────────┼──────────────────────────────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
   │      13 │ 1429079 │ N/A           │ s_waitcnt vmcnt(2)                                           │ INTERNAL           │                2 │ 0x3bbc   │     199 │                  100.00 │              0 │             199 │                    84.38 │ [('WAITCNT', 199)]       │ _Z22matmul_fp16_throughputPDv4_DF16_PDv4 │
   │         │         │               │                                                              │                    │                  │          │         │                         │                │                 │                          │                          │ _fi                                      │
   ├─────────┼─────────┼───────────────┼──────────────────────────────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────────────────────┤
   │      14 │ 1429079 │ N/A           │ v_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[20:21], v[8:11] │ MATRIX             │                2 │ 0x3bc0   │       1 │                   93.75 │              0 │               1 │                    62.50 │ [('ARBITER_NOT_WIN', 1)] │ void fma_throughput<int>(int             │
   │         │         │               │                                                              │                    │                  │          │         │                         │                │                 │                          │                          │ __vector(4)*, int)                       │
   ╘═════════╧═════════╧═══════════════╧══════════════════════════════════════════════════════════════╧════════════════════╧══════════════════╧══════════╧═════════╧═════════════════════════╧════════════════╧═════════════════╧══════════════════════════╧══════════════════════════╧══════════════════════════════════════════╛

Sorting a single kernel by sample ``count`` instead of ``offset``:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0 --pc-sampling-sorting-type count

   ╒═════════╤═════════╤═══════════════╤═══════════════════════════════════════════════════╤════════════════════╤══════════════════╤══════════╤═════════╤═════════════════════════╤════════════════╤═════════════════╤══════════════════════════╤══════════════════════════════════════════════════════════════╤══════════════════════════════════════╕
   │   index │     pid │ source_line   │ instruction                                       │ instruction_type   │   code_object_id │ offset   │   count │   active_thread_percent │   issue_count  │   stall_count   │   wave_occupancy_percent │ stall_reason                                                 │ Kernel_Name                          │
   ╞═════════╪═════════╪═══════════════╪═══════════════════════════════════════════════════╪════════════════════╪══════════════════╪══════════╪═════════╪═════════════════════════╪════════════════╪═════════════════╪══════════════════════════╪══════════════════════════════════════════════════════════════╪══════════════════════════════════════╡
   │     106 │ 1429079 │ N/A           │ global_load_dword v18, v[0:1], off                │ FLAT               │                2 │ 0x3f78   │   29715 │                  100.00 │              0 │           29715 │                    90.62 │ [('ARBITER_NOT_WIN', 26037), ('ARBITER_WIN_EX_STALL', 3678)] │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                                   │                    │                  │          │         │                         │                │                 │                          │                                                              │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼───────────────────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │     117 │ 1429079 │ N/A           │ v_mfma_f32_16x16x4_f32 v[8:11], v19, v19, v[8:11] │ VALU               │                2 │ 0x3fb0   │   21164 │                  100.00 │            169 │           20995 │                    87.50 │ [('ARBITER_NOT_WIN', 20995)]                                 │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                                   │                    │                  │          │         │                         │                │                 │                          │                                                              │ __vector(4)*, int)                   │
   ├─────────┼─────────┼───────────────┼───────────────────────────────────────────────────┼────────────────────┼──────────────────┼──────────┼─────────┼─────────────────────────┼────────────────┼─────────────────┼──────────────────────────┼──────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │     188 │ 1429079 │ N/A           │ global_store_dwordx4 v[4:5], v[0:3], off          │ FLAT               │                2 │ 0x4204   │   13821 │                   78.12 │              0 │           13821 │                    65.62 │ [('ARBITER_WIN_EX_STALL', 7300), ('ARBITER_NOT_WIN', 6521)]  │ matmul_fp32_throughput(float*, float │
   │         │         │               │                                                   │                    │                  │          │         │                         │                │                 │                          │                                                              │ __vector(4)*, int)                   │
   ╘═════════╧═════════╧═══════════════╧═══════════════════════════════════════════════════╧════════════════════╧══════════════════╧══════════╧═════════╧═════════════════════════╧════════════════╧═════════════════╧══════════════════════════╧══════════════════════════════════════════════════════════════╧══════════════════════════════════════╛

Analyze multi-process workloads
-------------------------------

Pass the workload directory to ``analyze`` as usual. No additional
multi-process option is required:

.. code-block:: shell

   $ rocprof-compute analyze -p <workload_dir>

Analyze mode loads the PC sampling data for every process in the workload
directory and reports it in a single table. A ``pid`` column identifies the
process each row came from.

Samples are not merged across processes. The same instruction offset can belong
to different code in two processes, so each process keeps its own rows and its
own counts.

Sorting and ``--pc-sampling-rows`` apply to the combined table. Sorting by
``offset`` orders on ``pid`` first, which keeps each process's rows together.
Sorting by ``count`` ranks on sample count alone, so rows from different
processes can interleave. In both cases, ``--pc-sampling-rows 10`` selects ten
rows from the workload as a whole, not ten rows per process.

.. _pc-sampling-per-kernel-csv:

PC-sampling files in the run folder
===================================

For a workload with PC samples, every analyze run writes ``pc-sampling-summary.csv`` into the
workload folder. It has one row for each kernel with samples. Each such kernel also gets an ISA
file and a source file in its own folder:

.. code-block:: none

   <run folder>/
       1-<workload>/
           pc-sampling-summary.csv
           kernel-000-vecCopy/
               kernel-000-vecCopy-metrics.csv
               kernel-000-vecCopy-pc-sampling-isa.csv
               kernel-000-vecCopy-pc-sampling-src.csv

The folder names are described in :ref:`analysis-output-format`, and ``kernels.csv`` gives the
full name of each kernel. One kernel has one ISA file even when it was compiled into several code
objects or run by several processes. The ``Code object id`` and ``Pid`` columns tell those rows
apart. Analyze does not copy source files into the run folder.

Each ISA file holds one row per instruction line of its kernel. The line numbers count from 1
across the whole file:

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

The ``Source`` column is populated only when the target app was built with debug info; see the
:ref:`note <pc-sampling-note>`.

``pc-sampling-summary.csv`` starts with a ``#`` line that says PC samples are narrowed by ``-k``
only, then its header. Its ``Total count``, ``Active count``, ``Stall count`` and
``Stall <REASON>`` columns are sums over the kernel's instruction lines, across every code object
and process.

.. _pc-sampling-per-source-line-csv:

Per-source-line counts in the source file
=========================================

Each kernel's ``<folder>-pc-sampling-src.csv`` has one row for each source line its samples reach.
The row sums the samples of every instruction compiled from that line, across processes:

.. list-table::
   :header-rows: 1

   * - Column
     - Description
   * - ``Source file``
     - Source path the profile recorded for the line.
   * - ``Source line number``
     - Source line number in the source file.
   * - ``Source line``
     - Text of the line, when the source is available.
   * - ``Total count``, ``Active count``, ``Stall count``, ``Stall <REASON>``
     - Sums over the instructions compiled from this line.

An instruction from inlined code counts in full toward every source line of its
inline stack. For example, an instruction that reads ``blockDim.x`` counts toward
both the HIP header lines that implement it and the application line that reads
it.

Some instructions have no source line number. Their samples go to one row with empty
``Source file`` and ``Source line number`` cells and the label ``Unattributable instructions`` in
``Source line``. Its counts are the samples lost because those instructions have no source line
number. See the :ref:`note <pc-sampling-note>` for why the compiler leaves out the source line
number.

.. _pc-sampling-note:

.. note::

  * PC sampling now only shows assembly instructions collected in our record of pc samples and not all instructions of compiled code are represented.
  * Source information requires the target app to be built with debug info (for example ``hipcc -g``). Without it, samples map to assembly only: profile mode captures no source files, the terminal table's ``source_line`` shows ``N/A``, and the ``Source`` column of the per-kernel ISA CSV (``-pc-sampling-isa.csv``) is empty.
  * The ``Source`` column can show ``?`` instead of a line number, such as ``kernel.cpp:?``. The compiler does this when it builds one instruction from several source lines. To find the source of such an instruction, look at the source lines of the instructions it depends on. For example, a wait instruction such as ``s_waitcnt`` depends on earlier memory and LDS operations, such as ``global_*``, ``buffer_*``, ``s_load*`` or ``ds_*`` instructions.
