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
* ``--pc-sampling-interval``: For ``stochastic`` sampling, the interval is in cycles; it must be a power of 2 and at least 65536 (DEFAULT: 1048576). These are hardware limits reported by the driver. For ``host_trap`` sampling, the interval is in microseconds and may be any positive integer (DEFAULT: 512). When omitted, the method-appropriate default is used.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute profile -n pc_test --no-roof --experimental --pc-sampling --pc-sampling-method stochastic -VVV -- target_app

Analysis options
================
For using analysis options for PC sampling the configuration needed are:

* ``--pc-sampling-sorting-type``: ``offset`` or ``count``. The default option is ``offset``. ``offset`` is an assembly instruction offset in the code object.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute analyze -p <workload_dir> -k 0 --pc-sampling-sorting-type offset

**Sample output:**

``source_line`` shows ``N/A`` because the example binary was built without
``-g`` (See the :ref:`note <pc-sampling-note>` at the end of this page).

Selecting a single kernel with ``host_trap`` PC sampling:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0

   ╒═════════╤═══════════════╤═════════════════════════════════════════════════════╤══════════════════╤══════════╤═════════╤══════════════════════════════════════╕
   │   index │ source_line   │ instruction                                         │   code_object_id │ offset   │   count │ Kernel_Name                          │
   ╞═════════╪═══════════════╪═════════════════════════════════════════════════════╪══════════════════╪══════════╪═════════╪══════════════════════════════════════╡
   │      83 │ N/A           │ v_add_u32_e32 v16, s2, v0                           │                2 │ 0x3f30   │   42959 │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼──────────────────────────────────────┤
   │      84 │ N/A           │ v_ashrrev_i32_e32 v17, 31, v16                      │                2 │ 0x3f38   │   10908 │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼──────────────────────────────────────┤
   │      85 │ N/A           │ s_load_dword s0, s[0:1], 0x0                        │                2 │ 0x3f40   │       1 │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │ __vector(4)*, int)                   │
   ╘═════════╧═══════════════╧═════════════════════════════════════════════════════╧══════════════════╧══════════╧═════════╧══════════════════════════════════════╛

Selecting a single kernel with ``stochastic`` PC sampling, which adds the
``count_issued``, ``count_stalled``, and ``stall_reason`` columns:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0

   ╒═════════╤═══════════════╤═════════════════════════════════════════════════════╤══════════════════╤══════════╤═════════╤════════════════╤═════════════════╤═════════════════════════════════════════════════════════════════════════════════════╤══════════════════════════════════════╕
   │   index │ source_line   │ instruction                                         │   code_object_id │ offset   │   count │   count_issued │   count_stalled │ stall_reason                                                                        │ Kernel_Name                          │
   ╞═════════╪═══════════════╪═════════════════════════════════════════════════════╪══════════════════╪══════════╪═════════╪════════════════╪═════════════════╪═════════════════════════════════════════════════════════════════════════════════════╪══════════════════════════════════════╡
   │      90 │ N/A           │ s_load_dword s8, s[0:1], 0x24                       │                2 │ 0x3f00   │       3 │              0 │               3 │ [('ARBITER_NOT_WIN', 3)]                                                            │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │      91 │ N/A           │ s_load_dwordx4 s[4:7], s[0:1], 0x0                  │                2 │ 0x3f08   │       2 │              0 │               2 │ [('ARBITER_NOT_WIN', 1), ('ARBITER_WIN_EX_STALL', 1)]                               │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │      92 │ N/A           │ s_load_dword s3, s[0:1], 0x10                       │                2 │ 0x3f10   │       2 │              0 │               2 │ [('ARBITER_WIN_EX_STALL', 1), ('ARBITER_NOT_WIN', 1)]                               │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ╘═════════╧═══════════════╧═════════════════════════════════════════════════════╧══════════════════╧══════════╧═════════╧════════════════╧═════════════════╧═════════════════════════════════════════════════════════════════════════════════════╧══════════════════════════════════════╛

Without a kernel filter, the same per-instruction table is shown across all
kernels, with a ``Kernel_Name`` column identifying each row's kernel:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir>

   ╒═════════╤═══════════════╤══════════════════════════════════════════════════════════════════╤══════════════════╤══════════╤═════════╤════════════════╤═════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════╤══════════════════════════════════════════╕
   │   index │ source_line   │ instruction                                                      │   code_object_id │ offset   │   count │   count_issued │   count_stalled │ stall_reason                                                                                       │ Kernel_Name                              │
   ╞═════════╪═══════════════╪══════════════════════════════════════════════════════════════════╪══════════════════╪══════════╪═════════╪════════════════╪═════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════╪══════════════════════════════════════════╡
   │      12 │ N/A           │ s_cmp_eq_u32 s1, 0                                               │                2 │ 0x3bb8   │     199 │            197 │               2 │ [('OTHER_WAIT', 197), ('ARBITER_NOT_WIN', 2)]                                                      │ _Z22matmul_fp16_throughputPDv4_DF16_PDv4 │
   │         │               │                                                                  │                  │          │         │                │                 │                                                                                                    │ _fi                                      │
   ├─────────┼───────────────┼──────────────────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
   │      13 │ N/A           │ s_waitcnt vmcnt(2)                                               │                2 │ 0x3bbc   │     199 │              0 │             199 │ [('WAITCNT', 199)]                                                                                 │ _Z22matmul_fp16_throughputPDv4_DF16_PDv4 │
   │         │               │                                                                  │                  │          │         │                │                 │                                                                                                    │ _fi                                      │
   ├─────────┼───────────────┼──────────────────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────────┤
   │      14 │ N/A           │ v_mfma_f32_16x16x16_f16 v[8:11], v[20:21], v[20:21], v[8:11]     │                2 │ 0x3bc0   │       1 │              0 │               1 │ [('ARBITER_NOT_WIN', 1)]                                                                           │ void fma_throughput<int>(int             │
   │         │               │                                                                  │                  │          │         │                │                 │                                                                                                    │ __vector(4)*, int)                       │
   ╘═════════╧═══════════════╧══════════════════════════════════════════════════════════════════╧══════════════════╧══════════╧═════════╧════════════════╧═════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════╧══════════════════════════════════════════╛

Sorting a single kernel by sample ``count`` instead of ``offset``:

.. code-block:: shell-session

   $ rocprof-compute analyze -p <workload_dir> -k 0 --pc-sampling-sorting-type count

   ╒═════════╤═══════════════╤═════════════════════════════════════════════════════╤══════════════════╤══════════╤═════════╤════════════════╤═════════════════╤═════════════════════════════════════════════════════════════════════════════════════╤══════════════════════════════════════╕
   │   index │ source_line   │ instruction                                         │   code_object_id │ offset   │   count │   count_issued │   count_stalled │ stall_reason                                                                        │ Kernel_Name                          │
   ╞═════════╪═══════════════╪═════════════════════════════════════════════════════╪══════════════════╪══════════╪═════════╪════════════════╪═════════════════╪═════════════════════════════════════════════════════════════════════════════════════╪══════════════════════════════════════╡
   │     106 │ N/A           │ global_load_dword v18, v[0:1], off                  │                2 │ 0x3f78   │   29715 │              0 │           29715 │ [('ARBITER_NOT_WIN', 26037), ('ARBITER_WIN_EX_STALL', 3678)]                        │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │     117 │ N/A           │ v_mfma_f32_16x16x4_f32 v[8:11], v19, v19, v[8:11]   │                2 │ 0x3fb0   │   21164 │            169 │           20995 │ [('ARBITER_NOT_WIN', 20995), ('OTHER_WAIT', 169)]                                   │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ├─────────┼───────────────┼─────────────────────────────────────────────────────┼──────────────────┼──────────┼─────────┼────────────────┼─────────────────┼─────────────────────────────────────────────────────────────────────────────────────┼──────────────────────────────────────┤
   │     188 │ N/A           │ global_store_dwordx4 v[4:5], v[0:3], off            │                2 │ 0x4204   │   13821 │              0 │           13821 │ [('ARBITER_WIN_EX_STALL', 7300), ('ARBITER_NOT_WIN', 6521)]                         │ matmul_fp32_throughput(float*, float │
   │         │               │                                                     │                  │          │         │                │                 │                                                                                     │ __vector(4)*, int)                   │
   ╘═════════╧═══════════════╧═════════════════════════════════════════════════════╧══════════════════╧══════════╧═════════╧════════════════╧═════════════════╧═════════════════════════════════════════════════════════════════════════════════════╧══════════════════════════════════════╛

.. _pc-sampling-note:

.. note::

  * PC sampling now only shows assembly instructions collected in our record of pc samples and not all instructions of compiled code are represented.
  * To associate PC sampling info back to HIP source code, you need to build the profiling target app with ``-g`` to keep the symbols. Otherwise, PC sampling info will be only associated with assembly lines.
