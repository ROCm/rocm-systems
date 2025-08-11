.. meta::
  :description: Documentation of the usage of thread trace with rocprofv3 command-line tool
  :keywords: rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, Thread Trace, SQTT, ATT, ROCprof Trace Decoder, ROCprof Compute Viewer

.. _using-thread-trace:

============================
Using thread trace
============================

Thread trace is a shader execution tracing technique capable of profiling wavefronts at the instruction timing level.
This is a low-level tracing and profiling feature that targets a single or a few kernel executions.

Thread trace features include:

* Near cycle-accurate instruction tracing
* Exact thread or wave execution path
* Wave scheduling and stall timing analysis
* Instruction and source level hotspots
* Extremely fast and granular counter collection (AMD Instinct)

Supported devices:

* AMD Instinct: MI200 and MI300 series
* AMD Radeon: gfx10, gfx11 and gfx12

Thread trace profiling is performed in the following steps:

1. Tracing (data collection) - Uses ROCprofiler-SDK thread trace service API
2. Decoding (analysis) - Uses ROCprof Trace Decoder API
3. Visualization - Requires ROCprof Compute Viewer

Tracing and decoding is handled by ``rocprofv3`` while visualization is handled by the ROCprof Compute Viewer.

Prerequisites
=========

- aqlprofile:

  * ROCm 7.x build, or

  * Early release can be `built from source <https://github.com/rocm/aqlprofile>`_

  * Otherwise, ``rocprofv3`` throws error "INVALID_SHADER_DATA" or "Agent not supported".

- Installation of ROCprof Trace Decoder component:

  * For binary files, see `ROCprof trace decoder release page <https://github.com/ROCm/rocprof-trace-decoder/releases>`_.

  * Default install location is ``/opt/rocm/lib``
   
  * For custom location, use:

      * Parameter ``--att-library-path``, or

      * Environment variable ``ROCPROF_ATT_LIBRARY_PATH``
   

.. _thread-trace-parameters:

rocprofv3 parameters for thread tracing
============================

To collect thread trace with default parameters, use:

.. code-block:: bash

  rocprofv3 --att -d <output_dir> -- <application_path>

The following table lists the parameters relevant to thread tracing:

+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| Parameter                | Type    | Range   | Typical   | Description                                                  |
+==========================+=========+=========+===========+==============================================================+
| att-target-cu            | Integer | 0 - 15  | 1         | Defines the CU used to gather detail tokens (WGP on Navi)    |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-shader-engine-mask   | Bitmask | 1 - ~0u | 0x1       | Defines the Shader Engines (SE) to be traced. Max 2^32 - 1   |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-simd-select          | Integer | 0 - 0xF | gfx9: 0xF | Defines one or more SIMDs to be traced, out of four.         |
|                          |         |         | Navi: 0x0 | Bitmask on GFX9 and SIMD_ID[0,3] on Navi.                    |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| kernel-iteration-range   | List    |         |           | Defines dispatch iteration of the kernel to be profiled      |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| kernel-include-regex     | String  | Any     |           | Profiles kernel names matching the regex                     |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| kernel-exclude-regex     | String  | Any     |           | Doesn't profile kernel names matching the regex              |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-buffer-size          | Bytes   | 1MB-2GB | 96MB      | Specifies the trace buffer size. This is shared for all SEs. |
|                          |         |         |           | Increase this value if the buffer tends to get full.         |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-serialize-all        | Bool    |         | False     | If set to "True", turns on serialization for untraced kernels|
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-perfcounter-ctrl     | Integer | 1 - 32  | 2~8       | Available only in gfx9. Streams SQ performance counters to   |
|                          |         |         |           | the thread trace buffer in the given relative period. As     |
|                          |         |         |           | this uses high bandwidth, a value too low can cause or worsen|
|                          |         |         |           | "Data Lost" events and warnings.                             |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-perfcounters         | String  | SQ-only |           | Available only in gfx9. Specifies the list of SQ counters.   |
|                          |         |         |           | To list all counters, use "rocprofv3 --list-avail``.         |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-activity             | Integer | 1 - 16  | 5~10      | Available only in gfx9.                                      |
|                          |         |         |           | Shorthand for att-perfcounter-ctrl and the att-perfcounters  |
|                          |         |         |           | related to compute unit activity such as VALU, SALU, etc.    |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+

For AMD Instinct accelerators, enable perfmon streaming using:

.. code-block:: bash

  rocprofv3 --att --att-activity 8 -- <application_path>

For AMD Radeon, the ``simd-select`` parameter is a SIMD ID defaulting to 3. For some applications it's best to use:

.. code-block:: bash

  rocprofv3 --att --att-simd-select 0x0 -- <application_path>


Using input file
===========

As explained in the preceding section, you can specify parameters on the command line or use a JSON input file:

.. code-block:: text

  {
      "jobs": [
          {
              "advanced_thread_trace": true,
              "att_target_cu": 1,
              "att_shader_engine_mask": "0x1",
              "att_simd_select": "0xF",
              "att_buffer_size": "0x6000000"
          }
      ]
  }

Thread tracing for multiple kernel instances
=============================

By default, ``rocprofv3`` enables thread trace only once per kernel instance. This implies that if an application launches the same kernel multiple times, only the first instance will be traced.
To enable thread trace for multiple kernel instances, use the ``kernel-iteration-range`` parameter.
It's recommended to use ``kernel-include-regex`` parameter to filter the desired kernel names instead of tracing everything.

.. _output-files:

rocprofv3 output files
===============

After the application finishes executing, ROCprof Trace Decoder runs automatically and the following output files are generated:

- stats_*.csv files:

  * Contains a summary of instruction latency per kernel.
  
- ui_output_agent_{agent_id}_dispatch_{dispatch_id} directory:
  
  * Contains detailed tracing information in the form of .json files.
    
  * This directory can be opened using the `ROCprof Compute Viewer <https://rocm.docs.amd.com/projects/rocprof-compute-viewer/en/amd-mainline/>`_.

- Raw files:

  * .att - Raw SQTT data. Can be used with the ROCprof Trace Decoder for further analysis.
  
  * .out - Code object binaries (executable). Can be used with ISA analysis tools.

.. _csv-content:

Stats CSV
------------

Here is a sample stats_*.csv file that is generated by the rocprofv3 tool.

+---------+-------+---------------------------------------------+----------+---------+-------+------+-------------------+
| Codeobj | Vaddr | Instruction                                 | Hitcount | Latency | Stall | Idle | Source            |
+=========+=======+=============================================+==========+=========+=======+======+===================+
| 11      | 5888  | s_load_dwordx4 s[40:43], s[0:1], 0x18       | 48       | 276     | 96    | 48   | kernel.py:391     |
+---------+-------+---------------------------------------------+----------+---------+-------+------+-------------------+
| 11      | 5896  | s_load_dwordx2 s[38:39], s[0:1], 0x28       | 48       | 192     | 0     | 0    | kernel.py:391     |
+---------+-------+---------------------------------------------+----------+---------+-------+------+-------------------+
| 11      | 5904  | s_ashr_i32 s3, s2, 31                       | 48       | 260     | 0     | 0    | kernel.py:395     |
+---------+-------+---------------------------------------------+----------+---------+-------+------+-------------------+
| 11      | 5908  | s_add_i32 s7, s2, s3                        | 48       | 196     | 0     | 0    | kernel.py:395     |
+---------+-------+---------------------------------------------+----------+---------+-------+------+-------------------+

The columns of the stats_*.csv file are described here:

* **Codeobj:** The code object load ID assigned by ROCprofiler-SDK.

* **Vaddr:** ELF vaddr.

* **Hitcount:** The number of times a particular instruction is executed while adding all the traced waves.

* **Latency:** Total latency in cycles, defined as "Stall time + Issue time" for gfx9 or "Stall time + Execute time" for gfx10+.

* **Stall:** The total number of cycles the hardware pipe couldn't issue an instruction. 

  * Usually caused when the hardware unit is busy, such as TCP or LDS backpressure.
    
* **Idle:** The total time gap between the completion of previous instruction and the beginning of the current instruction. The idle time can be caused by:

  * Arbiter loss
    
  * Source or destination register dependency
    
  * Instruction cache miss
    
* **Source:** The original source line of code assigned by the compiler.

  * Requires compiling with debug symbols.
    

Troubleshooting
===============

For some applications, stats_*.csv file could be empty even for a valid kernel dispatch.
Thread trace is limited to a single CU per SE (``att-target-cu``). If a kernel dispatch doesn't launch enough waves to populate the whole GPU, there's a possibility of no wave getting assigned to the ``target_cu``. In such cases, there's nothing to be traced. 
Here are some options to handle this:

* Launch more waves.

* Swap the ``target_cu``.

* Set the ``--att-shader-engine-mask`` to 0x11111111, or possibly to 0xFFFFFFFF

  * A number too high can cause packet losses and/or lead to a full buffer.
    
* Set the ``HSA_CU_MASK`` to mask out all CUs but the target. For more details, see `setting CUs <https://rocm.docs.amd.com/en/latest/how-to/setting-cus.html>`_.

  * If only the ``target_cu`` (or a few CUs) are not masked out, then all or most waves will be assigned to the ``target_cu``.
    
  * This can potentially cause low performance in high-demanding kernels.
    
