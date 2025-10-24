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
| kernel-iteration-range   | List    |         |           | Defines dispatch iteration of the kernel to be profiled.     |
|                          |         |         |           | Hyphen (-) specifies a range: 1-10 means iterations 1-10.    |
|                          |         |         |           | Comma (,) specifies individual items: 1,10 means only 1,10.  |
|                          |         |         |           | Can combine: [1,2,[5-8]] means iterations 1,2,5,6,7,8.       |
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
| att-gpu-index            | Integer |         |           | Comma-separated list of integers. If enabled, only the GPU   |
|                          | (List)  |         |           | indexes in the list will be profiled by thread trace.        |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+
| att-consecutive-kernels  | Integer | >=0     |           | Starting at the targeted kernel, enables thread trace for the|
|                          |         |         |           | next N kernel dispatches, sharing a single ATT file,         |
|                          |         |         |           | stats.csv and UI dir. See --kernel-include-regex and         |
|                          |         |         |           | --kernel-iteration-range. If multiple targeted kernels       |
|                          |         |         |           | overlap, the count for N next dispatches starts again from 0.|
|                          |         |         |           | Recommended use with --att-gpu-index due to thread trace     |
|                          |         |         |           | being enabled for all GPUs.                                  |
+--------------------------+---------+---------+-----------+--------------------------------------------------------------+

For AMD Instinct accelerators, enable perfmon streaming using:

.. code-block:: bash

  rocprofv3 --att --att-activity 8 -- <application_path>

For AMD Radeon, the ``simd-select`` parameter is a SIMD ID defaulting to 3. For some applications it's best to use:

.. code-block:: bash

  rocprofv3 --att --att-simd-select 0x0 -- <application_path>

Parameter usage examples
========================

This section provides examples of using various thread trace parameters.

**Targeting specific CU and shader engine:**

.. code-block:: bash

  # Trace CU 2 on shader engine 0 (mask 0x1)
  rocprofv3 --att --att-target-cu 2 --att-shader-engine-mask 0x1 -- <application_path>

  # Trace multiple shader engines (mask 0xF traces 4 SEs)
  rocprofv3 --att --att-shader-engine-mask 0xF -- <application_path>

**Adjusting buffer size:**

.. code-block:: bash

  # Increase buffer to 256MB to avoid data loss
  rocprofv3 --att --att-buffer-size 268435456 -- <application_path>

  # Use 1GB buffer for long-running kernels
  rocprofv3 --att --att-buffer-size 0x40000000 -- <application_path>

**Kernel filtering and iteration ranges:**

.. code-block:: bash

  # Trace only kernels matching "matmul"
  rocprofv3 --att --kernel-include-regex "matmul" -- <application_path>

  # Trace iterations 5 through 10 of matching kernels
  rocprofv3 --att --kernel-include-regex "matmul" --kernel-iteration-range 5-10 -- <application_path>

  # Trace specific iterations 1, 3, and 7 through 9
  rocprofv3 --att --kernel-include-regex "matmul" --kernel-iteration-range "1,3,[7-9]" -- <application_path>

  # Exclude specific kernels from profiling
  rocprofv3 --att --kernel-include-regex ".*" --kernel-exclude-regex "copy.*" -- <application_path>

**GPU selection:**

.. code-block:: bash

  # Trace only on GPU 0
  rocprofv3 --att --att-gpu-index 0 -- <application_path>

  # Trace on GPUs 0 and 2
  rocprofv3 --att --att-gpu-index 0,2 -- <application_path>

**Consecutive kernel profiling:**

.. code-block:: bash

  # Profile 5 consecutive kernels starting from the first "matmul" kernel
  rocprofv3 --att --kernel-include-regex "matmul" --att-consecutive-kernels 5 -- <application_path>

  # Combined with GPU selection
  rocprofv3 --att --kernel-include-regex "conv" --att-consecutive-kernels 3 --att-gpu-index 0 -- <application_path>

**Performance counter streaming (gfx9 only):**

.. code-block:: bash

  # Enable activity counters with period 8
  rocprofv3 --att --att-activity 8 -- <application_path>

  # Custom perfcounter control period
  rocprofv3 --att --att-perfcounter-ctrl 4 -- <application_path>

**Combined parameters:**

.. code-block:: bash

  # Comprehensive example: trace specific kernels on specific hardware
  rocprofv3 --att \
    --att-target-cu 1 \
    --att-shader-engine-mask 0x1 \
    --att-simd-select 0xF \
    --att-buffer-size 0x10000000 \
    --kernel-include-regex "mykernel.*" \
    --kernel-iteration-range "2-5" \
    --att-gpu-index 0 \
    -d output_dir \
    -- <application_path>

Using input file
================

As explained in the preceding section, you can specify parameters on the command line or use a JSON input file:

.. code-block:: text

  {
      "jobs": [
          {
              "advanced_thread_trace": true,
              "att_target_cu": 1,
              "att_shader_engine_mask": "0x1",
              "att_simd_select": "0xF",
              "att_buffer_size": "0x6000000",
              "kernel_include_regex": "matmul.*",
              "kernel_iteration_range": "[1,2,[5-8]]"
          }
      ]
  }

This example traces iterations 1, 2, 5, 6, 7, and 8 of kernels matching "matmul.*"

Thread tracing for multiple kernel instances
=============================

By default, ``rocprofv3`` enables thread trace only once per kernel instance. This implies that if an application launches the same kernel multiple times, only the first instance will be traced.
To enable thread trace for multiple kernel instances, use the ``kernel-iteration-range`` parameter.
It's recommended to use ``kernel-include-regex`` parameter to filter the desired kernel names instead of tracing everything.

Typically, each kernel profile has its own ATT file output.
To compile multiple kernel profiles into a single output file, use the ``att-consecutive-kernels`` parameter.
When using this parameter, the ``rocprofv3`` tool begins profiling kernels after encountering a targeted kernel.
The tool then continues profiling subsequent kernels until a total of ``n`` kernels are profiled including the initial targeted kernel
where ``n`` is the non-negative integer passed to ``att-consecutive-kernels``.
Note that the subsequent kernels encountered after the initial targeted kernel do not themselves have to be targeted.
If the subsequent kernels are targeted kernels, the profiler will then profile another ``n - 1`` kernels after encountering this
new targeted kernel, so it is possible for a generated ATT file to have more than ``n`` kernels profiled.
All the profiled kernels are then compiled into a single ATT file.
If a new targeted kernel is encountered after the ``rocprofv3`` tool has finished profiling a batch of kernels,
the profiler will restart profiling when encountering this new targeted kernel and create another ATT file with multiple kernels. 

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
    
Migrating from rocprofv2 thread trace
======================================

If you're familiar with rocprofv2 ATT (Advanced Thread Trace) parameters, this section maps the old configuration format to the new rocprofv3 command-line options.

**Old rocprofv2 input file format:**

.. code-block:: text

  att: TARGET_CU=0
  SE_MASK=0xffffffff
  SIMD_SELECT=0
  DISPATCH_RANGE=4000,4500
  BUFFER_SIZE=192
  PERFCOUNTERS_CTRL=8

**Parameter mapping table:**

+----------------------------+-----------------------------+-----------------------------------------------+
| rocprofv2 Parameter        | rocprofv3 Parameter         | Notes                                         |
+============================+=============================+===============================================+
| TARGET_CU                  | --att-target-cu             | Same value range (0-15)                       |
+----------------------------+-----------------------------+-----------------------------------------------+
| SE_MASK                    | --att-shader-engine-mask    | Same hex bitmask format                       |
+----------------------------+-----------------------------+-----------------------------------------------+
| SIMD_SELECT                | --att-simd-select           | Same value range (0-0xF)                      |
+----------------------------+-----------------------------+-----------------------------------------------+
| BUFFER_SIZE                | --att-buffer-size           | **v2: MB, v3: bytes**                         |
|                            |                             | v2: 192 = v3: 201326592 (or 0xC000000)        |
+----------------------------+-----------------------------+-----------------------------------------------+
| DISPATCH_RANGE             | --kernel-iteration-range    | Syntax changed: see iteration range section   |
|                            |                             | v2: 4000,4500 = v3: 4000-4500                 |
+----------------------------+-----------------------------+-----------------------------------------------+
| PERFCOUNTERS_CTRL          | --att-perfcounter-ctrl      | Same value range (1-32), gfx9 only            |
+----------------------------+-----------------------------+-----------------------------------------------+
| PERFCOUNTER                | --att-perfcounters          | Counter name, gfx9 only                       |
+----------------------------+-----------------------------+-----------------------------------------------+
| PERFCOUNTER_ID             | --att-perfcounters          | Counter ID, gfx9 only                         |
+----------------------------+-----------------------------+-----------------------------------------------+
| PERFCOUNTER_MASK           | N/A                         | Use --att-target-cu to select CU              |
+----------------------------+-----------------------------+-----------------------------------------------+
| ISA_CAPTURE_MODE           | N/A                         | Code object capture is automatic in v3        |
+----------------------------+-----------------------------+-----------------------------------------------+
| OCCUPANCY                  | N/A                         | For similar effect, see --att-no-detail       |
+----------------------------+-----------------------------+-----------------------------------------------+

**Equivalent rocprofv3 command:**

.. code-block:: bash

  # rocprofv2 config converted to rocprofv3
  rocprofv3 --att \
    --att-target-cu 0 \
    --att-shader-engine-mask 0xffffffff \
    --att-simd-select 0 \
    --kernel-iteration-range 4000-4500 \
    --att-buffer-size 201326592 \
    --att-perfcounter-ctrl 8 \
    -- <application_path>

  # Or using JSON input file:

.. code-block:: text

  {
      "jobs": [
          {
              "advanced_thread_trace": true,
              "att_target_cu": 0,
              "att_shader_engine_mask": "0xffffffff",
              "att_simd_select": "0x0",
              "att_buffer_size": "0xC000000",
              "kernel_iteration_range": "4000-4500",
              "att_perfcounter_ctrl": 8
          }
      ]
  }

**Key differences:**

- **Buffer size units:** rocprofv2 used megabytes, rocprofv3 uses bytes. Multiply your old value by 1,048,576 (or left-shift by 20 bits).
- **Dispatch range syntax:** rocprofv2 used comma notation for ranges (4000,4500), rocprofv3 uses hyphen for ranges (4000-4500) and comma for individual items.
- **No input.txt file:** rocprofv3 uses command-line options or JSON/YAML input files instead of the old input.txt format.
- **Kernel filtering:** Use --kernel-include-regex and --kernel-exclude-regex for kernel name filtering in rocprofv3.

