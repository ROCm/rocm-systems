.. meta::
  :description: Documentation of the usage of pc-sampling with rocprofv3 command-line tool
  :keywords: Sampling PC, Sampling program counter, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, PC sampling

.. _using-pc-sampling:

==================
Using PC sampling
*****************************
Using PC sampling
*****************************

PC (Program Counter) sampling is a GPU profiling technique that periodically samples the program counter during GPU kernel execution. PC sampling helps in understanding code execution patterns and identifying hotspots.

Here are the benefits of using PC sampling:

- Identify performance bottlenecks
- Understand kernel execution behavior
- Analyze code coverage
- Find heavily executed code paths

To try out the PC sampling feature, you can use the ``rocprofv3`` command-line tool or the ROCprofiler-SDK library on ROCm 6.4 or later.

For a comparison of the HOST_TRAP and STOCHASTIC sampling methods, see :ref:`Sampling methods <glance-pc-sampling-methods>` in the SDK overview.

.. _pc-sampling-supported-gpus:

Supported AMD Instinct GPUs
============================

The following table lists the AMD Instinct™ GPUs supporting PC sampling and shows whether they support stochastic PC sampling and host-trap PC sampling:

.. list-table::
  :header-rows: 1

  * - AMD Instinct GPU
    - Architecture
    - Stochastic PC sampling
    - Host-trap PC sampling

  * - MI355X
    - CDNA4
    - ✅
    - ✅

  * - MI350X
    - CDNA4
    - ✅
    - ✅

  * - MI350P
    - CDNA4
    - ✅
    - ✅

  * - MI325X
    - CDNA3
    - ✅
    - ✅

  * - MI300X
    - CDNA3
    - ✅
    - ✅

  * - MI300A
    - CDNA3
    - ✅
    - ✅

  * - MI250X
    - CDNA2
    - ❌
    - ✅

  * - MI250
    - CDNA2
    - ❌
    - ✅

  * - MI210
    - CDNA2
    - ❌
    - ✅

.. note::

   AMD Instinct MI100 (CDNA1, gfx908) is not listed because it lacks the hardware support that PC sampling requires.

PC sampling availability and configuration
===========================================

To check if the GPU supports PC sampling, use:

.. code-block:: bash

  rocprofv3 -L

Or

.. code-block:: bash

  rocprofv3 --list-avail

The output lists if ``rocprofv3`` supports PC sampling on the GPU and the supported configuration.

.. code-block:: bash

    GPU:0
    NAME:gfx90a
    configs:
       Method              :host_trap
       Unit                :time
       Min_Interval        :1
       Max_Interval        :18446744073709551615
       Flags               :none

The preceding output shows that the GPU supports PC sampling with the ``ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP`` method and the ``ROCPROFILER_PC_SAMPLING_UNIT_TIME`` unit. The minimum and maximum intervals are also displayed.

.. note::
   Important firmware fixes to host-trap and stochastic PC-sampling for AMD Instinct MI300A and MI300X
   shipped in ROCm 7.0. Before enabling PC sampling, confirm that the GPU firmware revisions meet the
   minimum levels below.

   The rocprofiler-sdk checks **firmware version** (not feature version) for ``gfx942`` agents when PC
   sampling is configured:

   - **Host-trap** (PSP TOS / SOS): firmware version >= ``0x00360259`` (version 00.36.02.59)
   - **Stochastic** (MEC): firmware version >= ``0x000000b9``

   These values are read from the same sysfs files that ROCm SMI uses
   (for example, ``.../fw_version/sos_fw_version`` and ``.../fw_version/mec_fw_version``). If a version
   cannot be read, PC sampling configuration returns ``ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_FIRMWARE``.

Checking firmware versions (matches SDK checks)
-----------------------------------------------

Use the ``rocm-smi`` tool (ROCm SMI) to inspect the firmware revisions that the SDK compares. This does
not require root access:

.. code-block:: bash

  rocm-smi --showfwinfo

On MI300 systems, look for **SOS firmware version** (host-trap) and **MEC firmware version**
(stochastic). Example output:

.. code-block:: text

  GPU[0]          : SOS firmware version:         0x0036025d
  GPU[0]          : MEC firmware version:         192

``rocm-smi`` may print MEC as a decimal value; ``192`` is ``0x000000c0`` in hexadecimal. Ensure SOS is
at least ``0x00360259`` and MEC is at least ``0x000000b9`` (decimal ``185``).

Checking firmware feature versions (debugfs, optional)
------------------------------------------------------

Stochastic PC sampling on MI300 is also associated with **MEC feature version 50** in release notes.
The SDK does **not** read feature version today—only the firmware revision fields above.

To view both feature and firmware version for every engine, use the AMDGPU debugfs interface. This
requires ``sudo`` and the correct DRI index under ``/sys/kernel/debug/dri/``:

.. code-block:: bash

  # List available debugfs DRI indices (one per DRM card)
  ls /sys/kernel/debug/dri/

  # Replace <N> with the index that corresponds to your GPU (often matches the card number)
  sudo grep -E "MEC|SOS" /sys/kernel/debug/dri/<N>/amdgpu_firmware_info

Example debugfs lines:

.. code-block:: text

  MEC feature version: 50, firmware version: 0x000000c0
  SOS feature version: 3539549, firmware version: 0x0036025d

The DRI index is not always the same as the ``GPU[i]`` index shown by ``rocm-smi``. On multi-GPU
systems, inspect each ``/sys/kernel/debug/dri/<N>/amdgpu_firmware_info`` file (or correlate
``/sys/class/drm/card<N>/`` with the target device) until the output matches the GPU you intend to
profile.

Based on the available PC-sampling configurations, use the following command to profile the application using PC-sampling:

.. code-block:: bash

  rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 1 --output-format csv -- <application_path>

The preceding command enables PC sampling with the ``host_trap`` method, ``time`` unit, and an interval of ``1`` μs (microsecond). Replace ``<application_path>`` with the path to the application you want to profile.

This generates two files, ``agent_info.csv`` and ``pc_sampling_host_trap.csv``. Both files are prefixed with the process ID.

Here are the contents of ``pc_sampling_host_trap.csv`` file generated for MatrixTranspose sample application:

.. _pc_sampling_host_trap:

.. csv-table:: PC sampling host trap
   :file: /data/pc_sampling_host_trap.csv
   :widths: 20,10,10,10,10,20
   :header-rows: 1


For description of the fields in the output file, see :ref:`pc-sampling-fields`.

If you find the ``Instruction_Comment`` field in the output file to be empty, populate this field by compiling your application with debug symbols.
Enabling debug symbols while compiling the application maps back to the source line. This helps in understanding the code execution pattern and hotspots.

.. csv-table:: PC sampling host trap with debug symbols
   :file: /data/pc_sampling_host_trap_debug.csv
   :widths: 20,10,10,10,10,20
   :header-rows: 1


The preceding output shows the ``Instruction_Comment`` field populated with the source-line information.

.. _pc-sampling-fields:

PC sampling fields
===================

Here are the fields in the output file generated by PC sampling:

- ``Sample_Timestamp``: Timestamp when sample is generated
- ``Exec_Mask``: Active SIMD lanes when sampled
- ``Dispatch_Id``: Originating kernel dispatch ID
- ``Instruction``: Assembly instruction such as ``s_load_dword s8, s[1:2], 0x10``
- ``Instruction_Comment``: Instruction comment that maps back to the source-line if debug symbols were enabled when application was compiled
- ``Correlation_Id``: API launch call ID that matches dispatch ID

 To dump samples in a more comprehensive format, use JSON through ``--output-format json``:

.. code-block:: bash

  rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method host_trap --pc-sampling-unit time --pc-sampling-interval 1 --output-format json -- <application_path>

The preceding command generates a JSON file with the comprehensive output. Here is a trimmed down output with multiple records:

.. code-block:: text

  {
    "pc_sample_host_trap": [
      {
        "record": {
          "hw_id": {
            "chiplet": 0,
            "wave_id": 0,
            "simd_id": 2,
            "pipe_id": 0,
            "cu_or_wgp_id": 1,
            "shader_array_id": 0,
            "shader_engine_id": 2,
            "workgroup_id": 0,
            "vm_id": 3,
            "queue_id": 2,
            "microengine_id": 1
          },
          "pc": {
            "code_object_id": 1,
            "code_object_offset": 20228
          },
          "exec_mask": 18446744073709551615,
          "timestamp": 51040126667689,
          "dispatch_id": 1,
          "corr_id": {
            "internal": 1,
            "external": 0
          },
          "wrkgrp_id": {
            "x": 182,
            "y": 0,
            "z": 0
          },
          "wave_in_grp": 1
        },
        "inst_index": 0
      },
      {
        "record": {
          "hw_id": {
            "chiplet": 0,
            "wave_id": 0,
            "simd_id": 2,
            "pipe_id": 0,
            "cu_or_wgp_id": 0,
            "shader_array_id": 0,
            "shader_engine_id": 2,
            "workgroup_id": 0,
            "vm_id": 3,
            "queue_id": 2,
            "microengine_id": 1
          },
          "pc": {
            "code_object_id": 1,
            "code_object_offset": 20236
          },
          "exec_mask": 18446744073709551615,
          "timestamp": 51040126667689,
          "dispatch_id": 1,
          "corr_id": {
            "internal": 1,
            "external": 0
          },
          "wrkgrp_id": {
            "x": 158,
            "y": 0,
            "z": 0
          },
          "wave_in_grp": 2
        },
        "inst_index": 1
      }
    ]
  }

For description of the fields in the JSON output, see :ref:`output-file-fields`.

Host-trap PC sampling and arbitrary sampling skid
==================================================

Host-trap PC sampling is a software-based technique that utilizes a background kernel thread
to periodically interrupt running waves to capture the program counter (PC).
This method is effective for gathering performance data without requiring specialized hardware
to snapshot the waves. However, this method has certain limitations due to the potential delay between
receiving and processing the interrupt by the wave to capture the PC.
This delay can lead to a sampling skid, where the PC samples might be attributed to the instructions
that are up to two instructions away from the actual source of latency.
This results in a non-precise intra-kernel sampling method.

When analyzing an application profile generated by host-trap PC sampling,
it is important to consider not only the costliest reported instruction but
also the instructions immediately preceding or following it.
If the costly instruction is near a branch instruction, it is important
to consider the instruction targeted by the branch and the one immediately following it as well.

To address the limitations of host-trap sampling, the hardware-based stochastic PC sampling method
has been developed. This method provides precise intra-kernel sampling with zero sampling skid,
offering more accurate performance insights.

It is important to note that the skid issue inherent in host-trap PC sampling is not likely to be resolved
in its current form. Therefore, to achieve more precise performance profiling, it is recommended to adopt stochastic PC sampling starting with the gfx942 architecture.

Hardware-based (stochastic) PC sampling method
===============================================

The ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` method has been introduced for the gfx942 architecture.
It employs a specific hardware for probing waves actively running on the GPU.
Besides the information already provided with ``ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP`` useful for
determining hotspots within the kernel, this method delivers additional information, which helps to determine
whether a sampled wave issued an instruction represented with the specified PC.
If not, this method provides the reason for not issuing the instruction (stall reason).
Such information is particularly useful for understanding stalls during kernel execution.

To use this method on gfx942, it is recommended to list available PC sampling configurations to verify if the latest ROCm stack is installed on the system using:

.. code-block:: bash

  rocprofv3 -L

An output similar to the following indicates that the ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` method is available:

.. code-block:: bash

    GPU:1
    NAME:gfx942
    configs:
       Method              :stochastic
       Unit                :cycle
       Min_Interval        :256
       Max_Interval        :2147483648
       Flags               :interval pow2

.. note::

  On gfx942, ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` requires intervals to be specified in cycles with values as powers of 2.

To profile an application with ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` PC sampling enabled on gfx942, use:

.. code-block:: bash

  rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic --pc-sampling-unit cycles --pc-sampling-interval 1048576 --output-format csv, json -- <application_path>

The preceding command serializes samples in both CSV and JSON output formats in the ``pc_sampling_stochastic.csv`` and ``out_results.json`` files, respectively.

On comparing the :ref:`pc_sampling_stochastic.csv <pc_sampling_stochastic>` to :ref:`pc_sampling_host_trap.csv <pc_sampling_host_trap>`, you can notice that the ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` method
generates the following additional fields:

- ``Wave_Issued_Instruction``: Indicates whether the wave issued an instruction represented with the specified PC. Value = 1 for yes and 0 for no.

- ``Instruction_Type``: If the value of ``Wave_Issued_Instruction`` is 1, this field indicates the type of the issued instruction. Otherwise, this field remains irrelevant.

- ``Stall_Reason``: If the value of ``Wave_Issued_Instruction`` is 0, this field indicates the reason for not issuing the instruction (stall reason). Otherwise, this field remains irrelevant.

- ``Wave_Count``: Total number of waves actively running on a compute unit when the sample is generated.

.. _pc_sampling_stochastic:

.. csv-table:: PC sampling stochastic with debug symbols
   :file: /data/pc_sampling_stochastic_debug.csv
   :widths: 20,10,10,10,10,20,10,20,20,10
   :header-rows: 1

Similarly, ``ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC`` method delivers additional information to every sample in the JSON output.

Here is a ``out_results.json`` file sample:

.. code-block:: text

  {
    "record": {
      "flags": {
        "has_mem_cnt": 0
      },
      "hw_id": {
        "chiplet": 4,
        "wave_id": 0,
        "simd_id": 2,
        "pipe_id": 3,
        "cu_or_wgp_id": 1,
        "shader_array_id": 0,
        "shader_engine_id": 3,
        "workgroup_id": 0,
        "vm_id": 3,
        "queue_id": 2,
        "microengine_id": 1
      },
      "pc": {
        "code_object_id": 2,
        "code_object_offset": 13880
      },
      "exec_mask": 18446744073709551615,
      "timestamp": 390705261924637,
      "dispatch_id": 29,
      "corr_id": {
        "internal": 29,
        "external": 0
      },
      "wrkgrp_id": {
        "x": 9,
        "y": 489,
        "z": 0
      },
      "wave_in_grp": 0,
      "wave_issued": 1,
      "inst_type": "ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU",
      "wave_cnt": 6,
      "snapshot": {
        "stall_reason": "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT",
        "dual_issue_valu": 0,
        "arb_state_issue_valu": 1,
        "arb_state_issue_matrix": 0,
        "arb_state_issue_lds": 0,
        "arb_state_issue_lds_direct": 0,
        "arb_state_issue_scalar": 0,
        "arb_state_issue_vmem_tex": 0,
        "arb_state_issue_flat": 0,
        "arb_state_issue_exp": 0,
        "arb_state_issue_misc": 0,
        "arb_state_issue_brmsg": 0,
        "arb_state_stall_valu": 0,
        "arb_state_stall_matrix": 0,
        "arb_state_stall_lds": 0,
        "arb_state_stall_lds_direct": 0,
        "arb_state_stall_scalar": 0,
        "arb_state_stall_vmem_tex": 0,
        "arb_state_stall_flat": 0,
        "arb_state_stall_exp": 0,
        "arb_state_stall_misc": 0,
        "arb_state_stall_brmsg": 0
      }
    },
    "inst_index": 1
  },

Fields starting with ``arb_state_`` are of particular interest as they indicate the state of the arbiter at the time of sampling.
For example, ``arb_state_issue_`` fields indicate the type of instructions issued by the arbiter at the time of sampling.
On the other hand, ``arb_state_stall_`` fields indicate the type of instructions stalled at the time of sampling.
This information is useful for understanding how many instructions per cycle (IPC) are issued.

For information about how to analyze stochastic PC sampling data, see :ref:`cdna3-cdna4-pc-sampling`.
