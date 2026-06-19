.. meta::
  :description: Analyzing PC sampling data on gfx9 architecture
  :keywords: Sampling PC, Sampling program counter, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, PC sampling

.. _cdna3-cdna4-pc-sampling:

****************************************************************
Analyzing PC sampling data on CDNA3 and CDNA4 GPU architectures
****************************************************************

Program Counter (PC) sampling periodically samples waves running on a compute unit (CU) and reports whether the sampled wave issued an instruction in the sampled cycle. If the wave couldn't proceed to issue an instruction, a stall reason is recorded.
In addition to the wave's state, PC sampling captures the state of the sampled SIMD's arbiter (also referred to as the `scheduler <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html#scheduler>`_). The term arbiter is used in accordance with the PC sampling data fields. The arbiter state indicates whether any wavefront on the SIMD was issued to a given pipeline during the given cycle, and if so, whether that instruction began execution.

PC sampling on the CDNA3 (AMD Instinct™ MI300 Series) and CDNA4 (AMD Instinct MI350 Series) architectures primarily focuses on the frontend of shader execution, examining which waves are running and what prevents them from issuing instructions. It provides a limited view of the backend (execution pipelines), examining whether they are stalled and backpressuring the frontend, but not the underlying cause of the stall. For the list of pipelines available on the CDNA3 and CDNA4 architecture, see :ref:`execution-pipelines`.

Stall reasons
==============

The stall reason reported for a sample indicates why the sampled wave couldn't issue or execute an instruction during the given cycle. This contrasts with the arbiter state fields, which report the entire SIMD state. To learn more about the arbiter state, see :ref:`arbiter_state`.

The following table lists the stall reasons:

.. list-table:: Stall reasons
  :header-rows: 1

  * - Stall reason
    - Description

  * - NO_INSTRUCTION_AVAILABLE
    - The wave is stalled waiting for instructions. For example, at the branch target, I$ miss, and others.

  * - ALU_DEPENDENCY
    - The sampled wave's instruction couldn't be issued due to an internal hardware dependency, such as an inter-pipeline dependency or a data hazard. For more information, see 4.4 Data dependency resolution in `AMD Instinct MI300 (CDNA3) instruction set architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>`_ or `CDNA4 instruction set architecture <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>`_.

  * - WAITCNT
    - The sampled wave is waiting due to memory dependency (``waitcnt``).

  * - INTERNAL_INSTRUCTION
    - The sampled wave is issuing an internal instruction, such as a ``NOP``.

  * - BARRIER_WAIT
    - The sampled wave is waiting at a barrier for the other waves in the workgroup to reach that barrier.

  * - ARBITER_NOT_WIN
    - The sampled wave isn't selected to issue instructions. This typically occurs when multiple waves compete to issue instructions of the same type (served by the same execution pipeline) simultaneously. Only one wave wins the arbitration. For more information about wave arbitration, see `Scheduler documentation <https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/conceptual/pipeline-descriptions.html#scheduler>`_.

  * - ARBITER_WIN_EX_STALL
    - The wave was ready to issue an instruction, which the arbiter then selected to issue. However, the execution pipeline backpressured the wave, preventing it from issuing the instruction because it couldn't accept more instructions.

  * - OTHER
    - Other reasons for stalling a wave, such as a recoverable page-fault (``XNACK``).

Occupancy
==========

``wave_cnt`` reports the number of active waves present on the CU (across all four SIMDs) at the time of sampling. This information helps explain the evolution of CU occupancy over time. The information includes details, such as whether the CU was fully loaded throughout kernel execution, whether waves were draining towards the end, how occupancy affects the cost of stalls, and the occupancy at specific source or assembly lines at the time of sampling.

Note that ``wave_cnt`` doesn't provide a full timeline trace. For full timeline tracing, use `Thread trace <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>`_.

.. _execution-pipelines:

Execution pipelines
====================

The following table lists the execution pipelines available on CDNA3 and CDNA4 architectures:

.. list-table:: Execution pipelines
  :header-rows: 1

  * - Execution pipeline
    - Description

  * - VALU
    - Vector ALU pipeline

  * - Matrix
    - Matrix (MFMA) pipeline

  * - LDS
    - Local data share pipeline

  * - Scalar
    - Scalar ALU/memory pipeline

  * - Tex (VMEM_TEX)
    - Texture/vector memory pipeline

  * - Flat
    - Flat memory pipeline

  * - Exp
    - Export pipeline

  * - Misc
    - Miscellaneous pipeline

.. _arbiter_state:

Arbiter state
==============

The arbiter state (``arbiter_state_*`` or ``arb_state_*``) describes the arbiter activity during the sampled cycle. These variables can be used to analyze each pipeline activity and the overall CU utilization.

- ``arbiter_state_issue_PIPE``: Indicates whether the arbiter picked any instruction from the ready instructions of all active waves, implying that the waves weren't blocked from issuing an instruction for execution on the given execution pipeline due to reasons such as NO_INSTRUCTION_AVAILABLE, ALU_DEPENDENCY, WAITCNT, INTERNAL_INSTRUCTION, or BARRIER_WAIT. Note that PIPE is the execution pipeline responsible for servicing the instruction type of the sampled wave.

- ``arbiter_state_stall_PIPE``: Indicates whether a given execution pipeline backpressured an issued instruction, implying that the pipeline couldn't accept the instruction and the arbiter will try issuing it again in a later cycle.

The following table lists the conditions based on the ``arbiter_state_*`` values to determine the arbiter activity during the sampled cycle:

.. list-table:: Diagnosing arbiter activity
  :header-rows: 1

  * - Condition
    - Diagnosis

  * - ``arbiter_state_issue_PIPE`` = true/false
    - ARBITER_NOT_WIN stalls. For details, see :ref:`arbiter_not_win`.

  * - ``arbiter_state_issue_PIPE`` == 0 OR ``arbiter_state_stall_PIPE`` == 1
    - | For one pipe, Pipe latency.
      | For all pipes, SIMD latency.
      | For details, see :ref:`latency_stalls`.

  * - ``arbiter_state_issue_PIPE`` == 0 AND ``arbiter_state_stall_PIPE`` == 0
    - Frontend pipe latency. For details, see :ref:`latency_stalls`.

  * - ``arbiter_state_issue_PIPE`` == 1 AND ``arbiter_state_stall_PIPE`` == 0
    - Instructions issued per cycle (IPC) estimation. For details, see :ref:`ipc_approximation`.

  * - ``arbiter_state_issue_PIPE`` == 1 AND ``arbiter_state_stall_PIPE`` == 0 with the wave stalled due to ``ARBITER NOT WIN``
    - Pipeline oversubscription. For details, see :ref:`pipeline_oversubscribe`.

  * - ``arbiter_state_stall_PIPE`` == 1 with the wave stalled due to ``ARBITER_WIN_EX_STALL``
    - Pipeline backpressuring. For details, see :ref:`pipeline-backpressure`.

.. _arbiter_not_win:

ARBITER_NOT_WIN stalls
-----------------------

To diagnose the ARBITER_NOT_WIN stalls, check the ``arbiter_state_issue_PIPE`` value:

- ``arbiter_state_issue_PIPE`` = true: Another wave won the arbitration for that PIPE in the sampled cycle. This confirms the classic multiwave contention scenario. To determine whether the winning wave's instruction actually began execution, check ``arbiter_state_stall_PIPE``. Also, look for pipeline hotspotting, as high contention on a single pipe is not always beneficial.

- ``arbiter_state_issue_PIPE`` = false: No wave was issued on that PIPE in the sampled cycle. This indicates that the PIPE wasn't ready to accept an instruction in the sampled cycle and is a rare scenario.

.. _latency_stalls:

Latency stalls
---------------

To detect different types of latency stalls, check the ``arbiter_state_issue_PIPE`` and ``arbiter_state_stall_PIPE`` values:

- **Pipe latency:** This is caused when one PIPE is stalled. It is indicated by the condition where ``arbiter_state_issue_PIPE == 0`` OR ``arbiter_state_stall_PIPE == 1``. This implies that no instruction made forward progress on that PIPE in the sampled cycle.

- **SIMD latency:** This is caused when all PIPEs are stalled. It is indicated by the condition where for all the pipes, ``arbiter_state_issue_PIPE == 0`` OR ``arbiter_state_stall_PIPE == 1``. This implies that no instruction made forward progress on any PIPE in the sampled cycle. This is a full SIMD stall where no wave from the sampled SIMD made progress.

- **Frontend pipe latency:** This implies that no wave was issued to the PIPE in the sampled cycle. It is indicated by the condition where ``arbiter_state_issue_PIPE == 0`` AND ``arbiter_state_stall_PIPE == 0``.

.. _ipc_approximation:

IPC approximation
------------------

Arbiter states can also be used to estimate the number of IPC, which is specifically the count of pipes with ``arbiter_state_issue_PIPE == 1`` AND ``arbiter_state_stall_PIPE == 0``.

.. note::

    Under specific circumstances, the GPU can co-issue two VALU instructions in the same clock cycle.
    When this occurs, the PC sampling data records ``dual_issue_valu`` = 1 in the sample. This has implications for IPC estimation; a sample with ``arbiter_state_issue_valu`` == 1 AND ``arbiter_state_stall_valu`` == 0 AND ``dual_issue_valu`` == 1 represents two VALU instructions issued in that cycle. To obtain an accurate IPC estimate, VALU contributions from such samples should be counted as two instead of one. For more information, see `Why does VALU utilization exceed the theoretical peak? <https://github.com/ROCm/rocm-systems/blob/7bd3b0f4870adcec97fabaf7442566345da105e7/projects/rocprofiler-compute/docs/reference/faq.rst#why-does-valu-utilization-exceed-the-theoretical-peak>`_.

.. _pipeline_oversubscribe:

Execution pipeline oversubscription
------------------------------------

To identify pipeline oversubscription, look for samples where a wave is stalled due to ARBITER_NOT_WIN, despite the corresponding pipe having simultaneously accepted an instruction (indicated by ``arbiter_state_issue_PIPE`` == 1 AND ``arbiter_state_stall_PIPE`` == 0). This indicates that multiple waves on the sampled SIMD were competing to issue on the same pipe, and the pipe was able to serve at least one of those requests.

.. _pipeline-backpressure:

Pipeline backpressuring
------------------------

Sometimes, the execution pipeline backpressures a wave from issuing an instruction, even when the arbiter has chosen that wave. This is indicated by ``arbiter_state_stall_PIPE`` == 1 .
A backpressured wave will be stalled due to ARBITER_WIN_EX_STALL.

Pipeline backpressuring can occur due to the following conditions:

- **Oversubscription:** When a pipeline is oversubscribed, it might start backpressuring waves. Some pipes can accept only a limited number of outstanding instructions at a time. When that limit is exceeded, the pipe stops accepting new instructions and backpressures waves trying to issue instructions.

- **long-latency instructions:** Backpressuring can also occur without pipeline oversubscription. For example, waves issuing instructions targeting different pipelines don't tend to cause contention. However, some instructions, such as MFMA operations and vector transcendentals, are long latency and require multiple quad-cycles to execute. If a wave attempts to issue long-latency instructions back-to-back, the pipe might backpressure it.
