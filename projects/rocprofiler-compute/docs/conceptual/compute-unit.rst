.. meta::
   :description: ROCm Compute Profiler performance model: Compute unit (CU)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, GCN, compute, unit, pipeline, workgroup, wavefront,
              CDNA

*****************
Compute unit (CU)
*****************

The compute unit (CU) is responsible for executing a user's kernels on
CDNA™-based accelerators. All :ref:`wavefronts <desc-wavefront>` of a
:ref:`workgroup <desc-workgroup>` are scheduled on the same CU.

.. image:: ../data/performance-model/gcn_compute_unit.png
   :align: center
   :alt: AMD CDNA accelerator compute unit diagram
   :width: 800

The CU consists of several independent execution pipelines and functional units.
The :doc:`/conceptual/pipeline-descriptions` section details the various
execution pipelines -- VALU, SALU, LDS, scheduler, and so forth. The metrics
presented by ROCm Compute Profiler for these pipelines are described in
:doc:`pipeline-metrics`. The :doc:`vL1D <vector-l1-cache>` cache and
:doc:`LDS <local-data-share>` are described in their own sections.

* The :ref:`desc-valu` is composed of multiple SIMD (single
  instruction, multiple data) vector processors, vector general purpose
  registers (VGPRs) and instruction buffers. The VALU is responsible for
  executing much of the computational work on CDNA accelerators, including but
  not limited to floating-point operations (FLOPs) and integer operations
  (IOPs).

* The vector memory (VMEM) unit is responsible for issuing loads, stores and
  atomic operations that interact with the memory system.

* The :ref:`desc-salu` is shared by all threads in a
  :ref:`wavefront <desc-wavefront>`, and is responsible for executing
  instructions that are known to be uniform across the wavefront at compile
  time. The SALU has a memory unit (SMEM) for interacting with memory, but it
  cannot issue separately from the SALU.

* The :doc:`local-data-share` is an on-CU software-managed scratchpad memory
  that can be used to efficiently share data between all threads in a
  :ref:`workgroup <desc-workgroup>`.

* The :ref:`desc-scheduler` is responsible for issuing and decoding instructions
  for all the :ref:`wavefronts <desc-wavefront>` on the compute unit.

* The :doc:`vector L1 data cache (vL1D) <vector-l1-cache>` is the first level
  cache local to the compute unit. On current CDNA accelerators, the vL1D is
  write-through. The vL1D caches from multiple compute units are kept coherent
  with one another through software instructions.

* CDNA accelerators -- that is, AMD Instinct™ MI100 and newer -- contain
  specialized matrix-multiplication accelerator pipelines known as the
  :ref:`desc-mfma`.

For a more in-depth description of a compute unit on a CDNA accelerator, see
:hip-training-pdf:`22` and :gcn-crash-course:`27`.
