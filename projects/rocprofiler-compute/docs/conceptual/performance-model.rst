.. meta::
   :description: ROCm Compute Profiler performance model
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, performance, model, profiler, tool, Instinct,
              accelerator, AMD, CDNA

*****************
Performance model
*****************

ROCm Compute Profiler makes available an extensive list of metrics to better understand
achieved application performance on AMD Instinct™ MI-series accelerators
including Graphics Core Next™ (GCN) GPUs like the AMD Instinct MI50, CDNA™
accelerators like the MI100, CDNA2 accelerators such as the AMD Instinct MI250X, MI250,
and MI210, CDNA3 accelerators such as the AMD Instinct MI300A, MI300X, MI325X, and CDNA4 accelerators such as MI350X and MI355X.

The table provides key details and support available for the different architectures:

✅: Supported
❌: Unsupported

**Architecture details**

.. table::
  :widths: 30 30 30 30 30

  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+
  |Architecture     |CDNA       |CDNA 2                           |CDNA 3                               |CDNA 4                   |
  +=================+===========+=================================+=====================================+=========================+
  |Chip packaging   |Single Die |Two graphics Compute Dies (GCDs) |One logical processor with dozen     |Similar to CDNA3,        |
  |                 |           |into single package.             |chiplets, configurable with partition|Multi-Die chiplet, but   |
  |                 |           |                                 |modes.                               |with two I/O Dies (IODs) |
  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+
  |Supported series |MI100      |MI200                            |MI300A                               |MI350X                   |
  |                 |           +---------------------------------+-------------------------------------+-------------------------+
  |                 |           |MI210                            |MI300X                               |MI355X                   |
  |                 |           +---------------------------------+-------------------------------------+-------------------------+
  |                 |           |MI250                            |MI325X                               |                         |
  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+
  |Spatial partition|❌         |❌                               |Compute partition mode and           |Compute partition mode   |
  |mode             |           |                                 |Memory partition mode                |and Memory partition mode|
  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+

**Data type support**

.. list-table::
      :header-rows: 1

      *
        - Architecture
        - FP32
        - FP64
        - FP16
        - INT32 ADD/LOGIC/MAD
        - INT8 DOT
        - INT4 DOT
        - FP32 GEMM
        - FP64 GEMM
        - FP16 GEMM
        - BF16 GEMM
        - INT8 GEMM
        - Packed FP32
        - TF32 GEMM
        - FP8/BF8
      *
        - CDNA
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
        - ❌
      *
        - CDNA2
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ❌
        - ❌
      *
        - CDNA3
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
      *
        - CDNA4
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ✅
        - ❌
        - ✅

To best use profiling data, it's important to understand the role of various
hardware blocks of AMD Instinct accelerators. Refer to the following top level GPU architecture diagram to understand the hardware blocks of each architectures. 

.. tab-set::

  .. tab-item:: CDNA

    .. image:: ../data/conceptual/CDNA.png
      :alt: CDNA top level architecture diagram with zoomed view of Compute unit

  .. tab-item:: CDNA2

    .. image:: ../data/conceptual/CDNA2.png
      :alt: CDNA2 top level architecture diagram with zoomed view of Compute unit

  .. tab-item:: CDNA3

    .. image:: ../data/conceptual/CDNA3.png
      :alt: CDNA3 top level architecture diagram with zoomed view of Accelerator Complex Dies (XCDs)

  .. tab-item:: CDNA4

    .. image:: ../data/conceptual/CDNA4.png
      :alt: CDNA4 top level architecture diagram

This section describes each hardware block on the accelerator as interacted with by a software developer to
give a deeper understanding of the metrics reported by profiling data. Refer to
:doc:`/tutorial/profiling-by-example` for more practical examples and details on how
to use ROCm Compute Profiler to optimize your code.

.. _mixxx-note:

.. note::

   In this documentation, **MI2XX** refers to any of the CDNA2 architecture-based MI200 series accelerators such as AMD
   Instinct MI250X, MI250, and MI210 accelerators interchangeably in cases
   where the exact product at hand is not relevant. For product details, see `AMD Instinct GPUs <https://www.amd.com/en/products/accelerators/instinct.html>`_.

   For a comparison of AMD Instinct accelerator specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`.

In this chapter, the AMD Instinct performance model used by ROCm Compute Profiler is divided into a handful of
key hardware blocks, each detailed in the following sections:

* :doc:`compute-unit`

* :doc:`l2-cache`

* :doc:`shader-engine`

* :doc:`command-processor`

* :doc:`system-speed-of-light`