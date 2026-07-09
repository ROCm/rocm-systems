.. meta::
   :description: Understand the AMD CDNA architectures and hardware blocks with ROCm Compute Profiler to analyze and optimize performance on AMD Instinct MI series GPUs.
   :keywords: ROCm Compute Profiler, CDNA, Instinct, performance model, MI-series

.. _cdna-performance-model:

***********************************
AMD CDNA architecture (CDNA-CDNA4)
***********************************

ROCm Compute Profiler makes available an extensive list of metrics to better understand the
achieved application performance on AMD Instinct™ MI-series GPUs
including: 

* CDNA™ architecture based GPUs: AMD Instinct MI100
* CDNA2 architecture based GPUs: AMD Instinct MI210, MI250, and MI250X
* CDNA3 architecture based GPUs: AMD Instinct MI300A, MI300X, and MI325X, and 
* CDNA4 architecture based GPUs: AMD Instinct MI350X and MI355X.

.. note::

  * For AMD Ryzen™ / RDNA™ APUs (e.g. gfx115x/RDNA3.5), see :doc:`RDNA3 <../rdna/rdna-performance-model>`.

  * For top-level metrics details on CDNA and RDNA architecture, see :doc:`../performance-model`.

The table provides key details and support available for the different CDNA architectures:

✅: Supported
❌: Unsupported

**Architecture details**

.. table::
  :widths: 30 30 30 30 30

  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+
  |Architecture     |CDNA       |CDNA 2                           |CDNA 3                               |CDNA 4                   |
  +=================+===========+=================================+=====================================+=========================+
  |Chip packaging   |Single Die |Up to two graphics Compute Dies  |One logical processor with dozen     |Similar to CDNA3,        |
  |                 |           |(GCDs) into single package.      |chiplets, configurable with partition|Multi-Die chiplet, but   |
  |                 |           |                                 |modes.                               |with two I/O Dies (IODs) |
  +-----------------+-----------+---------------------------------+-------------------------------------+-------------------------+
  |Supported series |MI100      |MI210                            |MI300A                               |MI350X                   |
  |                 |           +---------------------------------+-------------------------------------+-------------------------+
  |                 |           |MI250                            |MI300X                               |MI355X                   |
  |                 |           +---------------------------------+-------------------------------------+-------------------------+
  |                 |           |MI250X                           |MI325X                               |                         |
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
        - FP8/BF8 GEMM
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
hardware blocks of AMD Instinct GPUs. Refer to the following top-level GPU architecture diagram to understand the hardware blocks of each architecture.

.. tab-set::

  .. tab-item:: CDNA

    .. image:: ../../data/conceptual/CDNA.png
      :alt: CDNA top level architecture diagram with zoomed view of Compute unit

  .. tab-item:: CDNA2

    .. image:: ../../data/conceptual/CDNA2.png
      :alt: CDNA2 top level architecture diagram with zoomed view of Compute unit

  .. tab-item:: CDNA3

    .. image:: ../../data/conceptual/CDNA3.png
      :alt: CDNA3 top level architecture diagram with zoomed view of Accelerator Complex Dies (XCDs)

  .. tab-item:: CDNA4

    .. image:: ../../data/conceptual/CDNA4.png
      :alt: CDNA4 top level architecture diagram

This section describes each hardware block on the GPUs as interacted with by a software developer to
give a deeper understanding of the metrics reported by profiling data. Refer to
:doc:`/tutorial/profiling-by-example` for more practical examples and details on how
to use ROCm Compute Profiler to optimize your code.

.. _mixxx-note:

.. note::

   In this documentation, **MI2XX** refers to any of the CDNA2 architecture-based MI200 series GPUs, such as AMD
   Instinct MI250X, MI250, and MI210 GPUs interchangeably in cases
   where the exact product at hand is not relevant. For product details, see `AMD Instinct GPUs <https://www.amd.com/en/products/accelerators/instinct.html>`_.

   For a comparison of AMD Instinct GPU specifications, refer to
   :doc:`Hardware specifications <rocm:reference/gpu-arch-specs>`.

Hardware blocks
================

The AMD Instinct performance model is divided into the following blocks:

* :doc:`system-speed-of-light`

* :doc:`compute-unit`

* :doc:`l2-cache`

* :doc:`shader-engine`

* :doc:`command-processor`

* :doc:`references`
