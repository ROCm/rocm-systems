.. meta::
   :description: ROCm Compute Profiler performance model: Shader engine (SE)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, shader, engine, sL1D, L1I, workgroup manager, SPI

******************
Shader engine (SE)
******************

The :doc:`compute units <compute-unit>` on a CDNA™ architecture-based GPUs are grouped
together into a higher-level organizational unit called a shader engine (SE):

.. figure:: ../../data/performance-model/selayout.png
   :align: center
   :alt: Example of CU-grouping into shader engines
   :width: 800

The number of CUs on a SE varies from chip to chip (For example, see
:hip-training-pdf:`20`).

For the purposes of ROCm Compute Profiler, resources that are shared between
multiple CUs on a single SE as part of the SE's metrics are considered.

These include:

* :ref:`Scalar L1 data cache <desc-sl1d>`

* :ref:`L1 instruction cache <desc-l1i>`

* :ref:`Workgroup manager <desc-spi>`

.. _desc-sl1d:

Scalar L1 data cache (sL1D)
===========================

The Scalar L1 Data cache (sL1D) can cache data accessed from scalar load
instructions (and scalar store instructions on GPUs where they exist)
from wavefronts in the :doc:`CUs <compute-unit>`. The sL1D is shared between
multiple CUs (:gcn-crash-course:`36`). The exact number of CUs depends on the
architecture (3 CUs in GCN™ GPUs and AMD Instinct MI100 GPUs, 2 CUs in
AMD Instinct :ref:`MI2XX <mixxx-note>` GPUs), and is backed by the :doc:`L2 cache <l2-cache>`.

In typical usage, the data in the sL1D is comprised of:

* Kernel arguments, such as pointers,
  `non-populated <https://llvm.org/docs/AMDGPUUsage.html#amdgpu-amdhsa-sgpr-register-set-up-order-table>`_
  grid and block dimensions, and others

* HIP's ``__constant__`` memory, when accessed in a provably uniform manner
  [#uniform-access]_

* Other memory, when accessed in a provably uniform manner, *and* the backing
  memory is provably constant [#uniform-access]_

.. _desc-sl1d-sol:

Scalar L1D Speed-of-Light
-------------------------

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for a GPU. This may not be realistic for all
   workloads.

The Scalar L1D speed-of-light chart shows some key metrics of the sL1D
cache as a comparison with the peak achievable values of those metrics:

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-sl1d-sol-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-sl1d-sol-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-sl1d-sol-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-sl1d-sol-gfx950
         :file: _templates/metrics_table.j2

.. _desc-sl1d-stats:

Scalar L1D cache accesses
-------------------------

This section details the types of accesses made to the sL1D,
and the hit/miss statistics.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-sl1d-stats-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-sl1d-stats-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-sl1d-stats-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-sl1d-stats-gfx950
         :file: _templates/metrics_table.j2

.. _desc-sl1d-l2-interface:

sL1D ↔ L2 Interface
-------------------

This section details the data requested across the sL1D↔:doc:`L2 <l2-cache>` interface.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-sl1d-l2-interface-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-sl1d-l2-interface-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-sl1d-l2-interface-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-sl1d-l2-interface-gfx950
         :file: _templates/metrics_table.j2

.. rubric:: Footnotes

.. [#uniform-access] The scalar data cache is used when the compiler generates
   scalar load instructions to access data. This requires that the data can have a guaranteed
   uniform accesses (the compiler can verify that all work-items in a
   wavefront access the same data), and that the data can be proven to be
   read-only (for instance, HIP's ``__constant__`` memory, or properly
   ``__restrict__``\ed pointers to avoid write-aliasing). For example, access of
   ``__constant__`` memory is not guaranteed to go through the sL1D
   if the wavefront loads a non-uniform value.

.. [#sl1d-cache] Unlike the :doc:`vL1D <vector-l1-cache>` and
   :doc:`L2 <l2-cache>` caches, the sL1D cache on CDNA architecture-based AMD Instinct MI-Series GPUs doesn't use the "hit-on-miss" approach to report the cache
   hits. While satisfying a miss, if another request comes in that
   would hit on the same pending cache line, the subsequent request will be
   counted as a duplicated miss.

.. _desc-l1i:

L1 Instruction Cache (L1I)
==========================

As with the :ref:`sL1D <desc-sL1D>`, the L1 Instruction (L1I) cache is shared
between multiple CUs on a shader-engine, where the precise number of CUs
sharing a L1I depends on the architecture in question (:gcn-crash-course:`36`)
and is backed by the :doc:`L2 cache <l2-cache>`. Unlike the sL1D, the
instruction cache is read-only.

.. _desc-l1i-sol:

L1I Speed-of-Light
------------------

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for a GPU. This may not be realistic for all
   workloads.

The L1 Instruction Cache speed-of-light chart shows some key metrics of
the L1I cache as a comparison with the peak achievable values of those
metrics:

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-l1i-sol-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-l1i-sol-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-l1i-sol-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-l1i-sol-gfx950
         :file: _templates/metrics_table.j2

.. _desc-l1i-stats:

L1I cache accesses
------------------

This section details the hit/miss statistics of the L1I:

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-l1i-stats-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-l1i-stats-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-l1i-stats-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-l1i-stats-gfx950
         :file: _templates/metrics_table.j2

.. _desc-l1i-l2-interface:

L1I - L2 interface
------------------

This section details the data requested across the L1I-:doc:`L2 <l2-cache>` interface.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: desc-l1i-l2-interface-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: desc-l1i-l2-interface-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: desc-l1i-l2-interface-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: desc-l1i-l2-interface-gfx950
         :file: _templates/metrics_table.j2

.. rubric:: Footnotes

.. [#l1i-cache] Unlike the :doc:`vL1D <vector-l1-cache>` and
   :doc:`L2 <l2-cache>` caches, the L1I cache on CDNA architecture-based AMD Instinct MI-Series GPUs doesn't use the "hit-on-miss" approach to report the cache
   hits. While satisfying a miss, if another request comes in that
   would hit on the same pending cache line, the subsequent request will be
   counted as a duplicated miss.

.. _desc-spi:

Workgroup manager (SPI)
=======================

The workgroup manager (SPI) is the bridge between the
:doc:`command processor <command-processor>` and the
:doc:`compute units <compute-unit>`. After the command processor processes a
kernel dispatch, it will then pass the dispatch off to the workgroup manager,
which then schedules :ref:`workgroups <desc-workgroup>` onto the compute units.
As workgroups complete execution and resources become available, the
workgroup manager will schedule new workgroups onto compute units. The workgroup
manager's metrics therefore are focused on reporting the following:

*  Utilizations of various parts of the GPU that the workgroup
   manager interacts with (and the workgroup manager itself)

*  How many workgroups were dispatched, their size, and how many
   resources they used

*  Percent of scheduler opportunities (cycles) where workgroups failed
   to dispatch, and

*  Percent of scheduler opportunities (cycles) where workgroups failed
   to dispatch due to lack of a specific resource on the CUs (for instance, too
   many VGPRs allocated)

This gives you an idea of why the workgroup manager couldn’t schedule more
wavefronts onto the device, and is most useful for workloads that you suspect to
be limited by scheduling or launch rate.

The command processor on AMD Instinct MI-Series GPUs contains four hardware
scheduler-pipes, each with eight software threads (:mantor-vega10-pdf:`19`). Each
scheduler-pipe can issue a kernel dispatch to the workgroup manager to schedule
concurrently. Therefore, some workgroup manager metrics are presented relative
to the utilization of these scheduler-pipes (for instance, whether all four are
issuing concurrently). For more details, see :doc:`Command processor <command-processor>`.

.. note::

   Current versions of the profiling libraries underlying ROCm Compute Profiler attempt to
   serialize concurrent kernels running on the GPU, as the performance
   counters on the device are global (shared between concurrent
   kernels). For example, this means that these scheduler-pipe utilization metrics are
   expected to reach a maximum of one pipe active.

.. _spi-util:

Workgroup manager utilizations
------------------------------

This section details the utilization of the workgroup manager, and the
hardware components it interacts with.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: spi-util-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: spi-util-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: spi-util-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: spi-util-gfx950
         :file: _templates/metrics_table.j2

.. _spi-resc-util:

Resource allocation
-------------------

This panel gives more detail on how workgroups and wavefronts are scheduled
onto compute units, and what occupancy limiters they hit, if any. When
analyzing these metrics, you should also take into account their
achieved occupancy, such as
:ref:`wavefront occupancy <wavefront-runtime-stats>`. A kernel may be occupancy
limited by LDS usage, but as an example, may still achieve high occupancy levels
such that improving occupancy further may not improve performance. For more details, see
:ref:`occupancy-example`.

.. tab-set::

   .. tab-item:: CDNA

      .. jinja:: spi-resc-util-gfx908
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 2

      .. jinja:: spi-resc-util-gfx90a
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 3

      .. jinja:: spi-resc-util-gfx942
         :file: _templates/metrics_table.j2

   .. tab-item:: CDNA 4
      :selected:

      .. jinja:: spi-resc-util-gfx950
         :file: _templates/metrics_table.j2
