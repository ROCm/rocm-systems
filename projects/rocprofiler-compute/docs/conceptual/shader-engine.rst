.. meta::
   :description: ROCm Compute Profiler performance model: Shader engine (SE)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, shader, engine, sL1D, L1I, workgroup manager, SPI

******************
Shader engine (SE)
******************

The :doc:`compute units <compute-unit>` on a CDNA™ accelerator are grouped
together into a higher-level organizational unit called a shader engine (SE):

.. figure:: ../data/performance-model/selayout.png
   :align: center
   :alt: Example of CU-grouping into shader engines
   :width: 800

   Example of CU-grouping into shader engines on AMD Instinct MI-series
   accelerators.

The number of CUs on a SE varies from chip to chip -- see for example
:hip-training-pdf:`20`. In addition, newer accelerators such as the AMD
Instinct™ MI 250X have 8 SEs per accelerator.

For the purposes of ROCm Compute Profiler, we consider resources that are shared between
multiple CUs on a single SE as part of the SE's metrics.

These include:

* The :ref:`scalar L1 data cache <desc-sl1d>`

* The :ref:`L1 instruction cache <desc-l1i>`

* The :ref:`workgroup manager <desc-spi>`

.. _desc-sl1d:

Scalar L1 data cache (sL1D)
===========================

The Scalar L1 Data cache (sL1D) can cache data accessed from scalar load
instructions (and scalar store instructions on architectures where they exist)
from wavefronts in the :doc:`CUs <compute-unit>`. The sL1D is shared between
multiple CUs (:gcn-crash-course:`36`) -- the exact number of CUs depends on the
architecture in question (3 CUs in GCN™ GPUs and MI100, 2 CUs in
:ref:`MI2XX <mixxx-note>`) -- and is backed by the :doc:`L2 cache <l2-cache>`.

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
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

The Scalar L1D speed-of-light chart shows some key metrics of the sL1D
cache as a comparison with the peak achievable values of those metrics:

.. jinja:: desc-sl1d-sol
   :file: _templates/metrics_table.j2

.. _desc-sl1d-stats:

Scalar L1D cache accesses
-------------------------

This panel gives more detail on the types of accesses made to the sL1D,
and the hit/miss statistics.

.. jinja:: desc-sl1d-stats
   :file: _templates/metrics_table.j2

.. _desc-sl1d-l2-interface:

sL1D ↔ L2 Interface
-------------------

This panel gives more detail on the data requested across the
sL1D↔
:doc:`L2 <l2-cache>` interface.

.. jinja:: desc-sl1d-l2-interface
   :file: _templates/metrics_table.j2

.. rubric:: Footnotes

.. [#uniform-access] The scalar data cache is used when the compiler emits
   scalar loads to access data. This requires that the data be *provably*
   uniformly accesses (that is, the compiler can verify that all work-items in a
   wavefront access the same data), *and* that the data can be proven to be
   read-only (for instance, HIP's ``__constant__`` memory, or properly
   ``__restrict__``\ed pointers to avoid write-aliasing). Access of
   ``__constant__`` memory for example is not guaranteed to go through the sL1D
   if the wavefront loads a non-uniform value.

.. [#sl1d-cache] Unlike the :doc:`vL1D <vector-l1-cache>` and
   :doc:`L2 <l2-cache>` caches, the sL1D cache on AMD Instinct MI-series CDNA
   accelerators does *not* use the "hit-on-miss" approach to reporting cache
   hits. That is, if while satisfying a miss, another request comes in that
   would hit on the same pending cache line, the subsequent request will be
   counted as a *duplicated miss*.

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
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

The L1 Instruction Cache speed-of-light chart shows some key metrics of
the L1I cache as a comparison with the peak achievable values of those
metrics:

.. jinja:: desc-l1i-sol
   :file: _templates/metrics_table.j2

.. _desc-l1i-stats:

L1I cache accesses
------------------

This panel gives more detail on the hit/miss statistics of the L1I:

.. jinja:: desc-l1i-stats
   :file: _templates/metrics_table.j2

.. _desc-l1i-l2-interface:

L1I - L2 interface
------------------

This panel gives more detail on the data requested across the
L1I-:doc:`L2 <l2-cache>` interface.

.. jinja:: desc-l1i-l2-interface
   :file: _templates/metrics_table.j2

.. rubric:: Footnotes

.. [#l1i-cache] Unlike the :doc:`vL1D <vector-l1-cache>` and
   :doc:`L2 <l2-cache>` caches, the L1I cache on AMD Instinct MI-series CDNA
   accelerators does *not* use the "hit-on-miss" approach to reporting cache
   hits. That is, if while satisfying a miss, another request comes in that
   would hit on the same pending cache line, the subsequent request will be
   counted as a *duplicated miss*.

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
manager’s metrics therefore are focused on reporting the following:

*  Utilizations of various parts of the accelerator that the workgroup
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

As discussed in :doc:`Command processor <command-processor>`, the command
processor on AMD Instinct MI-series architectures contains four hardware
scheduler-pipes, each with eight software threads (:mantor-vega10-pdf:`19`). Each
scheduler-pipe can issue a kernel dispatch to the workgroup manager to schedule
concurrently. Therefore, some workgroup manager metrics are presented relative
to the utilization of these scheduler-pipes (for instance, whether all four are
issuing concurrently).

.. note::

   Current versions of the profiling libraries underlying ROCm Compute Profiler attempt to
   serialize concurrent kernels running on the accelerator, as the performance
   counters on the device are global (that is, shared between concurrent
   kernels). This means that these scheduler-pipe utilization metrics are
   expected to reach (for example) a maximum of one pipe active -- only 25%.

.. _spi-util:

Workgroup manager utilizations
------------------------------

This section describes the utilization of the workgroup manager, and the
hardware components it interacts with.

.. jinja:: spi-util
   :file: _templates/metrics_table.j2

.. _spi-resc-util:

Resource allocation
-------------------

This panel gives more detail on how workgroups and wavefronts were scheduled
onto compute units, and what occupancy limiters they hit -- if any. When
analyzing these metrics, you should also take into account their
achieved occupancy -- such as
:ref:`wavefront occupancy <wavefront-runtime-stats>`. A kernel may be occupancy
limited by LDS usage, for example, but may still achieve high occupancy levels
such that improving occupancy further may not improve performance. See
:ref:`occupancy-example` for details.

.. jinja:: spi-resc-util
   :file: _templates/metrics_table.j2
