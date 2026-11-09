.. meta::
   :description: ROCm Compute Profiler RDNA4 GL0 (TCP vector cache) and LDS metrics
   :keywords: ROCm Compute Profiler, RDNA, RDNA4, gfx1201, GL0, TCP, L0, LDS

.. _rdna4-gl0:

=========================
GL0 (TCP Vector Cache)
=========================

**GL0** is the vector L0 cache in front of the GL1 write-combining cache. Navi 48 (gfx1201) has 64 GL0
instances, one per CU, each 32 KB with a 128-byte cache line. The RDNA4 ISA shows
two L0 read/write caches per WGP. Hardware counters keep the ``TCP_*`` prefix on gfx1201.

Each WGP also has a 128 KB Local Data Share (LDS) with 64 banks. A workgroup can
allocate up to 64 KB of LDS.

For the GL1 write-combining cache and the GL1-GL2 interface, see :doc:`gl1-cache`.
The handoff toward GL2 is under :doc:`gl2-cache`.

.. note::

   On RDNA4, GL0 and TCP refer to the same cache. Hardware counter names
   (for example, ``TCP_REQ_sum``) retain the TCP prefix.

Memory chart: GL0 and LDS
=========================

The following memory chart tables follow the vector path from GL0 and LDS to the
GL0-GL1 interface.

Memory chart - GL0 cache
------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl0-cache-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - LDS
------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-lds-gfx1201
         :file: _templates/metrics_table.j2

Memory chart - GL0-GL1 interface
--------------------------------

.. tab-set::

   .. tab-item:: RDNA4 (gfx1201)
      :selected:

      .. jinja:: rdna4-memory-chart-gl0-gl1-interface-gfx1201
         :file: _templates/metrics_table.j2

GL0 cache and LDS panel
=======================

The GL0 Cache (TCP Cache) panel (panel 900) has tables for GL0 utilization,
requests, cache performance, the GL0-GL1 interface, and GL0 stalls. These
tables have metric descriptions in the analysis report. Run
``rocprof-compute analyze`` with ``--block 9`` to show the panel.
