
.. meta::
   :description: How to build and run the ROCm fio fork with the libhipfile engine to benchmark hipFile IO performance.
   :keywords: hipFile, fio, benchmark, ROCm, GPU IO, libhipfile, direct IO


****************************
Benchmark hipFile with fio
****************************

`fio <https://github.com/axboe/fio>`_ is a flexible IO tester commonly used for storage benchmarking. AMD maintains a fork at `ROCm/fio <https://github.com/ROCm/fio>`_ that adds a ``libhipfile`` engine, letting you drive hipFile IO workloads from fio job files. This page explains how to build the fork, configure it against a local hipFile build, and run an example workload.

Before you begin, install hipFile by following the steps in :doc:`/install/install`. The instructions below assume hipFile has been built but not necessarily installed system-wide.

Build fio with hipFile support:

.. code:: shell

   git clone https://github.com/ROCm/fio.git
   cd fio
   git checkout hipFile
   mkdir build && cd build
   ../configure --enable-libhipfile
   make -j
   make install


The resulting ``fio`` binary in the current directory includes the ``libhipfile`` IO engine.
