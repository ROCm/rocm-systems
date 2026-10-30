.. meta::
   :description: How to build fio with the libhipfile engine to benchmark hipFile I/O performance.
   :keywords: hipFile, fio, benchmark, ROCm, GPU I/O, libhipfile, direct I/O


***************************
Benchmark hipFile with fio
***************************

`fio <https://github.com/axboe/fio>`_ is a flexible I/O tester commonly used for storage benchmarking. fio versions 3.43 and later includes a ``libhipfile`` engine that lets you drive hipFile I/O workloads from fio job files. This page explains how to build fio with hipFile support.

Before you begin, install hipFile by following the steps in :doc:`/install/install`.

To build fio with the ``libhipfile`` engine, use the following commands:

.. code:: shell

   git clone https://github.com/axboe/fio.git
   mkdir fio/build && cd fio/build
   ../configure --enable-libhipfile
   make -j
   make install


The resulting ``fio`` binary includes the ``libhipfile`` I/O engine.
