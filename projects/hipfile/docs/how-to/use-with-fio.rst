hipFile I/O performance.
   :keywords: hipFile, fio, benchmark, ROCm, GPU I/O, libhipfile, direct I/O


***************************
Benchmark hipFile with fio
***************************

`fio <https://github.com/axboe/fio>`_ is a flexible I/O tester commonly used for storage benchmarking. fio versions 3.43 and later include a ``libhipfile`` engine that lets you drive hipFile I/O workloads from fio job files.

Before you begin, install hipFile by following the steps in :doc:`/install/install`. hipFile doesn't need to be installed system-wide to build fio.

To build fio with the ``libhipfile`` engine, use the following commands:

.. code:: shell

   git clone https://github.com/ROCm/fio.git
   cd fio
   git checkout hipFile
   mkdir build && cd build
   ../configure --enable-libhipfile
   make -j
   make install

The resulting ``fio`` binary will include the ``libhipfile`` I/O engine.