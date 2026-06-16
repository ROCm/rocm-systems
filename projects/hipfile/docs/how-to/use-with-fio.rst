
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

Run the example workload
************************

The hipFile repository ships a ready-made fio job file at ``util/fio/write-read-verify.fio``. Run it by setting two environment variables:

``GPU_DEV_IDS``
   Comma-separated list of GPU device indices to use (for example, ``0``).

``FIO_DIR``
   Path to a directory on an ext4 or xfs filesystem where fio creates its test files.

.. code-block:: shell

   GPU_DEV_IDS=0 FIO_DIR=/path/to/ext4_or_xfs_directory ~/fio/fio ~/fio/examples/libhipfile-hipfile.fio

Replace ``/path/to/ext4_or_xfs_directory`` with an actual directory path. 

You can also point fio at the job file bundled with hipFile:

.. code-block:: shell

   GPU_DEV_IDS=0 FIO_DIR=/path/to/ext4_or_xfs_directory ~/fio/fio $HOME/hipFile/util/fio/write-read-verify.fio

Unofficial pre-built releases of the ``hipFile`` branch are available on the `ROCm/fio releases page <https://github.com/ROCm/fio/releases>`_.
