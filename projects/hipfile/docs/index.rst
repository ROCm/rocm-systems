.. meta::
   :description: hipfile is AMD's Infinity Storage library that enables direct-to-GPU I/O for the ROCm platform, bypassing the CPU for data transfers between storage and GPU memory. It provides a C API (and Python bindings) for registering files and GPU buffers, then performing synchronous, asynchronous, and batch read/write operations directly into device memory. The AMD backend uses a fastpath (via HIP runtime extensions) with automatic fallback to POSIX I/O; an NVIDIA backend wraps cuFile for compatibility.
   :keywords: hipfile, AMD, ROCm, GPU

=====================
hipfile documentation
=====================

hipfile is AMD's Infinity Storage library that enables direct-to-GPU I/O for the ROCm platform, bypassing the CPU for data transfers between storage and GPU memory. It provides a C API (and Python bindings) for registering files and GPU buffers, then performing synchronous, asynchronous, and batch read/write operations directly into device memory. The AMD backend uses a fastpath (via HIP runtime extensions) with automatic fallback to POSIX I/O; an NVIDIA backend wraps cuFile for compatibility.

The hipfile public repository is located at
`/mnt/c/Users/spolifro/Documents/rocmsystems/projects/hipfile <https://github.com//mnt/c/Users/spolifro/Documents/rocmsystems/projects/hipfile/tree/develop//mnt/c/Users/spolifro/Documents/rocmsystems/projects/hipfile>`_.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Install

      * :doc:`Installation overview <install/install>`
      * :doc:`Build hipfile from source <install/build-from-source>`
      * :doc:`Install the hipfile Python bindings <install/python-bindings>`

   .. grid-item-card:: Conceptual

      * :doc:`IO backend architecture <conceptual/io-backends>`
      * :doc:`Filesystem and file type requirements <conceptual/filesystem-requirements>`
      * :doc:`Driver lifecycle and reference counting <conceptual/driver-lifecycle>`
      * :doc:`IO statistics collection <conceptual/statistics-collection>`
      * :doc:`NVIDIA cuFile compatibility <conceptual/nvidia-compatibility>`

   .. grid-item-card:: How to

      * :doc:`Register a file and GPU buffer for GPU IO <how-to/register-file-and-buffer>`
      * :doc:`Configure the IO backend <how-to/configure-io-backend>`
      * :doc:`Collect IO statistics with ais-stats <how-to/collect-io-statistics>`

   .. grid-item-card:: Tutorials

      * :doc:`Copy a file via GPU memory using hipfile <tutorials/copy-a-file>`
      * :doc:`Query the hipfile version <tutorials/get-version>`
      * :doc:`Perform GPU I/O with the Python bindings <tutorials/python-gpu-io>`

   .. grid-item-card:: Reference

      * :doc:`CMake build options reference <reference/cmake-options>`
      * :doc:`Environment variables reference <reference/environment-variables>`
      * :doc:`API reference <reference/api-reference>`
      * :doc:`Error codes reference <reference/error-codes>`
      * :doc:`Python bindings API reference <reference/python-api-reference>`
      * :doc:`ais-stats tool reference <reference/ais-stats-tool>`

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

Licensing information can be found on the :doc:`License <license>` page.
