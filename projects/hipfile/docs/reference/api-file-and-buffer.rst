.. meta::
   :description: API reference for hipFile file handle registration, buffer registration, RDMA types, and related macros.
   :keywords: hipFile, ROCm, API, file handle, buffer, RDMA, hipFileHandleRegister, hipFileBufRegister, GPU I/O

********************************************
File handle, buffer, and RDMA API reference
********************************************

This page documents the hipFile functions and types for registering and deregistering file handles and GPU memory buffers with the hipFile driver.

For a walkthrough of synchronous reads and writes using registered handles and buffers, see :doc:`/tutorials/copy-a-file`. For the synchronous read and write function signatures, see :doc:`/reference/api-synchronous-io`.

File handle and buffer types and functions
============================================

File handle types, the filesystem operations structure, and the functions for
registering and deregistering file handles and GPU memory buffers with the
hipFile driver.

.. doxygengroup:: file
   :content-only:
