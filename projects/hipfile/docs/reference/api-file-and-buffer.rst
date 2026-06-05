.. meta::
   :description: API reference for hipFile file handle registration, buffer registration, RDMA types, and related macros.
   :keywords: hipFile, ROCm, API, file handle, buffer, RDMA, hipFileHandleRegister, hipFileBufRegister, GPU IO

===========================================
File handle, buffer, and RDMA API reference
===========================================

This page documents the hipFile functions and types for registering and deregistering file handles and GPU memory buffers with the hipFile driver. It also covers the RDMA-related types and macros used with userspace RDMA filesystems.

For a walkthrough of synchronous reads and writes using registered handles and buffers, see :doc:`/tutorials/synchronous-file-copy`. For the synchronous read and write function signatures, see :doc:`/reference/api-synchronous-io`.

.. note::

   The ``hipFileHandleTypeOpaqueWin32`` and ``hipFileHandleTypeUserspaceFS`` handle types and RDMA functionality are not currently supported on AMD. Setting RDMA options in hipFile API calls has no effect.

File handle types
*****************

.. doxygenenum:: hipFileFileHandleType_t

.. doxygenstruct:: hipFileDescr_t
   :members:

.. doxygentypedef:: hipFileHandle_t

RDMA types and macros
*********************

.. doxygengroup:: rdma
   :content-only:

.. doxygengroup:: rdma_flags
   :content-only:

Filesystem operations
*********************

.. doxygenstruct:: hipFileFSOps_t
   :members:

File handle registration
************************

.. doxygenfunction:: hipFileHandleRegister
.. doxygenfunction:: hipFileHandleDeregister

Buffer registration
*******************

.. doxygenfunction:: hipFileBufRegister
.. doxygenfunction:: hipFileBufDeregister
