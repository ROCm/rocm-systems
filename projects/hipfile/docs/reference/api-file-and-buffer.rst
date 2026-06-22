.. meta::
   :description: API reference for hipFile file handle registration, buffer registration, RDMA types, and related macros.
   :keywords: hipFile, ROCm, API, file handle, buffer, RDMA, hipFileHandleRegister, hipFileBufRegister, GPU IO

*******************************************
File handle, buffer, and RDMA API reference
*******************************************

This page documents the hipFile functions and types for registering and deregistering file handles and GPU memory buffers with the hipFile driver. 

For a walkthrough of synchronous reads and writes using registered handles and buffers, see :doc:`/tutorials/copy-a-file`. For the synchronous read and write function signatures, see :doc:`/reference/api-synchronous-io`.


File handle types
*****************

.. doxygenenum:: hipFileFileHandleType_t

.. doxygenstruct:: hipFileDescr_t
   :members:

.. doxygentypedef:: hipFileHandle_t


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
