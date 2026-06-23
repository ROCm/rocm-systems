.. meta::
   :description: Synchronous read and write API reference for hipFile, covering hipFileRead, hipFileWrite, and the hoff_t offset type.
   :keywords: hipFile, ROCm, synchronous I/O, hipFileRead, hipFileWrite, hoff_t, GPU I/O, API reference

****************************************
Synchronous read and write API reference
****************************************

``hipFileRead`` and ``hipFileWrite`` perform blocking transfers between a file and registered GPU memory. Both functions accept file and buffer offsets for fine-grained control over where data is read from or written to.

Before calling these functions, register the file handle with ``hipFileHandleRegister`` and the GPU buffer with ``hipFileBufRegister``. See :doc:`/reference/api-file-and-buffer` for registration details and :doc:`/tutorials/copy-a-file` for a usage walkthrough.

Offset type and functions
************************

The offset type used by the synchronous read and write functions, and the
functions themselves.

.. doxygengroup:: sync
   :content-only:
