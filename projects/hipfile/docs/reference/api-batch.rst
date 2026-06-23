.. meta::
   :description: Batch I/O API reference for hipFile, covering batch handle lifecycle, I/O parameter and event types, opcodes, and status codes.
   :keywords: hipFile, batch, API, ROCm, GPU, I/O, hipFileBatchIOSetUp, hipFileBatchIOSubmit, hipFileBatchIOGetStatus, hipFileBatchIOCancel, hipFileBatchIODestroy

***********************
Batch I/O API reference
***********************

The hipFile batch I/O API lets you create batch contexts, submit multiple I/O
requests in a single call, poll for completion, cancel pending operations, and
tear down batch handles. All batch types and functions belong to the
``batch`` Doxygen group declared in ``hipfile.h``.

.. warning::

   The batch API is not currently supported on the AMD backend.
   ``hipFileBatchIOGetStatus``, ``hipFileBatchIOCancel``, and
   ``hipFileBatchIODestroy`` are not implemented and return errors or perform
   no operation. The maximum batch size is 128 operations.

For related API surfaces, see :doc:`/reference/api-synchronous-io`,
:doc:`/reference/api-async`, and :doc:`/reference/api-errors`.

Batch I/O types and functions
*****************************

Enumerations, structures, and handles used by batch I/O operations, together
with the functions to create, submit, poll, cancel, and destroy batch I/O
contexts.

.. doxygengroup:: batch
   :content-only:
