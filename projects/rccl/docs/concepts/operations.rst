.. meta::
   :description: How to use RCCL
   :keywords: RCCL, ROCm, library, API,

.. _operations:

**************************
RCCL collective operations
**************************

Collective operations have to be called for each rank (hence CUDA device), using the same count and the same datatype, to form a complete collective operation.
Failure to do so will result in undefined behavior, including hangs, crashes, or data corruption. 

