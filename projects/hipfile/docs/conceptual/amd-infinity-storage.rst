.. meta::
   :description: How AMD Infinity Storage moves file data between storage and GPU memory
   :keywords: hipFile, AMD Infinity Storage, AIS, ROCm, GPU I/O, P2PDMA, NVMe, direct I/O

***********************
AMD Infinity Storage
***********************

AMD Infinity Storage (AIS) lets the storage device read and write GPU memory with direct memory access (DMA) for each hipFile fastpath transfer. The hipFile fastpath uses AIS and skips the copy to host memory.

Transfers that use AIS move data directly between storage and GPU memory, bypassing the host staging buffer. The CPU submits and completes the request without copying the payload to an intermediate buffer.

A transfer that doesn't use AIS stages the payload in host memory when the buffer is GPU device memory and the fallback is enabled. In the fallback case, a read copies the payload to host memory and then the payload is copied into GPU memory. Similarly, a write copies GPU memory into host memory and then copies it again into storage.

Disabling AIS drops the direct storage-to-GPU transfer and hipFile uses the fallback for GPU device memory.