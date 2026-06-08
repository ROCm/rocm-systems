.. meta::
  :description: hipFile I/O fastpath and fallback backends
  :keywords: hipFile, ROCm, I/O backend, fastpath, fallback, AMD

****************************************
hipFile fastpath and fallback backends
****************************************

When an I/O operation is submitted to hipFile, the operation can either be completed by the fastpath or the fallback. 

The fastpath transfers data between storage and the GPU without the use of a buffer. The fallback method routes the operation through a host-side buffer first before transferring data to the GPU.

At submission time, both the fastpath and fallback backends inspect the I/O and provide an eagerness score. The fastpath backend will return an eagerness score of 100 if it can handle the request. The fallback backend will return an eagerness score of 1 if it can handle the request. In the case where a backend can't handle the request, it will return a score of -1. 

hipFile will use the backend with the highest score. In most situations, the fastpath backend will have the highest score.

If I/O is routed through the fastpath but an unexpected error occurs and the I/O request can't be fulfilled by the fastpath backend, the I/O request will be retried using the fallback backend. If the I/O operation fails on the fallback, it won't be retried. 

.. note::

   If the ``HIPFILE_ALLOW_COMPAT_MODE`` environment variable is set to ``false``, I/O that fails with the fastpath backend won't be retried on the fallback backend.

   If the ``HIPFILE_FORCE_COMPAT_MODE`` environment variable is set to ``true``, the fastpath backend will never be used and all I/O will go through the fallback backend.

Fastpath will only return 100 if the file was open with ``O_DIRECT`` and if the destination or the source of the transfer is device memory. If the file doesn't reside on a block device, or if it isn't stored on ext4 or XFS, fastpath will return an eagerness score of -1.

File offsets, buffer offsets, and I/O sizes must be aligned to the file system's direct I/O alignment for fastpath to return a score of 100.

The fallback backend uses POSIX ``pread`` and ``pwrite`` system calls combined with ``hipMemcpy``. Because it does not require ``O_DIRECT``, specific filesystem types, or aligned offsets, it can handle I/O request that the fastpath backend can't.
