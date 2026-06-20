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

Fastpath will only return 100 if the file was open with ``O_DIRECT`` and if the destination or the source of the transfer is device memory. The file's type must be either a regular file or a block device. If the file is a regular file, the file must reside on an ext4 filesystem with ordered journaling mode or on an xfs filesystem. Files on other filesystems are rejected by the fastpath backend unless ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` is set to ``true``.

File offsets, buffer offsets, and I/O sizes must be aligned to the file system's direct I/O alignment for fastpath to return a score of 100.

The fallback backend uses POSIX ``pread`` and ``pwrite`` system calls combined with ``hipMemcpy``. It will reject the I/O operation if the buffer type is not ``hipMemoryTypeDevice``.

I/O fails when both backends reject the request, or when a fastpath runtime error isn't eligible for automatic fallback retry. Fastpath rejects a request when:

- ``HIPFILE_FORCE_COMPAT_MODE`` is ``true``, which disables the fastpath entirely.
- The HIP runtime doesn't expose ``hipAmdFileRead()`` or ``hipAmdFileWrite()``.
- The file doesn't have an ``O_DIRECT`` file descriptor. hipFile tries to open one at registration, but if the file or file system doesn't support ``O_DIRECT``, fastpath can't service any requests for that file.
- The file isn't a regular file or block device.
- The file is a regular file on a file system other than ext4 with ordered journaling or XFS, and ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` isn't set to ``true``.
- The buffer isn't ``hipMemoryTypeDevice``. Both fastpath and fallback require device memory. Non-device buffers cause both backends to reject the request.
- The file offset or I/O size isn't aligned to the file system's offset alignment, or the device buffer address isn't aligned to the memory alignment. These two alignment values can differ depending on what the kernel reports through ``statx()``.

When the fastpath accepts a request but encounters a runtime error, only ``ENODEV`` and ``EREMOTEIO`` trigger an automatic retry on the fallback backend. All other errors, including ``EINVAL``, ``EBADF``, and ``EIO``, cause the I/O to fail immediately. Setting ``HIPFILE_ALLOW_COMPAT_MODE`` to ``false`` disables the fallback backend entirely, which also prevents automatic retry.

