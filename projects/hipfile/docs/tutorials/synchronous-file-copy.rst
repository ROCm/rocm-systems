.. meta::
   :description: Tutorial walking through the aiscp example, a synchronous GPU-mediated file copy built with hipFile.
   :keywords: hipFile, ROCm, GPU, file copy, synchronous I/O, aiscp, tutorial, O_DIRECT, hipFileRead, hipFileWrite

*************************************************
Synchronous GPU-mediated file copy with ``aiscp``
*************************************************

`aiscp.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_ copies a source file to a destination by routing
every byte through GPU memory. The same sources live under ``examples/aiscp/aiscp.cpp`` in a hipFile checkout. The program opens files with ``O_DIRECT``,
registers them with hipFile, allocates a GPU buffer with ``hipMalloc()``, and
copies data in a chunk loop using ``hipFileRead()`` and ``hipFileWrite()``.

When to use this pattern
************************

Use synchronous GPU-mediated copy when you need to:

- Move bulk data between storage and GPU memory without staging through host
  DRAM.
- Keep the control flow on the host in a single thread with blocking
  ``hipFileRead()`` and ``hipFileWrite()`` calls.
- Match a minimal reference path before layering async or Python bindings on
  top.

Prerequisites
*************

Verify you have:

- AMD ROCm installed with the HIP runtime available.
- hipFile built and installed. See :doc:`/install/install`.
- A GPU visible to the HIP runtime.
- A file system that supports ``O_DIRECT``, for example ext4 or XFS. The hipFile
  fast path expects direct I/O to aligned host storage. See
  :doc:`/reference/hipFile-io-backends` for backend rules and alignment
  constraints.

Step-by-step walkthrough
*************************

Define the chunk size limit
---------------------------

``AISCP_CHUNK_SIZE`` defaults to ``0x7ffff000``, roughly 2 GiB minus one page.
That value matches the Linux kernel ``MAX_RW_COUNT`` cap on a single ``read()``
or ``write()``-sized transfer. Each loop iteration moves at most that many
bytes.

.. code-block:: cpp

   #ifndef AISCP_CHUNK_SIZE
   #define AISCP_CHUNK_SIZE 0x7ffff000LU
   #endif

You can override the macro at compile time if smaller chunks help limit GPU
buffer size.

Open and register files
-----------------------

Both paths use ``O_DIRECT`` so hipFile can use its direct GPU I/O fast path.
Each POSIX descriptor is wrapped in a ``hipFileDescr_t`` and registered with
``hipFileHandleRegister()``. On success the call fills in an opaque
``hipFileHandle_t`` for all later hipFile I/O on that file.

.. code-block:: cpp

   static int
   open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
   {
       hipFileError_t hipfile_err;
       hipFileDescr_t descr;

       *fd = open(path, flags | O_DIRECT, mode);
       if (-1 == *fd) {
           fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
           return 1;
       }

       descr.type      = hipFileHandleTypeOpaqueFD;
       descr.handle.fd = *fd;

       hipfile_err = hipFileHandleRegister(handle, &descr);
       if (hipFileSuccess != hipfile_err.err) {
           fprintf(stderr, "Could not register %s (%s)\n", path,
                   hipFileGetOpErrorString(hipfile_err.err));
           close(*fd);
           return 1;
       }

       return 0;
   }

Important details:

- ``hipFileDescr_t.type`` is ``hipFileHandleTypeOpaqueFD`` for a POSIX file
  descriptor.
- ``hipFileHandleRegister()`` opens the hipFile driver automatically if it was
  not already opened with ``hipFileDriverOpen()``. See
  :doc:`/reference/hipFile-reference-count` for reference counting and lifecycle
  rules.
- The error path compares ``hipfile_err.err`` to ``hipFileSuccess`` and prints
  with ``hipFileGetOpErrorString()`` when registration fails.

The destination opens with ``O_WRONLY | O_CREAT``. The source opens with
``O_RDONLY``:

.. code-block:: cpp

   if (open_file(dst_path, O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH,
                 &dst_fd, &dst_handle)) {
       goto program_exit;
   }

   if (open_file(src_path, O_RDONLY, 0, &src_fd, &src_handle)) {
       goto close_dst;
   }

If the source size is zero, the program skips the read loop and exits after
closing the destination handle.

Allocate a GPU buffer
---------------------

``hipMalloc()`` receives a buffer sized to ``min`` of the source file size and
``AISCP_CHUNK_SIZE``, rounded up to the file system block size so ``O_DIRECT``
alignment rules hold.

.. code-block:: cpp

   buffer_size = align_up(std::min(file_size, AISCP_CHUNK_SIZE), block_size);
   hip_err     = hipMalloc(&devbuf, buffer_size);
   if (hipSuccess != hip_err) {
       fprintf(stderr, "Could not allocate device buffer (%d)", hip_err);
       goto close_src;
   }

The ``align_up`` helper rounds up to the next multiple of a power-of-two
alignment:

.. code-block:: cpp

   static inline size_t
   align_up(size_t value, size_t align)
   {
       return (value + align - 1) & ~(align - 1);
   }

``aiscp`` does not call ``hipFileBufRegister()``. Registration is optional.
When you skip it, hipFile may use an internal bounce buffer. Registering with
``hipFileBufRegister()`` can increase throughput when you reuse the same buffer for
many operations. See :doc:`/reference/api-file-and-buffer` for registration
and I/O entry points.

Read and write in a chunk loop
-------------------------------

The copy is a ``do`` / ``while`` loop that reads from the source into device
memory, then writes from device memory to the destination. ``file_offset``
advances by the number of bytes read each iteration.

.. code-block:: cpp

   do {
       nread = hipFileRead(src_handle, devbuf, buffer_size, file_offset, 0);
       if (nread < 0) {
           fprintf(stderr, "Could not read from %s (%zd) (%s)\n", src_path, nread,
                   IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
           goto free_devbuf;
       }

       nwrite = 0;
       while (nwrite < nread) {
           nbytes =
               hipFileWrite(dst_handle, devbuf, align_up(static_cast<size_t>(nread - nwrite), block_size),
                            file_offset + nwrite, nwrite);
           if (nbytes < 0) {
               fprintf(stderr, "Could not write to %s (%zd) (%s)\n", dst_path, nbytes,
                       IS_HIPFILE_ERR(nbytes) ? HIPFILE_ERRSTR(nbytes) : strerror(errno));
               goto free_devbuf;
           }
           nwrite += nbytes;
       }
       file_offset += nread;
   } while (nread > 0);

``hipFileRead()`` and ``hipFileWrite()`` return:

- A non-negative byte count on success.
- ``-1`` for a POSIX error. Check ``errno``.
- A negative hipFile error code, the negated ``hipFileOpError_t`` value.

The sample uses ``IS_HIPFILE_ERR()`` to pick between ``HIPFILE_ERRSTR()`` and
``strerror()`` for logging:

.. code-block:: cpp

   IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno)

The inner ``while`` loop handles partial writes. Each ``hipFileWrite()`` call
uses ``align_up`` on the remaining byte count and passes ``nwrite`` as
``buffer_offset`` so the write starts at the correct offset inside the GPU
buffer.

Truncate to the exact size
--------------------------

``O_DIRECT`` forces block-aligned transfer sizes. The last write can overshoot the
true file size. ``ftruncate()`` trims the destination to match the source.

.. code-block:: cpp

   if (-1 == ftruncate(dst_fd, static_cast<off_t>(file_size))) {
       fprintf(stderr, "Could not truncate %s (%zu) (%s)\n", dst_path, file_size,
               strerror(errno));
   }

Clean up resources
------------------

Teardown reverses acquisition order: ``hipFree()`` releases device memory,
``hipFileHandleDeregister()`` drops hipFile tracking for each handle, then
``close()`` closes the POSIX descriptors.

``hipFileHandleDeregister()`` takes the ``hipFileHandle_t`` from registration.
The POSIX ``close()`` call is separate.

Running the example
*******************

.. code:: shell

   ./aiscp /path/to/source_file /path/to/dest_file

Both paths should sit on a file system that supports ``O_DIRECT``. If the source
is empty, the program exits successfully without entering the read loop.

