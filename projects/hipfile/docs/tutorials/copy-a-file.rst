.. meta::
   :description: Tutorial walking through the aiscp example program, which copies a file via GPU memory using hipFile synchronous read and write operations.
   :keywords: hipFile, ROCm, GPU I/O, direct-to-GPU, file copy, synchronous I/O, hipFileRead, hipFileWrite, aiscp, tutorial, example

*****************************************
Copy a file via GPU memory using hipFile
*****************************************

`aiscp.cpp <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_ copies a source file to a destination by routing every byte through GPU memory. In a hipFile checkout, the same sources are in ``examples/aiscp/aiscp.cpp``. The program opens files with ``O_DIRECT``, registers them with hipFile, allocates a GPU buffer with ``hipMalloc()``, and copies data in a chunk loop using ``hipFileRead()`` and ``hipFileWrite()``.

The walkthrough covers every required step of a synchronous hipFile workflow:

1. Opening files with ``O_DIRECT`` and registering them for GPU I/O
2. Allocating a device buffer with ``hipMalloc()``
3. Performing chunked reads and writes with ``hipFileRead()`` and ``hipFileWrite()``
4. Handling errors with the ``IS_HIPFILE_ERR`` and ``HIPFILE_ERRSTR`` macros
5. Cleaning up handles, buffers, and file descriptors

When to use this pattern
===========================

Use synchronous GPU-mediated copy when you need to:

- Move bulk data between storage and GPU memory without staging through host DRAM.
- Keep the control flow on the host in a single thread with blocking ``hipFileRead()`` and ``hipFileWrite()`` calls.
- Match a minimal reference path before layering async or Python bindings on top.

Prerequisites
===============

Verify you have:

- A supported AMD GPU with the ROCm stack installed and visible to the HIP runtime.
- hipFile built and installed. See :doc:`/install/install`.
- A file system that supports ``O_DIRECT``, for example ext4 or XFS. The hipFile fast path expects direct I/O to aligned host storage. See :doc:`/reference/hipFile-io-backends` for backend rules and alignment constraints.
- The HIP runtime development headers from ``hip/hip_runtime_api.h``.

Build the example
==================

Create a build directory and point CMake at ``examples/aiscp`` in your hipFile checkout. If ROCm or hipFile are in non-standard locations, pass them through ``CMAKE_PREFIX_PATH``:

.. code:: shell

   cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipFile" /path/to/hipfile/examples/aiscp
   cmake --build .

Run the resulting binary the same way as the Linux ``cp`` command:

.. code:: shell

   ./aiscp SOURCE DEST

Both paths should sit on a file system that supports ``O_DIRECT``. If the source is empty, the program exits successfully without entering the read loop.

Step-by-step walkthrough
===========================

Define the chunk size limit
---------------------------

``AISCP_CHUNK_SIZE`` defaults to ``0x7ffff000``, roughly 2 GiB minus one page. That value matches the Linux kernel ``MAX_RW_COUNT`` cap on a single ``read()`` or ``write()``-sized transfer. Each loop iteration moves at most that many bytes.

.. code-block:: cpp

   #ifndef AISCP_CHUNK_SIZE
   #define AISCP_CHUNK_SIZE 0x7ffff000LU
   #endif

You can override the macro at compile time if smaller chunks help limit GPU buffer size.

Open and register files
-----------------------

Both paths use ``O_DIRECT`` so hipFile can use its direct GPU I/O fast path. Each POSIX descriptor is wrapped in a ``hipFileDescr_t`` and registered with ``hipFileHandleRegister()``. On success the call fills in an opaque ``hipFileHandle_t`` for all later hipFile I/O on that file.

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

- ``hipFileDescr_t.type`` is ``hipFileHandleTypeOpaqueFD`` for a POSIX file descriptor.
- ``hipFileHandleRegister()`` opens the hipFile driver automatically if it was not already opened with ``hipFileDriverOpen()``. See :doc:`/reference/hipFile-reference-count` for reference counting and lifecycle rules.
- The error path compares ``hipfile_err.err`` to ``hipFileSuccess`` and prints with ``hipFileGetOpErrorString()`` when registration fails.

The destination opens with ``O_WRONLY | O_CREAT``. The source opens with ``O_RDONLY``:

.. code-block:: cpp

   if (open_file(dst_path, O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH,
                 &dst_fd, &dst_handle)) {
       goto program_exit;
   }

   if (open_file(src_path, O_RDONLY, 0, &src_fd, &src_handle)) {
       goto close_dst;
   }

For a detailed procedure on registering files and GPU buffers, see :doc:`/how-to/register-file-and-buffer`.

Determine file size and block size
----------------------------------

.. code-block:: cpp

   struct stat statbuf;
   stat(src_path, &statbuf);
   file_size  = static_cast<size_t>(statbuf.st_size);
   block_size = static_cast<size_t>(statbuf.st_blksize);

The program uses ``stat()`` to obtain the source file's total size and the file system block size. The block size is critical because writes performed with ``O_DIRECT`` must be aligned to the block size.

If the source size is zero, the program skips the read loop and exits after closing the destination handle.

Allocate a GPU buffer
---------------------

``hipMalloc()`` receives a buffer sized to the minimum of the source file size and ``AISCP_CHUNK_SIZE``, rounded up to the file system block size so ``O_DIRECT`` alignment rules hold.

.. code-block:: cpp

   buffer_size = align_up(std::min(file_size, AISCP_CHUNK_SIZE), block_size);
   hip_err     = hipMalloc(&devbuf, buffer_size);
   if (hipSuccess != hip_err) {
       fprintf(stderr, "Could not allocate device buffer (%d)", hip_err);
       goto close_src;
   }

``align_up`` is a helper in ``aiscp`` to round up to the next multiple of a power-of-two alignment:

.. code-block:: cpp

   static inline size_t
   align_up(size_t value, size_t align)
   {
       return (value + align - 1) & ~(align - 1);
   }

``aiscp`` does not call ``hipFileBufRegister()``. Registration is optional. When you skip it, hipFile may use an internal bounce buffer. Registering with ``hipFileBufRegister()`` can increase throughput when you reuse the same buffer for many operations. See :doc:`/reference/api-file-and-buffer` for registration and I/O entry points.

Read and write in a chunk loop
------------------------------

The copy is a ``do`` / ``while`` loop that reads from the source into device memory, then writes from device memory to the destination. ``file_offset`` advances by the number of bytes read each iteration.

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

This is the core copy logic:

- Outer loop: reads a chunk of up to ``buffer_size`` bytes from the source file into the GPU buffer. When ``hipFileRead()`` returns ``0``, all data has been read.
- Inner loop: writes the chunk from the GPU buffer to the destination file. A partial write is possible, so the inner loop continues until every byte from the current chunk is written.

``hipFileRead()`` and ``hipFileWrite()`` return:

- A non-negative byte count on success.
- ``-1`` for a POSIX error. Check ``errno``.
- A negative hipFile error code, the negated ``hipFileOpError_t`` value.

The sample uses ``IS_HIPFILE_ERR()`` to pick between ``HIPFILE_ERRSTR()`` and ``strerror()`` for logging:

.. code-block:: cpp

   IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno)

The ``IS_HIPFILE_ERR`` macro tests whether the absolute value of the return code falls in the hipFile error range at or above ``HIPFILE_BASE_ERR``. In the public headers, ``HIPFILE_BASE_ERR`` is 5000. When the macro matches, ``HIPFILE_ERRSTR`` converts the code to a human-readable string via ``hipFileGetOpErrorString()``. Otherwise, the program falls back to ``strerror(errno)`` for standard system errors.

The inner ``while`` loop handles partial writes. Each ``hipFileWrite()`` call uses ``align_up`` on the remaining byte count and passes ``nwrite`` as ``buffer_offset`` so the write starts at the correct offset inside the GPU buffer.

Two alignment details:

1. Maximum I/O size per call: because each ``hipFileRead()`` or ``hipFileWrite()`` call can transfer at most ``0x7ffff000`` bytes, the buffer is sized accordingly, and larger files are handled by iterating.
2. Block-size alignment on writes: the write size is rounded up to the file system block size with ``align_up()``. This is required because the file is opened with ``O_DIRECT``, which mandates that offsets and sizes are aligned to the block size.

For a full list of error codes, see :doc:`/reference/api-errors`.

Truncate to the exact size
--------------------------

``O_DIRECT`` forces block-aligned transfer sizes. The last write can overshoot the true file size. ``ftruncate()`` trims the destination to match the source.

.. code-block:: cpp

   if (-1 == ftruncate(dst_fd, static_cast<off_t>(file_size))) {
       fprintf(stderr, "Could not truncate %s (%zu) (%s)\n", dst_path, file_size,
               strerror(errno));
   }

Clean up resources
------------------

Teardown reverses acquisition order: ``hipFree()`` releases device memory, ``hipFileHandleDeregister()`` drops hipFile tracking for each handle, then ``close()`` closes the POSIX descriptors.

.. code-block:: cpp

   hipFree(devbuf);

   hipFileHandleDeregister(handle);
   close(fd);

``hipFileHandleDeregister()`` takes the ``hipFileHandle_t`` from registration. The POSIX ``close()`` call is separate.

There is no need to call ``hipFileDriverClose()`` explicitly. The hipFile library cleans up its internal state automatically at program exit.
