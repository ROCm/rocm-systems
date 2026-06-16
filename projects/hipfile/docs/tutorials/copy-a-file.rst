.. meta::
  :description: Tutorial walking through the aiscp example program, which copies a file via GPU memory using hipFile synchronous read and write operations.
  :keywords: hipFile, ROCm, GPU I/O, direct-to-GPU, file copy, hipFileRead, hipFileWrite, tutorial, example

****************************************
Copy a file via GPU memory using hipFile
****************************************

This tutorial walks through the ``aiscp`` example program shipped with hipFile.
The program copies a source file to a destination file by streaming data
through GPU memory, bypassing the CPU data path. It demonstrates every
required step of a hipFile workflow:

1. Opening files with ``O_DIRECT`` and registering them for GPU I/O
2. Allocating a device buffer with ``hipMalloc``
3. Performing chunked reads and writes with ``hipFileRead`` and ``hipFileWrite``
4. Handling errors with the ``IS_HIPFILE_ERR`` and ``HIPFILE_ERRSTR`` macros
5. Cleaning up handles, buffers, and file descriptors

Use this pattern whenever you need to move bulk data between storage and GPU
memory without staging it through host RAM.

Prerequisites
*************

- A supported AMD GPU with the ROCm stack installed
- hipFile built and installed (see the project ``INSTALL.md``)
- A file system that supports ``O_DIRECT``, such as ext4 or XFS
- The HIP runtime development headers (``hip/hip_runtime_api.h``)

Build the example
*****************

Create a build directory and point CMake at the ``aiscp`` source tree. If ROCm
or hipFile are in non-standard locations, pass them through
``CMAKE_PREFIX_PATH``:

.. code:: shell

   cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipFile" /path/to/aiscp/dir
   cmake --build .

Run the resulting binary the same way as the Linux ``cp`` command:

.. code:: shell

   ./aiscp SOURCE DEST

Complete source code
********************

The listing later in this page is the complete ``aiscp.cpp`` file. Subsequent sections
explain each piece.

.. code-block:: cpp

   #include <hipfile.h>
   #include <hip/hip_runtime_api.h>

   #include <algorithm>
   #include <cerrno>
   #include <cstdio>
   #include <cstdlib>
   #include <cstring>
   #include <fcntl.h>
   #include <sys/stat.h>
   #include <sys/types.h>
   #include <unistd.h>

   #ifndef AISCP_CHUNK_SIZE
   #define AISCP_CHUNK_SIZE 0x7ffff000LU
   #endif

   static int
   open_file(const char *path, int flags, mode_t mode, int *fd, hipFileHandle_t *handle)
   {
       hipFileError_t hipFile_err;
       hipFileDescr_t descr;

       *fd = open(path, flags | O_DIRECT, mode);
       if (-1 == *fd) {
           fprintf(stderr, "Could not open %s (%s)\n", path, strerror(errno));
           return 1;
       }

       descr.type      = hipFileHandleTypeOpaqueFD;
       descr.handle.fd = *fd;

       hipFile_err = hipFileHandleRegister(handle, &descr);
       if (hipFileSuccess != hipFile_err.err) {
           fprintf(stderr, "Could not register %s (%s)\n", path,
                   hipFileGetOpErrorString(hipFile_err.err));
           close(*fd);
           return 1;
       }

       return 0;
   }

   static int
   close_file(const char *path, int fd, hipFileHandle_t handle)
   {
       hipFileHandleDeregister(handle);
       if (-1 == close(fd)) {
           fprintf(stderr, "Could not close %s (%s)\n", path, strerror(errno));
           return 1;
       }
       return 0;
   }

   static inline size_t
   align_up(size_t value, size_t align)
   {
       return (value + align - 1) & ~(align - 1);
   }

   static inline bool
   is_power_of_two(size_t value)
   {
       return (value > 0) && ((value & (value - 1)) == 0);
   }

   int
   main(int argc, char *argv[])
   {
       const char     *src_path, *dst_path;
       int             src_fd, dst_fd;
       hipFileHandle_t src_handle, dst_handle;
       void           *devbuf;
       hipError_t      hip_err;
       int             exit_status = EXIT_FAILURE;
       size_t          buffer_size, file_size, block_size;
       ssize_t         nwrite{}, nread{}, nbytes{};
       hoff_t          file_offset{};

       if (argc != 3) {
           fprintf(stderr, "Usage: %s SOURCE DEST\n", argv[0]);
           exit(1);
       }

       src_path = argv[1];
       dst_path = argv[2];

       {
           struct stat statbuf;
           if (stat(src_path, &statbuf)) {
               fprintf(stderr, "Could not stat file: %s (%s)\n", src_path, strerror(errno));
               goto program_exit;
           }
           file_size  = static_cast<size_t>(statbuf.st_size);
           block_size = static_cast<size_t>(statbuf.st_blksize);
           if (!is_power_of_two(block_size)) {
               fprintf(stderr, "Blocksize is not a power of two (%zu)", block_size);
               goto program_exit;
           }
       }

       if (open_file(dst_path, O_WRONLY | O_CREAT, S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH,
                      &dst_fd, &dst_handle)) {
           goto program_exit;
       }

       if (0 == file_size) {
           exit_status = EXIT_SUCCESS;
           goto close_dst;
       }

       if (open_file(src_path, O_RDONLY, 0, &src_fd, &src_handle)) {
           goto close_dst;
       }

       buffer_size = align_up(std::min(file_size, AISCP_CHUNK_SIZE), block_size);
       hip_err     = hipMalloc(&devbuf, buffer_size);
       if (hipSuccess != hip_err) {
           fprintf(stderr, "Could not allocate device buffer (%d)", hip_err);
           goto close_src;
       }

       do {
           nread = hipFileRead(src_handle, devbuf, buffer_size, file_offset, 0);
           if (nread < 0) {
               fprintf(stderr, "Could not read from %s (%zd) (%s)\n", src_path, nread,
                       IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
               goto free_devbuf;
           }

           nwrite = 0;
           while (nwrite < nread) {
               nbytes = hipFileWrite(dst_handle, devbuf,
                           align_up(static_cast<size_t>(nread - nwrite), block_size),
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

       if (-1 == ftruncate(dst_fd, static_cast<off_t>(file_size))) {
           fprintf(stderr, "Could not truncate %s (%zu) (%s)\n", dst_path, file_size,
                   strerror(errno));
       }

       exit_status = EXIT_SUCCESS;

   free_devbuf:
       hip_err = hipFree(devbuf);
       if (hipSuccess != hip_err) {
           fprintf(stderr, "Could not free device buffer (%d)\n", hip_err);
           exit_status = EXIT_FAILURE;
       }

   close_src:
       if (close_file(src_path, src_fd, src_handle))
           exit_status = EXIT_FAILURE;

   close_dst:
       if (close_file(dst_path, dst_fd, dst_handle))
           exit_status = EXIT_FAILURE;

   program_exit:
       return exit_status;
   }

Step-by-step explanation
************************

Define the maximum I/O size
--------------------------

.. code-block:: cpp

   #ifndef AISCP_CHUNK_SIZE
   #define AISCP_CHUNK_SIZE 0x7ffff000LU
   #endif

Each call to ``hipFileRead`` or ``hipFileWrite`` can transfer at most
``0x7ffff000`` bytes (2 GiB minus one page). This value matches the Linux
kernel's ``MAX_RW_COUNT`` limit. If you need to transfer more data, you must
loop and issue multiple calls, as shown later in the read-write loop.

Open a file and register it for GPU I/O
--------------------------------------

.. code-block:: cpp

   *fd = open(path, flags | O_DIRECT, mode);

   descr.type      = hipFileHandleTypeOpaqueFD;
   descr.handle.fd = *fd;

   hipFile_err = hipFileHandleRegister(handle, &descr);

Every file you want to use with hipFile must be:

1. Opened with the standard POSIX ``open()`` call. The ``O_DIRECT`` flag is
   included to enable direct I/O, which avoids the kernel page cache.
2. Wrapped in a ``hipFileDescr_t``, setting the ``type`` to
   ``hipFileHandleTypeOpaqueFD`` and storing the file descriptor in
   ``handle.fd``.
3. Registered with ``hipFileHandleRegister()``, which returns an opaque
   ``hipFileHandle_t`` for use with all subsequent hipFile I/O calls.

.. note::

   The first call to ``hipFileHandleRegister()`` also initializes the hipFile
   library automatically, so an explicit ``hipFileDriverOpen()`` call is
   optional.

For a detailed procedure on registering files and GPU buffers, see
:doc:`/how-to/register-file-and-buffer`.

Determine file size and block size
----------------------------------

.. code-block:: cpp

   struct stat statbuf;
   stat(src_path, &statbuf);
   file_size  = static_cast<size_t>(statbuf.st_size);
   block_size = static_cast<size_t>(statbuf.st_blksize);

The program uses ``stat()`` to obtain the source file's total size and the
file system block size. The block size is critical because writes performed with
``O_DIRECT`` must be aligned to the block size.

Allocate a GPU buffer
---------------------

.. code-block:: cpp

   buffer_size = align_up(std::min(file_size, AISCP_CHUNK_SIZE), block_size);
   hip_err     = hipMalloc(&devbuf, buffer_size);

The GPU buffer is allocated with ``hipMalloc``. Its size is the smaller of the
file size and ``AISCP_CHUNK_SIZE`` (``0x7ffff000``), rounded up to the
file system block size. This ensures every I/O operation meets alignment
requirements.

.. note::

   The example does not call ``hipFileBufRegister()``. The hipFile library
   accepts unregistered GPU buffers by creating a temporary internal buffer
   object. Explicitly registering buffers with ``hipFileBufRegister()`` is
   recommended for repeated operations because it avoids per-call overhead.
   See :doc:`/how-to/register-file-and-buffer` for details.

Chunked read-write loop
-----------------------

.. code-block:: cpp

   do {
       nread = hipFileRead(src_handle, devbuf, buffer_size, file_offset, 0);
       if (nread < 0) { /* handle error */ }

       nwrite = 0;
       while (nwrite < nread) {
           nbytes = hipFileWrite(dst_handle, devbuf,
                       align_up(static_cast<size_t>(nread - nwrite), block_size),
                       file_offset + nwrite, nwrite);
           if (nbytes < 0) { /* handle error */ }
           nwrite += nbytes;
       }
       file_offset += nread;
   } while (nread > 0);

This is the core copy logic:

- Outer loop: reads a chunk of up to ``buffer_size`` bytes from the source
  file into the GPU buffer. When ``hipFileRead()`` returns ``0``, all data has
  been read.
- Inner loop: writes the chunk from the GPU buffer to the destination file.
  A partial write is possible, so the inner loop continues until every byte from
  the current chunk is written.

Two important details:

1. Maximum I/O size per call: Because each ``hipFileRead()`` or
   ``hipFileWrite()`` call can transfer at most ``0x7ffff000`` bytes, the buffer
   is sized accordingly, and larger files are handled by iterating.
2. Block-size alignment on writes: The write size is rounded up to the
   file system block size with ``align_up()``. This is required because the file
   is opened with ``O_DIRECT``, which mandates that offsets and sizes are
   aligned to the block size.

Error handling with IS_HIPFILE_ERR and HIPFILE_ERRSTR
-----------------------------------------------------

.. code-block:: cpp

   if (nread < 0) {
       fprintf(stderr, "Could not read from %s (%zd) (%s)\n", src_path, nread,
               IS_HIPFILE_ERR(nread) ? HIPFILE_ERRSTR(nread) : strerror(errno));
   }

``hipFileRead()`` and ``hipFileWrite()`` encode their status in the return value:

- ``>= 0``: number of bytes successfully transferred.
- ``-1``: a POSIX or system error occurred. Inspect ``errno``.
- Other negative value: the negative of a ``hipFileOpError_t`` code.

The ``IS_HIPFILE_ERR`` macro tests whether the absolute value of the return code
falls in the hipFile error range (at or above ``HIPFILE_BASE_ERR``, which is 5000).
If it does, ``HIPFILE_ERRSTR`` converts it to a human-readable string via
``hipFileGetOpErrorString()``. Otherwise, the program falls back to
``strerror(errno)`` for standard system errors.

For a full list of error codes, see the
:doc:`/reference/api-errors`.

Truncate the destination file
-----------------------------

.. code-block:: cpp

   ftruncate(dst_fd, static_cast<off_t>(file_size));

Because writes are rounded up to the block size, the destination file may be
slightly larger than the source. A final ``ftruncate()`` trims it to the exact
original size.

Teardown
--------

.. code-block:: cpp

   hipFree(devbuf);

   hipFileHandleDeregister(handle);
   close(fd);

Resources are released in reverse order:

1. Free the GPU buffer with ``hipFree()``.
2. Deregister each file handle with ``hipFileHandleDeregister()``.
3. Close each file descriptor with ``close()``.

There is no need to call ``hipFileDriverClose()`` explicitly. The hipFile
library cleans up its internal state automatically at program exit.

Next steps
**********

- Explore asynchronous and batch I/O operations for overlapping transfers with
  computation. See the :doc:`/reference/api-async` and
  :doc:`/reference/api-batch`. On the AMD backend, batch status and cancel
  return ``hipFileInternalError``; ``hipFileBatchIODestroy`` is a no-op.
  Synchronous I/O, batch setup/submit, and async read/write are available.
- Learn more about registering files and buffers in
  :doc:`/how-to/register-file-and-buffer`.
