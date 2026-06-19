.. meta::
   :description: Tutorial walking through the aiscp example program, which copies a file via GPU memory using hipFile synchronous read and write operations.
   :keywords: hipFile, ROCm, GPU I/O, direct-to-GPU, file copy, hipFileRead, hipFileWrite, tutorial, example

****************************************
Copy a file via GPU memory using hipFile
****************************************

`aiscp.cpp  <https://github.com/ROCm/rocm-systems/blob/develop/projects/hipfile/examples/aiscp/aiscp.cpp>`_ copies a source file to a destination by streaming data through GPU
memory, bypassing the CPU data path. It demonstrates every required step of a hipFile workflow:

1. Opening files with ``O_DIRECT`` and registering them for GPU I/O
2. Allocating a device buffer with ``hipMalloc()``
3. Performing chunked reads and writes with ``hipFileRead()`` and ``hipFileWrite()``
4. Handling errors with the ``IS_HIPFILE_ERR`` and ``HIPFILE_ERRSTR`` macros
5. Cleaning up handles, buffers, and file descriptors

Use this pattern whenever you need to move bulk data between storage and GPU
memory without staging it through host RAM.

Prerequisites
*************

- A supported AMD GPU with the ROCm stack installed.
- hipFile built and installed. See :doc:`/install/install`.
- A file system that supports ``O_DIRECT``, for example, ext4 or XFS.
- The HIP runtime development headers from ``hip/hip_runtime_api.h``.

Build the example
*****************

Create a build directory and point CMake at ``examples/aiscp`` in your hipFile
checkout. If ROCm or hipFile are in non-standard locations, pass them through
``CMAKE_PREFIX_PATH``:

.. code:: shell

   cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipFile" /path/to/hipfile/examples/aiscp
   cmake --build .

Run the resulting binary the same way as the Linux ``cp`` command:

.. code:: shell

   ./aiscp SOURCE DEST

Step-by-step explanation
************************

Define the maximum I/O size
--------------------------

.. code-block:: cpp

   #ifndef AISCP_CHUNK_SIZE
   #define AISCP_CHUNK_SIZE 0x7ffff000LU
   #endif

Each call to ``hipFileRead()`` or ``hipFileWrite()`` can transfer at most
``0x7ffff000`` bytes. That equals 2 GiB minus one page. This value matches the Linux
kernel's ``MAX_RW_COUNT`` limit. If you need to transfer more data, you must
loop and issue multiple calls, as shown later in the read-write loop.

Open a file and register it for GPU I/O
--------------------------------------

.. code-block:: cpp

   *fd = open(path, flags | O_DIRECT, mode);

   descr.type      = hipFileHandleTypeOpaqueFD;
   descr.handle.fd = *fd;

   hipfile_err = hipFileHandleRegister(handle, &descr);

Every file you want to use with hipFile must be:

1. Opened with the standard POSIX ``open()`` call. The ``O_DIRECT`` flag selects
   direct I/O, which avoids the kernel page cache.
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

The GPU buffer is allocated with ``hipMalloc()``. Its size is the smaller of the
file size and ``AISCP_CHUNK_SIZE``, rounded up to the file system block size.
``AISCP_CHUNK_SIZE`` is ``0x7ffff000``. That sizing ensures every I/O operation meets alignment
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

Two details:

1. Maximum I/O size per call: because each ``hipFileRead()`` or
   ``hipFileWrite()`` call can transfer at most ``0x7ffff000`` bytes, the buffer
   is sized accordingly, and larger files are handled by iterating.
2. Block-size alignment on writes: the write size is rounded up to the
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
falls in the hipFile error range at or above ``HIPFILE_BASE_ERR``. In the public
headers, ``HIPFILE_BASE_ERR`` is 5000. When the macro matches, ``HIPFILE_ERRSTR`` converts the code to a human-readable string via
``hipFileGetOpErrorString()``. Otherwise, the program falls back to
``strerror(errno)`` for standard system errors.

For a full list of error codes, see the error handling section of the
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

