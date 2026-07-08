.. meta::
   :description: Step-by-step instructions for registering a file and GPU buffer for direct GPU I/O using hipFile.
   :keywords: hipFile, GPU I/O, hipFileHandleRegister, hipFileBufRegister, O_DIRECT, ROCm, register file, GPU buffer

*******************************************
Register a file and GPU buffer for GPU I/O
*******************************************

This guide walks you through registering a file and a GPU memory buffer for direct GPU I/O with hipFile. After registration, you can perform read and write operations directly between storage and GPU memory. Buffer-only semantics (registered versus unregistered device memory) are summarized on :doc:`/reference/hipFile-buffer-registration`.

Prerequisites
==============

- hipFile is installed and available on your system.
- A GPU is available and the HIP runtime is functional.
- You have a file to register for GPU I/O.

.. note::

   Using ``O_DIRECT`` when opening files is recommended for optimal performance. If the client fd lacks ``O_DIRECT``, hipFile attempts to reopen the file via ``/proc/self/fd`` with ``O_DIRECT`` before rejecting fastpath selection.

Open a file with ``O_DIRECT``
=============================

Use the POSIX ``open(2)`` system call to open the file. Include the ``O_DIRECT`` flag to enable direct I/O:

.. code-block:: c

   #include <fcntl.h>
   #include <stdio.h>
   #include <string.h>
   #include <errno.h>

   int fd = open("/path/to/file", O_RDONLY | O_DIRECT);
   if (fd == -1) {
       fprintf(stderr, "Could not open file (%s)\n", strerror(errno));
       return 1;
   }

For writable files, use ``O_WRONLY | O_DIRECT`` or ``O_RDWR | O_DIRECT`` as appropriate. When creating a new file, add ``O_CREAT`` and specify a mode:

.. code-block:: c

   int fd = open("/path/to/output", O_WRONLY | O_CREAT | O_DIRECT,
                 S_IWUSR | S_IRUSR | S_IRGRP | S_IROTH);

Register the file handle
===========================

1. Populate a ``hipFileDescr_t`` structure with the file descriptor and handle type:

   .. code-block:: c

      hipFileDescr_t descr;
      descr.type      = hipFileHandleTypeOpaqueFD;
      descr.handle.fd = fd;

   The ``hipFileHandleTypeOpaqueFD`` handle type indicates a POSIX file descriptor.

2. Call ``hipFileHandleRegister()`` to register the file for GPU I/O:

   .. code-block:: c

      hipFileHandle_t handle;
      hipFileError_t err = hipFileHandleRegister(&handle, &descr);
      if (err.err != hipFileSuccess) {
          fprintf(stderr, "Could not register file (%s)\n",
                  HIPFILE_ERRSTR(err.err));
          close(fd);
          return 1;
      }

   On success, ``handle`` contains an opaque file handle for use with hipFile I/O APIs.

   .. note::

      If the hipFile library has not already been initialized, the first call to ``hipFileHandleRegister()`` initializes the library and increments its reference count.

Allocate GPU memory
======================

Allocate a GPU memory buffer using ``hipMalloc()``:

.. code-block:: c

   void *devbuf = NULL;
   size_t buffer_size = 1024 * 1024;  /* 1 MiB */

   hipError_t hip_err = hipMalloc(&devbuf, buffer_size);
   if (hip_err != hipSuccess) {
       fprintf(stderr, "Could not allocate device buffer (%d)\n", hip_err);
       hipFileHandleDeregister(handle);
       close(fd);
       return 1;
   }

Register the GPU buffer
========================

Call ``hipFileBufRegister()`` to register the GPU memory region for use with GPU I/O:

.. code-block:: c

   hipFileError_t buf_err = hipFileBufRegister(devbuf, buffer_size, 0);
   if (buf_err.err != hipFileSuccess) {
       fprintf(stderr, "Could not register buffer (%s)\n",
               HIPFILE_ERRSTR(buf_err.err));
       hipFree(devbuf);
       hipFileHandleDeregister(handle);
       close(fd);
       return 1;
   }

The third parameter is a flags field. Pass ``0`` for default behavior.

.. note::

   If the hipFile library has not already been initialized, the first call to ``hipFileBufRegister()`` also initializes the library and increments its reference count.

After both the file handle and buffer are registered, you can perform GPU I/O operations such as ``hipFileRead()`` and ``hipFileWrite()``. For a complete worked example, see :doc:`/tutorials/copy-a-file`.

Error handling
=================

hipFile API calls return a ``hipFileError_t`` struct containing two fields:

- ``err``: a ``hipFileOpError_t`` value indicating a hipFile-specific or GPU I/O driver error.
- ``hip_drv_err``: a ``hipError_t`` value for GPU driver errors.

Use the ``HIPFILE_ERRSTR`` macro to get a human-readable description of a hipFile error:

.. code-block:: c

   hipFileError_t err = hipFileHandleRegister(&handle, &descr);
   if (err.err != hipFileSuccess) {
       fprintf(stderr, "hipFile error: %s\n", HIPFILE_ERRSTR(err.err));
   }

To check whether an error is a GPU driver error, use the ``IS_HIP_DRV_ERR`` macro, and retrieve the underlying ``hipError_t`` with ``HIP_DRV_ERR``:

.. code-block:: c

   if (IS_HIP_DRV_ERR(err)) {
       hipError_t drv_err = HIP_DRV_ERR(err);
       fprintf(stderr, "HIP driver error: %d\n", drv_err);
   }

For synchronous I/O functions such as ``hipFileRead()`` and ``hipFileWrite()``, the return value is a ``ssize_t``. A non-negative value indicates the number of bytes transferred. A return value of ``-1`` indicates a system error (check ``errno``). Any other negative value is the negated ``hipFileOpError_t`` error code, which you can inspect with ``IS_HIPFILE_ERR`` and ``HIPFILE_ERRSTR``:

.. code-block:: c

   ssize_t nread = hipFileRead(handle, devbuf, buffer_size, 0, 0);
   if (nread < 0) {
       if (IS_HIPFILE_ERR(nread)) {
           fprintf(stderr, "hipFile read error: %s\n", HIPFILE_ERRSTR(nread));
       } else {
           fprintf(stderr, "System error: %s\n", strerror(errno));
       }
   }

For the complete list of error codes and their descriptions, see the :doc:`/reference/api-errors`.

Teardown
============

When you are finished with GPU I/O, release all resources in the following order:

1. Deregister the GPU buffer:

   .. code-block:: c

      hipFileBufDeregister(devbuf);

2. Deregister the file handle:

   .. code-block:: c

      hipFileHandleDeregister(handle);

3. Free the GPU memory:

   .. code-block:: c

      hipFree(devbuf);

4. Close the file descriptor:

   .. code-block:: c

      close(fd);

Complete example
================

The following example shows the full registration and teardown workflow:

.. code-block:: c

   #include <hipfile.h>
   #include <hip/hip_runtime_api.h>
   #include <fcntl.h>
   #include <stdio.h>
   #include <string.h>
   #include <errno.h>
   #include <unistd.h>

   int main(void)
   {
       int fd = open("/path/to/file", O_RDONLY | O_DIRECT);
       if (fd == -1) {
           fprintf(stderr, "Could not open file (%s)\n", strerror(errno));
           return 1;
       }

       hipFileDescr_t descr;
       descr.type      = hipFileHandleTypeOpaqueFD;
       descr.handle.fd = fd;

       hipFileHandle_t handle;
       hipFileError_t err = hipFileHandleRegister(&handle, &descr);
       if (err.err != hipFileSuccess) {
           fprintf(stderr, "Could not register file (%s)\n",
                   HIPFILE_ERRSTR(err.err));
           close(fd);
           return 1;
       }

       size_t buffer_size = 1024 * 1024;
       void *devbuf = NULL;
       hipError_t hip_err = hipMalloc(&devbuf, buffer_size);
       if (hip_err != hipSuccess) {
           fprintf(stderr, "Could not allocate device buffer (%d)\n", hip_err);
           hipFileHandleDeregister(handle);
           close(fd);
           return 1;
       }

       hipFileError_t buf_err = hipFileBufRegister(devbuf, buffer_size, 0);
       if (buf_err.err != hipFileSuccess) {
           fprintf(stderr, "Could not register buffer (%s)\n",
                   HIPFILE_ERRSTR(buf_err.err));
           hipFree(devbuf);
           hipFileHandleDeregister(handle);
           close(fd);
           return 1;
       }

       /* Perform GPU I/O operations here (hipFileRead, hipFileWrite, etc.) */

       /* Teardown */
       hipFileBufDeregister(devbuf);
       hipFileHandleDeregister(handle);
       hipFree(devbuf);
       close(fd);

       return 0;
   }
