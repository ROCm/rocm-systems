.. meta::
   :description: Tutorial walking through direct-to-GPU I/O using the hipFile Python bindings, covering Driver, FileHandle, Buffer, read, write, and error handling.
   :keywords: hipFile, Python, GPU I/O, tutorial, ROCm, direct I/O, Buffer, Driver, FileHandle

*****************************************
Perform GPU I/O with the Python bindings
*****************************************

This tutorial walks you through a complete example that uses the hipFile Python
bindings to read data from a file directly into GPU memory and write it back to
a different file. You will learn how to:

- Initialize the hipFile driver with the ``Driver`` context manager
- Open and register files with ``FileHandle``
- Register GPU memory with ``Buffer``
- Perform synchronous ``read()`` and ``write()`` operations
- Handle errors with ``HipFileException``
- Clean up resources automatically through context managers

This workflow is useful whenever you need to move large datasets between
storage and GPU memory without an intermediate copy through host RAM.

Prerequisites
===============

Verify you have:

- An AMD GPU supported by ROCm.
- ROCm installed and configured.
- The hipFile C library built and installed.
- The hipFile Python package installed in a virtual environment.

See :doc:`/install/python-bindings` for installation.

Complete example
==================

The following script reads a source file into GPU memory and writes its
contents to a destination file. It uses the same API sequence as
``python/main.py`` in the hipFile source tree, adapted for chunked copying
with CLI arguments. Save it as ``gpu_copy.py``:

.. code-block:: python

   #!/usr/bin/env python3
   """Copy a file through GPU memory using hipFile."""

   import os
   import sys

   from hipfile import Buffer, Driver, FileHandle, HipFileException

   CHUNK_SIZE = 64 * 1024  # 64 KiB per I/O operation

   def gpu_copy(src_path: str, dst_path: str) -> None:
       gpu_ptr = hipMalloc(CHUNK_SIZE)

       with Driver():
           with Buffer.from_ctypes_void_p(gpu_ptr, CHUNK_SIZE, 0) as buf:
               with FileHandle(
                   src_path, os.O_RDONLY | os.O_DIRECT
               ) as src, FileHandle(
                   dst_path,
                   os.O_WRONLY | os.O_DIRECT | os.O_CREAT | os.O_TRUNC,
               ) as dst:
                   file_offset = 0
                   while True:
                       bytes_read = src.read(
                           buf, CHUNK_SIZE, file_offset, 0
                       )
                       if bytes_read == 0:
                           break
                       dst.write(buf, bytes_read, file_offset, 0)
                       file_offset += bytes_read

       hipFree(gpu_ptr)
       print(f"Copied {file_offset} bytes from {src_path} to {dst_path}")

   if __name__ == "__main__":
       if len(sys.argv) != 3:
           print(f"Usage: {sys.argv[0]} SOURCE DEST", file=sys.stderr)
           sys.exit(1)
       try:
           gpu_copy(sys.argv[1], sys.argv[2])
       except HipFileException as exc:
           print(f"hipFile error: {exc}", file=sys.stderr)
           sys.exit(1)

Step-by-step walkthrough
==========================

Import modules
--------------

.. code-block:: python

   import os
   from hipfile import Buffer, Driver, FileHandle, HipFileException

The Python package name is ``hipfile``. For the full API, see
:doc:`/reference/api-python`.

Initialize the driver
---------------------

.. code-block:: python

   with Driver():
       ...

``Driver`` is a context manager that calls ``hipFileDriverOpen()`` on entry and
``hipFileDriverClose()`` on exit.

Open files with FileHandle
--------------------------

.. code-block:: python

   with FileHandle(src_path, os.O_RDONLY | os.O_DIRECT) as src, \
        FileHandle(dst_path, os.O_WRONLY | os.O_DIRECT | os.O_CREAT | os.O_TRUNC) as dst:
       ...

``FileHandle`` takes a file system path and ``os.open()`` flags. Include
``os.O_DIRECT`` for direct I/O. It opens the file, registers the handle with
hipFile, and deregisters on exit.

Allocate and register GPU memory
--------------------------------

.. code-block:: python

   gpu_ptr = hipMalloc(CHUNK_SIZE)
   with Buffer.from_ctypes_void_p(gpu_ptr, CHUNK_SIZE, 0) as buf:
       ...

``hipMalloc()`` allocates device memory. ``Buffer.from_ctypes_void_p()`` wraps the
pointer for registration with hipFile. ``Buffer`` doesn't free the allocation.
Call ``hipFree()`` after the buffer is deregistered.

Read data into the GPU buffer
-----------------------------

.. code-block:: python

   bytes_read = src.read(buf, CHUNK_SIZE, file_offset, 0)

``read()`` performs a synchronous read from the registered file into device
memory. It returns the number of bytes read. A return value of ``0`` indicates
end-of-file.

Write data from the GPU buffer
------------------------------

.. code-block:: python

   dst.write(buf, bytes_read, file_offset, 0)

``write()`` performs a synchronous write from device memory to the registered
file. It returns the number of bytes written.

Handle errors
-------------

.. code-block:: python

   except HipFileException as exc:
       print(f"hipFile error: {exc}", file=sys.stderr)

hipFile C API failures raise ``HipFileException``. ``FileHandle.read()`` and
``FileHandle.write()`` may also raise ``RuntimeError`` when the handle isn't open
or ``OSError`` when the platform sets ``errno``.

Clean up resources
------------------

``Driver``, ``FileHandle``, and ``Buffer`` are context managers. Call
``hipFree()`` to release the GPU allocation after the ``Buffer`` context exits.

Running the example
*******************

.. code:: shell

   python gpu_copy.py /path/to/source_file /path/to/dest_file

On success, the script prints the number of bytes copied:

.. code-block:: text

   Copied 1048576 bytes from /path/to/source_file to /path/to/dest_file

