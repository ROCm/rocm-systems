.. meta::
   :description: Tutorial walking through direct-to-GPU I/O using the hipfile Python bindings, covering Driver, FileHandle, Buffer, read, write, and error handling.
   :keywords: hipfile, Python, GPU I/O, tutorial, ROCm, direct I/O, hipMalloc, Buffer, Driver, FileHandle

========================================
Perform GPU I/O with the Python bindings
========================================

This tutorial walks you through a complete example that uses the hipfile Python
bindings to read data from a file directly into GPU memory and write it back to
a different file. You will learn how to:

* Initialize the hipfile driver with the ``Driver`` context manager
* Open and register files with ``FileHandle``
* Allocate GPU memory with ``Buffer`` (backed by ``hipMalloc``)
* Perform synchronous ``read`` and ``write`` operations
* Handle errors with ``HipFileException``
* Clean up resources automatically through context managers

This workflow is useful whenever you need to move large datasets between
storage and GPU memory without an intermediate copy through host RAM.

Prerequisites
*************

Before you begin, make sure you have:

* An AMD GPU supported by ROCm
* ROCm installed and configured
* The hipfile C library built and installed
* The hipfile Python package installed in a virtual environment

For installation instructions, see :doc:`/install/python-bindings`.

Complete example
****************

The following script reads a source file into GPU memory and writes its
contents to a destination file. Save it as ``gpu_copy.py``:

.. code-block:: python

   #!/usr/bin/env python3
   """Copy a file through GPU memory using hipfile."""

   import sys
   import hipfile

   CHUNK_SIZE = 64 * 1024  # 64 KiB per I/O operation

   def gpu_copy(src_path: str, dst_path: str) -> None:
       # 1. Initialize the GPU I/O driver
       with hipfile.Driver() as drv:

           # 2. Open and register the source and destination files
           with hipfile.FileHandle(src_path, "r") as src, \
                hipfile.FileHandle(dst_path, "w") as dst:

               # 3. Allocate a GPU buffer
               buf = hipfile.Buffer(CHUNK_SIZE)

               file_offset = 0
               while True:
                   # 4. Read from the source file into the GPU buffer
                   bytes_read = src.read(buf, size=CHUNK_SIZE,
                                         file_offset=file_offset,
                                         buffer_offset=0)
                   if bytes_read == 0:
                       break

                   # 5. Write from the GPU buffer to the destination file
                   bytes_written = dst.write(buf, size=bytes_read,
                                             file_offset=file_offset,
                                             buffer_offset=0)

                   file_offset += bytes_read

               # 6. Free the GPU buffer
               buf.free()

       print(f"Copied {file_offset} bytes from {src_path} to {dst_path}")

   if __name__ == "__main__":
       if len(sys.argv) != 3:
           print(f"Usage: {sys.argv[0]} SOURCE DEST", file=sys.stderr)
           sys.exit(1)
       try:
           gpu_copy(sys.argv[1], sys.argv[2])
       except hipfile.HipFileException as exc:
           print(f"hipfile error: {exc}", file=sys.stderr)
           sys.exit(1)

Step-by-step explanation
************************

Import hipfile
--------------

.. code-block:: python

   import hipfile

The ``hipfile`` package exposes the high-level Python classes and helpers. For
the full list of available types and functions, see
:doc:`/reference/python-api-reference`.

Initialize the driver
---------------------

.. code-block:: python

   with hipfile.Driver() as drv:
       ...

``Driver`` is a context manager that calls the underlying
``hipFileDriverOpen`` on entry and ``hipFileDriverClose`` on exit. Using it as
a context manager guarantees that the driver is properly shut down even if an
exception occurs.

Open files with FileHandle
--------------------------

.. code-block:: python

   with hipfile.FileHandle(src_path, "r") as src, \
        hipfile.FileHandle(dst_path, "w") as dst:
       ...

``FileHandle`` wraps the C-level file registration workflow
(``hipFileHandleRegister`` and ``hipFileHandleDeregister``). Opening a file as
a context manager ensures that the handle is deregistered and the underlying
file descriptor is closed when the block exits.

Allocate a GPU buffer
---------------------

.. code-block:: python

   buf = hipfile.Buffer(CHUNK_SIZE)

``Buffer`` allocates device memory with ``hipMalloc`` and registers it for
GPU I/O with the hipfile driver. The size is specified in bytes. Call
``buf.free()`` when you no longer need the buffer to release the GPU memory.

Read data into the GPU buffer
-----------------------------

.. code-block:: python

   bytes_read = src.read(buf, size=CHUNK_SIZE,
                         file_offset=file_offset,
                         buffer_offset=0)

``read`` performs a synchronous read from the registered file directly into
device memory. It returns the number of bytes actually read, which may be less
than ``size`` near the end of the file. A return value of ``0`` indicates
end-of-file.

Write data from the GPU buffer
------------------------------

.. code-block:: python

   bytes_written = dst.write(buf, size=bytes_read,
                             file_offset=file_offset,
                             buffer_offset=0)

``write`` performs a synchronous write from device memory to the registered
file. It returns the number of bytes written.

Handle errors
-------------

.. code-block:: python

   except hipfile.HipFileException as exc:
       print(f"hipfile error: {exc}", file=sys.stderr)

All hipfile Python calls raise ``HipFileException`` when the underlying C API
returns an error. Wrapping your top-level call in a ``try``/``except`` block
lets you catch and report these errors cleanly.

Clean up resources
------------------

Because ``Driver`` and ``FileHandle`` are context managers, exiting their
``with`` blocks automatically releases driver state and deregisters file
handles. Call ``buf.free()`` explicitly to release the GPU allocation when you
are finished with it.

Running the example
*******************

.. code-block:: bash

   python gpu_copy.py /path/to/source_file /path/to/dest_file

On success, the script prints the number of bytes copied:

.. code-block:: text

   Copied 1048576 bytes from /path/to/source_file to /path/to/dest_file

Next steps
**********

* Explore the full :doc:`/reference/python-api-reference` for additional
  classes such as ``FileHandleType`` and ``OpError``, as well as utility
  functions like ``get_version`` and ``driver_get_properties``.
* Review :doc:`/install/python-bindings` if you need to customize build paths
  or install an editable development version of the package.
* For C-level examples, including asynchronous and batch I/O, see the
  ``examples/`` directory in the hipfile source tree.
