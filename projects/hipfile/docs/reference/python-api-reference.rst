.. meta::
   :description: Python bindings API reference for hipfile, documenting Driver, FileHandle, Buffer, enums, exceptions, and helper functions.
   :keywords: hipfile, Python, API, bindings, ROCm, GPU, direct I/O, Cython

=============================
Python bindings API reference
=============================

.. warning::

   The hipfile Python bindings are experimental. The API may change in future releases without notice. Pin your dependency versions accordingly.

The hipfile Python package provides Cython-based bindings to the hipfile C API. All classes that manage resources support the Python context manager protocol. The Global Interpreter Lock (GIL) is released during all underlying C calls, allowing multi-threaded Python programs to issue concurrent I/O operations.

For installation instructions, see :doc:`/install/python-bindings`. For the corresponding C API, see :doc:`/reference/api-reference`.

.. contents:: On this page
   :local:
   :depth: 2

Enumerations
************

``FileHandleType``
------------------

Enum specifying the type of file handle used when registering a file for GPU I/O. Maps to the C ``hipFileFileHandleType_t`` enumeration.

.. code-block:: python

   class FileHandleType(enum.Enum):
       OpaqueFD = 1
       OpaqueWin32 = 2
       UserspaceFS = 3

:``OpaqueFD``: POSIX file descriptor.
:``OpaqueWin32``: Win32 ``HANDLE`` file handle.
:``UserspaceFS``: Userspace RDMA filesystem.

``OpError``
-----------

Enum representing hipfile operation error codes. Maps to the C ``hipFileOpError_t`` enumeration.

.. code-block:: python

   class OpError(enum.Enum):
       ...

Each member corresponds to a ``hipFileOpError_t`` value (for example, ``hipFileSuccess``, ``hipFileDriverNotInitialized``, ``hipFileInvalidValue``). Use this enum to inspect the ``err`` field of a :ref:`HipFileException <hipfileexception>`.

Exception handling
******************

.. _hipfileexception:

``HipFileException``
--------------------

Exception raised when a hipfile operation fails. Carries both the hipfile-level and HIP driver-level error codes, mirroring the C ``hipFileError_t`` struct.

.. code-block:: python

   class HipFileException(Exception):
       err: OpError
       hip_drv_err: int

:``err``: An ``OpError`` enum value describing the hipfile or GPU I/O driver error.
:``hip_drv_err``: The ``hipError_t`` value from the HIP driver. Relevant when ``err`` is ``hipFileHipDriverError``.

Driver class
************

``Driver``
----------

Manages the lifecycle of the GPU I/O driver for the current process. ``Driver`` implements the context manager protocol, calling ``hipFileDriverOpen()`` on entry and ``hipFileDriverClose()`` on exit.

.. code-block:: python

   class Driver:
       def __init__(self) -> None: ...
       def open(self) -> None: ...
       def close(self) -> None: ...
       def __enter__(self) -> "Driver": ...
       def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

``Driver.open()``
^^^^^^^^^^^^^^^^^

Initialize the GPU I/O driver and increment the library reference count. Corresponds to ``hipFileDriverOpen()``.

:raises HipFileException: If the driver cannot be initialized.

``Driver.close()``
^^^^^^^^^^^^^^^^^^

Decrement the library reference count. When the count reaches zero, library state is destroyed. Corresponds to ``hipFileDriverClose()``.

:raises HipFileException: If an error occurs during shutdown.

**Context manager usage:**

.. code-block:: python

   from hipfile import Driver

   with Driver() as drv:
       # Driver is open; perform GPU I/O operations
       pass
   # Driver is closed automatically

FileHandle class
****************

``FileHandle``
--------------

Wraps a file descriptor for use with GPU I/O. ``FileHandle`` implements the context manager protocol, calling ``hipFileHandleRegister()`` on entry and ``hipFileHandleDeregister()`` on exit.

.. code-block:: python

   class FileHandle:
       def __init__(self, fd: int, handle_type: FileHandleType) -> None: ...
       def open(self) -> None: ...
       def close(self) -> None: ...
       def read(self, buffer, size: int, file_offset: int, buffer_offset: int) -> int: ...
       def write(self, buffer, size: int, file_offset: int, buffer_offset: int) -> int: ...
       def __enter__(self) -> "FileHandle": ...
       def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

``FileHandle.__init__``
^^^^^^^^^^^^^^^^^^^^^^^

:param fd: An open file descriptor (for ``OpaqueFD`` or ``UserspaceFS`` handle types).
:param handle_type: A ``FileHandleType`` enum value specifying the handle type.

``FileHandle.open()``
^^^^^^^^^^^^^^^^^^^^^

Register the file for GPU I/O. Corresponds to ``hipFileHandleRegister()``.

:raises HipFileException: If registration fails.

``FileHandle.close()``
^^^^^^^^^^^^^^^^^^^^^^

Deregister the file from GPU I/O. Corresponds to ``hipFileHandleDeregister()``.

``FileHandle.read``
^^^^^^^^^^^^^^^^^^^

Synchronously read data from the file into a GPU buffer.

:param buffer: A registered GPU buffer (typically a ``Buffer`` instance or device pointer).
:param size: Number of bytes to read.
:param file_offset: Byte offset into the file to begin reading from.
:param buffer_offset: Byte offset into the GPU buffer where data is written.
:returns: The number of bytes read.
:raises HipFileException: On I/O error.

Corresponds to ``hipFileRead()``. The GIL is released for the duration of the underlying C call.

``FileHandle.write``
^^^^^^^^^^^^^^^^^^^^

Synchronously write data from a GPU buffer to the file.

:param buffer: A registered GPU buffer (typically a ``Buffer`` instance or device pointer).
:param size: Number of bytes to write.
:param file_offset: Byte offset into the file to begin writing to.
:param buffer_offset: Byte offset into the GPU buffer where data is read from.
:returns: The number of bytes written.
:raises HipFileException: On I/O error.

Corresponds to ``hipFileWrite()``. The GIL is released for the duration of the underlying C call.

**Context manager usage:**

.. code-block:: python

   import os
   from hipfile import FileHandle, FileHandleType

   fd = os.open("/path/to/file", os.O_RDONLY | os.O_DIRECT)
   with FileHandle(fd, FileHandleType.OpaqueFD) as fh:
       nbytes = fh.read(gpu_buffer, size, file_offset=0, buffer_offset=0)
   os.close(fd)

Buffer class
************

``Buffer``
----------

Wraps a GPU memory region for use with GPU I/O. ``Buffer`` implements the context manager protocol, calling ``hipFileBufRegister()`` on entry and ``hipFileBufDeregister()`` on exit.

.. code-block:: python

   class Buffer:
       def __init__(self, ptr, size: int, flags: int = 0) -> None: ...
       def register(self) -> None: ...
       def deregister(self) -> None: ...
       def __enter__(self) -> "Buffer": ...
       def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

``Buffer.__init__``
^^^^^^^^^^^^^^^^^^^

:param ptr: A device pointer to GPU memory (for example, the return value of ``hipMalloc``).
:param size: Size of the GPU memory region in bytes.
:param flags: Control flags for the buffer (default ``0``).

``Buffer.register()``
^^^^^^^^^^^^^^^^^^^^^

Register the GPU memory region for GPU I/O. Corresponds to ``hipFileBufRegister()``.

:raises HipFileException: If registration fails (for example, pointer already registered or invalid memory type).

``Buffer.deregister()``
^^^^^^^^^^^^^^^^^^^^^^^

Deregister the GPU memory region. Corresponds to ``hipFileBufDeregister()``.

:raises HipFileException: If deregistration fails.

**Context manager usage:**

.. code-block:: python

   from hipfile import Buffer, hipMalloc, hipFree

   ptr = hipMalloc(1024 * 1024)
   with Buffer(ptr, 1024 * 1024) as buf:
       # buf is registered; use with FileHandle.read/write
       pass
   hipFree(ptr)

GPU memory helpers
******************

``hipMalloc``
-------------

Allocate GPU device memory.

.. code-block:: python

   def hipMalloc(size: int) -> int: ...

:param size: Number of bytes to allocate.
:returns: An integer device pointer to the allocated memory.
:raises HipFileException: If allocation fails.

``hipFree``
-----------

Free GPU device memory previously allocated with ``hipMalloc``.

.. code-block:: python

   def hipFree(ptr: int) -> None: ...

:param ptr: Device pointer returned by ``hipMalloc``.
:raises HipFileException: If deallocation fails.

Module-level functions
**********************

``get_version``
---------------

Return the runtime version of the hipfile library.

.. code-block:: python

   def get_version() -> tuple: ...

:returns: A tuple representing the hipfile version (for example, ``(major, minor, patch)``).

Corresponds to ``hipFileGetVersion()`` in the C API.

``driver_get_properties``
-------------------------

Retrieve GPU I/O driver properties for the current system.

.. code-block:: python

   def driver_get_properties() -> dict: ...

:returns: A dictionary describing the driver properties. Fields correspond to the ``hipFileDriverProps_t`` structure in the C API.
:raises HipFileException: If the driver is not initialized or properties cannot be retrieved.

Corresponds to ``hipFileDriverGetProperties()`` in the C API.

GIL release behavior
********************

All Python bindings release the GIL before calling into the hipfile C library. This means that:

- Multiple Python threads can issue concurrent hipfile operations without blocking each other at the Python level.
- Long-running I/O operations (such as large reads or writes) do not block other Python threads from executing.

.. note::

   While the GIL is released during C calls, the hipfile C library has its own internal synchronization. Refer to the :doc:`/reference/api-reference` for thread-safety guarantees of individual operations.
