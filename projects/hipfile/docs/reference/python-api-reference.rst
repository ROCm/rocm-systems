.. meta::
  :description: Python bindings API reference for hipFile, documenting Driver, FileHandle, Buffer, enums, exceptions, and helper functions.
  :keywords: hipFile, Python, API, bindings, ROCm, GPU, direct I/O, Cython

*****************************
Python bindings API reference
*****************************

.. warning::

   The hipFile Python bindings are experimental. The API may change in future releases without notice. Pin your dependency versions accordingly.

The hipFile Python package provides Cython-based bindings to the hipFile C API. All classes that manage resources support the Python context manager protocol. The Global Interpreter Lock (GIL) is released during all underlying C calls, allowing multi-threaded Python programs to issue concurrent I/O operations.

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

   class FileHandleType(enum.IntEnum):
       OPAQUE_FD = 1
       OPAQUE_WIN32 = 2
       USERSPACE_FS = 3

:``OPAQUE_FD``: POSIX file descriptor.
:``OPAQUE_WIN32``: Win32 ``HANDLE`` file handle.
:``USERSPACE_FS``: Userspace RDMA file system.

``OpError``
-----------

Enum representing hipFile operation error codes. Maps to the C ``hipFileOpError_t`` enumeration.

.. code-block:: python

   class OpError(enum.IntEnum):
       ...

Each member corresponds to a ``hipFileOpError_t`` value, such as ``hipFileSuccess``, ``hipFileDriverNotInitialized``, ``hipFileInvalidValue``). Compare ``HipFileException.hipfile_err`` to ``OpError`` members.

Exception handling
******************

.. _hipfileexception:

``HipFileException``
--------------------

Exception raised when a hipFile operation fails. Carries both the hipFile-level and HIP driver-level error codes, mirroring the C ``hipFileError_t`` struct.

.. code-block:: python

   class HipFileException(Exception):
       ...

:``hipfile_err``: The ``hipFileOpError_t`` value as an ``int``.
:``hip_err``: The ``hipError_t`` value from the HIP driver. Relevant when ``hipfile_err`` equals ``OpError.HIP_DRIVER_ERROR.value``.

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

``Driver.use_count()``
^^^^^^^^^^^^^^^^^^^^^^

Return the current driver reference count. Corresponds to ``hipFileUseCount()``.

:returns: The library reference count as an ``int``.

Context manager usage
---------------------

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

Wraps a file path for use with GPU I/O. ``FileHandle`` implements the context manager protocol, opening the file with ``os.open``, calling ``hipFileHandleRegister()`` on entry, and ``hipFileHandleDeregister()`` on exit.

.. code-block:: python

   class FileHandle:
       def __init__(
           self,
           path: str | os.PathLike[str],
           flags: int,
           mode: int = 0o644,
           handle_type: FileHandleType = FileHandleType.OPAQUE_FD,
       ) -> None: ...
       def open(self) -> None: ...
       def close(self) -> None: ...
       def read(self, buffer, size: int, file_offset: int, buffer_offset: int) -> int: ...
       def write(self, buffer, size: int, file_offset: int, buffer_offset: int) -> int: ...
       def __enter__(self) -> "FileHandle": ...
       def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

``FileHandle.__init__``
^^^^^^^^^^^^^^^^^^^^^^^

:param path: Filesystem path to open.
:param flags: Flags passed to ``os.open``, such as ``os.O_RDONLY | os.O_DIRECT``).
:param mode: Permission bits used when creating a file. Defaults to ``0o644``.
:param handle_type: A ``FileHandleType`` enum value. Defaults to ``FileHandleType.OPAQUE_FD``.

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

:param buffer: A registered GPU buffer, typically a ``Buffer`` instance or device pointer).
:param size: Number of bytes to read.
:param file_offset: Byte offset into the file to begin reading from.
:param buffer_offset: Byte offset into the GPU buffer where data is written.
:returns: The number of bytes read.
:raises RuntimeError: If the file handle is not open.
:raises OSError: On a system-level I/O error.
:raises HipFileException: On a hipFile or HIP driver error.

Corresponds to ``hipFileRead()``. The GIL is released for the duration of the underlying C call.

``FileHandle.write``
^^^^^^^^^^^^^^^^^^^^

Synchronously write data from a GPU buffer to the file.

:param buffer: A registered GPU buffer, typically a ``Buffer`` instance or device pointer).
:param size: Number of bytes to write.
:param file_offset: Byte offset into the file to begin writing to.
:param buffer_offset: Byte offset into the GPU buffer where data is read from.
:returns: The number of bytes written.
:raises RuntimeError: If the file handle is not open.
:raises OSError: On a system-level I/O error.
:raises HipFileException: On a hipFile or HIP driver error.

Corresponds to ``hipFileWrite()``. The GIL is released for the duration of the underlying C call.

Context manager usage
---------------------

.. code-block:: python

   import os
   from hipfile import FileHandle

   with FileHandle("/path/to/file", os.O_RDONLY | os.O_DIRECT) as fh:
       nbytes = fh.read(gpu_buffer, size, file_offset=0, buffer_offset=0)

Buffer class
************

``Buffer``
----------

Wraps a GPU memory region for use with GPU I/O. ``Buffer`` implements the context manager protocol, calling ``hipFileBufRegister()`` on entry and ``hipFileBufDeregister()`` on exit. It does not own the underlying GPU allocation.

.. code-block:: python

   class Buffer:
       @classmethod
       def from_ctypes_void_p(cls, ctypes_void_p, length: int, flags: int) -> "Buffer": ...
       def __init__(self, buffer_ptr: int, length: int, flags: int) -> None: ...
       def register(self) -> None: ...
       def deregister(self) -> None: ...
       def __enter__(self) -> "Buffer": ...
       def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

``Buffer.from_ctypes_void_p``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a ``Buffer`` from a ``ctypes.c_void_p`` returned by ``hipMalloc``.

:param ctypes_void_p: Pointer to GPU memory. Must not be null.
:param length: Size of the buffer in bytes.
:param flags: Registration flags (pass ``0`` for default behavior).

``Buffer.__init__``
^^^^^^^^^^^^^^^^^^^

:param buffer_ptr: Integer address of the GPU memory.
:param length: Size of the GPU memory region in bytes.
:param flags: Control flags for the buffer (pass ``0`` for default behavior).

``Buffer.register()``
^^^^^^^^^^^^^^^^^^^^^

Register the GPU memory region for GPU I/O. Corresponds to ``hipFileBufRegister()``.

:raises HipFileException: If registration fails, such as pointer already registered or invalid memory type).

``Buffer.deregister()``
^^^^^^^^^^^^^^^^^^^^^^^

Deregister the GPU memory region. Corresponds to ``hipFileBufDeregister()``.

:raises HipFileException: If deregistration fails.

Context manager usage
---------------------

.. code-block:: python

   from hipfile import Buffer
   from hipfile.hipMalloc import hipFree, hipMalloc

   ptr = hipMalloc(1024 * 1024)
   with Buffer.from_ctypes_void_p(ptr, 1024 * 1024, 0) as buf:
       # buf is registered; use with FileHandle.read/write
       pass
   hipFree(ptr)

GPU memory helpers
******************

``hipMalloc`` and ``hipFree`` are defined in ``hipfile.hipMalloc``, not the top-level ``hipfile`` package.

``hipMalloc``
-------------

Allocate GPU device memory.

.. code-block:: python

   def hipMalloc(size: int) -> ctypes.c_void_p: ...

:param size: Number of bytes to allocate.
:returns: A ``ctypes.c_void_p`` device pointer to the allocated memory.
:raises RuntimeError: If allocation fails.

``hipFree``
-----------

Free GPU device memory previously allocated with ``hipMalloc``.

.. code-block:: python

   def hipFree(ptr: ctypes.c_void_p) -> None: ...

:param ptr: Device pointer returned by ``hipMalloc``.
:raises RuntimeError: If deallocation fails.

Module-level functions
**********************

``get_version``
---------------

Return the runtime version of the hipFile library.

.. code-block:: python

   def get_version() -> tuple: ...

:returns: A tuple representing the hipFile version, such as ``(major, minor, patch)``).

Corresponds to ``hipFileGetVersion()`` in the C API.

``driver_get_properties``
-------------------------

Retrieve GPU I/O driver properties for the current system.

.. code-block:: python

   def driver_get_properties() -> dict: ...

:returns: A dictionary describing the driver properties. Fields correspond to the ``hipFileDriverProps_t`` structure in the C API.
:raises HipFileException: If the driver is not initialized or properties cannot be retrieved. On the AMD backend, this call currently returns ``hipFileInternalError``.

Corresponds to ``hipFileDriverGetProperties()`` in the C API.

GIL release behavior
********************

All Python bindings release the GIL before calling into the hipFile C library. This means that:

- Multiple Python threads can issue concurrent hipFile operations without blocking each other at the Python level.
- Long-running I/O operations (such as large reads or writes) do not block other Python threads from executing.

.. note::

   While the GIL is released during C calls, the hipFile C library has its own internal synchronization. Refer to the :doc:`/reference/api-reference` for thread-safety guarantees of individual operations.
