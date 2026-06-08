.. meta::
  :description: Explains the hipFile driver lifecycle, reference counting model, implicit initialization, and cleanup behavior.
  :keywords: hipFile, driver, lifecycle, reference counting, hipFileDriverOpen, hipFileDriverClose, hipFileUseCount, ROCm, GPU I/O

=======================================
Driver lifecycle and reference counting
=======================================

hipFile uses a reference counting model to manage the lifecycle of its internal state. This page describes how the reference count controls driver initialization and teardown, the APIs that interact with it, and the cleanup behavior when the count reaches zero.

Reference count model
*********************

The hipFile library maintains an internal reference count that tracks how many active users the driver has. The count determines whether the library's state (registered files, buffers, streams, and backends) is live or torn down.

- A reference count of zero means the driver is uninitialized. API calls that require an initialized driver return ``hipFileDriverNotInitialized``, though synchronous ``hipFileRead()`` and ``hipFileWrite()`` may surface a POSIX error (``-1`` with ``errno`` set) instead.
- A positive reference count means the driver is initialized and ready for normal use.
- When the reference count transitions from one to zero, the library clears the ``FileMap`` and ``BufferMap``.

.. mermaid::

   stateDiagram-v2
       [*] --> Uninitialized: ref_count = 0
       Uninitialized --> Initialized: hipFileDriverOpen() or\nimplicit init (ref 0→1)
       Initialized --> Initialized: hipFileDriverOpen() (ref +1)\nhipFileDriverClose() (ref -1, stays > 0)
       Initialized --> Uninitialized: hipFileDriverClose() (ref 1→0)\nFileMap & BufferMap cleared

Incrementing the reference count
********************************

``hipFileDriverOpen()`` increments the reference count each time it is called. If the count transitions from zero to one, the library initializes its internal state.

.. code-block:: cpp

   hipFileError_t err = hipFileDriverOpen();
   // Reference count is now at least 1

Multiple calls to ``hipFileDriverOpen()`` are valid and each one increments the count. Every increment must eventually be balanced by a corresponding ``hipFileDriverClose()`` call if you want the library to release its resources before process exit.

Decrementing the reference count
********************************

``hipFileDriverClose()`` decrements the reference count. When the count transitions from one to zero, the library destroys all internal state:

- The ``BufferMap`` is cleared (all registered GPU buffer entries are removed).
- The ``FileMap`` is cleared (all registered file handle entries are removed).

.. code-block:: cpp

   hipFileError_t err = hipFileDriverClose();
   // If the reference count reached zero, all registrations are now invalid

.. warning::

   After the reference count reaches zero, any previously obtained file handles or buffer registrations are no longer valid. Attempting to use them results in errors such as ``hipFileHandleNotRegistered`` or ``hipFileMemoryNotRegistered``.

Implicit initialization
***********************

Calling ``hipFileDriverOpen()`` explicitly is optional. The first call to either ``hipFileHandleRegister()`` or ``hipFileBufRegister()`` implicitly initializes the library and increments the reference count if the driver is not already initialized.

From the ``hipFileHandleRegister()`` documentation:

   If the library has not already been initialized, the first call to ``hipFileHandleRegister()`` will initialize the library and increment the reference count.

The same applies to ``hipFileBufRegister()``. This means a minimal application can skip ``hipFileDriverOpen()`` entirely and let registration calls handle initialization:

.. code-block:: cpp

   // No explicit hipFileDriverOpen() needed
   hipFileError_t err = hipFileBufRegister(device_ptr, length, 0);
   // The library is now initialized with ref_count >= 1

Internally, ``registerFile()`` and ``registerBuffer()`` increment the reference count from zero to one when the driver is uninitialized, so the driver is ready for use without an explicit ``hipFileDriverOpen()`` call.

Querying the reference count
****************************

Use ``hipFileUseCount()`` to obtain the current reference count at any time:

.. code-block:: cpp

   int64_t count = hipFileUseCount();
   printf("Current reference count: %ld\n", count);

A return value of zero indicates that the driver is uninitialized. A positive value indicates that the driver is active and has that many outstanding references.

For the complete function signatures, see the GPU I/O driver API group in :doc:`/reference/api-reference`.

Automatic cleanup at process exit
*********************************

Explicitly closing the library is not required. The ``DriverState`` destructor runs automatically at process exit and destroys all remaining state. This means that if your application does not need to reclaim resources during execution, you can omit calls to ``hipFileDriverClose()`` and let the process-exit cleanup handle teardown.

Balanced open and close
********************************

The following example from the ``aiscp`` sample program demonstrates a typical pattern.

.. code-block:: cpp

   #include <hipfile.h>

   int main() {
       // Explicitly initialize
       hipFileError_t err = hipFileDriverOpen();
       if (err.err != hipFileSuccess) { /* handle error */ }

       // Register files and buffers, perform I/O...

       // Deregister files and buffers...

       // Explicitly close: decrements ref count
       err = hipFileDriverClose();
       // If this was the last reference, all state is destroyed

       return 0;
   }

Internal state cleared on teardown
**********************************

When the reference count reaches zero (whether through explicit ``hipFileDriverClose()`` calls or process exit), the following internal structures are cleared:

``FileMap``
   Stores all registered file handles created by ``hipFileHandleRegister()``. Each entry maps a file descriptor and opaque handle to an internal ``File`` object containing file metadata (file descriptors, file system type, alignment requirements).

``BufferMap``
   Stores all registered GPU memory regions created by ``hipFileBufRegister()``. Each entry maps a device pointer to an internal ``Buffer`` object containing the pointer, length, memory type, and GPU ID.

The ``BufferMap`` is cleared before the ``FileMap`` to make sure teardown ordering is correct ordering.
