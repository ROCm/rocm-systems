.. meta::
   :description: hipFile buffer registration
   :keywords: hipFile, ROCm, buffer, device memory, registration

****************************
hipFile buffer registration
****************************

Buffer registration stores the device base pointer and length in bytes in an internal map. During ``hipFileRead()`` and ``hipFileWrite()``, offsets and sizes are validated against the map without requiring HIP pointer lookup. This is useful for situations where the buffer is reused for multiple transfers or when predictable bounds checks are needed.

Buffers are registered with ``hipFileBufRegister()``. If ``hipFileBufRegister()`` is called, it must be called before a call to ``hipFileRead()`` or ``hipFileWrite()``. Buffers with either ``hipMemoryTypeDevice`` or ``hipMemoryTypeHost`` can be registered.

Registering a buffer located in device memory is optional. Buffers referring to host memory must always be registerd, just like :doc:`file registration <./hipFile-file-registration>`. When an unregistered device buffer is passed, hipFile resolves the pointer at the time of the I/O operation.

When a registered device buffer is passed to ``hipFileRead()`` or ``hipFileWrite()``, hipFile looks up the tracked entry and validates the I/O range against the registered length.

Call ``hipFileBufDeregister()`` when the buffer is no longer needed.
