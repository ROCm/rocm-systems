.. meta::
   :description: hipFile driver reference counting
   :keywords: hipFile, driver, lifecycle, reference counting, ROCm, GPU I/O

***************************
hipFile reference counting
***************************

The hipFile runtime, or driver, maintains the internal state for a hipFile process, including the backends, the registered files and buffers, and the internal maps that track those objects.

When the reference count is 0, the hipFile driver's internal state isn't initialized and any calls to ``hipFileRead()`` or ``hipFileWrite()`` will fail. The internal state is only initialized when the reference count increases to 1.

``hipFileHandleRegister()`` and ``hipFileBufRegister()`` increase the reference count to 1 if it was previously 0. ``hipFileDriverOpen()`` increments the reference count by 1 each time it's called.

hipFile clears its registered buffers and file handles when the process exits or the reference count drops to 0.

.. note::

   The reference count can only be decremented through calls to ``hipFileDriverClose()``. Calls to ``hipFileHandleDeregister()`` and ``hipFileBufDeregister()`` don't decrement the reference count.

If the reference count drops to 0, the hipFile driver will be uninitialized and calls to ``hipFileRead()`` or ``hipFileWrite()`` will return an error. ``hipFileUseCount()`` can be used to get the current driver reference count.






