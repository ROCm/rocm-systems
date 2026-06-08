***************
File Handle API
***************

File Handles
------------
hipFile files are accessed via an opaque ``hipFileHandle_t`` pointer
obtained from ``hipFileHandleRegister()``. This registration API call
takes a ``hipFileDescr_t`` struct, which contains the file system file
descriptor to be used for hipFile I/O. This hipFile handle must be closed
using ``hipFileHandleDeregister()`` to avoid leaking resources.

Ideally, ``hipFileDriverOpen()`` should be used to initialize the driver
before registering hipFile handles. hipFile will automatically perform
this initialization for the caller, though, if the driver has not been
initialized when ``hipFileHandleRegister()`` is called. See the driver
section of the documentation for a more thorough discussion of this.

Buffers
-------
Memory buffers that will be used with multiple hipFile I/O operations
should be registered via ``hipFileBufRegister()``. If this is not
done, a temporary internal buffer will be used for I/O, though this
may not be as performant as using registered buffers.

When registering a buffer, no special handle or pointer is returned
to the caller. Instead, the registered buffer will be tracked internally
and used for I/O when appropriate. When no longer necessary for I/O,
registered buffers should be freed using ``hipFileBufDeregister()``.

As in ``hipFileHandleRegister()``, the driver will automatically be
initialized by ``hipFileBufRegister()`` if it has not already.

I/O Operations
-------------
hipFile read and write operations are performed using ``hipFileRead()``
and ``hipFileWrite()``, respectively. These API calls take a hipFile
handle, a buffer (registered or not), the size of the I/O operation
in bytes, and the file and buffer offsets. If using a registered buffer,
the buffer pointer should be the one that was registered and not
a pointer inside that buffer.
