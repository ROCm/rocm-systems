Driver API
==========

Basic API Usage
---------------

The hipFile driver API is used to initialize and shut down the "driver" (i.e.,
the library state). The driver is reference counted, and calls to open and
close the driver simply increase and decrease the reference count. When the
reference count drops to zero, the driver's data structures are cleaned up,
including any open files, buffers, etc., which will be closed. The driver
does not currently maintain significant state, so there is little need to be
aggressive about closing the driver. All driver state will be properly
closed when the process exits.

The term "driver" is essentially synonymous with "library". The driver
is implemented as a library singleton and it is not possible to have multiple
drivers open in a single process.

The basic driver API calls:

* ``hipFileError_t hipFileDriverOpen(void)``
* ``hipFileError_t hipFileDriverClose(void)``
* ``int64_t hipFileUseCount(void)``

The open call should be called before making any other hipFile API calls
and the close call should be called when all hipFile API calls are complete.
The ``hipFileUseCount()`` call can be used to check the current reference
count of the driver.

Note that some API calls will automatically open the driver and bump the
reference count if it has not yet been explicitly opened:

* ``hipFileHandleRegister()``
* ``hipFileBufRegister()``

Also note that this "implicit opening" will not be tracked, so it's up the
application to add an extra call to ``hipFileDriverClose()`` to fully
close the driver.

These API calls will only increment the driver count if they succeed.
Because of this hidden driver opening, the reference count should be checked
if the intent is to completely shut down the driver, as simply matching open
and close calls may not reduce the reference count to zero.

These API calls have been coded to behave like cuFile's calls, regarding
initialization and errors, but there is no guarantee that hipFile's driver
calls will match unpublished cuFile API behaviour.

Driver Getters and Setters
--------------------------

These API calls exist in hipFile, but currently have no effect and will return
an error:

* ``hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props)``
* ``hipFileError_t hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size)``
* ``hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)``
* ``hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size)``
* ``hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)``

These API calls also exist in hipFile, but have no effect, cannot be used to set
driver properties, and will return an error:

* ``hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)``
* ``hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value)``
* ``hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str)``
