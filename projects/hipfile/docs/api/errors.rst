Errors and Error Handling
=========================

Functions that return a hipFileOpError_t struct
-----------------------------------------------
Errors are handled identically to cuFile. Most API calls return
a ``hipFileError_t`` struct, which includes ``hipFileOpError_t``
field for returning hipFile error codes, and a ``hipError_t``
field for returning GPU error codes. These fields should be checked
for ``hipFileSuccess`` and ``hipSuccess``, respectively. Any other
values indicate an error.

::

    typedef struct __HIPFILE_NODISCARD hipFileError {
        hipFileOpError_t err;         //!< Errors related to hipFile or the GPU IO driver
        hipError_t       hip_drv_err; //!< Errors related to the GPU driver
    } hipFileError_t;

Note that the struct is marked with the ``[[nodiscard]]`` attribute
so in C++17 / C23 or greater, the compiler will complain if you do
not check error values.

If a GPU driver error occurs the ``hipFileOpError_t`` value will be
``hipFileHipDriverError`` and the ``hipError_t`` field will be set to
the appropriate HIP driver error value.

When any other hipFile error is returned, the ``hipFileOpError_t`` field will be
set to the appropriate hipFile error and the ``hipError_t`` field will
be set to ``hipSuccess``.

Several helper macros are included in ``hipfile.h`` that help with error checking:

* ``IS_HIPFILE_ERR()``
* ``HIPFILE_ERRSTR()``
* ``IS_HIP_DRV_ERR()``
* ``HIP_DRV_ERR()``

See the error section of the hipFile reference manual for documentation for
each of these macros.

Functions that return an integer value
--------------------------------------
Several read and write functions (e.g., ``hipFileRead()``) return a ``ssize_t`` value.
Like the POSIX ``read(3)`` call, these API calls return negative values for errors.
Unlike the POSIX call, however, which only returns -1 on errors, the hipFile IO
calls return a value that reflects the ``hipFileOpError_t`` or ``hipError_t``
value that would have been returned.

``hipError_t`` and its values are defined in ``hip/hip_runtime_api.h``. The enum
values are assigned integers up to ~1000. ``hipFileOpError_t`` enum values are
assigned values of 5000+.

When hipFile IO calls that return a ``size_t`` fail, the returned value is the
negative of the ``hipFileOpError_t`` value (hipFile errors) or the negative of
the ``hipError_t`` value (GPU driver errors) that would normally have been returned
via the ``hipFileError_t`` struct.

Other functions
---------------
* ``hipFileGetOpErrorString()`` returns a string that corresponds to a ``hipFileOpError_t`` value. It cannot fail.
* ``hipFileUseCount()`` returns the reference count of the library. It returns -1 on errors.
