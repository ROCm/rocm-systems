Core API
========

API Versioning
--------------

hipFile uses `semantic versioning <https://semver.org/>`_ so library
releases have major, minor, and patch versions. The soversion number
on POSIX systems matches the major API number, so we guarantee that minor
and patch releases will not break the ABI.

These version numbers are defined in ``hipfile.h``:

* ``HIPFILE_VERSION_MAJOR``
* ``HIPFILE_VERSION_MINOR``
* ``HIPFILE_VERSION_PATCH``

These symbols can be used to ifdef around API differences. They are always set
to integer values and will never contain text strings like "-patch1".

hipFile also includes an API call for determining the library version at
runtime:

.. code-block:: c

    hipFileError_t hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch);

Any of the parameters can be ignored by passing in a NULL pointer.

We do not provide a hipFile equivalent to cuFile's ``cuFileGetVersion()``
via the ``hipify`` tool. This is because any logic involving the obtained
version number would be platform-specific and have to be customized regardless.

Parameter Getters and Setters
-----------------------------

Like cuFile, hipFile includes a set of functions for getting and setting library
parameters. These comprise several API calls that get/set parameters
of a particular type based on an enum selector (e.g., ``hipFileGetParameterSizeT()``).
These API calls are all unsupported at this time and will return errors
if called.

* ``hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value)``
* ``hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value)``
* ``hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len)``
* ``hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value)``
* ``hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value)``
* ``hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char *desc_str)``
