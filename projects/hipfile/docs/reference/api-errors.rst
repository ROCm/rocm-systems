.. meta::
   :description: Error handling API reference for hipFile, covering the hipFileError_t struct, hipFileOpError_t enum, helper macros, and the ssize_t return convention.
   :keywords: hipFile, error handling, API, ROCm, hipFileError_t, hipFileOpError_t, hipFileGetOpErrorString, error codes

*****************************
Error handling API reference
*****************************

hipFile functions communicate errors through two mechanisms: a dual-field
``hipFileError_t`` struct returned by most API calls, and a signed ``ssize_t``
return value used by ``hipFileRead`` and ``hipFileWrite``. This page documents
the types, constants, function, and helper macros that form the error handling
surface.

For guidance on interpreting errors in practice, see
:doc:`/reference/api-synchronous-io`.

``nodiscard`` attribute
========================

The ``hipFileError_t`` struct carries the ``[[nodiscard]]`` attribute when
compiled with C++ 17 (``__cplusplus >= 201703L``) or C23
(``__STDC_VERSION__ >= 202311L``). The compiler emits a warning if a caller
discards the return value of any function that returns ``hipFileError_t``.

``ssize_t`` return convention
==============================

``hipFileRead`` and ``hipFileWrite`` return an ``ssize_t`` instead of a
``hipFileError_t``. A non-negative value indicates the number of bytes
transferred. A return value of ``-1`` signals a system error; check
``errno`` for details. Any other negative value is the negation of the
corresponding ``hipFileOpError_t`` code. Use ``IS_HIPFILE_ERR`` and
``HIPFILE_ERRSTR`` (passing the absolute value) to inspect such errors.

Error types
===============

.. doxygengroup:: error
   :content-only:
