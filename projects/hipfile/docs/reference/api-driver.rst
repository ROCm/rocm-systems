.. meta::
   :description: API reference for hipFile driver lifecycle and configuration functions, including initialization, shutdown, property queries, and configuration parameter management.
   :keywords: hipFile, ROCm, driver, API, configuration, GPU I/O, AMD, reference

*************************************************
Driver lifecycle and configuration API reference
*************************************************

Functions and types for initializing and shutting down the GPU I/O driver, querying and setting driver properties, managing the library reference count, and reading or writing configuration parameters.

For a walkthrough of performing I/O after the driver is open, see :doc:`/tutorials/copy-a-file`.

Driver lifecycle and properties
================================

Driver property and flag types and the functions to open and close the GPU I/O
driver, query the library reference count, and query or modify driver
properties. See the Doxygen descriptions for current implementation status.

.. doxygengroup:: driver
   :content-only:

Core versioning and configuration
====================================

Version macros, the offset type, configuration parameter selectors, and the
functions to query the library version and read or write configuration
parameters.

.. doxygengroup:: core
   :content-only:
