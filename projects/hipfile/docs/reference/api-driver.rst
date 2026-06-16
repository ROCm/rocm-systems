.. meta::
   :description: API reference for hipFile driver lifecycle and configuration functions, including initialization, shutdown, property queries, and configuration parameter management.
   :keywords: hipFile, ROCm, driver, API, configuration, GPU IO, AMD, reference

************************************************
Driver lifecycle and configuration API reference
************************************************

Functions and types for initializing and shutting down the GPU IO driver, querying and setting driver properties, managing the library reference count, and reading or writing configuration parameters.

For a walkthrough of performing I/O after the driver is open, see :doc:`/tutorials/copy-a-file`.

Version macros
**************

.. doxygendefine:: HIPFILE_VERSION_MAJOR
.. doxygendefine:: HIPFILE_VERSION_MINOR
.. doxygendefine:: HIPFILE_VERSION_PATCH

Data types
**********

Driver property and flag types used by the lifecycle and configuration functions.

.. doxygenstruct:: hipFileDriverProps_t
   :members:

.. doxygenenum:: hipFileDriverStatusFlags_t
.. doxygenenum:: hipFileDriverControlFlags_t
.. doxygenenum:: hipFileFeatureFlags_t

Configuration parameter enumerations
************************************

Selectors passed to the parameter getter and setter functions.

.. doxygenenum:: hipFileSizeTConfigParameter_t
.. doxygenenum:: hipFileBoolConfigParameter_t
.. doxygenenum:: hipFileStringConfigParameter_t

Driver lifecycle
****************

Open and close the GPU IO driver and query the reference count.

.. doxygenfunction:: hipFileDriverOpen
.. doxygenfunction:: hipFileDriverClose
.. doxygenfunction:: hipFileUseCount

Driver properties
*****************

Query and modify driver properties. See the Doxygen descriptions for current implementation status.

.. doxygenfunction:: hipFileDriverGetProperties
.. doxygenfunction:: hipFileDriverSetPollMode
.. doxygenfunction:: hipFileDriverSetMaxDirectIOSize
.. doxygenfunction:: hipFileDriverSetMaxCacheSize
.. doxygenfunction:: hipFileDriverSetMaxPinnedMemSize

Version query
*************

.. doxygenfunction:: hipFileGetVersion

Configuration parameter getters
*******************************

.. doxygenfunction:: hipFileGetParameterSizeT
.. doxygenfunction:: hipFileGetParameterBool
.. doxygenfunction:: hipFileGetParameterString

Configuration parameter setters
*******************************

.. doxygenfunction:: hipFileSetParameterSizeT
.. doxygenfunction:: hipFileSetParameterBool
.. doxygenfunction:: hipFileSetParameterString
