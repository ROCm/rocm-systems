.. meta::
  :description: ROCprofiler-SDK-ROCTx API reference page
  :keywords: AMD, ROCm, HSA

.. _rocprofiler_sdk_roctx_api_reference:

********************************************************************************
ROCTx API
********************************************************************************

Introduction
============

ROCTx is a comprehensive library that implements the AMD code annotation API. It provides
essential functionality for:

- Event annotation and marking
- Code range tracking and management
- Profiler control and customization
- Thread and device naming capabilities

Key features:

- Nested range tracking with push/pop functionality
- Process-wide range management
- Thread-specific and global profiler control
- Device and stream naming support
- HSA agent and HIP device management

The API is divided into several main components:

1. **Markers** - For single event annotations
2. **Ranges** - For tracking code execution spans
3. **Profiler Control** - For managing profiling tool behavior
4. **Naming Utilities** - For labeling threads, devices, and streams

Thread Safety:

- Range operations are thread-local and thread-safe
- Marking operations are thread-safe
- Profiler control operations are process-wide

Integration:

- Compatible with HIP runtime
- Supports HSA (Heterogeneous System Architecture)
- Provides both C and C++ interfaces

Performance Considerations:

- Minimal overhead for marking and range operations
- Thread-local storage for efficient range stacking
- Lightweight profiler control mechanisms

.. note::

   All string parameters must be null-terminated.

.. warning::

   Proper nesting of range push/pop operations is the user's responsibility.

This ROCTx API topic broadly covers:

* :ref:`roctx_modules_reference`
* :ref:`global_roctx_data_structures_topics_files_reference`
