.. meta::
  :description: Documentation of the installation, configuration, use of the ROCprofiler-SDK, and rocprofv3 command-line tool 
  :keywords: ROCprofiler-SDK tool, ROCprofiler-SDK library, rocprofv3, ROCm, API, reference

.. _comparing-with-legacy-tools:

========================================================
Comparing ROCprofiler-SDK to other ROCm profiling tools
========================================================

ROCprofiler-SDK is an improved version of ROCm profiling tools that enables more efficient implementations and better thread safety while avoiding problems that plague the former implementations of ROCProfiler and ROCTracer.
Here are the distinct ROCprofiler-SDK features, which also highlight the improvements over ROCProfiler and ROCTracer:

- Improved tool initialization
- Support for simultaneous use of the same services by multiple tools
- Simplified control of one or more data collection services
- Improved error checking and logging
- Backward ABI compatibility
- PC sampling (beta implementation)

The former implementations allow a tool to access any of the services provided by ROCProfiler or ROCTracer, such as API tracing and kernel tracing, by calling ``roctracer_init()`` when an ROCm runtime is initially loaded.
As the calling tool is not required to specify during initialization, the services it needs to use, the libraries must be effectively prepared for any service to be available anytime. 
This behavior introduces unnecessary overhead and makes thread-safe data management difficult, as tools generally don't use all the available services.
For example, ROCTracer always installs wrappers around every runtime API and adds indirection overhead through the ROCTracer library to check for the current service configuration in a thread-safe manner.

ROCprofiler-SDK introduces `context` to solve the preceding issues. Contexts are effectively bundles of service configurations. ROCprofiler-SDK provides a single opportunity for a tool to create as many contexts as required. 
A tool can group all services into one context, create one context per service, or choose a mix.
This change in the design allows ROCprofiler-SDK to be aware of the services that might be requested by a tool at any given time.
The design change empowers ROCprofiler-SDK to:

- Avoid unnecessary preparation for services that are never used. If no registered contexts request HSA API tracing, no wrappers need to be generated.
- Perform more extensive checks during service specification and inform a tool about potential issues early.
- Allow multiple tools to use certain services simultaneously.
- Improve thread safety without introducing parallel bottlenecks.
- Manage internal data and allocations more efficiently.
