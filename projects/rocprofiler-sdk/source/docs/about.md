# About

```eval_rst
.. toctree::
   :glob:
   :maxdepth: 4
```

## Important Changes

[Roctracer](https://github.com/ROCm/roctracer) and [rocprofiler (v1)](https://github.com/ROCm/rocprofiler)
have been combined into a single rocprofiler SDK and re-designed from scratch. The new rocprofiler API has been designed with some
new restrictions to avoid problems that plagued the former implementations. These restrictions enable more efficient implementations
and much better thread-safety. The most important restriction is the window for tools to inform rocprofiler about which services
the tool wishes to use (where "services" refers to the capabilities for API tracing, kernel tracing, etc.).

In the former implementations, when one of the ROCm runtimes were initially loaded, a tool only had
to inform roctracer/rocprofiler that it wished to use its services at some point (e.g. calling `roctracer_init()`)
and were not required to specify which services it would eventually or potentially use. Thus, these libraries had to effectively prepare for
any service to be enable at any point in time -- which introduced unnecessary overhead when tools had no desire to use certain features and
made thread-safe data management difficult. For example, roctracer was required to _always_ install wrappers around _every_ runtime API function
and _always_ added extra overhead of indirection through the roctracer library and checks for the current service configuration (in a thread-safe manner).

In the re-designed implementation, rocprofiler introduces the concept of a "context". Contexts are effectively
bundles of service configurations. Rocprofiler gives each tool _one_ opportunity to create as many contexts as necessary --
for example, a tool can group all of the services into one context, create individual contexts for each service, or somewhere in between.
Due to this design choice change, rocprofiler now knows _exactly_ which services might be requested by the tool clients at any point in time.
This has several important implications:

- rocprofiler does not have to unnecessarily prepare for services that are never used -- if no registered contexts requested tracing the HSA API, no wrappers need to be generated
- rocprofiler can perform more extensive checks during service specification and inform tools about potential issues very early on
- rocprofiler can allow multiple tools to use certain services simulatenously
- rocprofiler was able to improve thread-safety without introducing parallel bottlenecks
- rocprofiler can manage internal data and allocations more efficiently
