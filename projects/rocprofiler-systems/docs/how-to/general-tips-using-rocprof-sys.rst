.. meta::
   :description: ROCm Systems Profiler general tips and usage documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, tips, how to, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

********************************************
General tips for using ROCm Systems Profiler
********************************************

Follow these general guidelines when using ROCm Systems Profiler. For an explanation of the terms used in this topic, see
the :doc:`ROCm Systems Profiler glossary <../reference/rocprof-sys-glossary>`.

* Use ``rocprof-sys-avail`` to look up configuration settings, hardware counters, and data collection components

  * Use the ``-d`` flag for descriptions

* Generate a default configuration with ``rocprof-sys-avail -G ${HOME}/.rocprof-sys.cfg`` and adjust it
  to the desired default behavior
* **Decide whether binary instrumentation, statistical sampling, or both** provides the desired performance data (for non-Python applications)
* Compile code with optimization enabled (``-O2`` or higher), disable asserts (i.e. ``-DNDEBUG``), and include debug info (for instance, ``-g1`` at a minimum)

  * Compiling with debug info does not slow down the code, it only increases compile time and the size of the binary
  * In CMake, this is generally done with the settings ``CMAKE_BUILD_TYPE=RelWithDebInfo`` or ``CMAKE_BUILD_TYPE=Release`` and ``CMAKE_<LANG>_FLAGS=-g1``

* **Use binary instrumentation for characterizing the performance of every invocation of specific functions**
* **Use statistical sampling to characterize the performance of the entire application while minimizing overhead**
* Enable statistical sampling after binary instrumentation to help "fill in the gaps" between instrumented regions
* Use the user API to create custom regions and enable/disable ROCm Systems Profiler for specific processes, threads, and regions
* Dynamic symbol interception, callback APIs, and the user API are always available with binary instrumentation and sampling

  * Dynamic symbol interception and callback APIs are (generally) controlled through ``ROCPROFSYS_USE_<API>``
    options, for example, ``ROCPROFSYS_USE_KOKKOSP`` and ``ROCPROFSYS_USE_OMPT`` enable Kokkos-Tools and OpenMP-Tools
    callbacks, respectively

* When generically seeking regions for performance improvement:

  * **Start off by collecting a flat profile**
  * Look for functions with high call counts, large cumulative runtimes/values, or large standard deviations

    * When call counts are high, improving the performance of this function or "inlining" the function can result in quick and easy performance improvements
    * When the standard deviation is high, collect a hierarchical profile and see if the high variation can be attributable to the calling context.
      In this scenario, consider creating a specialized version of the function for the longer-running contexts

  * **Collect a hierarchical profile** and verify the functions that are part of the "critical path" of your
    application, as indicated in the flat profile

    * For example, functions with high call counts but which are part of a "setup" or "post-processing"
      phase that does not consume much time relative to the overall time are generally a lower priority for optimization

* **Use the information from the profiles when analyzing detailed traces**
* When using binary instrumentation in "trace" mode, **binary rewrites are preferable to runtime instrumentation**.

  * Binary rewrites only instrument the functions defined in the target binary, whereas runtime instrumentation might instrument functions defined in the shared libraries which are linked into the target binary

* When using binary instrumentation with MPI, avoid runtime instrumentation

  * Runtime instrumentation requires a fork and a ``ptrace``, which is generally incompatible with how MPI applications spawn processes
  * Perform a binary rewrite of the executable (and optionally, libraries used by the executable) using MPI and run
    the generated instrumented executable using ``rocprof-sys-run`` instead of the original.
    For example, instead of ``mpirun -n 2 ./myexe``, use ``mpirun -n 2 rocprof-sys-run -- ./myexe.inst``, where
    ``myexe.inst`` is the instrumented ``myexe`` executable that was generated.
