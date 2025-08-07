.. meta::
   :description: ROCm Systems Profiler development documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, development, developers guide, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

****************************************************
Development guide
****************************************************

This guide discusses the `ROCm Systems Profiler <https://github.com/ROCm/rocprofiler-systems>`_ design.
It includes a list of the executables and libraries, along with a discussion of the application's
memory, sampling, and time-window constraint models.

Executables
========================================

This section lists the ROCm Systems Profiler executables.

rocprof-sys-avail: `source/bin/rocprof-sys-avail <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/bin/rocprof-sys-avail>`_
-----------------------------------------------------------------------------------------------------------------------------------------------

The ``main`` routine of ``rocprof-sys-avail`` has three important sections:

* Printing components
* Printing options
* Printing hardware counters

rocprof-sys-sample: `source/bin/rocprof-sys-sample <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/bin/rocprof-sys-sample>`_
--------------------------------------------------------------------------------------------------------------------------------------------------

* Requires a command-line format of ``rocprof-sys-sample <options> -- <command> <command-args>``
* Translates command-line options into environment variables
* Adds ``librocprof-sys-dl.so`` to ``LD_PRELOAD``
* Is launched by using ``execvpe`` with ``<command> <command-args>`` and a modified environment

rocprof-sys-causal: `source/bin/rocprof-sys-causal <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/bin/rocprof-sys-causal>`_
---------------------------------------------------------------------------------------------------------------------------------------------------

When there is exactly one causal profiling configuration variant (which enables debugging),
``rocprof-sys-casual`` has a nearly identical design to ``rocprof-sys-sample``

When the command-line options produce more than one causal profiling configuration variant,
the following actions take place for each variant:

* ``rocprof-sys-causal`` calls ``fork()``
* the child process launches ``<command> <command-args>`` using ``execvpe``, which modifies the environment for the variant
* the parent process waits for the child process to finish

rocprof-sys-instrument: `source/bin/rocprof-sys-instrument <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/bin/rocprof-sys-instrument>`_
--------------------------------------------------------------------------------------------------------------------------------------------------------------

* Requires a command-line format of ``rocprof-sys-instrument <options> -- <command> <command-args>``
* Allows the user to provide options specifying whether to perform runtime instrumentation, use binary rewrite, or
  attach to process
* Either opens the instrumentation target (for binary rewrite), launches the target and stops it
  before it starts executing ``main``, or attaches to a running executable and pauses it
* Finds all functions in the targets
* Finds ``librocprof-sys-dl`` and locates the functions
* Iterates over and instruments all the functions, provided they satisfy the
  defined criteria (such as a minimum number of instructions)

  * See the ``module_function`` class

* Until this point, the workflow has been the same for the different options,
  but it diverges after instrumentation is complete:

  * For a binary rewrite: it produces a new instrumented binary and exits
  * For runtime instrumentation or attaching to a process: it instructs the application
    to resume and then waits for it to exit

Libraries
========================================

Common library: `source/lib/common <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/common>`_
--------------------------------------------------------------------------------------------------------------------------------

* General header-only functionality used in multiple executables and/or libraries.
* Not installed or exported outside of the build tree.

Core library: `source/lib/core <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/core>`_
--------------------------------------------------------------------------------------------------------------------------------

* Static PIC library with functionality that does not depend on any components.
* Not installed or exported outside of the build tree.

Binary library: `source/lib/binary <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/binary>`_
--------------------------------------------------------------------------------------------------------------------------------

* Static PIC library with functionality for reading/analyzing binary info.
* Mostly used by the causal profiling sections of ``librocprof-sys``.
* Not installed or exported outside of the build tree.

librocprof-sys: `source/lib/rocprof-sys <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/rocprof-sys>`_
--------------------------------------------------------------------------------------------------------------------------------

This is the main library encapsulating all the capabilities.

librocprof-sys-dl: `source/lib/rocprof-sys-dl <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/rocprof-sys-dl>`_
-----------------------------------------------------------------------------------------------------------------------------------------

This is a lightweight, front-end library for ``librocprof-sys`` which serves three primary purposes:

* Dramatically speeds up instrumentation time compared to using ``librocprof-sys`` directly because
  Dyninst must parse the entire library in order to find the instrumentation functions
  (a ``dlopen`` call is made on ``librocprof-sys`` when the instrumentation functions get called)
* Prevents re-entry if ``librocprof-sys`` calls an instrumented function internally
* Coordinates communication between ``librocprof-sys-user`` and ``librocprof-sys``

librocprof-sys-user: `source/lib/rocprof-sys-user <https://github.com/ROCm/rocprofiler-systems/tree/amd-mainline/source/lib/rocprof-sys-user>`_
-----------------------------------------------------------------------------------------------------------------------------------------------

* Provides a set of functions and types for the users to add to their code, for example,
  disabling data collection globally or on a specific thread or
  user-defined region
* If ``librocprof-sys-dl`` is not loaded, the user API is effectively a set of no-op function calls.

Testing tools
========================================

* `CDash Testing Dashboard <https://my.cdash.org/index.php?project=rocprofiler-systems>`_ (requires a login)

Components
========================================

Most measurements and capabilities are encapsulated into a "component" with the following definitions:

Measurement
   A recording of some data relevant to performance, for instance, the current call-stack,
   hardware counter values, current memory usage, or timestamp

Capability
   Handles the implementation or orchestration of some feature which is used
   to collect measurements, for example, a component which handles setting up function wrappers
   around various functions such as ``pthread_create`` or ``MPI_Init``.

Components are designed to either hold no data at all or only the data for both an instantaneous
measurement and a phase measurement.

Components which store data typically implement a static ``record()`` function
for getting a record of the measurement,
``start()`` and ``stop()`` member functions for calculating a phase measurement,
and a ``sample()`` member function for storing an
instantaneous measurement. In reality, there are several more "standard" functions
but these are the most commonly-used ones.

Components which do not store data might also have ``start()``, ``stop()``, and ``sample()``
functions. However, components which
implement function wrappers typically provide a call operator or ``audit(...)``
functions. These are invoked with the
wrapped function's arguments before the wrapped function gets called and with the return value
after the wrapped function gets called.

.. note::

   The goal of this design is to provide relatively small and resuable lightweight objects
   for recording measurements and implementing capabilities.

Wall-clock component example
--------------------------------------

A component for computing the elapsed wall-clock time looks like this:

.. code-block:: cpp

   struct wall_clock
   {
      using value_type = int64_t;

      static value_type record() noexcept
      {
         return std::chrono::steady_clock::now().time_since_epoch().count();
      }

      void sample() noexcept
      {
         value = record();
      }

      void start() noexcept
      {
         value = record();
      }

      void stop() noexcept
      {
         auto _start_value = value;
         value = record();
         accum += (value - _start_value);
      }

   private:
      int64_t value = 0;
      int64_t accum = 0;
   };

Function wrapper component example
--------------------------------------

A component which implements wrappers around ``fork()`` and ``exit(int)`` (and stores no data)
could look like this:

.. code-block:: cpp

   struct function_wrapper
   {
      pid_t operator()(const gotcha_data&, pid_t (*real_fork)())
      {
         // disable all collection before forking
         categories::disable_categories(config::get_enabled_categories());

         auto _pid_v = real_fork();

         // only re-enable collection on parent process
         if(_pid_v != 0)
               categories::enable_categories(config::get_enabled_categories());

         return _pid_v;
      }

      void operator()(const gotcha_data&, void (*real_exit)(int), int _exit_code)
      {
         // catch the call to exit and finalize before truly exiting
         rocprofsys_finalize();

         real_exit(_exit_code);
      }
   };

Component member functions
--------------------------------------

There are no real restrictions or requirements on the member functions a component needs to provide.
Unless the component is being used directly, the invocation of component member functions via a "component bundler"
(provided by Timemory) makes extensive use of template metaprogramming concepts. This finds the best match, if any,
for calling a component's member function. This is a bit easier to demonstrate using an example:

.. code-block:: cpp

   struct foo
   {
      void sample() { puts("foo::sample()"); }
   };

   struct bar
   {
      void sample(int) { puts("bar::sample(int)"); }
   };

   struct spam
   {
      void start(int) { puts("spam::start()"); }
      void stop()     { puts("spam::stop()"); }
   };

   int main()
   {
      auto _bundle = component_tuple<foo, bar, spam>{ "main" };

      puts("A");
      _bundle.start();

      puts("B");
      _bundle.sample(10);

      puts("C");
      _bundle.sample();

      puts("D");
      _bundle.stop();
   }

When the preceding code runs, the following messages are printed:

.. code-block:: shell

   A
   spam::start()
   B
   foo::sample()
   bar::sample(int)
   C
   foo::sample()
   D
   spam::stop()

In section A, the bundle determined that only the ``spam`` object has a ``start`` function. Since this is determined
via template metaprogramming instead of dynamic polymorphism, this effectively omits any code related to
the ``foo`` or ``bar`` objects. In section B, because the integer ``10`` is passed to the bundle,
the bundle forwards this value to ``bar::sample(int)`` after it invokes ``foo::sample()``. ``foo::sample()`` is
invoked because the bundle recognizes that the call to the ``sample`` member function is still possible without
the argument.

Memory model
========================================

Collected data is generally handled in one of the three following ways:

* It is handed directly to, and stored by, Perfetto
* It is managed implicitly by Timemory and accessed as needed
* As thread-local data

In general, only instrumentation for relatively simple data is directly passed to
Perfetto and/or Timemory during runtime.
For example, the callbacks from binary instrumentation, user API instrumentation,
and rocprofiler-sdk directly invoke
calls to Perfetto or Timemory's storage model. Otherwise, the data is stored
by ROCm Systems Profiler in the thread-data model
which is more persistent than simply using ``thread_local`` static data, which gets deleted
when the thread stops.

Thread identification
--------------------------------------

Each CPU thread is assigned two integral identifiers. One identifier, the ``internal_value``, is
atomically incremented every time a new thread is created.
The other identifier, known as the ``sequent_value``, tries to account for the fact that ROCm Systems Profiler, Perfetto, ROCm, and other applications
start background threads. When a thread is created as a by-product of ROCm Systems Profiler,
the index is offset by a large value. This serves
two purposes:

* Accessing the data for threads created by the user is closer in memory
* When log messages are printed, the index approximately correlates to the order of thread creation from the user's perspective.

The ``sequent_value`` identifier is typically used to access the thread-data.

Thread-data class
--------------------------------------

Currently, most thread data is effectively stored in a static
``std::array<std::unique_ptr<T>, ROCPROFSYS_MAX_THREADS>`` instance.
``ROCPROFSYS_MAX_THREADS`` is a value defined a compile-time and set to ``2048``
for release builds. During finalization,
ROCm Systems Profiler iterates through the thread-data and transforms that data
into something that can be passed along to Perfetto and/or Timemory.
The downside of the current model is that if the user exceeds ``ROCPROFSYS_MAX_THREADS``,
a segmentation fault occurs. To fix this issue,
a new model is being adopted which has all the benefits of this model
but permits dynamic expansion.

Sampling model
========================================

The general structure for the sampling is within Timemory (``source/timemory/sampling``).
Currently, all sampling is done per-thread
via POSIX timers. ROCm Systems Profiler supports both a real-time timer and a CPU-time timer.
Both have adjustable frequencies, delays, and durations.
By default, only CPU-time sampling is enabled. Initial settings are inherited from
the settings starting with ``ROCPROFSYS_SAMPLING_``.

For each type of timer, timer-specific settings can be used to
override the common and inherited timer settings.
These settings begin with ``ROCPROFSYS_SAMPLING_CPUTIME`` for the CPU-time sampler
and ``ROCPROFSYS_SAMPLING_REALTIME`` for
the real-time sampler. For example, ``ROCPROFSYS_SAMPLING_FREQ=500`` initially sets the
sampling frequency to 500 interrupts per second. Adding the setting ``ROCPROFSYS_SAMPLING_REALTIME_FREQ=10``
lowers the sampling frequency for the real-time sampler
to 10 interrupts per second of real-time.

The ROCm Systems Profiler-specific implementation can be found in
`source/lib/rocprof-sys/library/sampling.cpp <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/rocprof-sys/library/sampling.cpp>`_.
Within `sampling.cpp <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/rocprof-sys/library/sampling.cpp>`_,
there is a bundle of three sampling components:

* `backtrace_timestamp <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/rocprof-sys/library/components/backtrace_timestamp.hpp>`_ simply
  records the wall-clock time of the sample.
* `backtrace <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/rocprof-sys/library/components/backtrace.hpp>`_
  records the call-stack via libunwind.
* `backtrace_metrics <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/rocprof-sys/library/components/backtrace_metrics.hpp>`_
  records the sample metrics, such as peak RSS and the hardware counters.

These three components are bundled together in
a tuple-like ``struct`` (``tuple<backtrace_timestamp, backtrace, backtrace_metrics>``).
A buffer of at least 1024 instances of this tuple is mapped using ``mmap``
per-thread. When this buffer is full,
the sampler hands the buffer off to its allocator thread and maps a new buffer with ``mmap``
before taking the next sample. The allocator thread takes this data
and either dynamically stores it in memory or writes it to a file depending on the
value of ``ROCPROFSYS_USE_TEMPORARY_FILES``.
This schema avoids all allocations in the signal handler, lets the data grow
dynamically, avoids potentially slow I/O within the signal handler, and also enables
the capability of avoiding I/O altogether.
The maximum number of samplers handled by each allocator is governed by the
``ROCPROFSYS_SAMPLING_ALLOCATOR_SIZE`` setting (the default is eight). Whenever an allocator
has reached its limit,
a new internal thread is created to handle the new samplers.

Time-window constraint model
========================================

With the recent introduction of tracing delay and duration, the
`constraint namespace <https://github.com/ROCm/rocprofiler-systems/blob/amd-mainline/source/lib/core/constraint.hpp>`_
was introduced to improve the management of delays and duration limits for
data collection. The ``spec`` class accepts a clock identifier, a delay value, a duration value, and an
integer indicating how many times to repeat the delay and duration cycle. It is therefore
possible to perform tasks such as periodically enabling tracing for brief periods
of time in between long periods without data collection while the application runs. The
syntax follows the format ``clock_identifier:delay:capture_duration:cycles``, so a value of
``10:1:3`` for the last three parameters represents the following sequence of operations:

* Ten seconds where no data is collected, then one second where it is
* Ten seconds where no data is collected, then one second where it is
* Ten seconds where no data is collected, then one second where it is
* Stop

As another example, ``ROCPROFSYS_TRACE_PERIODS = realtime:10:1:5 process_cputime:10:2:20`` translates
to this sequence:

* Five cycles of: no data collection for ten seconds of real-time followed by one second of data collection
* Twenty cycles of: no data collection for ten seconds of process CPU time followed by two CPU-time seconds of data collection

Eventually, the goal is to migrate all subsets of data collection which currently support
more rudimentary models of time window constraints, such as process sampling and causal profiling,
to this model.
