.. meta::
   :description: ROCm Compute Profiler: using Live Attach Detach
   :keywords: ROCm Compute Profiler, Attach Detach

********************************************
Using Live Attach Detach in ROCm Compute Profiler
********************************************

Live Attach/Detach is a new way for rocprofiler-compute of coupling with the workload process, which does not control the start and end of the workload application. When it runs, the application is already running. What the profiler does is only to attach to the workload process, collect the needed counter and detach, without changing the lifecycle of the workload.

Due to a specific attach is not repeatable, it can only collect a set of counter that the hardware is capable of doing during one run. Therefore, in current implementation, the user needs to specific the subset of goupe of counter and ensure that it's able to acquire during only one run. It can be achieved either by using "--block" parameter, for example, "--block 3.1.1 4.1.1 5.1.1", or by using single path argument such as "--set launch_stats".
The method of detach can be achieve by two way: 1. by setting parameter "--attach-duration-msec" of a specific time in terms of milliseconds, the detach will happen after this preset time since the start of subprocess of rocprofiler. 2. By clicking "Enter" key after the successful attach.

---------------------
Profiling options
---------------------
For using profiling options for PC sampling the configuration needed are:

* ``--pid``: Should be the process ID of the process of workload's application.
* ``--attach-duration-msec``: (Optional) The is for setting up the synchronised detach and it's optional. Its unit is in milliseconds. When setting up, the detach will happen adter this time since rocprof starts. For example, setting it to 6000 yields 1 mins.

**Sample command:**

.. code-block:: shell
   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --pid <process id of workload> --attach-duration-msec <time before detach>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --pid <process id of workload> --attach-duration-msec <time before detach>

-----------------------
Analysis options
-----------------------
The analyze options for attach/detach are completely compatible with the non-attach/detach option

.. note::

  * Live Attach Detach feature is currently in BETA version. To enable Live Attach Detach, you have to have proper verson of rocprofiler-sdk and rocprofiler-register.
  * To make Live Attach Detach work, you must use "--block" or single path to limit the number of counter input files to 1. This limitation will be release in later version with implementation such as Iteration Mutiplexing.
  * Due to limitation of rocporfiler-sdk, the attach can now only happen before HSA initialization. This will be solved in later release of rocprofiler-sdk