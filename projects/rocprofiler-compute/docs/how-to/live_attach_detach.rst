.. meta::
   :description: ROCm Compute Profiler: using Live Attach Detach
   :keywords: ROCm Compute Profiler, Attach Detach

********************************************
Using Live Attach Detach in ROCm Compute Profiler
********************************************

Live Attach/Detach is a new feature of rocprofiler-compute that allows coupling with a workload process, without controlling its start or end. Instead, the application is already running when the profiler is invoked. The profiler simply attaches to the process, collects the required counters, and then detaches—without altering the lifecycle of the workload.

Since a specific attach is not repeatable, it can only collect the set of counters that the hardware is capable of capturing in a single run. Therefore, in the current implementation, the user must specify a subset of counter groups that can be collected within one run. This can be done either by using the --block (for example, --block 3.1.1 4.1.1 5.1.1), or by providing a predefined set through the use of single pass counter collection --set.

Detachment can be achieved in two ways:
By setting the --attach-duration-msec parameter to a specific duration (in milliseconds). In this case, detachment occurs automatically after the specified time has elapsed since the rocprofiler subprocess started.
By pressing the Enter key after a successful attach within the same profiling terminal session. Upon a successful attach, a confirmation message is displayed in the terminal log of the workload process.

---------------------
Profiling options
---------------------
For using profiling options for PC sampling the configuration needed are:

* ``--attach-pid``: Should be the process ID of the process of workload's application.
* ``--attach-duration-msec``: (Optional) The is for setting up the synchronised detach and it's optional. Its unit is in milliseconds. When setting up, the detach will happen adter this time since rocprof starts. For example, setting it to 60000 yields 1 mins.

**Sample command:**

.. code-block:: shell
   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --attach-pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id of workload> --attach-duration-msec <time before detach>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --attach-pid <process id of workload> --attach-duration-msec <time before detach>

-----------------------
Analysis options
-----------------------
The analyze options for attach/detach are completely compatible with the non-attach/detach option

.. note::

  * Live Attach Detach feature is currently in BETA version. To enable Live Attach Detach, you have to have proper verson of rocprofiler-sdk and rocprofiler-register.
  * To make Live Attach Detach work, you must use "--block" or single path to limit the number of counter input files to 1. This limitation will be release in later version with implementation such as Iteration Mutiplexing.
  * Due to limitation of rocporfiler-sdk, the attach can now only happen before HSA initialization. HSA initialization happens before the execution of the first HIP kernel call. It only happens once to save all the kernels' function signature, like fucntion name and other launch parameters. Attaching after this stage misses all crucial infomations of HIP kernel and make it impossible to store output. This limitation will be solved in later releases of rocprofiler-sdk.