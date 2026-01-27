.. meta::
   :description: Dynamic process attachment in ROCm Compute Profiler
   :keywords: ROCm Compute Profiler, Attach, Detach, Dynamic process attachment

***********************************************************
Dynamic process attachment in ROCm Compute Profiler
***********************************************************

Dynamic process attachment is a new feature of ROCm Compute Profiler that allows coupling with a workload process, without controlling its start or end. The application can already be running before the profiler application is invoked. The profiler simply attaches to the process, collects the required counters, and then detaches—without altering the lifecycle of the workload.

A specific attach is not repeatable, and it can only collect the set of counters that the hardware is capable of capturing in a single run. As such, in the current implementation, you must specify a subset of counter groups that can be collected within one run. This can be done either by using the ``--block`` option (for example, --block 3.1.1 4.1.1 5.1.1) or by providing a predefined set through the use of single pass counter collection ``--set``.

Detachment can be achieved in two ways:

a) By setting the ``--attach-duration-msec`` parameter to a specific duration (in milliseconds). In this case, the detachment occurs automatically after the specified time has elapsed since the ``rocprof`` subprocess started.
b) By pressing the Enter key after a successful attach within the same profiling terminal session. Upon a successful attach, a confirmation message is displayed in the terminal log of the workload process.

---------------------
Profiling options
---------------------
For using profiling options for PC sampling the configuration needed are:

* ``--attach-pid``: Should be the process ID of the process of workload's application.
* ``--attach-duration-msec``: (Optional) This is for setting up the synchronized detach, and is optional. Its unit is in milliseconds. When setting up, the detach will happen after this time has elapsed since the ``rocprof`` subprocess started. For example, setting it to 60000 yields 1 minute.

**Sample command:**

.. code-block:: shell
   
   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --attach-pid <process id of workload>

   $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id of workload> --attach-duration-msec <time before detach>

   $ rocprof-compute profile -n try_live_attach_detach --set launch_stats --no-roof -VVV --attach-pid <process id of workload> --attach-duration-msec <time before detach>

-----------------------
Analysis options
-----------------------

The analyze options for Dynamic process attachment are completely compatible with other non-Dynamic process attachment options.

.. note::

  * Dynamic process attachment feature is currently in BETA version. To enable Dynamic process attachment, you need to have the correct supported version of ROCprofiler-SDK and rocprofiler-register.
  * To make the Dynamic process attachment feature work, you must use "--block" or "--set" to limit the number of counter input files to ensure single application run. You can also use "--iteration-multiplexing" to ensure single application run.
  * Due to the limitation of ROCprofiler-SDK, the attach can now only happen before Heterogeneous System Architecture (HSA) initialization. HSA initialization happens before the execution of the first HIP kernel call. It only happens once to save all the kernels' function signature, such as the function name and other launch parameters. Attaching after this stage misses all crucial information of the HIP kernel and makes it impossible to store the output. This limitation will be solved in later releases of ROCprofiler-SDK.
