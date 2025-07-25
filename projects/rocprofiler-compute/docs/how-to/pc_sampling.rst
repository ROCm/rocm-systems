.. meta::
   :description: ROCm Compute Profiler: using PC sampling
   :keywords: ROCm Compute Profiler, PC sampling

********************************************
Using PC sampling in ROCm Compute Profiler
********************************************

Program Counter (PC) sampling service for GPU profiling is a profiling technique that periodically samples the program counter during the GPU kernel execution to understand code execution patterns and hotspots.

ROCm Compute Profiler supports Host Trap PC sampling and Stochastic (Hardware-Based) PC sampling.
Host Trap PC sampling is enabled for AMD Instinct MI200 series and later
accelerators. Stochastic (hardware-based) PC sampling is enabled for
AMD Instinct MI300 series and later accelerators. Stochastic PC sampling provides additional information that tells whether a sampled wave issued an instruction for a particular PC. It also provides the reason
for not issuing the instruction (stall reason). This type of information is
particularly useful for understanding stalls during the kernel execution. The PC sampling can be used with profiling and analysis options.

---------------------
Profiling options
---------------------
For using profiling options for PC sampling the configuration needed are:

* ``--pc-sampling-method``: Should be either ``stochastic`` or ``host_trap``, (DEFAULT: stochastic)
* ``--pc-sampling-interval``: For stochastic sampling, the interval is in cycles. The finest granularity is 1 cycle. For ``host_trap`` sampling, the interval is in microsecond (DEFAULT: 1048576). The interval should be the power of 2. You are recommended try starting from 1048576, and lowering until reaching 65536.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute profile -n pc_test -b 21 --no-roof --pc-sampling-method stochastic --pc-sampling-interval 1048576 -VVV -- target_app

-----------------------
Analysis options
-----------------------
For using analysis options for PC sampling the configuration needed are:

* ``--pc-sampling-sorting-type``: ``offset`` or ``count``. The default option is ``offset``. ``offset`` is an assembly instruction offset in the code object.

**Sample command:**

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/pc_test/MI300A_A1/ -b 21 -k 0 --pc-sampling-sorting-type offset

**Sample output:**

Selecting single kernel host trap PC sampling:

.. image:: ../data/pc_sampling/pc_sampling_host_trap_single_kernel.png
   :align: left
   :alt: Host trap PC sampling snapshot

Selecting single kernel stochastic PC sampling:

.. image:: ../data/pc_sampling/pc_sampling_stochastic_single_kernel.png
   :align: left
   :alt: Stochastic PC sampling snapshot

If you don't filter by kernel, the output will fall back to the original data from ``rocprofv3`` csv output for all the kernels:

.. image:: ../data/pc_sampling/pc_sampling_no_kernel_filtering.png
   :align: left
   :alt: Host trap PC sampling snapshot no_kernel_filtering

Selecting single kernel sorting by PC count:

.. image:: ../data/pc_sampling/pc_sampling_sort_by_count.png
   :align: left
   :alt: Host trap PC sampling sorting snapshot

.. note::

  * PC sampling feature is currently in BETA version. To enable PC sampling, you have to explicitly enable it with block index 21.
  * To associate PC sampling info back to HIP source code, you need to build the profiling target app with ``-g`` to keep the symbols. Otherwise, PC sampling info will be only associated with assembly lines.
