.. meta::
  :description: Documentation of the usage of rocprofv3-avail
  :keywords: ROCprofiler-SDK tool usage, rocprofv3-avail usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _using-rocprofv3-avail:

======================
Using rocprofv3-avail
======================

``rocprofv3-avail`` is a CLI tool used to query the features supported by the hardware and ROCprofiler-SDK.

The following sections explain the ``rocprofv3-avail`` command-line options for querying various supported features.

``rocprofv3-avail`` is installed with ROCm under ``/opt/rocm/bin``. To use the tool from anywhere in the system, export ``PATH`` variable:

.. code-block:: bash

   export PATH=$PATH:/opt/rocm/bin

.. _rocprofv3-avail_cli-options:

Command-line options
---------------------

The following table lists ``rocprofv3-avail`` command-line options:

.. list-table:: rocprofv3-avail options
   :header-rows: 1

   * - Option
     - Description

   * - ``info``
     - Lists options for detailed information on counters, agents, and pc-sampling configurations.

   * - ``list``
     - Lists options for hardware counters, agents and pc-sampling support.

   * - ``pmc-check``
     -  Checks if a set of counters can be collected together on an agent.

list option
------------

The ``rocprofv3-avail list`` command displays the list of agents and hardware counters supported on them.

.. code-block:: bash

   rocprofv3-avail list

The output contains the logical node id, name, and the list of Performance Monitoring Counters (PMC) supported on the agent.

Sample output (truncated):

.. code-block:: bash

   GPU                           : 0
   Name                          : gfx1100
   PMC                           :

   processor_id_low               capability                     local_mem_size                size
   unique_id                      min_latency                    weight                        node_from
   version_major                  version_minor                  mem_clk_max                   num_xcc
   width                          flags                          size_in_bytes                 array_count

- To list basic info for all the agents, use:

  .. code-block:: bash

   rocprofv3-avail list --agent

- To list basic info only for the device, use the preceding command with ``-d <device_index>``.

  .. code-block:: bash

   rocprofv3-avail list -d 0

- To list the counters on all the agents, use:

  .. code-block:: bash

   rocprofv3-avail list --pmc

  The preceding command when used with ``-d <device_index>`` lists the counters on the device (specified with device_index).

- To list the agents supporting any kind of PC Sampling, use:

  .. code-block:: bash

   rocprofv3-avail list --pc-sampling

  Note that the option ``-d`` is not applicable here.

- To list the agents supporting SPM, use:

  .. code-block:: bash

   rocprofv3-avail list --spm-config

  Note that the option ``-d`` is not applicable here.

info option
------------

The ``rocprofv3-avail info`` command displays the agent information and lists all the counters supported on each agent.

.. code-block:: bash

   rocprofv3-avail info

Sample output (truncated):

.. code-block:: bash

   GPU:0

   cpu_cores_count               : 0
   simd_count                    : 192
   max_waves_per_simd            : 16
   runtime_visibility            : {'hsa': 1, 'hip': 1, 'rccl': 1, 'rocdecode': 1}
   wave_front_size               : 32
   num_xcc                       : 1
   cu_count                      : 96
   array_count                   : 12
   num_shader_banks              : 6
   simd_arrays_per_engine        : 2
   cu_per_simd_array             : 8
   simd_per_cu                   : 2
   gfx_target_version            : 110000
   max_waves_per_cu              : 32
   gpu_id                        : 54057
   workgroup_max_dim             : {'x': 1024, 'y': 1024, 'z': 1024}
   grid_max_dim                  : {'x': 2147483647, 'y': 65535, 'z': 65535}
   name                          : gfx1100
   vendor_name                   : AMD
   product_name                  : AMD Radeon PRO W7900
   model_name                    : ip discovery
   node_id                       : 1
   logical_node_id               : 1
   logical_node_type_id          : 0
   PMC                           :

   processor_id_low               capability                     local_mem_size                size
   unique_id                      min_latency                    weight                        node_from
   version_major                  version_minor                  mem_clk_max                   num_xcc
   width                          flags                          size_in_bytes                 array_count

- To list the pmc info, use:

  .. code-block:: bash

   rocprofv3-avail info --pmc

  The output includes: logical node id, name, counter_name, description of the counter, dimensions, and block/expression for every counter.

  Sample output (truncated):

  .. code-block:: bash

   GPU:0
   Name:gfx1100
   Counter_Name        :   processor_id_low
   Description         :   Constant value processor_id_low from agent properties

   Counter_Name        :   ALUStalledByLDS
   Description         :   The percentage of GPUTime ALU units are stalled by the LDS input queue being full or the output queue being not ready. If there are LDS bank conflicts, reduce them. Otherwise, try reducing the number of LDS accesses if possible. Value range: 0% (optimal) to 100% (bad).
   Expression          :   400*reduce(SQ_WAIT_INST_LDS,sum)/reduce(SQ_WAVES,sum)/reduce(GRBM_GUI_ACTIVE,max)
   Dimensions          :   DIMENSION_INSTANCE[0:0]

  The preceding command when used with ``-d <device_index>`` displays pmc information for the device (specified with device_index).

- To list the supported PC sampling configurations for each agent that supports PC sampling, use:

  .. code-block:: bash

   rocprofv3-avail info --pc-sampling

  The output includes: logical node id, method supported, unit, minimum sampling interval, and maximum sampling interval flags.

  Note that ``-d`` option is not applicable here.

- To list the supported SPM configurations for each agent that supports SPM, use:

   .. code-block:: bash

   rocprofv3-avail info --spm-config

  Note that ``-d`` option is not applicable here.
   
pmc-check option
-----------------

To check if the pmc can be collected together, use:

.. code-block:: bash

   rocprofv3-avail pmc-check  [pmc [pmc...]]

For example, the following command checks if pmc1 and pmc2 can be collected together on agent 0 (specified with ``-d 0``) and pmc3 on agent 1

.. code-block:: bash

   rocprofv3-avail pmc-check -d 0 <pmc1> <pmc2> <pmc3>:device=1

.. note::

   All commands write to the standard output.
