.. meta::
  :description: Documentation of the usage of rocprofv3-avail
  :keywords: ROCprofiler-SDK tool usage, rocprofv3-avail usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _using-rocprofv3-avail:

======================
Using rocprofv3-avail
======================

``rocprofv3-avail`` is a CLI tool that helps you to query the features supported by the hardware and Rocprofiler SDK.

The following sections demonstrate the use of ``rocprofv3-avail`` for querying features using various command-line options.

``rocprofv3-avail`` is installed with ROCm under ``/opt/rocm/bin``. To use the tool from anywhere in the system, export ``PATH`` variable:

.. code-block:: bash

   export PATH=$PATH:/opt/rocm/bin

.. _rocprofv3-avail_cli-options:

Command-line options
--------------------

The following table lists ``rocprofv3-avail`` command-line options categorized according to their purpose.

.. # COMMENT: The following lines define a line break for use in the table below.
.. |br| raw:: html

    <br />

.. list-table:: rocprofv3-avail options
   :header-rows: 1

   * - Purpose
     - Option
     - Description
   
   * -  avail-aptions commands
     -  | ``info`` 
        | ``list`` 
        | ``pmc-check``  
     -  | Info options for detailed information of counters, agents, and pc-sampling configurations. 
        | List options for hw counters, agents and pc-sampling support". 
        | Checking counters collection support on agents. 
     
  
Available Hardware Counters
++++++++++++++++++++++++++++
.. code-block:: bash

   rocprofv3-avail -d 0

The preceding command selects a device with logical node type id as 0 in the node.
The option is applied to further sub commands and options

.. code-block:: bash

   rocprofv3-avail list 

The preceding command generates an output listing agents and hardware counters

.. code-block:: bash

   rocprofv3-avail list --agent
      
The preceding command generates an output listing basic info for all agents, if used with -d only basic info for device -d is listed.

.. code-block:: bash

   rocprofv3-avail list --pmc
      
The preceding command generates an output listing counters for all agents, if used with -d only counters on the the -d device is listed

.. code-block:: bash

   rocprofv3-avail list --pc-sampling
      
The preceding command generates an output listing agents that supports any kind of PC Sampling. -d option is not applicable here.
.. code-block:: bash

   rocprofv3-avail info 
       
The preceding command generates an output with agent information and listing all counters 
supported on each

.. code-block:: bash

   rocprofv3-avail info --pmc
       
The preceding command generates an output with the pmc info, if used with -d information of pmc for device -d is generated.

.. code-block:: bash

   rocprofv3-avail info --pc-sampling 
       
The preceding command generates list of supported PC sampling configurations for each agent that supports PC sampling. -d option is not applicable here.

.. code-block:: bash

   rocprofv3-avail pmc-check  [pmc [pmc...]] 
       
The preceding command checks if the pmc can be collected together

.. code-block:: bash

   rocprofv3-avail pmc-check -d 0 <pmc1> <pmc2> <pmc3>:device=1

  The preceding command checks if the pmc1 and pmc2 can be collected together on agent 0
  and pmc3 on agent 1

.. note::

  The above command writes the ouptut to the standard output.
  

