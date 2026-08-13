.. meta::
  :description: Documentation of the usage of streaming performance monitor(SPM) with rocprofv3 command-line tool
  :keywords: Sampling counters, streaming performance monitors, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, SPM

.. _using-spm:

==================
Using SPM
==================

SPM (Streaming Performance Monitor) sampling service for GPU profiling is a profiling technique to periodically sample performance counters with GPU timestamp.

Here are the benefits of using SPM to sample counters:

- Identify performance bottlenecks 
- Understand kernel execution behavior
- fine-grained, time-resolved performance data.

To try out the SPM, you can use the command-line tool ``rocprofv3`` or the ROCprofiler-SDK library.

SPM availability and configuration
===========================================

To check counters that can be sampled, use:

.. code-block:: bash

  rocprofv3 -L

Or

.. code-block:: bash

  rocprofv3 --list-avail

The output lists if ``rocprofv3`` supports SPM

.. code-block:: bash

      Counter_Name        :   TCC_MISS
      Description         :   Number of cache misses. UC reads count as misses.
      Block               :   TCC 
      SPM                 :   Supported
      Dimensions          :   DIMENSION_INSTANCE[0:15] DIMENSION_XCC[0:7]

The preceding output shows that the TCC_MISS counter can be sampled.

.. note::
   For proper functioning, SPM requires AMD GPU Driver version **6.19.14.31400000** or later.
   Before using SPM, verify the loaded ``amdgpu`` kernel module version.

To check the driver version, use:

.. code-block:: bash

  cat /sys/module/amdgpu/version

  # Example output:
  # 6.19.14.31400000

You can also check this driver version using ``amd-smi version`` command on DKMS-built systems.

Use the following command to use SPM:

.. code-block:: bash

 rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200  --output-format json -- <application_path>

The preceding command enables SPM for SQ_WAVES and sample interval with unit as sclk cycle counts. Replace ``<application_path>`` with the path to the application you want to profile.
This generates a JSON results file prefixed with the process ID.

.. code-block:: bash

 rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200  -- <application_path>

The preceding command enables SPM for SQ_WAVES and sample interval with unit as sclk cycle counts. Replace ``<application_path>`` with the path to the application you want to profile.
This generates a rocpd results file prefixed with the process ID.

SPM is emitted directly only to the JSON and ``rocpd`` (default) output formats. Only CSV output is supported from the rocpd database via ``rocpd convert``.

Input parameters
===================

Here are the input parameters used to configure SPM

  - ``Metrics List``:  The list of counters that you want to sample with SPM.  Currently, basic counters are supported.
  - ``SPM Sample Interval``:  Specifies the sampling interval for SPM counter collection. It is used with spm-sample-interval-unit to define how frequently counters are sampled.
  - ``SPM Sample Interval Unit``: Specifies the unit for the SPM sample interval. Used with --spm-sample-interval to define the sampling interval. Currently, sclk_cycles unit is supported and the sample interval used with this unit is rounded to nearest multiple of 32.


.. code-block:: bash

  rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200  --output-format json -- <application_path>
  rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval-unit sclk_cycles --spm-sample-interval 1200 -- <application_path>

CSV format conversion
=====================

.. code-block:: bash

  rocpd convert -i <input-file>.db --output-format csv

The CSV conversion process generates a output file ``rocpd-output-data/out_spm_counter_collection_trace.csv`` path relative to the current working directory.

