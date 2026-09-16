.. meta::
  :description: Documentation of the usage of streaming performance monitor(SPM) with rocprofv3 command-line tool
  :keywords: Sampling counters, streaming performance monitors, rocprofv3, rocprofv3 tool usage, Using rocprofv3, ROCprofiler-SDK command line tool, SPM

.. _using-spm:

==================
Using SPM
*****************************
Using SPM
*****************************

SPM (Streaming Performance Monitor) sampling service for GPU profiling is a profiling technique to periodically sample performance counters with GPU timestamp.

Here are the benefits of using SPM to sample counters:

- Identify performance bottlenecks 
- Understand kernel execution behavior
- fine-grained, time-resolved performance data.

To try out the SPM, you can use the command-line tool ``rocprofv3`` or the ROCprofiler-SDK library.

SPM is one of three hardware counter collection mechanisms in ROCprofiler-SDK, alongside dispatch PMC and device counter collection. For a comparison of granularity, timing, and use cases, see :ref:`How SPM differs from other counter services <glance-spm-comparison>` in the SDK overview.

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

.. _spm-cli-options:

CLI options
===================

Use the following options to enable and configure SPM collection with ``rocprofv3``, or query SPM support per agent with ``rocprofv3-avail``.

.. list-table::
   :header-rows: 1

   * - Tool
     - Option
     - Description
     - Example
   * - ``rocprofv3``
     - ``--spm-beta-enabled``
     - Required to enable SPM collection
     - ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES -- ./my_app``
   * -
     - ``--spm <COUNTERS>``
     - Counters to collect; all must fit in a single hardware pass
     - ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES,SQ_BUSY_CYCLES -- ./my_app``
   * -
     - ``--spm-sample-interval <N>``
     - Sampling interval in ``sclk_cycles``
     - ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval 500 -- ./my_app``
   * -
     - ``--spm-sample-interval-unit <UNIT>``
     - Interval unit; currently accepts ``sclk_cycles`` only
     - ``rocprofv3 --spm-beta-enabled --spm SQ_WAVES --spm-sample-interval-unit sclk_cycles -- ./my_app``
   * - ``rocprofv3-avail``
     - ``list --spm``
     - Lists SPM-capable counters per agent
     - ``rocprofv3-avail list --spm``
   * -
     - ``list --spm-config``
     - Lists agents with SPM configuration support, including minimum and maximum sampling intervals
     - ``rocprofv3-avail list --spm-config``

``rocprofv3-avail`` is a companion tool that queries which counters support SPM, and which agents support SPM configuration, without collecting a trace. See :ref:`using-rocprofv3-avail` for full usage details.
