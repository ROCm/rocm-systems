.. meta::
   :description: Using ROCProfiler-Compute in Jupyter Notebooks
   :keywords: ROCm, profiler, rocprof-compute, Jupyter, notebook, analysis

*************************************
Jupyter Notebook Analysis
*************************************

ROCProfiler-Compute provides a Jupyter-compatible interface for interactive performance analysis in notebooks. This allows you to analyze profiling data with a simple Python API and visualize results inline.

Overview
========

The Jupyter interface provides:

* Simple API with ``open()`` and ``analysis()`` functions
* Interactive Plotly visualizations
* Direct access to pandas DataFrames for custom analysis
* Support for filtering by kernel, GPU, and dispatch
* Automatic roofline analysis when available

Installation
============

Prerequisites
-------------

Ensure you have the following installed:

* Python 3.9 or later
* Jupyter Notebook or JupyterLab
* ROCProfiler-Compute with all dependencies

Additional Dependencies
-----------------------

The Jupyter interface requires:

.. code-block:: shell

   pip install jupyter ipython plotly pandas

Basic Usage
===========

Quick Start
-----------

Here's a minimal example to get started:

.. code-block:: python

   import sys
   sys.path.insert(0, '/path/to/rocprofiler-compute/src')
   
   import rocprof_compute_jupyter as rc
   
   # Load performance data
   rc.open('/path/to/workload_dir')
   
   # Display analysis
   rc.analysis()

Step-by-Step Guide
------------------

1. **Collect Profiling Data**

   First, profile your application using the standard rocprof-compute workflow:

   .. code-block:: shell

      rocprof-compute profile -w <workload_dir> -- <your_application>

2. **Import the Module**

   In your Jupyter notebook:

   .. code-block:: python

      import rocprof_compute_jupyter as rc

3. **Load Data**

   Load your profiling data:

   .. code-block:: python

      rc.open('/path/to/workload_dir')

4. **View Results**

   Display the analysis:

   .. code-block:: python

      rc.analysis()

Advanced Features
=================

Filtering Results
-----------------

Filter by Kernel
^^^^^^^^^^^^^^^^

Show analysis for specific kernels:

.. code-block:: python

   # Filter by kernel IDs
   rc.analysis(filter_kernel=['0', '1', '2'])

Filter by GPU
^^^^^^^^^^^^^

Analyze specific GPUs:

.. code-block:: python

   # Filter by GPU ID
   rc.analysis(filter_gpu=[0])

Filter by Dispatch
^^^^^^^^^^^^^^^^^^

Focus on specific dispatches:

.. code-block:: python

   # Filter by dispatch IDs
   rc.analysis(filter_dispatch=[0, 1, 2])

Combined Filters
^^^^^^^^^^^^^^^^

You can combine multiple filters:

.. code-block:: python

   rc.analysis(
       filter_kernel=['0'],
       filter_gpu=[0],
       filter_dispatch=[0, 1]
   )

Working with DataFrames
-----------------------

List Available Tables
^^^^^^^^^^^^^^^^^^^^^

See all available data tables:

.. code-block:: python

   rc.list_tables()

This displays a list of table IDs with their descriptions.

Get Specific DataFrames
^^^^^^^^^^^^^^^^^^^^^^^

Retrieve specific tables as pandas DataFrames:

.. code-block:: python

   # Get kernel statistics (table ID 1)
   kernel_stats = rc.get_dataframe(1)
   
   # Get system information (table ID 101)
   sys_info = rc.get_dataframe(101)
   
   # Get L2 cache metrics (table ID 1701)
   l2_cache = rc.get_dataframe(1701)

Common Table IDs
^^^^^^^^^^^^^^^^

* **1**: Kernel Top Statistics
* **2**: Dispatch Information
* **101**: System Information
* **1101**: Command Processor Speed-of-Light
* **1201**: Wavefront Speed-of-Light
* **1301**: Compute Unit Speed-of-Light
* **1401**: Local Data Share Speed-of-Light
* **1601**: Vector L1 Cache Speed-of-Light
* **1701**: L2 Cache Speed-of-Light

Custom Analysis
---------------

Once you have DataFrames, perform custom analysis:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Get kernel statistics
   kernel_df = rc.get_dataframe(1)
   
   # Find top 5 kernels by duration
   top_kernels = kernel_df.nlargest(5, 'Mean_Duration')
   print(top_kernels[['Kernel_Name', 'Mean_Duration']])
   
   # Create custom visualization
   plt.figure(figsize=(12, 6))
   top_kernels.plot(x='Kernel_Name', y='Mean_Duration', kind='bar')
   plt.title('Top 5 Kernels by Duration')
   plt.ylabel('Mean Duration (ns)')
   plt.xticks(rotation=45, ha='right')
   plt.tight_layout()
   plt.show()

Configuration Options
=====================

Loading with Options
--------------------

Customize data loading with various options:

.. code-block:: python

   rc.open(
       '/path/to/workload_dir',
       time_unit='us',           # Time unit: 'ns', 'us', 'ms', 's'
       normal_unit='per_cycle',  # Normalization: 'per_wave', 'per_cycle', 'per_sec'
       max_stat_num=20,          # Max statistics to show
       decimal=3,                # Decimal places for display
       kernel_verbose=1,         # Kernel verbosity level
       no_roof=False             # Disable roofline analysis
   )

Available Options
-----------------

* **time_unit**: Time unit for display (``'ns'``, ``'us'``, ``'ms'``, ``'s'``)
* **normal_unit**: Normalization unit (``'per_wave'``, ``'per_cycle'``, ``'per_sec'``, ``'per_kernel'``)
* **max_stat_num**: Maximum number of statistics to display (default: 10)
* **decimal**: Number of decimal places (default: 2)
* **kernel_verbose**: Kernel name verbosity (0-4, default: 0)
* **no_roof**: Disable roofline analysis (default: False)
* **kernel_filter**: Initial kernel filter (list of kernel IDs)
* **gpu_filter**: Initial GPU filter (list of GPU IDs)
* **dispatch_filter**: Initial dispatch filter (list of dispatch IDs)

Complete Workflow Example
==========================

Here's a complete analysis workflow:

.. code-block:: python

   import rocprof_compute_jupyter as rc
   import pandas as pd
   
   # 1. Load profiling data
   rc.open('/path/to/workload_dir')
   
   # 2. Show basic overview
   rc.analysis(show_basic_only=True)
   
   # 3. List available tables
   rc.list_tables()
   
   # 4. Get kernel statistics
   kernel_df = rc.get_dataframe(1)
   
   # 5. Identify top kernels by duration
   if kernel_df is not None:
       top_3 = kernel_df.nlargest(3, 'Mean_Duration')
       print("Top 3 kernels by duration:")
       print(top_3[['Kernel_Name', 'Mean_Duration', 'Pct_of_Total_Duration']])
       
       # Get kernel IDs
       top_kernel_ids = [str(k) for k in top_3.index.tolist()]
       
       # 6. Show detailed analysis for top kernels
       rc.analysis(filter_kernel=top_kernel_ids)
   
   # 7. Analyze specific metrics
   l2_cache = rc.get_dataframe(1701)
   if l2_cache is not None:
       print("\nL2 Cache Metrics:")
       print(l2_cache)

Comparison with GUI Mode
=========================

The Jupyter interface provides similar functionality to the GUI mode (``--gui``), but with key differences:

**Jupyter Interface Advantages:**

* Programmatic access to all data
* Custom analysis and visualization
* Integration with other Python tools
* Reproducible analysis workflows
* Easy sharing via notebooks

**GUI Mode Advantages:**

* Standalone web application
* Real-time interactive filtering
* No coding required
* Better for quick exploration

Use the Jupyter interface when you need:

* Custom analysis or visualizations
* Integration with other tools
* Reproducible workflows
* Programmatic data access

Use GUI mode when you need:

* Quick interactive exploration
* No programming required
* Standalone web interface

Troubleshooting
===============

Module Not Found
----------------

If you get ``ModuleNotFoundError``, ensure the path is correct:

.. code-block:: python

   import sys
   sys.path.insert(0, '/correct/path/to/rocprofiler-compute/src')
   import rocprof_compute_jupyter as rc

No Data Loaded
--------------

If you see "No performance data loaded", ensure you called ``open()`` first:

.. code-block:: python

   rc.open('/path/to/workload_dir')  # Must call this first
   rc.analysis()                      # Then analyze

Empty DataFrames
----------------

If tables are empty, check:

1. Profiling data was collected successfully
2. Correct workload directory path
3. Filters aren't too restrictive

Display Issues
--------------

If visualizations don't display:

1. Ensure Jupyter is running properly
2. Check that plotly is installed: ``pip install plotly``
3. Try restarting the kernel

API Reference
=============

open()
------

.. code-block:: python

   rc.open(perf_data_dir, **kwargs)

Load performance data from a directory.

**Parameters:**

* ``perf_data_dir`` (str): Path to performance data directory
* ``**kwargs``: Additional options (see Configuration Options)

analysis()
----------

.. code-block:: python

   rc.analysis(filter_kernel=None, filter_gpu=None, 
               filter_dispatch=None, show_basic_only=False)

Display analysis results.

**Parameters:**

* ``filter_kernel`` (list[str], optional): Kernel IDs to filter
* ``filter_gpu`` (list[int], optional): GPU IDs to filter
* ``filter_dispatch`` (list[int], optional): Dispatch IDs to filter
* ``show_basic_only`` (bool): Show only basic metrics

get_dataframe()
---------------

.. code-block:: python

   df = rc.get_dataframe(table_id)

Get a specific DataFrame by table ID.

**Parameters:**

* ``table_id`` (int): Table ID to retrieve

**Returns:**

* ``pd.DataFrame`` or ``None``: DataFrame if found

list_tables()
-------------

.. code-block:: python

   rc.list_tables()

List all available table IDs and descriptions.

See Also
========

* :doc:`use` - General usage guide
* :doc:`analyze/analyze` - CLI analysis mode
* :doc:`../tutorial/profiling-by-example` - Profiling examples
