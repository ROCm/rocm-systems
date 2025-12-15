.. meta::
   :description: A guide to troubleshooting the RCCL library of multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API, debug

.. _troubleshooting-rccl:

*********************
Troubleshooting RCCL
*********************

This topic explains the steps to troubleshoot functional and performance issues with RCCL.
While debugging, collect the output from the commands in this guide. This data
can be used as supporting information when submitting an issue report to AMD.

.. _debugging-system-info:

Collecting system information
=============================

Collect this information about the ROCm version, GPU/accelerator, platform, and configuration.

*  Verify the ROCm version. This might be a release version or a
   mainline or staging version. Use this command to display the version:

   .. code:: shell

      cat /opt/rocm/.info/version

   Run the following command and collect the output:

   .. code:: shell

      rocm_agent_enumerator

   Also, collect the name of the GPU or accelerator:

   .. code:: shell

      rocminfo

*  Run these ``rocm-smi`` commands to display the system topology.

   .. code:: shell

      rocm-smi
      rocm-smi --showtopo
      rocm-smi --showdriverversion

*  Determine the values of the ``PATH`` and ``LD_LIBRARY_PATH`` environment variables.

   .. code:: shell

      echo $PATH
      echo $LD_LIBRARY_PATH

*  Collect the HIP configuration.

   .. code:: shell

      /opt/rocm/bin/hipconfig --full

*  Verify the network settings and setup. Use the ``ibv_devinfo`` command 
   to display information about the available RDMA devices and determine 
   whether they are installed and functioning properly. Run ``rdma link``
   to print a summary of the network links.

   .. code:: shell

      ibv_devinfo
      rdma link

Isolating the issue
-------------------

The problem might be a general issue or specific to the architecture or system.
To narrow down the issue, collect information about the GPU or accelerator and other
details about the platform and system. Some issues to consider include:

*  Is ROCm running on:

   *  A bare-metal setup
   *  In a Docker container (determine the name of the Docker image)
   *  In an SR-IOV virtualized
   *  Some combination of these configurations

*  Is the problem only seen on a specific GPU architecture?
*  Is it only seen on a specific system type?
*  Is it happening on a single node or multinode setup?
*  Use the following troubleshooting techniques to attempt to isolate the issue.

   *  Build or run the develop branch version of RCCL and see if the problem persists.
   *  Try an earlier RCCL version (minor or major).
   *  If you recently changed the ROCm runtime configuration, AMD Kernel-mode GPU Driver (KMD), or compiler,
      rerun the test with the previous configuration.

.. _collecting-rccl-info:

Collecting RCCL information
=============================

Collect the following information about the RCCL installation and configuration.

*  Run the ``ldd`` command to list any dynamic dependencies for RCCL.

   .. code:: shell

      ldd <specify-path-to-librccl.so>

*  Determine the RCCL version. This might be the pre-packaged component in
   ``/opt/rocm/lib`` or a version that was built from source. To verify the RCCL version,
   enter the following command, then run either rccl-tests or an e2e application.

   .. code:: shell

      export NCCL_DEBUG=VERSION

*  Run rccl-tests and collect the results. For information on how to build and run rccl-tests, see the
   `rccl-tests GitHub <https://github.com/ROCm/rccl-tests/blob/develop/README.md>`_.

*  Collect the RCCL logging information. Enable the debug logs, 
   then run rccl-tests or any e2e workload to collect the logs. Use the 
   following command to enable the logs.

   .. code:: shell

      export NCCL_DEBUG=INFO

.. _use-rccl-replayer:

Using the RCCL Replayer
------------------------

The RCCL Replayer is a debugging tool designed to analyze and replay the collective logs obtained from RCCL runs. 
It can be helpful when trying to reproduce problems, because it uses dummy data and doesn't have any dependencies 
on non-RCCL calls. For more information, 
see `RCCL Replayer GitHub documentation <https://github.com/ROCm/rccl/tree/develop/tools/RcclReplayer>`_.

You must build the RCCL Replayer before you can use it. To build it, run these commands. Ensure ``MPI_DIR`` is set to 
the path where MPI is installed.

.. code:: shell

   cd rccl/tools/rccl_replayer
   MPI_DIR=/path/to/mpi make

To use the RCCL Replayer, follow these steps: 

#. Collect the per-rank logs from the RCCL run by adding the following environment variables.
   This prevents any race conditions that might cause ranks to interrupt the output from other ranks.

   .. code:: shell

      NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=COLL NCCL_DEBUG_FILE=some_name_here.%h.%p.log

#. Combine all the logs into a single file. This will become the input to the RCCL Replayer.

   .. code:: shell

      cat some_name_here_*.log > some_name_here.log

#. Run the RCCL Replayer using the following command. Replace ``<numProcesses>`` with the number of MPI processes to 
   run, ``</path/to/logfile>`` with the path to the collective log file generated during 
   the RCCL runs, and ``<numGpusPerMpiRank>`` with the number of GPUs per MPI rank used in the application.

   .. code:: shell

      mpirun -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>

   In a multi-node application environment, you can replay the collective logs on multiple nodes
   using the following command:

   .. code:: shell

      mpirun --hostfile <path/to/hostfile.txt> -np <numProcesses> ./rcclReplayer </path/to/logfile> <numGpusPerMpiRank>

   .. note::

      Depending on the MPI library you're using, you might need to modify the ``mpirun`` command.

.. _analyze-performance-info:

Analyzing performance issues
=============================

If the issues involve performance issues in an e2e workload, try the following 
microbenchmarks and collect the results. Follow the instructions in the subsequent sections
to run these benchmarks and provide the results to the support team.

*  TransferBench
*  RCCL Unit Tests
*  rccl-tests
  
Collect the TransferBench data
---------------------------------

TransferBench allows you to benchmark simultaneous copies between
user-specified devices. For more information, 
see the :doc:`TransferBench documentation <transferbench:index>`.

To collect the TransferBench data, follow these steps:

#. Clone the TransferBench Git repository.

   .. code:: shell

      git clone https://github.com/ROCm/TransferBench.git 

#. Change to the new directory and build the component.

   .. code:: shell

      cd TransferBench
      make

#. Run the TransferBench utility with the following parameters and save the results.

   .. code:: shell

      USE_FINE_GRAIN=1 GFX_UNROLL=2 ./TransferBench a2a 64M 8

Collect the RCCL microbenchmark data
-------------------------------------

To use the RCCL tests to collect the RCCL benchmark data, follow these steps:

#. Disable NUMA auto-balancing using the following command:

   .. code:: shell

      sudo sysctl kernel.numa_balancing=0

   Run the following command to verify the setting. The expected output is ``0``.

   .. code:: shell

      cat /proc/sys/kernel/numa_balancing

#. Build MPI, RCCL, and rccl-tests. To download and install MPI, see either 
   `OpenMPI <https://www.open-mpi.org/software/ompi/v5.0/>`_ or `MPICH <https://www.mpich.org/>`_.
   To learn how to build and run rccl-tests, see the `rccl-tests GitHub <https://github.com/ROCm/rccl-tests/blob/develop/README.md>`_.

#. Run rccl-tests with MPI and collect the performance numbers.

RCCL and NCCL comparisons
=============================

If you are also using NVIDIA hardware or NCCL and notice a performance gap between the two systems,
collect the system and performance data on the NVIDIA platform. 
Provide both sets of data to the support team.
