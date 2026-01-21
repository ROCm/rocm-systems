.. meta::
   :description: How to use RCCL
   :keywords: RCCL, ROCm, library, API,

.. _integrate_rccl:

*********************
Use the RCCL Replayer
*********************

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
