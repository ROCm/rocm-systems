.. meta::
   :description: Build and run your first RCCL AllReduce program on AMD GPUs, from installation through single-node and multi-node execution, with correctness validation.
   :keywords: RCCL, tutorial, AllReduce, ncclCommInitAll, ncclCommInitRank, ncclGroupStart, HIP, ROCm, getting started, multi-GPU, multi-node

.. _your-first-rccl-program:

Run your first RCCL program
===========================

This tutorial walks you through writing, building, and running a complete RCCL
program from scratch. By the end, you will have a working AllReduce benchmark
that runs in both single-process (multiple GPUs, one process) and
multi-process (one GPU per process, multiple nodes) modes, and you will
understand the core RCCL concepts that every distributed workload depends on.

The program you build here mirrors what deep learning frameworks do internally
every time they synchronize gradients across GPUs. Building it yourself makes
the framework behavior transparent and gives you a foundation for debugging
performance issues in real workloads.

Prerequisites
-------------

Before you begin, ensure the following are available on your system:

- ROCm is installed (ROCm 6.0 or later). See the
  `ROCm installation guide <https://rocm.docs.amd.com/en/latest/deploy/linux/index.html>`_.
- RCCL is installed. See :doc:`Install RCCL <../install/installation>`.
- A C++20-capable compiler (``hipcc`` from ROCm, which wraps ``amdclang++``).
- At least two AMD Instinct GPUs on the same node for the single-process
  examples, or multiple nodes each with at least one GPU for the multi-process
  example.
- For multi-process: MPI is installed (Open MPI or MPICH) and passwordless SSH
  is configured between nodes.

You can verify your GPU setup with:

.. code-block:: bash

   rocminfo | grep "Device Type"
   amd-smi topology

Overview of the approach
------------------------

RCCL follows a communicator-centric model inherited from MPI. Before any
collective can run, every participating GPU must be enrolled in a
**communicator** — a handle (``ncclComm_t``) that encodes the group
membership, the rank of this GPU within the group, and the communication
topology. Once the communicator is initialized, calling a collective is a
single function call per rank.

The program you build performs this sequence for AllReduce:

1. Initialize the ROCm/HIP runtime and choose a GPU for each rank.
2. Create communicators so all ranks can find each other.
3. Allocate GPU memory for send and receive buffers.
4. Warm up the collective to prime hardware caches and firmware state.
5. Time a batch of AllReduce iterations and report bandwidth.
6. Validate that the result is numerically correct.
7. Release all resources.

For an AllReduce summing ``N`` floats where every rank contributes ``1.0f``,
the correct output at every index is ``numRanks × 1.0f = numRanks``. This
makes correctness checking trivial without requiring a reference computation.

Step 1: Set up the header and helper macros
-------------------------------------------

Create a header file ``HelloRccl.hpp`` that defines error-checking wrappers
for HIP and RCCL calls. Wrapping every API call is not optional for production
code — silent failures are the most common source of hard-to-diagnose hangs
and incorrect results.

.. code-block:: cpp

   // HelloRccl.hpp
   #pragma once
   #include <cstdio>
   #include <cstdlib>
   #include <hip/hip_runtime.h>
   #include <rccl/rccl.h>

   #define HIP_CALL(cmd)                                                   \
     do {                                                                  \
       hipError_t e = (cmd);                                               \
       if (e != hipSuccess) {                                              \
         fprintf(stderr, "HIP error %s:%d '%s'\n",                        \
                 __FILE__, __LINE__, hipGetErrorString(e));                \
         exit(EXIT_FAILURE);                                               \
       }                                                                   \
     } while (0)

   #define NCCL_CALL(cmd)                                                  \
     do {                                                                  \
       ncclResult_t e = (cmd);                                             \
       if (e != ncclSuccess) {                                             \
         fprintf(stderr, "RCCL error %s:%d '%s'\n",                       \
                 __FILE__, __LINE__, ncclGetErrorString(e));               \
         exit(EXIT_FAILURE);                                               \
       }                                                                   \
     } while (0)

``HIP_CALL`` checks every HIP runtime call and prints the file, line, and
error string before exiting. ``NCCL_CALL`` does the same for RCCL. This
pattern ensures you never silently ignore a failed allocation or initialization.

Step 2: Initialize communicators
---------------------------------

Communicator initialization is the step that connects all ranks into a group.
RCCL provides two approaches depending on how many processes are involved.

**Single-process mode** — all GPUs in one process. Use ``ncclCommInitAll``,
which handles the bootstrapping internally:

.. code-block:: cpp

   int nranks = 4;  // number of GPUs
   std::vector<ncclComm_t> comm(nranks);

   // NULL uses GPU indices 0, 1, 2, ... nranks-1
   NCCL_CALL(ncclCommInitAll(comm.data(), nranks, NULL));

Each element of ``comm`` is an independent ``ncclComm_t`` for one GPU. After
this call, every GPU in the vector is enrolled in the same communicator group
and ready for collectives.

**Multi-process mode** — one GPU per process, one process per rank. This
requires a rendezvous mechanism so ranks can find each other. RCCL uses a
``ncclUniqueId`` — an opaque blob that encodes the root rank's network address:

.. code-block:: cpp

   ncclUniqueId commId;

   // Only rank 0 generates the ID; all other ranks receive it
   // via your application's bootstrap (environment variable, MPI broadcast, etc.)
   if (rank == 0) NCCL_CALL(ncclGetUniqueId(&commId));

   // In this example, NCCL_COMM_ID in the environment provides
   // the address of rank 0 — RCCL reads it automatically
   ncclComm_t comm;
   HIP_CALL(hipSetDevice(rank));
   NCCL_CALL(ncclCommInitRank(&comm, nranks, commId, rank));

``ncclCommInitRank`` blocks until all ``nranks`` processes have called it,
making it a collective initialization barrier. After it returns, every rank
holds a valid communicator and knows the topology of the entire group.

.. note::

   ``ncclGetUniqueId`` must be called exactly once and the result shared with
   all ranks before any rank calls ``ncclCommInitRank``. The most common way to
   distribute it is via ``MPI_Bcast``. In this tutorial, the ``NCCL_COMM_ID``
   environment variable is used instead, which RCCL's bootstrap reads directly.

For a deeper explanation of communicators and the rank model, see
:doc:`Collective operations in RCCL <../conceptual/collective-operations>`.

Step 3: Allocate GPU memory and initialize data
------------------------------------------------

Each rank allocates its own send and receive buffers on its GPU. RCCL requires
that buffers be allocated in GPU memory — it does not accept host (CPU) memory
pointers for collective operations.

.. code-block:: cpp

   int N = 1 << 24;  // 16M floats = 64 MB

   float *sendBuf, *recvBuf;
   HIP_CALL(hipSetDevice(rank));
   HIP_CALL(hipMalloc((void **)&sendBuf, N * sizeof(float)));
   HIP_CALL(hipMalloc((void **)&recvBuf, N * sizeof(float)));

   // Initialize send buffer on CPU, then copy to GPU
   std::vector<float> hostBuf(N, 1.0f);
   HIP_CALL(hipMemcpy(sendBuf, hostBuf.data(),
                      N * sizeof(float), hipMemcpyHostToDevice));

Every rank contributes ``1.0f`` at every index. After a sum AllReduce across
``nranks`` GPUs, every element of ``recvBuf`` should equal ``nranks``.

Step 4: Create a HIP stream
----------------------------

RCCL collectives are always launched onto a HIP stream. The stream controls
ordering: collectives on the same stream execute in submission order; collectives
on different streams can overlap. For this tutorial, one stream per rank is
sufficient.

.. code-block:: cpp

   hipStream_t stream;
   HIP_CALL(hipStreamCreate(&stream));

In production, you would typically pass the same stream that your compute
kernels use, so that gradient synchronization happens in the correct sequence
relative to the backward pass.

Step 5: Run warmup iterations
------------------------------

GPU hardware, firmware, and the RCCL runtime all have initialization costs
that only appear in the first few calls — JIT compilation of kernels, cache
warming, connection establishment between ranks. Timing any of these first
calls would give a misleading result. Run several untimed warmup iterations
first.

.. code-block:: cpp

   int numWarmups = 3;

   for (int i = 0; i < numWarmups; i++) {
     NCCL_CALL(ncclAllReduce(sendBuf, recvBuf, N,
                             ncclFloat, ncclSum, comm, stream));
   }
   HIP_CALL(hipStreamSynchronize(stream));

``hipStreamSynchronize`` blocks the CPU until all GPU work on ``stream``
has completed. This ensures the warmup is fully finished before timing starts.

Step 6: Time the AllReduce
---------------------------

Use HIP events to measure GPU-side execution time. HIP events are recorded
directly on the GPU timeline and give a more accurate picture of actual
communication time than CPU-side wall clocks, which include host overhead
and scheduling jitter.

.. code-block:: cpp

   int numIterations = 10;

   hipEvent_t startEvent, stopEvent;
   HIP_CALL(hipEventCreate(&startEvent));
   HIP_CALL(hipEventCreate(&stopEvent));

   // Record start on the stream — fires when GPU reaches this point
   HIP_CALL(hipEventRecord(startEvent, stream));

   for (int i = 0; i < numIterations; i++) {
     NCCL_CALL(ncclAllReduce(sendBuf, recvBuf, N,
                             ncclFloat, ncclSum, comm, stream));
   }

   HIP_CALL(hipEventRecord(stopEvent, stream));
   HIP_CALL(hipStreamSynchronize(stream));

   float gpuTimeMs;
   HIP_CALL(hipEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));

   double avgGpuTimeMs = gpuTimeMs / numIterations;
   double bytes = (double)N * sizeof(float);
   double algBwGBs = bytes / (avgGpuTimeMs * 1e-3) / 1e9;

   if (rank == 0)
     printf("N=%d  avgTime=%.3f ms  algBw=%.2f GB/s\n",
            N, avgGpuTimeMs, algBwGBs);

The ``algBw`` (algorithm bandwidth) here divides the input data size by the
elapsed time. This is the application-visible bandwidth — the number that
matters for estimating how much your workload is bottlenecked on communication.
For how this relates to bus bandwidth and hardware limits, see
:doc:`Run RCCL-Tests <../how-to/running-rccl-tests>`.

Step 7: Use group calls for multi-rank single-process
------------------------------------------------------

When running multiple ranks within a single process (single-process mode),
you must wrap per-rank collective calls in ``ncclGroupStart`` /
``ncclGroupEnd``. Without the group call, RCCL would try to complete the
AllReduce for rank 0 before launching rank 1's call — a deadlock, because
AllReduce requires all ranks to participate before any can finish.

.. code-block:: cpp

   // Single-process mode: numIntraRank ranks, each with its own comm
   NCCL_CALL(ncclGroupStart());
   for (int r = 0; r < numIntraRank; r++) {
     HIP_CALL(hipSetDevice(intraRankStartId + r));
     NCCL_CALL(ncclAllReduce(sendBuf[r], recvBuf[r], N,
                             ncclFloat, ncclSum, comm[r], stream[r]));
   }
   NCCL_CALL(ncclGroupEnd());

``ncclGroupStart`` tells RCCL to batch all subsequent collective calls into a
single fused operation. ``ncclGroupEnd`` finalizes the batch and launches it.
The group call ensures all ranks' contributions are submitted together, avoiding
the deadlock.

In multi-process mode (one rank per process), each process calls
``ncclAllReduce`` directly without a group call, because each process
represents exactly one rank and there is no risk of deadlock.

Step 8: Validate correctness
------------------------------

Always validate that your collective produced the correct answer, at least
during development. For a sum AllReduce where every rank contributes
``1.0f``, every output element must equal ``numRanks``:

.. code-block:: cpp

   std::vector<float> result(N);
   HIP_CALL(hipMemcpy(result.data(), recvBuf,
                      N * sizeof(float), hipMemcpyDeviceToHost));

   float expected = static_cast<float>(numRanks);
   bool correct = true;
   for (int i = 0; i < N; i++) {
     if (result[i] != expected) {
       correct = false;
       fprintf(stderr, "[ERROR] rank %d: result[%d] = %f, expected %f\n",
               rank, i, result[i], expected);
       break;
     }
   }
   if (correct && rank == 0)
     printf("Correctness check passed.\n");

Step 9: Release resources
--------------------------

Release GPU memory, HIP objects, and the communicator in the correct order.
Destroying the communicator before synchronizing the stream can produce
undefined behavior.

.. code-block:: cpp

   HIP_CALL(hipStreamSynchronize(stream));
   HIP_CALL(hipFree(sendBuf));
   HIP_CALL(hipFree(recvBuf));
   HIP_CALL(hipStreamDestroy(stream));
   HIP_CALL(hipEventDestroy(startEvent));
   HIP_CALL(hipEventDestroy(stopEvent));
   NCCL_CALL(ncclCommDestroy(comm));

Complete example
-----------------

The following is the complete program combining all the steps above. It
supports both single-process and multi-process modes from the same binary.

**HelloRccl.hpp**

.. code-block:: cpp

   #pragma once
   #include <cstdio>
   #include <cstdlib>
   #include <hip/hip_runtime.h>
   #include <rccl/rccl.h>

   #define HIP_CALL(cmd)                                                   \
     do {                                                                  \
       hipError_t e = (cmd);                                               \
       if (e != hipSuccess) {                                              \
         fprintf(stderr, "HIP error %s:%d '%s'\n",                        \
                 __FILE__, __LINE__, hipGetErrorString(e));                \
         exit(EXIT_FAILURE);                                               \
       }                                                                   \
     } while (0)

   #define NCCL_CALL(cmd)                                                  \
     do {                                                                  \
       ncclResult_t e = (cmd);                                             \
       if (e != ncclSuccess) {                                             \
         fprintf(stderr, "RCCL error %s:%d '%s'\n",                       \
                 __FILE__, __LINE__, ncclGetErrorString(e));               \
         exit(EXIT_FAILURE);                                               \
       }                                                                   \
     } while (0)

**HelloRccl.cpp**

.. code-block:: cpp

   #include <sys/socket.h>
   #include <ifaddrs.h>
   #include <netdb.h>
   #include <unistd.h>
   #include <cstdio>
   #include <string>
   #include <chrono>
   #include <vector>
   #include <hip/hip_runtime.h>
   #include <rccl/rccl.h>
   #include "HelloRccl.hpp"

   void Usage(char *argv0);
   void ExecuteTest(int numIntraRank, int intraRankStartId,
                    int numTotalRanks, ncclComm_t *comm);

   int main(int argc, char **argv)
   {
     if (getenv("NCCL_COMM_ID") && argc == 3)
     {
       // Multi-process mode: one rank per process
       int nranks = atoi(argv[1]);
       int rank   = atoi(argv[2]);
       if (rank == 0) printf("Running in multi-process mode\n");

       ncclUniqueId commId;
       NCCL_CALL(ncclGetUniqueId(&commId));

       ncclComm_t comm;
       HIP_CALL(hipSetDevice(rank));
       NCCL_CALL(ncclCommInitRank(&comm, nranks, commId, rank));

       ExecuteTest(1, rank, nranks, &comm);
     }
     else if (argc == 2)
     {
       // Single-process mode: all ranks in one process
       printf("Running in single-process mode\n");

       int nranks = atoi(argv[1]);
       std::vector<ncclComm_t> comm(nranks);
       NCCL_CALL(ncclCommInitAll(comm.data(), nranks, NULL));

       ExecuteTest(nranks, 0, nranks, comm.data());
     }
     else
     {
       Usage(argv[0]);
       return 1;
     }
     return 0;
   }

   void ExecuteTest(int numIntraRank, int intraRankStartId,
                    int numTotalRanks, ncclComm_t *comm)
   {
     int minPow        = 10;   // 2^10 floats  = 4 KB
     int maxPow        = 28;   // 2^28 floats  = 1 GB
     int numWarmups    =  3;
     int numIterations = 10;

     std::vector<hipStream_t> stream(numIntraRank);
     std::vector<hipEvent_t>  startEvent(numIntraRank);
     std::vector<hipEvent_t>  stopEvent(numIntraRank);
     for (int i = 0; i < numIntraRank; i++) {
       HIP_CALL(hipSetDevice(intraRankStartId + i));
       HIP_CALL(hipStreamCreate(&stream[i]));
       HIP_CALL(hipEventCreate(&startEvent[i]));
       HIP_CALL(hipEventCreate(&stopEvent[i]));
     }

     if (intraRankStartId == 0)
       printf("AllReduce Performance (sum of floats):\n"
              "%10s %10s %10s\n", "Bytes", "CpuTime(ms)", "GpuTime(ms)");

     for (int power = minPow; power <= maxPow; power++) {
       int N = 1 << power;

       // Allocate GPU memory
       std::vector<float *> sendBuf(numIntraRank);
       std::vector<float *> recvBuf(numIntraRank);
       for (int r = 0; r < numIntraRank; r++) {
         HIP_CALL(hipSetDevice(intraRankStartId + r));
         HIP_CALL(hipMalloc((void **)&sendBuf[r], N * sizeof(float)));
         HIP_CALL(hipMalloc((void **)&recvBuf[r], N * sizeof(float)));
       }

       // Initialize host buffer and copy to each GPU
       std::vector<float> hostBuf(N, 1.0f);
       for (int r = 0; r < numIntraRank; r++) {
         HIP_CALL(hipSetDevice(intraRankStartId + r));
         HIP_CALL(hipMemcpy(sendBuf[r], hostBuf.data(),
                            N * sizeof(float), hipMemcpyHostToDevice));
       }

       // Warmup
       for (int iter = 0; iter < numWarmups; iter++) {
         NCCL_CALL(ncclGroupStart());
         for (int r = 0; r < numIntraRank; r++) {
           HIP_CALL(hipSetDevice(intraRankStartId + r));
           NCCL_CALL(ncclAllReduce(sendBuf[r], recvBuf[r], N,
                                   ncclFloat, ncclSum, comm[r], stream[r]));
         }
         NCCL_CALL(ncclGroupEnd());
       }
       for (int r = 0; r < numIntraRank; r++)
         HIP_CALL(hipStreamSynchronize(stream[r]));

       // Timed iterations
       auto cpuStart = std::chrono::high_resolution_clock::now();
       for (int r = 0; r < numIntraRank; r++)
         HIP_CALL(hipEventRecord(startEvent[r], stream[r]));

       for (int iter = 0; iter < numIterations; iter++) {
         NCCL_CALL(ncclGroupStart());
         for (int r = 0; r < numIntraRank; r++) {
           HIP_CALL(hipSetDevice(intraRankStartId + r));
           NCCL_CALL(ncclAllReduce(sendBuf[r], recvBuf[r], N,
                                   ncclFloat, ncclSum, comm[r], stream[r]));
         }
         NCCL_CALL(ncclGroupEnd());
       }

       for (int r = 0; r < numIntraRank; r++)
         HIP_CALL(hipEventRecord(stopEvent[r], stream[r]));
       for (int r = 0; r < numIntraRank; r++)
         HIP_CALL(hipStreamSynchronize(stream[r]));

       auto cpuDelta = std::chrono::high_resolution_clock::now() - cpuStart;
       double totalCpuMs =
         std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(cpuDelta).count();

       float totalGpuMs;
       HIP_CALL(hipEventElapsedTime(&totalGpuMs, startEvent[0], stopEvent[0]));

       if (intraRankStartId == 0)
         printf("%10lu %10.3f %10.3f\n",
                (unsigned long)(N * sizeof(float)),
                totalCpuMs / numIterations,
                totalGpuMs / numIterations);

       // Correctness validation
       std::vector<float> result(N);
       float expected = static_cast<float>(numTotalRanks);
       for (int r = 0; r < numIntraRank; r++) {
         HIP_CALL(hipMemcpy(result.data(), recvBuf[r],
                            N * sizeof(float), hipMemcpyDeviceToHost));
         for (int i = 0; i < N; i++) {
           if (result[i] != expected) {
             fprintf(stderr, "[ERROR] Rank %d: result[%d] = %f, expected %f\n",
                     intraRankStartId + r, i, result[i], expected);
             NCCL_CALL(ncclCommDestroy(comm[r]));
             exit(1);
           }
         }
       }

       for (int r = 0; r < numIntraRank; r++) {
         HIP_CALL(hipFree(sendBuf[r]));
         HIP_CALL(hipFree(recvBuf[r]));
       }
     }

     for (int r = 0; r < numIntraRank; r++) {
       HIP_CALL(hipStreamDestroy(stream[r]));
       HIP_CALL(hipEventDestroy(startEvent[r]));
       HIP_CALL(hipEventDestroy(stopEvent[r]));
       NCCL_CALL(ncclCommDestroy(comm[r]));
     }
   }

   void Usage(char *argv0)
   {
     printf("Single-process: %s <numRanks>\n", argv0);
     printf("Multi-process:  NCCL_COMM_ID=<host:port> %s <numRanks> <rank>\n\n",
            argv0);
     printf("To run as the root process, set NCCL_COMM_ID to one of:\n");

     char hostname[256];
     gethostname(hostname, 256);
     printf("  export NCCL_COMM_ID=%s:12345\n", hostname);

     struct ifaddrs *ifaddr;
     getifaddrs(&ifaddr);
     for (struct ifaddrs *ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
       int family = ifa->ifa_addr->sa_family;
       if (family != AF_INET && family != AF_INET6) continue;
       if (family == AF_INET6) {
         struct sockaddr_in6 *sa = (struct sockaddr_in6 *)(ifa->ifa_addr);
         if (IN6_IS_ADDR_LOOPBACK(&sa->sin6_addr)) continue;
       }
       socklen_t saLen = (family == AF_INET
                          ? sizeof(struct sockaddr_in)
                          : sizeof(struct sockaddr_in6));
       char host[NI_MAXHOST], service[NI_MAXSERV];
       getnameinfo(ifa->ifa_addr, saLen, host, NI_MAXHOST,
                   service, NI_MAXSERV, NI_NUMERICHOST | NI_NUMERICSERV);
       printf("  export NCCL_COMM_ID=%s:12345\n", host);
     }
     freeifaddrs(ifaddr);
   }

Build the program
------------------

Create a ``Makefile`` in the same directory as the source files:

.. code-block:: makefile

   ROCM_DIR ?= /opt/rocm
   RCCL_DIR ?= $(ROCM_DIR)

   CXXFLAGS = -std=c++20 -O2
   INCLUDES = -I$(RCCL_DIR)/include
   LDFLAGS  = -L$(RCCL_DIR)/lib -lrccl

   HelloRccl: HelloRccl.cpp HelloRccl.hpp
       $(ROCM_DIR)/bin/hipcc $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

   clean:
       rm -f HelloRccl

Build with:

.. code-block:: bash

   make

If RCCL is installed somewhere other than ``/opt/rocm``, pass the path:

.. code-block:: bash

   make RCCL_DIR=/path/to/rccl/build/release

Run in single-process mode
---------------------------

Single-process mode runs all ranks inside one process, one per GPU. Pass the
number of GPUs as the only argument:

.. code-block:: bash

   ./HelloRccl 8

Expected output (values will vary by hardware):

.. code-block:: text

   Running in single-process mode
   AllReduce Performance (sum of floats):
        Bytes CpuTime(ms) GpuTime(ms)
         4096       0.041       0.038
         8192       0.043       0.040
        ...
    268435456      32.451      31.887
   1073741824     129.203     128.614

Each row is the average of 10 iterations for that message size. The GPU time
is measured with HIP events on the device timeline. The CPU time includes
host-side launch overhead and is typically slightly higher.

Run in multi-process mode
--------------------------

Multi-process mode runs one process per GPU, which is the pattern used by
PyTorch, JAX, and MPI-based HPC applications. It requires ``NCCL_COMM_ID``
to be set so all ranks can find rank 0.

**Step 1 — Find a valid address for rank 0.** Run the binary with no arguments
to print available addresses:

.. code-block:: bash

   ./HelloRccl
   # Output includes lines like:
   #   export NCCL_COMM_ID=192.168.1.10:12345

**Step 2 — Set ``NCCL_COMM_ID`` and launch all ranks.** On a single node with
8 GPUs, launch 8 processes in the background:

.. code-block:: bash

   export NCCL_COMM_ID=192.168.1.10:12345
   for rank in 0 1 2 3 4 5 6 7; do
     ./HelloRccl 8 $rank &
   done
   wait

**Step 3 — Multi-node with MPI.** For a two-node run with 8 GPUs per node
(16 ranks total):

.. code-block:: bash

   export NCCL_COMM_ID=node01:12345
   mpirun -np 16 --hostfile hostfile.txt \
     --bind-to numa \
     -x NCCL_COMM_ID \
     ./HelloRccl 16 $OMPI_COMM_WORLD_RANK

.. note::

   In the multi-process example, ``ncclGetUniqueId`` is called by every rank
   because ``NCCL_COMM_ID`` is set — RCCL's bootstrap uses the environment
   variable to distribute the unique ID automatically. In production MPI
   code, you would typically call ``ncclGetUniqueId`` only on rank 0 and
   distribute the result via ``MPI_Bcast`` before calling
   ``ncclCommInitRank``.

Interpret the output
---------------------

The program prints one row per message size:

.. code-block:: text

   Bytes        — total bytes in the send buffer (N × 4 for float)
   CpuTime(ms)  — average wall-clock time per iteration (includes host overhead)
   GpuTime(ms)  — average GPU event time per iteration (device execution only)

To convert GPU time to algorithm bandwidth:

.. code-block:: text

   algBw (GB/s) = Bytes ÷ (GpuTime_ms × 10⁻³) ÷ 10⁹

For a deeper explanation of algorithm bandwidth versus bus bandwidth — and how
to compare your results against hardware limits — see
:doc:`Run RCCL-Tests <../how-to/running-rccl-tests>` and
:doc:`Collective algorithms in RCCL <../conceptual/collective-algorithms>`.

Summary
--------

You have built a complete RCCL program that:

- Initializes communicators in both single-process and multi-process modes.
- Allocates GPU memory and initializes data on device.
- Runs warmup and timed AllReduce iterations using ``ncclGroupStart`` /
  ``ncclGroupEnd`` to correctly batch multi-rank calls.
- Measures performance with HIP events for accurate GPU-side timing.
- Validates correctness by checking every output element.
- Releases all resources in the correct order.

The core pattern — initialize communicator, allocate buffers, group-start,
collective, group-end, synchronize, validate, destroy — is the same pattern
used by every distributed framework built on RCCL.

**Next steps:**

- :doc:`Collective operations in RCCL <../conceptual/collective-operations>` —
  understand AllGather, ReduceScatter, AllToAll, and point-to-point Send/Recv
- :doc:`Collective algorithms in RCCL <../conceptual/collective-algorithms>` —
  learn how RCCL chooses between Ring, Tree, Hierarchical, and other algorithms
- :doc:`Collective protocols in RCCL <../conceptual/collective-protocols>` —
  understand Simple, LL, and LL128 and when each is selected
- :doc:`Run RCCL-Tests <../how-to/running-rccl-tests>` — benchmark every
  collective across a full message-size sweep
- :doc:`Hardware-specific optimizations <../conceptual/hardware-specific-optimizations>` —
  understand MI300X ring topology, tuning models, and protocol availability
- :ref:`API reference <api-library>` — full C API documentation
- :ref:`Environment variables <env-variables>` — runtime knobs for debugging
  and tuning
