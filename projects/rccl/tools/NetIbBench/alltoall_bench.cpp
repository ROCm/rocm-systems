/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// RCCL Net Plugin All-to-All Benchmark
//
// Loads an RCCL net plugin via net_plugin_loader, creates per-direction
// send/recv communicators between all N MPI ranks, and runs an
// all-to-all exchange loop: irecv from all peers, isend to all peers,
// wait for all completions — with no synchronization between iterations.
//
// Each rank reports its own elapsed time and average time per call.
//
// Usage:
//   mpirun -np <N> ./alltoall_bench -p <plugin.so> [options]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdarg>
#include <cerrno>
#include <chrono>
#include <unistd.h>
#include <dlfcn.h>
#include <vector>
#include <mpi.h>

#include "nccl_common.h"
#include "net_device.h"
#include "nccl_net.h"
#include "net_plugin_loader.h"


// ─── Command line arguments ──────────────────────────────────────────────────
const char* pluginPath = "NetIb";
int numIters    = 1000;
size_t msgSize  = 64;
int devIdx      = -1; // auto-bound: local_rank % ndev
int batchSize   = 1;
int outerLoops  = 1;
bool postOnly   = false; // measure isend/irecv posting time only
int warmup      = 100;

void usage(const char* prog) {
    fprintf(stderr,
        "Usage: mpirun -np <N> %s -p <plugin.so> [options]\n"
        "\n"
        "  -p <path>   Path to RCCL net plugin shared library (required)\n"
        "  -m <int>    Number of all-to-all iterations        (default: 1000)\n"
        "  -s <int>    Message size in bytes per direction     (default: 64)\n"
        "  -d <int>    Network device index                    (default: auto = local_rank %% ndev)\n"
        "  -b <int>    Batch size (alltoall calls per wait)     (default: 1)\n"
        "  -N <int>    Number of outer benchmark loops         (default: 1)\n"
        "  -i          Measure isend/irecv posting time only    (default: off)\n"
        "  -w <int>    Warmup iterations                       (default: 100)\n"
        "  -h          Show this help\n",
        prog);
}

void parseArgs(int argc, char** argv) {
    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "-p") == 0 && i + 1 < argc) { pluginPath = argv[++i]; }
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) { numIters   = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) { msgSize    = (size_t)atol(argv[++i]); }
        else if (strcmp(argv[i], "-d") == 0 && i + 1 < argc) { devIdx     = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-b") == 0 && i + 1 < argc) { batchSize  = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-N") == 0 && i + 1 < argc) { outerLoops = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-i") == 0)                 { postOnly   = true; }
        else if (strcmp(argv[i], "-w") == 0 && i + 1 < argc) { warmup     = atoi(argv[++i]); }
        else if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); MPI_Finalize(); exit(0); }
    }
}

#include "benchmark_utils.h"
// ─── All-to-all iteration ────────────────────────────────────────────────────
double runAlltoall(ncclNet_t* net, std::vector<PeerConn>& peers,
                          int iters, bool timePostOnly) {
    std::vector<void*> requests(2 * numRemotePeers * batchSize);
    double postTime_us = 0;
    int tag = 42;
    const int maxRetries = 100000;

    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iters; iter++) {
        int nreq = 0;

        auto tPost0 = std::chrono::high_resolution_clock::now();

        for (int b = 0; b < batchSize; b++) {
            for (int p = 0; p < nranks; p++) {
                if (!isRemote[p]) continue;
                void* req = nullptr;
                CHECK_NCCL(net->irecv(peers[p].recvComms[0], 1,
                                      &peers[p].recvBuf, &msgSize, &tag,
                                      &peers[p].recvMh, nullptr, &req));
                requests[nreq++] = req;
            }

            for (int p = 0; p < nranks; p++) {
                if (!isRemote[p]) continue;
                void* req = nullptr;
                int attempts = 0;
                do {
                    CHECK_NCCL(net->isend(peers[p].sendComms[0], peers[p].sendBuf, msgSize,
                                          tag, peers[p].sendMh, nullptr, &req));
                    if (req) break;
                    if (++attempts >= maxRetries) {
                        fprintf(stderr, "[rank %d] isend to peer %d stuck at iter %d+%d\n",
                                rank, p, iter, b);
                        MPI_Abort(MPI_COMM_WORLD, 1);
                    }
                    usleep(10);
                } while (!req);
                requests[nreq++] = req;
            }
        }

        auto tPost1 = std::chrono::high_resolution_clock::now();
        postTime_us += std::chrono::duration<double, std::micro>(tPost1 - tPost0).count();

        int remaining = nreq;
        while (remaining > 0) {
            for (int r = 0; r < nreq; r++) {
                if (!requests[r]) continue;
                int done = 0, sz = 0;
                CHECK_NCCL(net->test(requests[r], &done, &sz));
                if (done) {
                    requests[r] = nullptr;
                    remaining--;
                }
            }
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (timePostOnly)
        return postTime_us;
    return std::chrono::duration<double, std::micro>(t1 - t0).count();
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    int ndev;
    MPI_Init(&argc, &argv);
    detectNodeTopology();
    parseArgs(argc, argv);
    if (nranks < 2) {
        if (rank == 0)
            fprintf(stderr, "Error: at least 2 MPI ranks required (got %d)\n", nranks);
        MPI_Finalize();
        return 1;
    }
    if (numRemotePeers == 0) {
        if (rank == 0)
            fprintf(stderr, "Error: all ranks are on the same node, nothing to benchmark\n");
        MPI_Finalize();
        return 1;
    }

    // ── Load plugin ──────────────────────────────────────────────────────
    NetPluginHandle pluginHandle;
    ncclNet_t* net = netPluginInit(&pluginHandle, pluginPath);
    if (!net) {
        fprintf(stderr, "[rank %d] Failed to load plugin: %s\n", rank, pluginPath);
        MPI_Abort(MPI_COMM_WORLD, 1);
    } else {
        if (rank == 0) {
            printf("Loaded plugin: %s (v%d)\n", net->name, pluginHandle.netVersion);
        }
    }

    // ── Init plugin ──────────────────────────────────────────────────────
    void* netCtx = nullptr;
    ncclNetCommConfig_t commConfig{};
    commConfig.trafficClass = NCCL_NET_TRAFFIC_CLASS_UNDEF;
    CHECK_NCCL(net->init(&netCtx, 0, &commConfig, pluginLogger, nullptr));

    bindToDevice(net, ndev);

    if (rank == 0) {
        printf("\nConfiguration:\n");
        printf("  Ranks          : %d\n", nranks);
        printf("  NICs           : %d  (auto-bound: local_rank %% ndev)\n", ndev);
        printf("  Ranks/node     : %d\n", nodeSize);
        printf("  Iterations     : %d\n", numIters);
        printf("  Batch size     : %d  (alltoall calls per wait)\n", batchSize);
        printf("  Message size   : %zu bytes  (per direction)\n", msgSize);
        printf("  Warmup iters   : %d\n", warmup);
        printf("  Local ranks    : %d  (skipped)\n", nodeSize - 1);
        printf("  Remote peers   : %d  (per rank)\n", numRemotePeers);
        printf("\n");
    }

    // ── Setup per-peer connections (remote only) ─────────────────────────
    std::vector<PeerConn> peers(nranks);
    establishConnections(net, netCtx, peers);
    allocateBuffers(net, peers, msgSize);
    MPI_Barrier(MPI_COMM_WORLD);

    // ── Warmup ───────────────────────────────────────────────────────────
    if (rank == 0)
        printf("Running warmup (%d iterations)...\n", warmup);

    runAlltoall(net, peers, warmup, false);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Warmup complete.\n");
        if (postOnly)
            printf("Mode: measuring isend/irecv posting time only (-i)\n");
        printf("\n");
    }

    // ── Timed runs ───────────────────────────────────────────────────────
    for (int loop = 0; loop < outerLoops; loop++) {
        if (outerLoops > 1 && rank == 0)
            printf("=== Outer loop %d / %d ===\n", loop + 1, outerLoops);

        double my_elapsed_us = runAlltoall(net, peers, numIters, postOnly);
        MPI_Barrier(MPI_COMM_WORLD);

        // ── Report ───────────────────────────────────────────────────────
        long long totalCalls = (long long)numIters * batchSize;
        double avg_per_call_us = my_elapsed_us / totalCalls;
        long long totalMsgs = totalCalls * 2 * numRemotePeers;
        double myMsgRate = (double)totalMsgs / (my_elapsed_us / 1e6);

        std::vector<double> allElapsed(nranks);
        MPI_Gather(&my_elapsed_us, 1, MPI_DOUBLE, allElapsed.data(), 1, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);

        for (int r = 0; r < nranks; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank) {
                printf("  [rank %2d]  total: %10.3f ms  avg/call: %8.2f us  "
                       "msgs: %lld  msg_rate: %10.0f msg/s\n",
                       rank, my_elapsed_us / 1000.0, avg_per_call_us,
                       totalMsgs, myMsgRate);
                fflush(stdout);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0) {
            printf("\n--- Summary ---\n");
            double minTime = allElapsed[0], maxTime = allElapsed[0], sumTime = 0;
            for (int r = 0; r < nranks; r++) {
                if (allElapsed[r] < minTime) minTime = allElapsed[r];
                if (allElapsed[r] > maxTime) maxTime = allElapsed[r];
                sumTime += allElapsed[r];
            }
            double avgTime = sumTime / nranks;
            printf("  Iterations       : %d\n", numIters);
            printf("  Batch size       : %d\n", batchSize);
            printf("  Total a2a calls  : %lld  (%d iters x %d batch)\n",
                   (long long)numIters * batchSize, numIters, batchSize);
            printf("  Message size     : %zu bytes\n", msgSize);
            printf("  Ranks            : %d\n", nranks);
            printf("  Remote peers/rank: %d\n", numRemotePeers);
            printf("  Msgs/rank/iter   : %d  (%d send + %d recv)\n",
                   2 * numRemotePeers, numRemotePeers, numRemotePeers);
            printf("  Min rank time    : %.3f ms  (%.2f us/call)\n",
                   minTime / 1000.0, minTime / totalCalls);
            printf("  Max rank time    : %.3f ms  (%.2f us/call)\n",
                   maxTime / 1000.0, maxTime / totalCalls);
            printf("  Avg rank time    : %.3f ms  (%.2f us/call)\n",
                   avgTime / 1000.0, avgTime / totalCalls);
            long long totalAllMsgs = totalCalls * 2 * numRemotePeers * nranks;
            double aggRate = (double)totalAllMsgs / (maxTime / 1e6);
            printf("  Agg msg rate     : %.0f msg/s  (based on slowest rank)\n", aggRate);
            printf("  Agg throughput   : %.3f MB/s  (%.3f Gbps)\n",
                   aggRate * msgSize / 1e6, aggRate * msgSize * 8.0 / 1e9);
            printf("\n");
        }
    }

    // ── Cleanup ──────────────────────────────────────────────────────────
    freeBuffers(net, peers);
    closeConnections(net, peers);
    net->finalize(netCtx);
    netPluginFinalize(&pluginHandle);
    MPI_Finalize();
    return 0;
}
