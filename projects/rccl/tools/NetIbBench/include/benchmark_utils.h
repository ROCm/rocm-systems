#pragma once
/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include <mpi.h>
#include <vector>
#include "nccl_common.h"
#include "net_device.h"
#include "nccl_net.h"

typedef char ncclNetHandle_t[NCCL_NET_HANDLE_MAXSIZE];

void pluginLogger(ncclDebugLogLevel level, unsigned long flags,
                          const char *file, int line, const char *fmt, ...) {
    if (level > NCCL_LOG_WARN) return;
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[plugin] ");
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

// ─── Node topology (set by detectNodeTopology) ───────────────────────────
int nodeRank;
int nodeSize;
std::vector<bool> isRemote;
int numRemotePeers;
int rank, nranks;

void detectNodeTopology() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &nodeComm);
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);

    int nodeLeader = (nodeRank == 0) ? rank : -1;
    MPI_Bcast(&nodeLeader, 1, MPI_INT, 0, nodeComm);

    std::vector<int> allNodeLeaders(nranks);
    MPI_Allgather(&nodeLeader, 1, MPI_INT, allNodeLeaders.data(), 1, MPI_INT,
                  MPI_COMM_WORLD);

    isRemote.resize(nranks, false);
    numRemotePeers = 0;
    for (int p = 0; p < nranks; p++) {
        if (p != rank && allNodeLeaders[p] != allNodeLeaders[rank]) {
            isRemote[p] = true;
            numRemotePeers++;
        }
    }
    MPI_Comm_free(&nodeComm);
}

// ─── Device selection & NIC info ─────────────────────────────────────────────
void bindToDevice(ncclNet_t* net, int& ndev, bool print = false) {
    ndev = 0;
    CHECK_NCCL(net->devices(&ndev));
    if (ndev == 0) {
        if (rank == 0) fprintf(stderr, "No network devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (devIdx < 0)
        devIdx = nodeRank % ndev;

    if (devIdx >= ndev) {
        fprintf(stderr, "[rank %d] Device %d out of range (found %d)\n", rank, devIdx, ndev);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    ncclNetProperties_t props{};
    CHECK_NCCL(net->getProperties(devIdx, &props));

    char nodeName[128];
    gethostname(nodeName, sizeof(nodeName));
    int pid = getpid();

    if (print) {
        for (int r = 0; r < nranks; r++) {
            MPI_Barrier(MPI_COMM_WORLD);
            if (r == rank) {
                printf("  [rank %2d @ %s:%d]  nodeRank=%d  NIC=%d/%d  name=%s\n",
                    rank, nodeName, pid,
                    nodeRank, devIdx, ndev,
                    props.name ? props.name : "?");
                fflush(stdout);
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

// ─── Per-peer state ──────────────────────────────────────────────────────────

struct PeerConn {
    std::vector<void*> sendComms;
    void* sendBuf = nullptr;
    void* sendMh  = nullptr;

    std::vector<void*> listenComms;
    std::vector<void*> recvComms;
    void* recvBuf = nullptr;
    void* recvMh  = nullptr;
};

// ─── Connection setup ────────────────────────────────────────────────────────
//
// For each remote peer, creates nConnections bidirectional connections.
// Each connection goes through a full listen → exchange → connect/accept cycle.
void establishConnections(ncclNet_t* net, void* netCtx,
                          std::vector<PeerConn>& peers, int nConnections = 1) {
    for (int p = 0; p < nranks; p++) {
        if (!isRemote[p]) continue;
        peers[p].listenComms.resize(nConnections, nullptr);
        peers[p].sendComms.resize(nConnections, nullptr);
        peers[p].recvComms.resize(nConnections, nullptr);
    }

    ncclNet_ctxt_t ncclNetCtxt = {};

    for (int c = 0; c < nConnections; c++) {
        std::vector<ncclNetHandle_t> myListenHandles(nranks);
        for (int p = 0; p < nranks; p++) {
            if (!isRemote[p]) continue;
            CHECK_NCCL(net->listen(netCtx, devIdx, &myListenHandles[p],
                                   &peers[p].listenComms[c]));
        }

        std::vector<ncclNetHandle_t> peerListenHandles(nranks);
        for (int p = 0; p < nranks; p++) {
            if (!isRemote[p]) continue;
            MPI_Sendrecv(&myListenHandles[p], sizeof(ncclNetHandle_t), MPI_BYTE,
                         p, 3000 + rank,
                         &peerListenHandles[p], sizeof(ncclNetHandle_t), MPI_BYTE,
                         p, 3000 + p,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        std::vector<bool> sendDone(nranks, true);
        std::vector<bool> recvDone(nranks, true);
        for (int p = 0; p < nranks; p++) {
            if (isRemote[p]) { sendDone[p] = false; recvDone[p] = false; }
        }

        bool allDone = false;
        while (!allDone) {
            allDone = true;
            for (int p = 0; p < nranks; p++) {
                if (!isRemote[p]) continue;
                if (!sendDone[p]) {
                    net->connect(netCtx, devIdx, &peerListenHandles[p],
                                 &peers[p].sendComms[c],
                                 (ncclNetDeviceHandle_t**)&ncclNetCtxt);
                    if (peers[p].sendComms[c]) sendDone[p] = true;
                    else allDone = false;
                }
                if (!recvDone[p]) {
                    net->accept(peers[p].listenComms[c], &peers[p].recvComms[c],
                                (ncclNetDeviceHandle_t**)&ncclNetCtxt);
                    if (peers[p].recvComms[c]) recvDone[p] = true;
                    else allDone = false;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("All connections established (%d remote peers x %d conns).\n\n",
               numRemotePeers, nConnections);
}

void closeConnections(ncclNet_t* net, std::vector<PeerConn>& peers) {
    for (int p = 0; p < nranks; p++) {
        if (!isRemote[p]) continue;
        for (auto* sc : peers[p].sendComms)   if (sc) net->closeSend(sc);
        for (auto* rc : peers[p].recvComms)   if (rc) net->closeRecv(rc);
        for (auto* lc : peers[p].listenComms) if (lc) net->closeListen(lc);
    }
}

// ─── Buffer allocation & registration ────────────────────────────────────────

void allocateBuffers(ncclNet_t* net, std::vector<PeerConn>& peers, size_t bufSize) {
    size_t sz = bufSize > 0 ? bufSize : 1;
    for (int p = 0; p < nranks; p++) {
        if (!isRemote[p]) continue;
        peers[p].sendBuf = malloc(sz);
        peers[p].recvBuf = malloc(sz);
        memset(peers[p].sendBuf, 0, sz);
        memset(peers[p].recvBuf, 0, sz);

        CHECK_NCCL(net->regMr(peers[p].sendComms[0], peers[p].sendBuf, sz,
                               NCCL_PTR_HOST, &peers[p].sendMh));
        CHECK_NCCL(net->regMr(peers[p].recvComms[0], peers[p].recvBuf, sz,
                               NCCL_PTR_HOST, &peers[p].recvMh));
    }
}

void freeBuffers(ncclNet_t* net, std::vector<PeerConn>& peers) {
    for (int p = 0; p < nranks; p++) {
        if (!isRemote[p]) continue;
        if (peers[p].sendMh) net->deregMr(peers[p].sendComms[0], peers[p].sendMh);
        if (peers[p].recvMh) net->deregMr(peers[p].recvComms[0], peers[p].recvMh);
        free(peers[p].sendBuf);
        free(peers[p].recvBuf);
    }
}
