/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// A connection that is not an NCCL peer (port scanner, a peer with the wrong magic,
// a plaintext peer reaching an encrypted root) must be dropped by the bootstrap root
// without ending its accept loop, or every rank's communicator creation fails.

#ifdef MPI_TESTS_ENABLED

#include "MPITestBase.hpp"
#include "MPIHelpers.hpp"
#include "TestChecks.hpp"
#include "common/EncryptionTestHelpers.hpp"

#include "bootstrap.h"

#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

using RcclUnitTesting::ClearPsk;
using RcclUnitTesting::kTlsBuild;
using RcclUnitTesting::SetPsk;

// Connects to the root listener named by `id` and sends bytes that are neither the
// socket hello nor a TLS ClientHello. Returns the descriptor (kept open) or -1.
int ConnectStrayPeer(const ncclUniqueId& id) {
  struct ncclBootstrapHandle handle;
  memcpy(&handle, &id, sizeof(handle));
  int fd = socket(handle.addr.sa.sa_family, SOCK_STREAM, 0);
  if (fd < 0) return -1;
  socklen_t len = handle.addr.sa.sa_family == AF_INET ? sizeof(struct sockaddr_in) : sizeof(struct sockaddr_in6);
  static const char probe[] = "GET / HTTP/1.0\r\n\r\n";
  if (connect(fd, &handle.addr.sa, len) != 0 || send(fd, probe, sizeof(probe) - 1, MSG_NOSIGNAL) < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

} // namespace

class BootstrapStrayPeerMPITest : public MPITestBase {
protected:
  void TearDown() override {
    MPITestBase::TearDown();
    (void)ClearPsk();
    MPI_Barrier(MPI_COMM_WORLD);
  }

  // Rank 0 queues the stray connection on the root listener before any rank checks in,
  // so the root has to reject it first; then every rank creates the communicator.
  void InitBehindStrayPeer() {
    int rank = MPIEnvironment::world_rank, nranks = MPIEnvironment::world_size;
    ncclUniqueId id = {};
    int strayFd = -1;
    int setup = 0; // 1: ncclGetUniqueId failed, 2: the root listener was unreachable
    if (rank == 0) {
      if (ncclGetUniqueId(&id) != ncclSuccess) setup = 1;
      else if ((strayFd = ConnectStrayPeer(id)) < 0) setup = 2;
    }
    MPI_Bcast(&setup, 1, MPI_INT, 0, MPI_COMM_WORLD);
    ASSERT_EQ(0, setup) << "rank 0 setup failed (1: ncclGetUniqueId, 2: could not reach the bootstrap root listener)";
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t comm = nullptr;
    EXPECT_EQ(ncclSuccess, ncclCommInitRank(&comm, nranks, id, rank));
    if (comm != nullptr) EXPECT_EQ(ncclSuccess, ncclCommDestroy(comm));
    if (strayFd >= 0) close(strayFd);
  }
};

TEST_F(BootstrapStrayPeerMPITest, RootSkipsNonNcclPeer) {
  SKIP_UNLESS_MPI_PREREQS(2);
  InitBehindStrayPeer();
}

TEST_F(BootstrapStrayPeerMPITest, EncryptedRootSkipsNonTlsPeer) {
  SKIP_UNLESS_MPI_PREREQS(2);
  if (!kTlsBuild) GTEST_SKIP() << "RCCL was built without TLS_BACKEND=OPENSSL3";
  ASSERT_EQ(ncclSuccess, SetPsk("rccl-bootstrap-stray-peer-test-key-0123456789"));
  InitBehindStrayPeer();
}

#endif // MPI_TESTS_ENABLED
