/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

#ifndef ROCSHMEM_IPC_HOST_AMO_GTEST_HPP
#define ROCSHMEM_IPC_HOST_AMO_GTEST_HPP

/**
 * @file ipc_host_amo_gtest.hpp
 *
 * Tests for JIRA-419: host-side AMO, fence, and quiet in IPC non-MPI mode.
 *
 * The fixture builds the exact same object graph that IPCBackend(TcpBootstrap*)
 * builds at runtime, but without the full backend init overhead:
 *
 *   HeapMemoryType      — allocates fine-grain GPU heap per PE
 *   RemoteHeapInfo<CommunicatorTCP>
 *                       — exchanges heap base pointers via TcpBootstrap
 *   IpcOnImpl           — opens IPC handles, builds ipc_bases[]
 *   HostInterface(bootstrap, heap)
 *                       — non-MPI constructor → plain WindowInfo pool
 *   IPCHostContext      — acquires window context + copies ipc_bases to host
 *
 * Before the JIRA-419 fix:
 *   calling amo_fetch_add / amo_fetch_cas / fence / quiet on the context
 *   hits dynamic_cast<WindowInfoMPI*> → nullptr → abort().
 *   The tests are written so that the pre-fix behavior produces a death
 *   (EXPECT_DEATH) and the post-fix behavior produces correct results.
 *
 * Launch with 2 MPI ranks (same node), IPC backend build:
 *   mpirun -np 2 ./rocshmem_unit_tests \
 *       --gtest_filter=IPCHostAMOTestFixture/*
 */

#include "gtest/gtest.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <mpi.h>

#include "../src/bootstrap/bootstrap.hpp"
#include "../src/hdp_proxy.hpp"
#include "../src/host/host.hpp"
#include "../src/host/host_templates.hpp"
#include "../src/ipc/context_ipc_host.hpp"
#include "../src/ipc/backend_ipc.hpp"
#include "../src/ipc_policy.hpp"
#include "../src/memory/default_allocator.hpp"
#include "../src/memory/heap_memory.hpp"
#include "../src/memory/hip_allocator.hpp"
#include "../src/memory/remote_heap_info.hpp"
#include "../src/memory/symmetric_heap.hpp"
#include "../src/mpi_instance.hpp"
#include "../src/util.hpp"
#include "../src/envvar.hpp"

#include "ipc_test_config.hpp"

namespace rocshmem {

/**
 * Fixture that constructs the non-MPI IPC host context stack using
 * TcpBootstrap for coordination, mirroring IPCBackend(TcpBootstrap*).
 *
 * Two MPI ranks are used purely as a process launcher so that this test
 * can share the existing unit-test driver infrastructure.  The bootstrap
 * object itself is TcpBootstrap — MPI is never consulted for the IPC
 * handle exchange, matching the non-MPI production path exactly.
 */
class IPCHostAMOFixture : public ::testing::Test {

 protected:
  void SetUp() override {
    MPIInstance::mpilib_dl_init();

    // Assign one GPU per rank using the OpenMPI local-rank env var.
    char* lr = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    int local_rank = lr ? atoi(lr) : 0;
    CHECK_HIP(hipSetDevice(local_rank));

    MPI_Comm_rank(MPI_COMM_WORLD, &my_pe_);
    MPI_Comm_size(MPI_COMM_WORLD, &num_pes_);

    // ----------------------------------------------------------------
    // Build TcpBootstrap for non-MPI IPC handle exchange.
    // PE 0 creates the uniqueid and broadcasts it via MPI (setup only).
    // ----------------------------------------------------------------
    rocshmem_uniqueid_t uid{};
    if (my_pe_ == 0) {
      uid = TcpBootstrap::createUniqueId();
    }
    MPI_Bcast(&uid, sizeof(uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    bootstrap_ = new TcpBootstrap(my_pe_, num_pes_);
    bootstrap_->initialize(uid);

    // ----------------------------------------------------------------
    // Allocate fine-grain symmetric heap and exchange base pointers
    // via TcpBootstrap (not MPI).
    // ----------------------------------------------------------------
    sym_heap_ = new SymmetricHeap(MPI_COMM_NULL, bootstrap_);
    assert(sym_heap_ != nullptr);

    // ----------------------------------------------------------------
    // Open IPC handles for all remote heaps — non-MPI path inside
    // ipcHostInit uses bootstrap_->groupAllGather.
    // ----------------------------------------------------------------
    ipc_impl_.ipcHostInit(my_pe_, sym_heap_->get_heap_bases(), bootstrap_);

    // ----------------------------------------------------------------
    // Build HostInterface with the TcpBootstrap constructor.
    // This creates a plain WindowInfo pool (no MPI_Win).
    // ----------------------------------------------------------------
    host_interface_ = std::make_shared<HostInterface>(
        hdp_proxy_.get(), bootstrap_, sym_heap_);

    // ----------------------------------------------------------------
    // Construct IPCHostContext — this is the object under test.
    // It copies ipc_bases from device to host and acquires a plain
    // WindowInfo from the pool.
    //
    // We cannot call the real IPCHostContext(Backend*, options) because
    // we don't have a full Backend.  Instead we wire the same members
    // that the constructor sets, using the internal API directly.
    // ----------------------------------------------------------------
    context_window_info_ = host_interface_->acquire_window_context();
    host_ipc_bases_ = new char*[num_pes_];
    CHECK_HIP(hipMemcpy(host_ipc_bases_,
                        ipc_impl_.ipc_bases,
                        num_pes_ * sizeof(char*),
                        hipMemcpyDeviceToHost));

    MPI_Barrier(MPI_COMM_WORLD);
  }

  void TearDown() override {
    MPI_Barrier(MPI_COMM_WORLD);

    delete[] host_ipc_bases_;
    host_interface_->release_window_context(context_window_info_);
    host_interface_.reset();

    ipc_impl_.ipcHostStop();
    delete sym_heap_;
    delete bootstrap_;

    MPIInstance::mpilib_dl_close();
  }

  // Translate a symmetric-heap pointer from local PE space to remote PE space.
  // Mirrors IPCHostContext::shmem_ptr / the address translation in all IPC ops.
  char* to_remote_ptr(void* local_ptr, int pe) {
    uint64_t offset = reinterpret_cast<char*>(local_ptr) - host_ipc_bases_[my_pe_];
    return host_ipc_bases_[pe] + offset;
  }

  // Allocate an int from the local fine-grain heap (base + fixed offset).
  // Returns a local pointer; to_remote_ptr() gives the view from another PE.
  int* alloc_int_on_heap(size_t slot = 0) {
    return reinterpret_cast<int*>(sym_heap_->get_local_heap_base()) + slot;
  }

  int my_pe_{-1};
  int num_pes_{-1};

  TcpBootstrap*                    bootstrap_{nullptr};
  SymmetricHeap*                   sym_heap_{nullptr};
  IpcOnImpl                        ipc_impl_{};
  HdpProxy<HIPHostAllocator>       hdp_proxy_{};
  std::shared_ptr<HostInterface>   host_interface_{nullptr};
  WindowInfo*                      context_window_info_{nullptr};
  char**                           host_ipc_bases_{nullptr};
};

}  // namespace rocshmem

#endif  // ROCSHMEM_IPC_HOST_AMO_GTEST_HPP
