/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "backend_bc.hpp"

#include "backend_type.hpp"
#include "context_incl.hpp"

#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/backend_ro.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/backend_ipc.hpp"
#endif

#include <cassert>

namespace rocshmem {

#define NET_CHECK(cmd)                                       \
  {                                                          \
    if (cmd != MPI_SUCCESS) {                                \
      fprintf(stderr, "Unrecoverable error: MPI Failure\n"); \
      abort() ;                                              \
    }                                                        \
  }

Backend::Backend(MPI_Comm comm) : heap(comm, nullptr) {
  init();
  init_mpi_once(comm);
  /*
   * Notify other threads that Backend has been initialized.
   */
  *done_init = 0;
}

Backend::Backend(TcpBootstrap* bootstrap) : heap(MPI_COMM_NULL, bootstrap) {
  init();
  backend_bootstr = bootstrap;
  backend_comm = MPI_COMM_NULL;

  my_pe = bootstrap->getRank();
  num_pes = bootstrap->getNranks();
  /*
   * Notify other threads that Backend has been initialized.
   */
  *done_init = 0;
}

void Backend::init(void) {
  CHECK_HIP(hipGetDevice(&hip_dev_id));

  int num_cus{};
  CHECK_HIP(hipDeviceGetAttribute(&num_cus, hipDeviceAttributeMultiprocessorCount, hip_dev_id));

  /*
   * Initialize 'print_lock' global and copy to the device memory space.
   */
  CHECK_HIP(hipMalloc(&print_lock, sizeof(*print_lock)));
  *print_lock = 0;

  int* print_lock_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&print_lock_addr),
                                HIP_SYMBOL(print_lock)));

  CHECK_HIP(hipMemcpy(print_lock_addr, &print_lock, sizeof(print_lock),
                      hipMemcpyDefault));

  /*
   * Copy this Backend object to 'backend_device_proxy' global in the
   * device memory space to provide a device-side handle to Backend.
   */
  int* device_backend_proxy_addr{nullptr};
  CHECK_HIP(
      hipGetSymbolAddress(reinterpret_cast<void**>(&device_backend_proxy_addr),
                          HIP_SYMBOL(device_backend_proxy)));

  Backend* this_temp_addr{this};
  CHECK_HIP(hipMemcpy(device_backend_proxy_addr, &this_temp_addr, sizeof(this),
                      hipMemcpyDefault));

  CHECK_HIP(
      hipHostMalloc(reinterpret_cast<void**>(&done_init), sizeof(uint8_t)));
}

void Backend::init_mpi_once(MPI_Comm comm) {
  if (comm == MPI_COMM_NULL) comm = MPI_COMM_WORLD;
  NET_CHECK(mpilib_ftable_.Comm_dup(comm, &backend_comm));
  NET_CHECK(mpilib_ftable_.Comm_size(backend_comm, &num_pes));
  NET_CHECK(mpilib_ftable_.Comm_rank(backend_comm, &my_pe));
}

void Backend::track_ctx(Context* ctx) {
  /**
   * TODO: Don't track CTX_PRIVATE when we support it
   * since destroying CTX_PRIVATE is the user's
   * responsibility.
   */
  list_of_ctxs.push_back(ctx);
}

void Backend::untrack_ctx(Context* ctx) {
  /* Get an iterator to this ctx in the vector */
  std::vector<Context*>::iterator it =
      std::find(list_of_ctxs.begin(), list_of_ctxs.end(), ctx);
  assert(it != list_of_ctxs.end());

  /* Remove the ctx from the vector */
  list_of_ctxs.erase(it);
}

void Backend::destroy_remaining_ctxs() {
  while (!list_of_ctxs.empty()) {
    ctx_destroy(list_of_ctxs.back());
    list_of_ctxs.pop_back();
  }
}

Backend::~Backend() {
  CHECK_HIP(hipFree(print_lock));
  if (backend_comm != MPI_COMM_NULL)
    NET_CHECK(mpilib_ftable_.Comm_free(&backend_comm));
}

void Backend::dump_stats() {
  printf("PE %d\n", my_pe);

  const auto& device_stats{globalStats};
  printf("DEVICE STATS\n");
  printf("Puts (Blocking/P/Nbi) %llu/%llu/%llu\n",
         device_stats.getStat(NUM_PUT), device_stats.getStat(NUM_P),
         device_stats.getStat(NUM_PUT_NBI));
  printf("WG_Puts (Blocking/Nbi) %llu/%llu\n", device_stats.getStat(NUM_PUT_WG),
         device_stats.getStat(NUM_PUT_NBI_WG));
  printf("WAVE_Puts (Blocking/Nbi) %llu/%llu\n",
         device_stats.getStat(NUM_PUT_WAVE),
         device_stats.getStat(NUM_PUT_NBI_WAVE));
  printf("Gets (Blocking/G/Nbi) %llu/%llu/%llu\n",
         device_stats.getStat(NUM_GET), device_stats.getStat(NUM_G),
         device_stats.getStat(NUM_GET_NBI));
  printf("WG_Gets (Blocking/Nbi) %llu/%llu\n", device_stats.getStat(NUM_GET_WG),
         device_stats.getStat(NUM_GET_NBI_WG));
  printf("WAVE_Gets (Blocking/Nbi) %llu/%llu\n",
         device_stats.getStat(NUM_GET_WAVE),
         device_stats.getStat(NUM_GET_NBI_WAVE));
  printf("Fences %llu\n", device_stats.getStat(NUM_FENCE));
  printf("Quiets %llu\n", device_stats.getStat(NUM_QUIET));
  printf("PE Quiets %llu\n", device_stats.getStat(NUM_PE_QUIET));
  printf("ToAll %llu\n", device_stats.getStat(NUM_TO_ALL));
  printf("BarrierAll %llu\n", device_stats.getStat(NUM_BARRIER_ALL));
  printf("WAVE_BarrierAll %llu\n", device_stats.getStat(NUM_BARRIER_ALL_WAVE));
  printf("WG_BarrierAll %llu\n", device_stats.getStat(NUM_BARRIER_ALL_WG));
  printf("Barrier %llu\n", device_stats.getStat(NUM_BARRIER));
  printf("WAVE_Barrier %llu\n", device_stats.getStat(NUM_BARRIER_WAVE));
  printf("WG_Barrier %llu\n", device_stats.getStat(NUM_BARRIER_WG));
  printf("Wait Until %llu\n", device_stats.getStat(NUM_WAIT_UNTIL));
  printf("Wait Until Any %llu\n", device_stats.getStat(NUM_WAIT_UNTIL_ANY));
  printf("Wait Until All %llu\n", device_stats.getStat(NUM_WAIT_UNTIL_ALL));
  printf("Wait Until Some %llu\n", device_stats.getStat(NUM_WAIT_UNTIL_SOME));
  printf("Wait Until All Vector %llu\n",
         device_stats.getStat(NUM_WAIT_UNTIL_ALL_VECTOR));
  printf("Wait Until Any Vector %llu\n",
         device_stats.getStat(NUM_WAIT_UNTIL_ANY_VECTOR));
  printf("Wait Until Some Vector %llu\n",
         device_stats.getStat(NUM_WAIT_UNTIL_SOME_VECTOR));
  printf("Finalizes %llu\n", device_stats.getStat(NUM_FINALIZE));
  printf("Coalesced %llu\n", device_stats.getStat(NUM_MSG_COAL));
  printf("Atomic_FAdd %llu\n", device_stats.getStat(NUM_ATOMIC_FADD));
  printf("Atomic_FCswap %llu\n", device_stats.getStat(NUM_ATOMIC_FCSWAP));
  printf("Atomic_FInc %llu\n", device_stats.getStat(NUM_ATOMIC_FINC));
  printf("Atomic_Fetch %llu\n", device_stats.getStat(NUM_ATOMIC_FETCH));
  printf("Atomic_Add %llu\n", device_stats.getStat(NUM_ATOMIC_ADD));
  printf("Atomic_Set %llu\n", device_stats.getStat(NUM_ATOMIC_SET));
  printf("Atomic_Cswap %llu\n", device_stats.getStat(NUM_ATOMIC_CSWAP));
  printf("Atomic_Inc %llu\n", device_stats.getStat(NUM_ATOMIC_INC));
  printf("Tests %llu\n", device_stats.getStat(NUM_TEST));
  printf("SHMEM_PTR %llu\n", device_stats.getStat(NUM_SHMEM_PTR));
  printf("SyncAll %llu\n", device_stats.getStat(NUM_SYNC_ALL));
  printf("WAVE_SyncAll %llu\n", device_stats.getStat(NUM_SYNC_ALL_WAVE));
  printf("WG_SyncAll %llu\n", device_stats.getStat(NUM_SYNC_ALL_WG));
  printf("Sync %llu\n", device_stats.getStat(NUM_SYNC));
  printf("WAVE_Sync %llu\n", device_stats.getStat(NUM_SYNC_WAVE));
  printf("WG_Sync %llu\n", device_stats.getStat(NUM_SYNC_WG));

  const auto& host_stats{globalHostStats};
  printf("HOST STATS\n");
  printf("Puts (Blocking/P/Nbi) %llu/%llu/%llu\n",
         host_stats.getStat(NUM_HOST_PUT), host_stats.getStat(NUM_HOST_P),
         host_stats.getStat(NUM_HOST_PUT_NBI));
  printf("Gets (Blocking/G/Nbi) (%llu/%llu/%llu)\n",
         host_stats.getStat(NUM_HOST_GET), host_stats.getStat(NUM_HOST_G),
         host_stats.getStat(NUM_HOST_GET_NBI));
  printf("Fences %llu\n", host_stats.getStat(NUM_HOST_FENCE));
  printf("Quiets %llu\n", host_stats.getStat(NUM_HOST_QUIET));
  printf("ToAll %llu\n", host_stats.getStat(NUM_HOST_TO_ALL));
  printf("BarrierAll %llu\n", host_stats.getStat(NUM_HOST_BARRIER_ALL));
  printf("Wait Until %llu\n", host_stats.getStat(NUM_HOST_WAIT_UNTIL));
  printf("Wait Until Any %llu\n", host_stats.getStat(NUM_HOST_WAIT_UNTIL_ANY));
  printf("Wait Until All %llu\n", host_stats.getStat(NUM_HOST_WAIT_UNTIL_ALL));
  printf("Wait Until Some %llu\n",
         host_stats.getStat(NUM_HOST_WAIT_UNTIL_SOME));
  printf("Wait Until All Vector %llu\n",
         host_stats.getStat(NUM_HOST_WAIT_UNTIL_ALL_VECTOR));
  printf("Wait Until Any Vector %llu\n",
         host_stats.getStat(NUM_HOST_WAIT_UNTIL_ANY_VECTOR));
  printf("Wait Until Some Vector %llu\n",
         host_stats.getStat(NUM_HOST_WAIT_UNTIL_SOME_VECTOR));
  printf("Finalizes %llu\n", host_stats.getStat(NUM_HOST_FINALIZE));
  printf("Atomic_FAdd %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FADD));
  printf("Atomic_FCswap %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FCSWAP));
  printf("Atomic_FInc %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FINC));
  printf("Atomic_Fetch %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_FETCH));
  printf("Atomic_Add %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_ADD));
  printf("Atomic_Set %llu\n", host_stats.getStat(NUM_ATOMIC_SET));
  printf("Atomic_Cswap %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_CSWAP));
  printf("Atomic_Inc %llu\n", host_stats.getStat(NUM_HOST_ATOMIC_INC));
  printf("Tests %llu\n", host_stats.getStat(NUM_HOST_TEST));
  printf("SHMEM_PTR %llu\n", host_stats.getStat(NUM_HOST_SHMEM_PTR));
  printf("SyncAll %llu\n", host_stats.getStat(NUM_HOST_SYNC_ALL));

  dump_backend_stats();
}

void Backend::reset_stats() {
  globalStats.resetStats();
  globalHostStats.resetStats();

  reset_backend_stats();
}

__device__ bool Backend::create_ctx(int64_t option, rocshmem_ctx_t* ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch(this->type) {
  case BackendType::GDA_BACKEND:
    return static_cast<GDABackend*>(this)->create_ctx(option, ctx);
    break;
  case BackendType::RO_BACKEND:
    return static_cast<ROBackend*>(this)->create_ctx(option, ctx);
    break;
  case BackendType::IPC_BACKEND:
  default:
      return static_cast<IPCBackend*>(this)->create_ctx(option, ctx);
      break;
  }
#elif defined(USE_GDA)
  return static_cast<GDABackend*>(this)->create_ctx(option, ctx);
#elif defined(USE_RO)
  return static_cast<ROBackend*>(this)->create_ctx(option, ctx);
#elif defined(USE_IPC)
  return static_cast<IPCBackend*>(this)->create_ctx(option, ctx);
#endif
}

__device__ void Backend::destroy_ctx(rocshmem_ctx_t* ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch(this->type) {
  case BackendType::GDA_BACKEND:
    static_cast<GDABackend*>(this)->destroy_ctx(ctx);
    break;
  case BackendType::RO_BACKEND:
    static_cast<ROBackend*>(this)->destroy_ctx(ctx);
    break;
  case BackendType::IPC_BACKEND:
  default:
    static_cast<IPCBackend*>(this)->destroy_ctx(ctx);
    break;
  }
#elif defined(USE_GDA)
  static_cast<GDABackend*>(this)->destroy_ctx(ctx);
#elif defined(USE_RO)
  static_cast<ROBackend*>(this)->destroy_ctx(ctx);
#elif defined(USE_IPC)
  static_cast<IPCBackend*>(this)->destroy_ctx(ctx);
#endif
}

}  // namespace rocshmem
