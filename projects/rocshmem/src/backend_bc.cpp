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

#include "log.hpp"

#include <cassert>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace rocshmem {

#define NET_CHECK(cmd) do {                                  \
  if (cmd != MPI_SUCCESS) {                                  \
    LOG_ERROR_ABORT("Unrecoverable error: MPI Failure");     \
  }                                                          \
} while(0)

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
   * Copy log state to device constant memory for device-side logging.
   */
  uint32_t log_flags = 0;
  if (envvar::log_flags.show_error) log_flags |= logd_constants::SHOW_ERROR;
  if (envvar::log_flags.show_warn)  log_flags |= logd_constants::SHOW_WARN;
  if (envvar::log_flags.show_info)  log_flags |= logd_constants::SHOW_INFO;
  if (envvar::log_flags.show_api)   log_flags |= logd_constants::SHOW_API;
  if (envvar::log_flags.show_trace) log_flags |= logd_constants::SHOW_TRACE;
  if (envvar::log_flags.show_color) log_flags |= logd_constants::SHOW_COLOR;
  struct logd_constants host_logd_constants{log_pe_number, log_flags};
  struct logd_constants* logd_constants_addr{nullptr};
  CHECK_HIP(hipGetSymbolAddress(reinterpret_cast<void**>(&logd_constants_addr),
                                HIP_SYMBOL(logd_constants)));
  CHECK_HIP(hipMemcpy(logd_constants_addr, &host_logd_constants, sizeof(host_logd_constants),
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

  psync_allocator_ = get_default_allocator();

  max_symm_regions_ = envvar::max_symm_regions;
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
  if (backend_comm != MPI_COMM_NULL)
    NET_CHECK(mpilib_ftable_.Comm_free(&backend_comm));

  if (done_init) {
    CHECK_HIP(hipHostFree(done_init));
  }
}

void Backend::dump_stats() {
#ifndef PROFILE
  LOG_WARN("dump_stats() called but PROFILE=ON was not set at build time; all counters are zero");
#endif
  // Accumulate per-context stats into the global accumulators before printing.
  // Reset first so repeated calls to dump_stats() do not double-count.
  globalStats.resetStats();
  globalHostStats.resetStats();

  // Device stats: each backend copies ctx_array[i].ctxStats via hipMemcpy.
  accumulate_ctx_device_stats();

  // Host stats: walk list_of_ctxs and accumulate ctxHostStats from each,
  // then include the default host context which is not in list_of_ctxs.
  for (auto* ctx : list_of_ctxs) {
    globalHostStats.accumulateStats(ctx->ctxHostStats);
  }
  accumulate_default_host_ctx_stats();

  // Build each stats section into a buffer and emit with a single LOG_INFO per section.
  char buf[8192];
  int pos = 0;
  auto append = [&](const char* fmt, ...) {
    if (pos >= static_cast<int>(sizeof(buf)) - 1) return;
    va_list args;
    va_start(args, fmt);
    int n = vsnprintf(buf + pos, sizeof(buf) - pos, fmt, args);
    va_end(args);
    if (n > 0) pos += n;
  };

  int n_printed = 0;
  auto pstat = [&](const char* name, StatType val) {
    if (val) { append("  %-30s %llu\n", name, static_cast<unsigned long long>(val)); ++n_printed; }
  };

  static_assert(NUM_STATS == 69,
    "rocshmem_stats enum changed; update dump_stats device section");
  const auto& device_stats{globalStats};
  uint64_t device_total = 0;
  for (int i = 0; i < NUM_STATS; i++) device_total += device_stats.getStat(i);
  if (device_total) {
    n_printed = 0;
    append("DEVICE STATS\n");
    pstat("Put",                   device_stats.getStat(NUM_PUT));
    pstat("Put_NBI",               device_stats.getStat(NUM_PUT_NBI));
    pstat("P",                     device_stats.getStat(NUM_P));
    pstat("WG_Put",                device_stats.getStat(NUM_PUT_WG));
    pstat("WG_Put_NBI",            device_stats.getStat(NUM_PUT_NBI_WG));
    pstat("WAVE_Put",              device_stats.getStat(NUM_PUT_WAVE));
    pstat("WAVE_Put_NBI",          device_stats.getStat(NUM_PUT_NBI_WAVE));
    pstat("Get",                   device_stats.getStat(NUM_GET));
    pstat("Get_NBI",               device_stats.getStat(NUM_GET_NBI));
    pstat("G",                     device_stats.getStat(NUM_G));
    pstat("WG_Get",                device_stats.getStat(NUM_GET_WG));
    pstat("WG_Get_NBI",            device_stats.getStat(NUM_GET_NBI_WG));
    pstat("WAVE_Get",              device_stats.getStat(NUM_GET_WAVE));
    pstat("WAVE_Get_NBI",          device_stats.getStat(NUM_GET_NBI_WAVE));
    pstat("Fence",                 device_stats.getStat(NUM_FENCE));
    pstat("Quiet",                 device_stats.getStat(NUM_QUIET));
    pstat("PE_Quiet",              device_stats.getStat(NUM_PE_QUIET));
    pstat("ToAll",                 device_stats.getStat(NUM_TO_ALL));
    pstat("BarrierAll",            device_stats.getStat(NUM_BARRIER_ALL));
    pstat("WAVE_BarrierAll",       device_stats.getStat(NUM_BARRIER_ALL_WAVE));
    pstat("WG_BarrierAll",         device_stats.getStat(NUM_BARRIER_ALL_WG));
    pstat("Barrier",               device_stats.getStat(NUM_BARRIER));
    pstat("WAVE_Barrier",          device_stats.getStat(NUM_BARRIER_WAVE));
    pstat("WG_Barrier",            device_stats.getStat(NUM_BARRIER_WG));
    pstat("SyncAll",               device_stats.getStat(NUM_SYNC_ALL));
    pstat("WAVE_SyncAll",          device_stats.getStat(NUM_SYNC_ALL_WAVE));
    pstat("WG_SyncAll",            device_stats.getStat(NUM_SYNC_ALL_WG));
    pstat("Sync",                  device_stats.getStat(NUM_SYNC));
    pstat("WAVE_Sync",             device_stats.getStat(NUM_SYNC_WAVE));
    pstat("WG_Sync",               device_stats.getStat(NUM_SYNC_WG));
    pstat("Wait_Until",            device_stats.getStat(NUM_WAIT_UNTIL));
    pstat("Wait_Until_Any",        device_stats.getStat(NUM_WAIT_UNTIL_ANY));
    pstat("Wait_Until_All",        device_stats.getStat(NUM_WAIT_UNTIL_ALL));
    pstat("Wait_Until_Some",       device_stats.getStat(NUM_WAIT_UNTIL_SOME));
    pstat("Wait_Until_All_Vector", device_stats.getStat(NUM_WAIT_UNTIL_ALL_VECTOR));
    pstat("Wait_Until_Any_Vector", device_stats.getStat(NUM_WAIT_UNTIL_ANY_VECTOR));
    pstat("Wait_Until_Some_Vector",device_stats.getStat(NUM_WAIT_UNTIL_SOME_VECTOR));
    pstat("Test",                  device_stats.getStat(NUM_TEST));
    pstat("SHMEM_PTR",             device_stats.getStat(NUM_SHMEM_PTR));
    pstat("Finalize",              device_stats.getStat(NUM_FINALIZE));
    pstat("Msg_Coal",              device_stats.getStat(NUM_MSG_COAL));
    pstat("Atomic_FAdd",           device_stats.getStat(NUM_ATOMIC_FADD));
    pstat("Atomic_FCswap",         device_stats.getStat(NUM_ATOMIC_FCSWAP));
    pstat("Atomic_FInc",           device_stats.getStat(NUM_ATOMIC_FINC));
    pstat("Atomic_Fetch",          device_stats.getStat(NUM_ATOMIC_FETCH));
    pstat("Atomic_Add",            device_stats.getStat(NUM_ATOMIC_ADD));
    pstat("Atomic_Set",            device_stats.getStat(NUM_ATOMIC_SET));
    pstat("Atomic_Swap",           device_stats.getStat(NUM_ATOMIC_SWAP));
    pstat("Atomic_Cswap",          device_stats.getStat(NUM_ATOMIC_CSWAP));
    pstat("Atomic_Inc",            device_stats.getStat(NUM_ATOMIC_INC));
    pstat("Atomic_FetchAnd",       device_stats.getStat(NUM_ATOMIC_FETCH_AND));
    pstat("Atomic_And",            device_stats.getStat(NUM_ATOMIC_AND));
    pstat("Atomic_FetchOr",        device_stats.getStat(NUM_ATOMIC_FETCH_OR));
    pstat("Atomic_Or",             device_stats.getStat(NUM_ATOMIC_OR));
    pstat("Atomic_FetchXor",       device_stats.getStat(NUM_ATOMIC_FETCH_XOR));
    pstat("Atomic_Xor",            device_stats.getStat(NUM_ATOMIC_XOR));
    pstat("Broadcast",             device_stats.getStat(NUM_BROADCAST));
    pstat("Alltoall",              device_stats.getStat(NUM_ALLTOALL));
    pstat("Alltoallv",             device_stats.getStat(NUM_ALLTOALLV));
    pstat("Fcollect",              device_stats.getStat(NUM_FCOLLECT));
    pstat("Create",                device_stats.getStat(NUM_CREATE));
    pstat("Put_Signal",            device_stats.getStat(NUM_PUT_SIGNAL));
    pstat("WG_Put_Signal",         device_stats.getStat(NUM_PUT_SIGNAL_WG));
    pstat("WAVE_Put_Signal",       device_stats.getStat(NUM_PUT_SIGNAL_WAVE));
    pstat("Put_Signal_NBI",        device_stats.getStat(NUM_PUT_SIGNAL_NBI));
    pstat("WG_Put_Signal_NBI",     device_stats.getStat(NUM_PUT_SIGNAL_NBI_WG));
    pstat("WAVE_Put_Signal_NBI",   device_stats.getStat(NUM_PUT_SIGNAL_NBI_WAVE));
    pstat("ReduceScatter",         device_stats.getStat(NUM_REDUCE_SCATTER));
    LOG_INFO("%s", buf);
  }

  static_assert(NUM_HOST_STATS == 39,
    "rocshmem_host_stats enum changed; update dump_stats host section");
  const auto& host_stats{globalHostStats};
  uint64_t host_total = 0;
  for (int i = 0; i < NUM_HOST_STATS; i++) host_total += host_stats.getStat(i);
  if (host_total) {
    pos = 0;
    n_printed = 0;
    append("HOST STATS\n");
    pstat("Put",                   host_stats.getStat(NUM_HOST_PUT));
    pstat("Put_NBI",               host_stats.getStat(NUM_HOST_PUT_NBI));
    pstat("P",                     host_stats.getStat(NUM_HOST_P));
    pstat("Get",                   host_stats.getStat(NUM_HOST_GET));
    pstat("Get_NBI",               host_stats.getStat(NUM_HOST_GET_NBI));
    pstat("G",                     host_stats.getStat(NUM_HOST_G));
    pstat("Fence",                 host_stats.getStat(NUM_HOST_FENCE));
    pstat("Quiet",                 host_stats.getStat(NUM_HOST_QUIET));
    pstat("ToAll",                 host_stats.getStat(NUM_HOST_TO_ALL));
    pstat("BarrierAll",            host_stats.getStat(NUM_HOST_BARRIER_ALL));
    pstat("SyncAll",               host_stats.getStat(NUM_HOST_SYNC_ALL));
    pstat("Wait_Until",            host_stats.getStat(NUM_HOST_WAIT_UNTIL));
    pstat("Wait_Until_Any",        host_stats.getStat(NUM_HOST_WAIT_UNTIL_ANY));
    pstat("Wait_Until_All",        host_stats.getStat(NUM_HOST_WAIT_UNTIL_ALL));
    pstat("Wait_Until_Some",       host_stats.getStat(NUM_HOST_WAIT_UNTIL_SOME));
    pstat("Wait_Until_All_Vector", host_stats.getStat(NUM_HOST_WAIT_UNTIL_ALL_VECTOR));
    pstat("Wait_Until_Any_Vector", host_stats.getStat(NUM_HOST_WAIT_UNTIL_ANY_VECTOR));
    pstat("Wait_Until_Some_Vector",host_stats.getStat(NUM_HOST_WAIT_UNTIL_SOME_VECTOR));
    pstat("Test",                  host_stats.getStat(NUM_HOST_TEST));
    pstat("SHMEM_PTR",             host_stats.getStat(NUM_HOST_SHMEM_PTR));
    pstat("Finalize",              host_stats.getStat(NUM_HOST_FINALIZE));
    pstat("Atomic_FAdd",           host_stats.getStat(NUM_HOST_ATOMIC_FADD));
    pstat("Atomic_FCswap",         host_stats.getStat(NUM_HOST_ATOMIC_FCSWAP));
    pstat("Atomic_FInc",           host_stats.getStat(NUM_HOST_ATOMIC_FINC));
    pstat("Atomic_Fetch",          host_stats.getStat(NUM_HOST_ATOMIC_FETCH));
    pstat("Atomic_Add",            host_stats.getStat(NUM_HOST_ATOMIC_ADD));
    pstat("Atomic_Set",            host_stats.getStat(NUM_HOST_ATOMIC_SET));
    pstat("Atomic_Swap",           host_stats.getStat(NUM_HOST_ATOMIC_SWAP));
    pstat("Atomic_Cswap",          host_stats.getStat(NUM_HOST_ATOMIC_CSWAP));
    pstat("Atomic_Inc",            host_stats.getStat(NUM_HOST_ATOMIC_INC));
    pstat("Atomic_FetchAnd",       host_stats.getStat(NUM_HOST_ATOMIC_FETCH_AND));
    pstat("Atomic_And",            host_stats.getStat(NUM_HOST_ATOMIC_AND));
    pstat("Atomic_FetchOr",        host_stats.getStat(NUM_HOST_ATOMIC_FETCH_OR));
    pstat("Atomic_Or",             host_stats.getStat(NUM_HOST_ATOMIC_OR));
    pstat("Atomic_FetchXor",       host_stats.getStat(NUM_HOST_ATOMIC_FETCH_XOR));
    pstat("Atomic_Xor",            host_stats.getStat(NUM_HOST_ATOMIC_XOR));
    pstat("Broadcast",             host_stats.getStat(NUM_HOST_BROADCAST));
    pstat("Alltoall",              host_stats.getStat(NUM_HOST_ALLTOALL));
    LOG_INFO("%s", buf);
  }

  dump_backend_stats();
}

void Backend::reset_stats() {
  globalStats.resetStats();
  globalHostStats.resetStats();

  reset_backend_stats();
}

int Backend::buffer_register(void *addr, size_t length) {
  LOG_TRACE("Backend::buffer_register addr=%p length=%zu", addr, length);

  if (addr == nullptr) {
    LOG_TRACE("Backend::buffer_register FAIL: addr is null");
    return ROCSHMEM_ERROR;
  }

  if (length == 0) {
    LOG_TRACE("Backend::buffer_register FAIL: length is 0");
    return ROCSHMEM_ERROR;
  }

  uintptr_t start = reinterpret_cast<uintptr_t>(addr);

  // Check for overflow when computing end address
  if (start > UINTPTR_MAX - length) {
    LOG_TRACE("Backend::buffer_register FAIL: overflow start=0x%lx length=%zu", start, length);
    return ROCSHMEM_ERROR;
  }

  uintptr_t end = start + length;

  // Find first entry with start >= our start
  auto it = user_buffer_regions.lower_bound(start);

  // Check entry at or after our start for overlap
  if (it != user_buffer_regions.end() && it->first < end) {
    LOG_TRACE("Backend::buffer_register FAIL: overlap with existing region "
              "[0x%lx, +%zu], new [0x%lx, +%zu]",
              it->first, it->second, start, length);
    return ROCSHMEM_ERROR;
  }

  // Check entry just before our start for overlap
  if (it != user_buffer_regions.begin()) {
    auto prev = std::prev(it);
    uintptr_t prev_end = prev->first + prev->second;
    if (prev_end > start) {
      LOG_TRACE("Backend::buffer_register FAIL: overlap with preceding region "
                "[0x%lx, +%zu] (ends at 0x%lx), new start=0x%lx",
                prev->first, prev->second, prev_end, start);
      return ROCSHMEM_ERROR;
    }
  }

  user_buffer_regions[start] = length;
  LOG_TRACE("Backend::buffer_register OK: registered [0x%lx, +%zu] (total regions: %zu)",
            start, length, user_buffer_regions.size());
  return ROCSHMEM_SUCCESS;
}

int Backend::buffer_unregister(void *addr) {
  if (addr == nullptr) {
    return ROCSHMEM_ERROR;
  }

  uintptr_t target = reinterpret_cast<uintptr_t>(addr);

  // Find first entry with start > target
  auto it = user_buffer_regions.upper_bound(target);

  if (it != user_buffer_regions.begin()) {
    auto prev = std::prev(it);
    uintptr_t start = prev->first;
    uintptr_t end = start + prev->second;

    // Check if target falls within [start, end)
    if (target < end) {
      user_buffer_regions.erase(prev);
      return ROCSHMEM_SUCCESS;
    }
  }

  return ROCSHMEM_ERROR;
}

void Backend::buffer_unregister_all() {
  user_buffer_regions.clear();
}

int Backend::buffer_register_symmetric(void *addr, size_t length,
                                       void **registered_addr) {
#if HIP_VERSION >= 70000000
  if (registered_addr == nullptr) {
    return ROCSHMEM_ERROR;
  }

  HIPAllocator *alloc = heap.get_allocator();

  /*
   * Symmetric registration is restricted to HIP VMM device memory. Validate
   * the buffer against the heap's own allocator (mirrors NVSHMEM's per-buffer
   * checks): this rejects non-VMM heaps, null/zero/granularity-misaligned
   * sizes, non-VMM pointers, memory on the wrong device, and handle-type
   * mismatches.
   */
  if (!alloc->ValidateVMMRegistration(addr, length, hip_dev_id)) {
    return ROCSHMEM_ERROR;
  }

  /* Enforce the configured maximum number of symmetric registrations. */
  if (symm_buffer_regions.size() >= max_symm_regions_) {
    return ROCSHMEM_ERROR;
  }

  /*
   * Overlap detection is performed against the user's *original* address
   * range. Aliases are freshly reserved virtual addresses and never overlap,
   * so checking them would be meaningless.
   */
  uintptr_t orig_start = reinterpret_cast<uintptr_t>(addr);

  // Check for overflow when computing end address
  if (orig_start > UINTPTR_MAX - length) {
    return ROCSHMEM_ERROR;
  }

  uintptr_t orig_end = orig_start + length;

  // Find first entry with base >= our base
  auto it = symm_orig_regions.lower_bound(orig_start);

  // Check entry at or after our base for overlap
  if (it != symm_orig_regions.end() && it->first < orig_end) {
    return ROCSHMEM_ERROR;
  }

  // Check entry just before our base for overlap
  if (it != symm_orig_regions.begin()) {
    auto prev = std::prev(it);
    uintptr_t prev_end = prev->first + prev->second;
    if (prev_end > orig_start) {
      return ROCSHMEM_ERROR;
    }
  }

  /*
   * Map the user's buffer to a fresh rocSHMEM-owned virtual address. This
   * alias refers to the same physical memory but at a distinct address that
   * the caller uses for RMA and unregistration.
   */
  void *alias = nullptr;
  if (create_symm_alias(addr, length, &alias) != hipSuccess) {
    return ROCSHMEM_ERROR;
  }

  uintptr_t alias_start = reinterpret_cast<uintptr_t>(alias);
  symm_buffer_regions[alias_start] = SymmRegion{length, orig_start};
  symm_orig_regions[orig_start] = length;
  *registered_addr = alias;
  return ROCSHMEM_SUCCESS;
#else
  (void)addr;
  (void)length;
  (void)registered_addr;
  return ROCSHMEM_ERROR;
#endif
}

int Backend::buffer_unregister_symmetric(void *addr) {
#if HIP_VERSION >= 70000000
  if (addr == nullptr) {
    return ROCSHMEM_ERROR;
  }

  uintptr_t base = reinterpret_cast<uintptr_t>(addr);

  auto it = symm_buffer_regions.find(base);
  if (it == symm_buffer_regions.end()) {
    return ROCSHMEM_ERROR;
  }

  /* Release the rocSHMEM-owned alias mapping created at registration. */
  destroy_symm_alias(addr, it->second.length);

  /* Drop the original-range entry used for overlap detection. */
  symm_orig_regions.erase(it->second.orig_base);

  symm_buffer_regions.erase(it);
  return ROCSHMEM_SUCCESS;
#else
  (void)addr;
  return ROCSHMEM_ERROR;
#endif
}

void Backend::symm_allgather(void* inout, size_t bytes_per_pe) {
  if (backend_comm != MPI_COMM_NULL) {
    NET_CHECK(mpilib_ftable_.Allgather(MPI_IN_PLACE, bytes_per_pe, MPI_CHAR,
                                       inout, bytes_per_pe, MPI_CHAR,
                                       backend_comm));
  } else {
    assert(backend_bootstr != nullptr);
    backend_bootstr->allGather(inout, bytes_per_pe);
  }
}

hipError_t Backend::create_symm_alias(void *addr, size_t length,
                                      void **alias) {
#if HIP_VERSION >= 70000000
  return heap.get_allocator()->MapLocalAlias(addr, length, alias);
#else
  (void)addr;
  (void)length;
  (void)alias;
  return hipErrorNotSupported;
#endif
}

void Backend::destroy_symm_alias(void *alias, size_t length) {
#if HIP_VERSION >= 70000000
  (void)heap.get_allocator()->UnmapLocalAlias(alias, length);
#else
  (void)alias;
  (void)length;
#endif
}

void Backend::alloc_ipc_symm_table() {
#if HIP_VERSION >= 70000000
  /*
   * Device-visible symmetric-registration table shared by all contexts via
   * IpcImpl (so registrations are observed without re-propagation).
   */
  int capacity = static_cast<int>(max_symm_regions_);
  if (capacity <= 0) {
    capacity = 1;
  }

  IpcSymmTable *table{nullptr};
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&table), sizeof(IpcSymmTable)));
  assert(table);
  CHECK_HIP(hipMemset(table, 0, sizeof(IpcSymmTable)));

  IpcSymmRegion *regions{nullptr};
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&regions),
                      sizeof(IpcSymmRegion) * capacity));
  assert(regions);
  CHECK_HIP(hipMemset(regions, 0, sizeof(IpcSymmRegion) * capacity));

  IpcSymmTable host_table{};
  host_table.count = 0;
  host_table.capacity = capacity;
  host_table.regions = regions;
  CHECK_HIP(hipMemcpy(table, &host_table, sizeof(IpcSymmTable),
                      hipMemcpyHostToDevice));
  ipcImpl.symm_table = table;
#endif
}

void Backend::free_ipc_symm_table() {
#if HIP_VERSION >= 70000000
  if (ipcImpl.symm_table == nullptr) {
    return;
  }
  IpcSymmTable host_table{};
  CHECK_HIP(hipMemcpy(&host_table, ipcImpl.symm_table, sizeof(IpcSymmTable),
                      hipMemcpyDeviceToHost));
  if (host_table.regions != nullptr) {
    CHECK_HIP(hipFree(host_table.regions));
  }
  CHECK_HIP(hipFree(ipcImpl.symm_table));
  ipcImpl.symm_table = nullptr;
#endif
}

int Backend::register_ipc_symm_region([[maybe_unused]] void *symm_addr,
                                      [[maybe_unused]] void *export_addr,
                                      [[maybe_unused]] size_t length,
                                      [[maybe_unused]] const std::vector<int>& peer_global,
                                      [[maybe_unused]] int self_index) {
#if HIP_VERSION >= 70000000
  IpcSymmTable *table = ipcImpl.symm_table;
  if (table == nullptr) {
    return ROCSHMEM_ERROR;
  }

  HIPAllocator *alloc = heap.get_allocator();
  int num_peers = static_cast<int>(peer_global.size());
  size_t handle_size = alloc->GetIpcHandleSize();

  /*
   * The handle vector spans all PEs so the all-gather below lands each PE's
   * handle at its global index; only the peers listed in peer_global are
   * subsequently opened.
   */
  HIPIpcHandleVec *vec = alloc->AllocateIpcHandleVec(num_pes);

  /* Export this PE's handle; make the outcome collective before the gather. */
  int export_ok = (alloc->GetIpcHandleFromPtr(export_addr, length,
                       vec->GetHandleVecElem(my_pe)) == hipSuccess) ? 1 : 0;
  if (!all_pes_succeeded(export_ok)) {
    if (export_ok) {
      (void)alloc->CloseExportedHandle(vec->GetHandleVecElem(my_pe));
    }
    delete vec;
    return ROCSHMEM_ERROR;
  }

  /* Keep this PE's handle bytes so the export can be released later. */
  std::vector<char> local_handle(handle_size);
  std::memcpy(local_handle.data(), vec->GetHandleVecElem(my_pe), handle_size);

  /* Collective exchange of IPC handles across all PEs. */
  symm_allgather(vec->GetHandleVecElem(0), handle_size);

  /* Open the handles of the requested peers into peer-mapped base addresses. */
  std::vector<char *> host_peer_bases(num_peers, nullptr);
  int open_ok = 1;
  for (int i = 0; i < num_peers; i++) {
    if (i == self_index) {
      host_peer_bases[i] = reinterpret_cast<char *>(symm_addr);
      continue;
    }
    void *p{nullptr};
    if (alloc->OpenIpcHandle(&p, vec->GetHandleVecElem(peer_global[i])) !=
        hipSuccess) {
      for (int j = 0; j < i; j++) {
        if (j == self_index) {
          continue;
        }
        (void)alloc->CloseIpcHandle(host_peer_bases[j]);
        host_peer_bases[j] = nullptr;
      }
      open_ok = 0;
      break;
    }
    host_peer_bases[i] = reinterpret_cast<char *>(p);
  }

  if (!all_pes_succeeded(open_ok)) {
    if (open_ok) {
      for (int i = 0; i < num_peers; i++) {
        if (i == self_index) {
          continue;
        }
        (void)alloc->CloseIpcHandle(host_peer_bases[i]);
      }
    }
    (void)alloc->CloseExportedHandle(local_handle.data());
    delete vec;
    return ROCSHMEM_ERROR;
  }

  delete vec;

  /* Publish the peer-base array into device memory for ipcPeerPtr. */
  char **peer_bases{nullptr};
  CHECK_HIP(hipMalloc(reinterpret_cast<void **>(&peer_bases),
                      num_peers * sizeof(char *)));
  CHECK_HIP(hipMemcpy(peer_bases, host_peer_bases.data(),
                      num_peers * sizeof(char *), hipMemcpyHostToDevice));

  IpcSymmTable host_table{};
  CHECK_HIP(hipMemcpy(&host_table, table, sizeof(IpcSymmTable),
                      hipMemcpyDeviceToHost));
  int slot = host_table.count;
  if (slot >= host_table.capacity) {
    for (int i = 0; i < num_peers; i++) {
      if (i == self_index) {
        continue;
      }
      (void)alloc->CloseIpcHandle(host_peer_bases[i]);
    }
    (void)hipFree(peer_bases);
    (void)alloc->CloseExportedHandle(local_handle.data());
    return ROCSHMEM_ERROR;
  }

  IpcSymmRegion host_region{};
  host_region.local_base = reinterpret_cast<uintptr_t>(symm_addr);
  host_region.length = length;
  host_region.peer_bases = peer_bases;
  CHECK_HIP(hipMemcpy(&host_table.regions[slot], &host_region,
                      sizeof(IpcSymmRegion), hipMemcpyHostToDevice));

  host_table.count = slot + 1;
  CHECK_HIP(hipMemcpy(table, &host_table, sizeof(IpcSymmTable),
                      hipMemcpyHostToDevice));

  IpcSymmRecord rec;
  rec.slot = slot;
  rec.num_peers = num_peers;
  rec.self_index = self_index;
  rec.dev_peer_bases = peer_bases;
  rec.peer_bases = std::move(host_peer_bases);
  rec.local_handle = std::move(local_handle);
  ipc_symm_records_[reinterpret_cast<uintptr_t>(symm_addr)] = std::move(rec);
  return ROCSHMEM_SUCCESS;
#else
  return ROCSHMEM_ERROR;
#endif
}

int Backend::unregister_ipc_symm_region([[maybe_unused]] void *symm_addr) {
#if HIP_VERSION >= 70000000
  IpcSymmTable *table = ipcImpl.symm_table;
  if (table == nullptr) {
    return ROCSHMEM_ERROR;
  }

  uintptr_t key = reinterpret_cast<uintptr_t>(symm_addr);
  auto it = ipc_symm_records_.find(key);
  if (it == ipc_symm_records_.end()) {
    return ROCSHMEM_ERROR;
  }

  HIPAllocator *alloc = heap.get_allocator();
  int slot = it->second.slot;
  int num_peers = it->second.num_peers;
  int self_index = it->second.self_index;
  std::vector<char *> &peer_bases = it->second.peer_bases;

  for (int i = 0; i < num_peers; i++) {
    if (i == self_index) {
      continue;
    }
    CHECK_HIP(alloc->CloseIpcHandle(peer_bases[i]));
  }
  (void)alloc->CloseExportedHandle(it->second.local_handle.data());

  /* Compact the device table by moving the last region into the freed slot. */
  IpcSymmTable host_table{};
  CHECK_HIP(hipMemcpy(&host_table, table, sizeof(IpcSymmTable),
                      hipMemcpyDeviceToHost));
  int last = host_table.count - 1;
  if (slot != last) {
    CHECK_HIP(hipMemcpy(&host_table.regions[slot], &host_table.regions[last],
                        sizeof(IpcSymmRegion), hipMemcpyDeviceToDevice));
    for (auto &kv : ipc_symm_records_) {
      if (kv.second.slot == last) {
        kv.second.slot = slot;
        break;
      }
    }
  }
  host_table.count = last;
  CHECK_HIP(hipMemcpy(table, &host_table, sizeof(IpcSymmTable),
                      hipMemcpyHostToDevice));

  CHECK_HIP(hipFree(it->second.dev_peer_bases));
  ipc_symm_records_.erase(it);
  return ROCSHMEM_SUCCESS;
#else
  return ROCSHMEM_ERROR;
#endif
}

}  // namespace rocshmem
