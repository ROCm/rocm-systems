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

#include "backend_ro.hpp"

#include <immintrin.h>
#include <smmintrin.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>  // NOLINT
#include <dlfcn.h>

#include "rocshmem/rocshmem.hpp"
#include "atomic_return.hpp"
#include "backend_type.hpp"
#include "context_incl.hpp"
#include "envvar.hpp"
#include "mpi_transport.hpp"
#include "ro_net_team.hpp"
#include "util.hpp"

namespace rocshmem {

extern rocshmem_ctx_t ROCSHMEM_HOST_CTX_DEFAULT;

ROBackend::ROBackend(MPI_Comm comm)
    : Backend(comm) {
  type = BackendType::RO_BACKEND;

  poll_block_count_ = envvar::max_num_contexts;

  profiler_proxy_ = ProfilerProxyT(envvar::max_num_contexts);

  int device_id;
  CHECK_HIP(hipGetDevice(&device_id));

  max_wg_size_ = get_threads_per_block(device_id);

  wf_size_ = get_wf_size(device_id);

  setup_default_ctx_buffers();

  size_t num_buff_elems = envvar::max_num_contexts * max_wg_size_;

  g_ret_buffer_ = RetBufferProxyT(num_buff_elems);

  atomic_ret_buffer_ = RetBufferProxyT(num_buff_elems);

  status_ = StatusProxyT(num_buff_elems);

  queue_ = Queue(envvar::max_num_contexts, queue_size_);

  transport_ = new MPITransport(backend_comm, &queue_);
  num_pes = transport_->getNumPes();
  my_pe = transport_->getMyPe();

  auto *bp{backend_proxy.get()};

  bp->hdp_policy = hdp_proxy_.get();

  bp->profiler = profiler_proxy_.get();

  bp->worker_thread_exit = false;

  bp->heap_ptr = &heap;

  ro_window_proxy_ = new WindowProxyT(&heap, transport_->get_world_comm(),
                                      num_windows_);

  bp->heap_window_info = ro_window_proxy_->get();

  initIPC();

  transport_->initTransport(envvar::max_num_contexts, &backend_proxy);

  host_interface = transport_->host_interface;

  default_host_ctx = std::make_unique<ROHostContext>(this, 0);

  ROCSHMEM_HOST_CTX_DEFAULT.ctx_opaque = default_host_ctx.get();

  team_world_proxy_ = new ROTeamProxy<HIPAllocator>(
      this, transport_->get_world_comm(), my_pe, num_pes);
  team_tracker.set_team_world(team_world_proxy_->get());

  ROCSHMEM_TEAM_WORLD =
      reinterpret_cast<rocshmem_team_t>(team_world_proxy_->get());

  default_block_handle_proxy_ = DefaultBlockHandleProxyT(
                                g_ret_buffer_.get(),
                                atomic_ret_buffer_.get(), &queue_,
                                status_.get(), default_ctx_status_.get(),
                                default_ctx_g_ret_buffer_.get(),
                                default_ctx_atomic_ret_buffer_.get());

  TeamInfo *tinfo = team_tracker.get_team_world()->tinfo_wrt_world;

  default_context_proxy_ = DefaultContextProxyT(this, tinfo);

  block_handle_proxy_ = BlockHandleProxyT(g_ret_buffer_.get(),
                        atomic_ret_buffer_.get(), &queue_,
                        max_wg_size_, status_.get(), envvar::max_num_contexts);
  setup_ctxs();

  worker_thread = std::thread(&ROBackend::ro_net_poll, this);

  *done_init = 1;
}

/* Currently we only check whether we can dlopen an MPI library.
 */
int ROBackend::backend_can_run() {
  auto handle = dlopen("libmpi.so", RTLD_LAZY);
  if (!handle) {
    printf("Could not open libmpi.so. Returning\n");
    return ROCSHMEM_ERROR;
  }
  //TODO dlsym MPI_Get_library_version and verify compat when HAVE_EXTERNAL_MPI is undef
  dlclose(handle);
  return ROCSHMEM_SUCCESS;
}

void ROBackend::setup_ctxs() {
  CHECK_HIP(hipMalloc(&ctx_array, sizeof(ROContext) * envvar::max_num_contexts));
  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    new (&ctx_array[i]) ROContext(this, i);
    ctx_free_list.get()->push_back(ctx_array + i);
  }
}

void ROBackend::setup_default_ctx_buffers() {
  size_t num_buff_elems = envvar::max_wavefront_buffers * wf_size_;

  g_ret_buffer_default_ctx_ = RetBufferProxyT(num_buff_elems);

  atomic_ret_buffer_default_ctx_ = RetBufferProxyT(num_buff_elems);

  status_default_ctx_ = StatusProxyT(num_buff_elems);

  default_ctx_status_.get()->allocate_queue(envvar::max_wavefront_buffers);
  default_ctx_g_ret_buffer_.get()->allocate_queue(envvar::max_wavefront_buffers);
  default_ctx_atomic_ret_buffer_.get()->allocate_queue(envvar::max_wavefront_buffers);


  char* status = status_default_ctx_.get();
  uint64_t* g_ret_buf = g_ret_buffer_default_ctx_.get();
  uint64_t* atomic_ret_buf = atomic_ret_buffer_default_ctx_.get();

  for (size_t i = 0; i < envvar::max_wavefront_buffers; i++) {
    size_t offset = i * wf_size_;
    default_ctx_status_.get()->push(status + offset);
    default_ctx_g_ret_buffer_.get()->push(g_ret_buf + offset);
    default_ctx_atomic_ret_buffer_.get()->push(atomic_ret_buf + offset);
  }
}

ROBackend::~ROBackend() {
  ro_net_free_runtime();
  CHECK_HIP(hipFree(ctx_array));
}

__device__ bool ROBackend::create_ctx(int64_t options, rocshmem_ctx_t *ctx) {
  ROContext *ctx_;

  auto pop_result = ctx_free_list.get()->pop_front();
  if (!pop_result.success) {
    return false;
  }
  ctx_ = pop_result.value;

  ctx->ctx_opaque = ctx_;
  return true;
}

__device__ void ROBackend::destroy_ctx(rocshmem_ctx_t *ctx) {
  ctx_free_list.get()->push_back(static_cast<ROContext *>(ctx->ctx_opaque));
}

void ROBackend::team_destroy(rocshmem_team_t team) {
  ROTeam *team_obj{get_internal_ro_team(team)};

  team_obj->~ROTeam();
  // CHECK_HIP(hipFree(team_obj));
}

void ROBackend::create_new_team(Team *parent_team,
                                TeamInfo *team_info_wrt_parent,
                                TeamInfo *team_info_wrt_world, int num_pes,
                                int my_pe_in_new_team, MPI_Comm team_comm,
                                rocshmem_team_t *new_team) {
  transport_->createNewTeam(this, parent_team, team_info_wrt_parent,
                            team_info_wrt_world, num_pes, my_pe_in_new_team,
                            team_comm, new_team);
}

void ROBackend::ctx_create(int64_t options, void **ctx) {
  ROHostContext *new_ctx{nullptr};
  new_ctx = new ROHostContext(this, options);
  *ctx = new_ctx;
}

ROHostContext *get_internal_ro_net_ctx(Context *ctx) {
  return reinterpret_cast<ROHostContext *>(ctx);
}

void ROBackend::ctx_destroy(Context *ctx) {
  ROHostContext *ro_net_host_ctx{get_internal_ro_net_ctx(ctx)};
  delete ro_net_host_ctx;
}

void ROBackend::reset_backend_stats() {
  auto *bp{backend_proxy.get()};

  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    bp->profiler[i].resetStats();
  }
}

void ROBackend::dump_backend_stats() {
  uint64_t total{0};
  for (int i = 0; i < NUM_STATS; i++) {
    total += globalStats.getStat(i);
  }

  int device_id;
  hipDeviceProp_t device_props;
  CHECK_HIP(hipGetDevice(&device_id));
  int wallClockMhz;
  CHECK_HIP(hipDeviceGetAttribute(&wallClockMhz, hipDeviceAttributeWallClockRate, device_id));
  int gpu_frequency_mhz{wallClockMhz};

  uint64_t us_wait_slot{0};
  uint64_t us_pack{0};
  uint64_t us_fence1{0};
  uint64_t us_fence2{0};
  uint64_t us_wait_host{0};

  auto *bp{backend_proxy.get()};

  for (size_t i = 0; i < envvar::max_num_contexts; i++) {
    // Average latency as perceived from a thread
    const ROStats &prof{bp->profiler[i]};
    us_wait_slot += prof.getStat(WAITING_ON_SLOT) / gpu_frequency_mhz;
    us_pack += prof.getStat(PACK_QUEUE) / gpu_frequency_mhz;
    us_fence1 += prof.getStat(THREAD_FENCE_1) / gpu_frequency_mhz;
    us_fence2 += prof.getStat(THREAD_FENCE_2) / gpu_frequency_mhz;
    us_wait_host += prof.getStat(WAITING_ON_HOST) / gpu_frequency_mhz;
  }

  constexpr int FIELD_WIDTH{20};
  constexpr int FLOAT_PRECISION{2};

  printf("%*s%*s%*s%*s%*s\n", FIELD_WIDTH + 1, "Wait On Slot (us)",
         FIELD_WIDTH + 1, "Pack Queue (us)", FIELD_WIDTH + 1, "Fence 1 (us)",
         FIELD_WIDTH + 1, "Fence 2 (us)", FIELD_WIDTH + 1, "Wait Host (us)");

  printf("%*.*f %*.*f %*.*f %*.*f %*.*f\n\n", FIELD_WIDTH, FLOAT_PRECISION,
         static_cast<double>(us_wait_slot) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_pack) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_fence1) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_fence2) / total, FIELD_WIDTH,
         FLOAT_PRECISION, static_cast<double>(us_wait_host) / total);
}

void ROBackend::ro_net_free_runtime() {
  /*
   * Validate that a handle was passed that is not a nullptr.
   */
  auto *bp{backend_proxy.get()};
  assert(bp);

  /*
   * Set this flag to denote that the runtime is being torn down.
   */
  bp->worker_thread_exit = true;

  /*
   * Tear down the worker threads.
   */
  worker_thread.join();

  /*
   * Tear down the transport object.
   */
  while (!transport_->readyForFinalize()) {
  }
  transport_->finalizeTransport();

  ro_window_proxy_->~WindowProxyT();
  team_world_proxy_->~ROTeamProxy<HIPAllocator>();
  transport_->~MPITransport();
  /*
   * Free the profiler statistics structure.
   */
  // CHECK_HIP(hipFree(bp->profiler));

  /*
   * Tear down team_world
   */
  auto *team_world{team_tracker.get_team_world()};
  team_world->~Team();
  // CHECK_HIP(hipFree(team_world));

  /*
   * Free the gpu_handle.
   */
  // CHECK_HIP(hipHostFree(bp));
}

void ROBackend::ro_net_poll() {
  auto *bp{backend_proxy.get()};
  while (!bp->worker_thread_exit) {
    for (size_t i{0}; i < poll_block_count_; i++) {
      int16_t request_count{0};
      const int16_t max_count{64};
      bool processed_req{true};
      while (processed_req && (request_count < max_count)) {
        processed_req = queue_.process(i, transport_);
        request_count++;
      }
    }
  }
}

void ROBackend::initIPC() {
  const auto &heap_bases{heap.get_heap_bases()};

  ipcImpl.ipcHostInit(transport_->getMyPe(), heap_bases,
                      transport_->get_world_comm());
}

void ROBackend::global_exit(int status) { transport_->global_exit(status); }

}  // namespace rocshmem
