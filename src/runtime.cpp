/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "runtime.h"

#include <unistd.h>

namespace rocdxg {

Runtime &Runtime::Instance() {
  static Runtime *runtime = new Runtime();
  return *runtime;
}

Runtime::Runtime()
    : hsakmt_mutex(PTHREAD_MUTEX_INITIALIZER) {
  ResetState();
}

void Runtime::ResetState() {
  dxg_fd = -1;
  parent_pid = getpid();
  is_forked = false;
  hsakmt_debug_level = HSAKMT_DEBUG_LEVEL_DEFAULT;
  dxg_open_count = 0;
  hsakmt_is_dgpu = false;
  is_svm_api_supported = false;
  zfb_support = 0;
  vendor_packet_process = 0;
  check_avail_sysram = false;
  max_single_alloc_size = 0;
  enable_thunk_sub_allocator = 0;
  local_heap_space_start_ = 0;
  local_heap_space_size_ = 0;
  local_heap_mgr_.reset();
  system_heap_space_start_ = 0;
  system_heap_space_size_ = 0;
  system_heap_mgr_.reset();
  handle_aperture_start_ = 0;
  handle_aperture_size_ = 0;
  handle_aperture_mgr_.reset();
  default_node = 1;
}

void Runtime::Reset() {
  if (dxg_fd >= 0) {
    close(dxg_fd);
  }
  ResetState();
}

void Runtime::ResetAfterFork() {
  Reset();
}

} // namespace rocdxg
