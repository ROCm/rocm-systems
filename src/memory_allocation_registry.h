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

#ifndef LIBROCDXG_MEMORY_ALLOCATION_REGISTRY_H_INCLUDED
#define LIBROCDXG_MEMORY_ALLOCATION_REGISTRY_H_INCLUDED

#include <cstddef>
#include <cstdint>
#include <map>
#include <mutex>

#include "hsakmt/hsakmt.h"
#include "impl/wddm/gpu_memory.h"

namespace rocdxg {

struct Allocation {
  Allocation()
      : handle(0), cpu_addr(0), gpu_addr(0), size(0), userptr(false),
        user_data(nullptr), size_requested(0), node_id(0), mem_flags_value(0),
        dmabuf_fd(-1), rocr_userdata(nullptr) {}

  Allocation(wsl::thunk::GpuMemoryHandle handle_arg, void *cpu_addr_arg,
             uint64_t gpu_addr_arg, size_t size_arg, bool userptr_arg = false,
             void *user_data_arg = nullptr, size_t user_size_arg = 0,
             HSAuint32 node_id_arg = 0, HSAuint32 mem_flags_value_arg = 0)
      : handle(handle_arg), cpu_addr(cpu_addr_arg), gpu_addr(gpu_addr_arg),
        size(size_arg), userptr(userptr_arg), user_data(user_data_arg),
        size_requested(user_size_arg), node_id(node_id_arg),
        mem_flags_value(mem_flags_value_arg), dmabuf_fd(-1),
        rocr_userdata(nullptr) {}

  wsl::thunk::GpuMemoryHandle handle;
  void *cpu_addr;
  uint64_t gpu_addr;
  bool userptr;
  size_t size; /* actual size = align_up(size_requested, granularity) */
  void *user_data;
  size_t size_requested; /* size requested by user */
  HSAuint32 node_id;
  HSAuint32 mem_flags_value;
  int dmabuf_fd;
  void *rocr_userdata;
};

class AllocationRegistry {
public:
  std::mutex &Mutex() { return mutex_; }

  void Clear();
  void Insert(const void *address, const Allocation &allocation);
  bool Erase(const void *address);
  Allocation *Find(const void *address);
  Allocation *FindPreceding(const void *address);
  Allocation *FindContaining(const void *address);

private:
  using AllocationMap = std::map<const void *, Allocation>;

  AllocationMap allocations_;
  std::mutex mutex_;
};

} // namespace rocdxg

#endif
