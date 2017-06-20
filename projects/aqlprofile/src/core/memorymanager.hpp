// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once
#include <vector>
#include <atomic>
#include <mutex>
#include <unordered_map>
#include <memory>
#include "include/aql_profile_v2.h"
#include <stdexcept>
#include "pm4/trace_config.h"

struct EventRequest : public aqlprofile_pmc_event_t {
  bool bInternal;

  auto GetOrder() const -> auto{
    uint64_t idx = bInternal ? 0 : 1;
    idx |= uint64_t(flags.raw) << 1;
    idx |= uint64_t(event_id) << 33;

    uint64_t blk = block_index;
    blk |= uint64_t(block_name) << 32;

    return std::pair<uint64_t, uint64_t>{blk, idx};
  }

  bool operator<(const EventRequest& other) const {
    auto idx1 = this->GetOrder();
    auto idx2 = other.GetOrder();
    if (idx1.first == idx2.first)
      return idx1.second < idx2.second;
    else
      return idx1.first < idx2.first;
  }

  bool operator==(const EventRequest& other) const {
    auto idx1 = this->GetOrder();
    auto idx2 = other.GetOrder();
    return idx1.second == idx2.second && idx1.first == idx2.first;
  }

  bool IsSameNoFlags(const EventRequest& other) const {
    auto idx1 = this->GetOrder();
    auto idx2 = other.GetOrder();
    return idx1.first == idx2.first && event_id == other.event_id;
  }
};

class MemoryManager {
 public:
  MemoryManager(hsa_agent_t agent, aqlprofile_memory_alloc_callback_t alloc,
                aqlprofile_memory_dealloc_callback_t dealloc, void* data)
      : agent(agent),
        alloc_cb(alloc),
        dealloc_cb(dealloc),
        userdata(data),
        handle(HANDLE_COUNTER.fetch_add(1)) {}

  MemoryManager(aqlprofile_agent_handle_t agent, aqlprofile_memory_alloc_callback_t alloc,
                aqlprofile_memory_dealloc_callback_t dealloc, void* data)
      : agent_handle(agent),
        alloc_cb(alloc),
        dealloc_cb(dealloc),
        userdata(data),
        handle(HANDLE_COUNTER.fetch_add(1)) {}

  virtual ~MemoryManager() {}

  void CheckStatus(hsa_status_t status) const {
    if (status != HSA_STATUS_SUCCESS) throw status;
  }

  void* GetCmdBuf() const { return cmdbuf.get(); }
  void* GetOutputBuf() const { return outputbuf.get(); }

  size_t GetOutputBufSize() const { return outputbuf_size; }

  size_t GetHandler() const { return handle; }
  hsa_agent_t GetAgent() const { return agent; }
  aqlprofile_agent_handle_t AgentHandle() const { return agent_handle; }

  void CreateCmdBuf(size_t size) {
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = true;
    flags.device_access = true;
    flags.memory_hint = AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT;
    cmdbuf = AllocMemory(size, flags);
  }

  virtual void CreateOutputBuf(size_t size) = 0;

  static void RegisterManager(const std::shared_ptr<MemoryManager>& shared) {
    std::lock_guard<std::mutex> lk(managers_map_mutex);
    managers[shared->handle] = shared;
  }

  static void DeleteManager(size_t handle) {
    std::lock_guard<std::mutex> lk(managers_map_mutex);
    managers.erase(handle);
  }

  static std::shared_ptr<MemoryManager> GetManager(size_t handle) {
    std::lock_guard<std::mutex> lk(managers_map_mutex);
    try {
      return managers.at(handle);
    } catch (std::exception& e) {
      return nullptr;
    }
  }

 protected:
  struct MemoryDeleter {
    aqlprofile_memory_dealloc_callback_t free_fn;
    void* userdata;
    void operator()(void* ptr) const {
      if (ptr && free_fn) free_fn(ptr, userdata);
    };
  };

  std::unique_ptr<void, MemoryDeleter> AllocMemory(size_t size,
                                                   aqlprofile_buffer_desc_flags_t flags) const {
    void* ptr;
    CheckStatus(alloc_cb(&ptr, size, flags, userdata));
    return std::unique_ptr<void, MemoryDeleter>{ptr, MemoryDeleter{dealloc_cb, userdata}};
  }

  aqlprofile_agent_handle_t agent_handle = {.handle = 0};
  hsa_agent_t agent = {.handle = 0};
  std::unique_ptr<void, MemoryDeleter> cmdbuf = nullptr;
  std::unique_ptr<void, MemoryDeleter> outputbuf = nullptr;
  size_t outputbuf_size = 0;

  void* const userdata;
  aqlprofile_memory_alloc_callback_t const alloc_cb;
  aqlprofile_memory_dealloc_callback_t const dealloc_cb;
  size_t handle;

  static std::atomic<size_t> HANDLE_COUNTER;
  static std::unordered_map<size_t, std::shared_ptr<MemoryManager>> managers;
  static std::mutex managers_map_mutex;
};

class CounterMemoryManager : public MemoryManager {
 public:
  CounterMemoryManager(hsa_agent_t agent, aqlprofile_memory_alloc_callback_t alloc,
                       aqlprofile_memory_dealloc_callback_t dealloc, void* data)
      : MemoryManager(agent, alloc, dealloc, data) {}

  CounterMemoryManager(aqlprofile_agent_handle_t agent, aqlprofile_memory_alloc_callback_t alloc,
                       aqlprofile_memory_dealloc_callback_t dealloc, void* data)
      : MemoryManager(agent, alloc, dealloc, data) {}

  void CreateOutputBuf(size_t size) override {
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = flags.device_access = true;
    flags.memory_hint = AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED;
    outputbuf = AllocMemory(size, flags);
    outputbuf_size = size;
  }

  std::vector<EventRequest>& GetEvents() { return events; }
  void CopyEvents(const aqlprofile_pmc_event_t* events, size_t count);

 protected:
  std::vector<EventRequest> events;
};

class TraceMemoryManager : public MemoryManager {
 public:
  TraceMemoryManager(hsa_agent_t agent, aqlprofile_memory_alloc_callback_t alloc,
                     aqlprofile_memory_dealloc_callback_t dealloc,
                     aqlprofile_memory_copy_t _copy_fn, void* data)
      : MemoryManager(agent, alloc, dealloc, data), copy_fn(_copy_fn) {}

  TraceMemoryManager(aqlprofile_agent_handle_t agent, aqlprofile_memory_alloc_callback_t alloc,
                     aqlprofile_memory_dealloc_callback_t dealloc, void* data)
      : MemoryManager(agent, alloc, dealloc, data) {}

  void CreateOutputBuf(size_t size) override {
    aqlprofile_buffer_desc_flags_t flags{};
    flags.device_access = true;
    flags.memory_hint = AQLPROFILE_MEMORY_HINT_DEVICE_NONCOHERENT;
    outputbuf = AllocMemory(size, flags);
    outputbuf_size = size;
  }

  void CreateTraceControlBuf(size_t size) {
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = flags.device_access = true;
    flags.memory_hint = AQLPROFILE_MEMORY_HINT_HOST;
    trace_control_buf = AllocMemory(size, flags);
  }

  const std::vector<hsa_ven_amd_aqlprofile_parameter_t>& GetATTParams() const { return att_params; }
  void CopyATTParams(hsa_ven_amd_aqlprofile_parameter_t* params, size_t count) {
    for (size_t i = 0; i < count; i++) this->att_params.push_back(params[i]);
    for (auto& param : att_params) {
      if (param.parameter_name == HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET)
        target_cu = param.value;
      else if (param.parameter_name == HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION)
        simd_mask = param.value;
    }
  }

  template <typename Type>
  Type* GetTraceControlBuf() const {
    return reinterpret_cast<Type*>(trace_control_buf.get());
  }

  void CopyMemory(void* dst, const void* src, size_t size) {
    this->copy_fn(dst, src, size, this->userdata);
  }

  int GetSimdMask() const { return simd_mask; }

  pm4_builder::TraceConfig config{};

 protected:
  int target_cu = -1;
  int simd_mask = 0xF;
  aqlprofile_memory_copy_t copy_fn;
  std::vector<hsa_ven_amd_aqlprofile_parameter_t> att_params;
  std::unique_ptr<void, MemoryDeleter> trace_control_buf = nullptr;
};

class CodeobjMemoryManager : public MemoryManager {
 public:
  CodeobjMemoryManager(hsa_agent_t agent, aqlprofile_memory_alloc_callback_t alloc,
                       aqlprofile_memory_dealloc_callback_t dealloc, size_t size, void* data)
      : MemoryManager(agent, alloc, dealloc, data) {
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = flags.device_access = true;
    this->cmd_buffer = AllocMemory(size, flags);
  }

  void CreateOutputBuf(size_t size) override{};
  std::unique_ptr<void, MemoryDeleter> cmd_buffer;
};
