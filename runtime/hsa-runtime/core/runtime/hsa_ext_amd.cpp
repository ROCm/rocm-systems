////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#include <algorithm>
#include <exception>
#include <map>
#include <memory>
#include <new>
#include <set>
#include <typeinfo>
#include <utility>
#include <vector>

#include "core/inc/agent.h"
#include "core/inc/amd_aie_agent.h"
#include "core/inc/amd_cpu_agent.h"
#include "core/inc/amd_gpu_agent.h"
#include "core/inc/amd_memory_region.h"
#include "core/inc/default_signal.h"
#include "core/inc/exceptions.h"
#include "core/inc/intercept_queue.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/ipc_signal.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"

namespace rocr {

template <class T>
struct ValidityError;
template <>
struct ValidityError<core::Signal*> {
  enum { value = HSA_STATUS_ERROR_INVALID_SIGNAL };
};

template <>
struct ValidityError<core::Agent*> {
  enum { value = HSA_STATUS_ERROR_INVALID_AGENT };
};

template <>
struct ValidityError<core::MemoryRegion*> {
  enum { value = HSA_STATUS_ERROR_INVALID_REGION };
};

template <>
struct ValidityError<AMD::MemoryRegion*> {
  enum { value = HSA_STATUS_ERROR_INVALID_REGION };
};

template <>
struct ValidityError<core::Queue*> {
  enum { value = HSA_STATUS_ERROR_INVALID_QUEUE };
};

template <class T>
struct ValidityError<const T*> {
  enum { value = ValidityError<T*>::value };
};

#define IS_TRUE(var)                                                                               \
  do {                                                                                             \
    if ((var) != true) return HSA_STATUS_ERROR_INVALID_ARGUMENT;                                   \
  } while (false)

#define IS_BAD_PTR(ptr)                                          \
  do {                                                           \
    if ((ptr) == NULL) return HSA_STATUS_ERROR_INVALID_ARGUMENT; \
  } while (false)

#define IS_ZERO(arg)                                                                               \
  do {                                                                                             \
    if ((arg) == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;                                      \
  } while (false)

#define IS_VALID_FD(fd)                                                                            \
  do {                                                                                             \
    if ((fd) < 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;                                        \
  } while (false)

#define IS_VALID(ptr)                                           \
  do {                                                          \
    if ((ptr) == NULL || !(ptr)->IsValid())                     \
      return hsa_status_t(ValidityError<decltype(ptr)>::value); \
  } while (false)

#define IS_NULL_OR_VALID(ptr)                                                                      \
  do {                                                                                             \
    if ((ptr) != NULL && !(ptr)->IsValid())                                                        \
      return hsa_status_t(ValidityError<decltype(ptr)>::value);                                    \
  } while (false)

#define CHECK_ALLOC(ptr)                                         \
  do {                                                           \
    if ((ptr) == NULL) return HSA_STATUS_ERROR_OUT_OF_RESOURCES; \
  } while (false)

#define IS_OPEN()                                     \
  do {                                                \
    if (!core::Runtime::runtime_singleton_->IsOpen()) \
      return HSA_STATUS_ERROR_NOT_INITIALIZED;        \
  } while (false)

template <class T>
static __forceinline bool IsValid(T* ptr) {
  return (ptr == NULL) ? NULL : ptr->IsValid();
}

#define TRY try {
#define CATCH } catch(...) { return AMD::handleException(); }
#define CATCHRET(RETURN_TYPE) } catch(...) { return AMD::handleExceptionT<RETURN_TYPE>(); }

namespace AMD {

hsa_status_t handleException() {
  try {
    throw;
  } catch (const std::bad_alloc& e) {
    debug_print("HSA exception: BadAlloc\n");
    return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  } catch (const hsa_exception& e) {
    ifdebug {
      if (!strIsEmpty(e.what())) debug_print("HSA exception: %s\n", e.what());
    }
    return e.error_code();
  } catch (const std::exception& e) {
    debug_print("Unhandled exception: %s\n", e.what());
    assert(false && "Unhandled exception.");
    return HSA_STATUS_ERROR;
  } catch (const std::nested_exception& e) {
    debug_print("Callback threw, forwarding.\n");
    e.rethrow_nested();
    return HSA_STATUS_ERROR;
  } catch (...) {
    assert(false && "Unhandled exception.");
    abort();
    return HSA_STATUS_ERROR;
  }
}

template <class T> static __forceinline T handleExceptionT() {
  handleException();
  abort();
  return T();
}

hsa_status_t hsa_amd_coherency_get_type(hsa_agent_t agent_handle, hsa_amd_coherency_type_t* type) {
  TRY;
  IS_OPEN();

  const core::Agent* agent = core::Agent::Convert(agent_handle);

  IS_VALID(agent);

  IS_BAD_PTR(type);

  if (agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  const AMD::GpuAgentInt* gpu_agent =
      static_cast<const AMD::GpuAgentInt*>(agent);

  *type = gpu_agent->current_coherency_type();

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_coherency_set_type(hsa_agent_t agent_handle,
                                        hsa_amd_coherency_type_t type) {
  TRY;
  IS_OPEN();

  core::Agent* agent = core::Agent::Convert(agent_handle);

  IS_VALID(agent);

  if (type < HSA_AMD_COHERENCY_TYPE_COHERENT ||
      type > HSA_AMD_COHERENCY_TYPE_NONCOHERENT) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  AMD::GpuAgent* gpu_agent = static_cast<AMD::GpuAgent*>(agent);

  if (!gpu_agent->current_coherency_type(type)) {
    return HSA_STATUS_ERROR;
  }

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_memory_fill(void* ptr, uint32_t value, size_t count) {
  TRY;
  IS_OPEN();

  if ((ptr == nullptr) || (uintptr_t(ptr) % 4 != 0)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (count == 0) {
    return HSA_STATUS_SUCCESS;
  }

  return core::Runtime::runtime_singleton_->FillMemory(ptr, value, count);
  CATCH;
}

hsa_status_t hsa_amd_memory_async_copy(void* dst, hsa_agent_t dst_agent_handle, const void* src,
                                       hsa_agent_t src_agent_handle, size_t size,
                                       uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal) {
  TRY;
  IS_BAD_PTR(dst);
  IS_BAD_PTR(src);

  if ((num_dep_signals == 0 && dep_signals != nullptr) ||
      (num_dep_signals > 0 && dep_signals == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  core::Agent* dst_agent = core::Agent::Convert(dst_agent_handle);
  IS_VALID(dst_agent);

  core::Agent* src_agent = core::Agent::Convert(src_agent_handle);
  IS_VALID(src_agent);

  std::vector<core::Signal*> dep_signal_list(num_dep_signals);
  if (num_dep_signals > 0) {
    for (size_t i = 0; i < num_dep_signals; ++i) {
      core::Signal* dep_signal_obj = core::Signal::Convert(dep_signals[i]);
      IS_VALID(dep_signal_obj);
      dep_signal_list[i] = dep_signal_obj;
    }
  }

  core::Signal* out_signal_obj = core::Signal::Convert(completion_signal);
  IS_VALID(out_signal_obj);

  bool rev_copy_dir = core::Runtime::runtime_singleton_->flag().rev_copy_dir();
  if (size > 0) {
    return core::Runtime::runtime_singleton_->CopyMemory(
        dst, (rev_copy_dir ? src_agent : dst_agent),
        src, (rev_copy_dir ? dst_agent : src_agent),
        size, dep_signal_list, *out_signal_obj);
  }

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_memory_async_copy_on_engine(void* dst, hsa_agent_t dst_agent_handle,
                                       const void* src, hsa_agent_t src_agent_handle, size_t size,
                                       uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                       hsa_signal_t completion_signal,
                                       hsa_amd_sdma_engine_id_t engine_id,
                                       bool force_copy_on_sdma) {
  TRY;
  IS_BAD_PTR(dst);
  IS_BAD_PTR(src);

  if ((num_dep_signals == 0 && dep_signals != nullptr) ||
      (num_dep_signals > 0 && dep_signals == nullptr)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  core::Agent* dst_agent = core::Agent::Convert(dst_agent_handle);
  IS_VALID(dst_agent);

  core::Agent* src_agent = core::Agent::Convert(src_agent_handle);
  IS_VALID(src_agent);

  std::vector<core::Signal*> dep_signal_list(num_dep_signals);
  if (num_dep_signals > 0) {
    for (size_t i = 0; i < num_dep_signals; ++i) {
      core::Signal* dep_signal_obj = core::Signal::Convert(dep_signals[i]);
      IS_VALID(dep_signal_obj);
      dep_signal_list[i] = dep_signal_obj;
    }
  }

  core::Signal* out_signal_obj = core::Signal::Convert(completion_signal);
  IS_VALID(out_signal_obj);

  bool rev_copy_dir = core::Runtime::runtime_singleton_->flag().rev_copy_dir();
  if (size > 0) {
    return core::Runtime::runtime_singleton_->CopyMemoryOnEngine(
        dst, (rev_copy_dir ? src_agent : dst_agent),
        src, (rev_copy_dir ? dst_agent : src_agent),
        size, dep_signal_list, *out_signal_obj, engine_id, force_copy_on_sdma);
  }

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_memory_copy_engine_status(hsa_agent_t dst_agent_handle,
                                               hsa_agent_t src_agent_handle,
                                               uint32_t *engine_ids_mask) {
  core::Agent* dst_agent = core::Agent::Convert(dst_agent_handle);
  IS_VALID(dst_agent);

  core::Agent* src_agent = core::Agent::Convert(src_agent_handle);
  IS_VALID(src_agent);

  return core::Runtime::runtime_singleton_->CopyMemoryStatus(dst_agent, src_agent,
                                                             engine_ids_mask);
}

hsa_status_t hsa_amd_memory_get_preferred_copy_engine(hsa_agent_t dst_agent_handle,
                                                      hsa_agent_t src_agent_handle,
                                                      uint32_t* recommended_ids_mask) {
  core::Agent* dst_agent = core::Agent::Convert(dst_agent_handle);
  IS_VALID(dst_agent);

  core::Agent* src_agent = core::Agent::Convert(src_agent_handle);
  IS_VALID(src_agent);

  return core::Runtime::runtime_singleton_->GetPreferredEngine(dst_agent, src_agent,
                                                               recommended_ids_mask);
}

hsa_status_t hsa_amd_memory_async_copy_rect(
    const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset, const hsa_pitched_ptr_t* src,
    const hsa_dim3_t* src_offset, const hsa_dim3_t* range, hsa_agent_t copy_agent,
    hsa_amd_copy_direction_t dir, uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
    hsa_signal_t completion_signal) {
  TRY;
  if (dst == nullptr || src == nullptr || dst_offset == nullptr || src_offset == nullptr ||
      range == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if ((num_dep_signals == 0 && dep_signals != NULL) ||
      (num_dep_signals > 0 && dep_signals == NULL)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (dir == hsaHostToHost) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  core::Agent* base_agent = core::Agent::Convert(copy_agent);
  IS_VALID(base_agent);
  if (base_agent->device_type() != core::Agent::DeviceType::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;
  AMD::GpuAgent* agent = static_cast<AMD::GpuAgent*>(base_agent);

  std::vector<core::Signal*> dep_signal_list(num_dep_signals);
  if (num_dep_signals > 0) {
    for (size_t i = 0; i < num_dep_signals; ++i) {
      core::Signal* dep_signal_obj = core::Signal::Convert(dep_signals[i]);
      IS_VALID(dep_signal_obj);
      dep_signal_list[i] = dep_signal_obj;
    }
  }

  core::Signal* out_signal_obj = core::Signal::Convert(completion_signal);
  IS_VALID(out_signal_obj);

  if ((range->x != 0) && (range->y != 0) && (range->z != 0)) {
    return agent->DmaCopyRect(dst, dst_offset, src, src_offset, range, dir, dep_signal_list,
                              *out_signal_obj);
  }

  return HSA_STATUS_SUCCESS;
  CATCH;
}


hsa_status_t hsa_amd_profiling_set_profiler_enabled(hsa_queue_t* queue, int enable) {
  TRY;
  IS_OPEN();

  core::Queue* cmd_queue = core::Queue::Convert(queue);
  IS_VALID(cmd_queue);

  cmd_queue->SetProfiling(enable);

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_profiling_async_copy_enable(bool enable) {
  TRY;
  IS_OPEN();

  hsa_status_t ret = HSA_STATUS_SUCCESS;
  for (core::Agent* agent : core::Runtime::runtime_singleton_->gpu_agents()) {
    hsa_status_t err = agent->profiling_enabled(enable);
    if (err != HSA_STATUS_SUCCESS) ret = err;
  }

  for (core::Agent* agent : core::Runtime::runtime_singleton_->cpu_agents()) {
    hsa_status_t err = agent->profiling_enabled(enable);
    if (err != HSA_STATUS_SUCCESS) ret = err;
  }
  return ret;

  CATCH;
}

hsa_status_t hsa_amd_profiling_get_dispatch_time(
    hsa_agent_t agent_handle, hsa_signal_t hsa_signal,
    hsa_amd_profiling_dispatch_time_t* time) {
  TRY;
  IS_OPEN();

  IS_BAD_PTR(time);

  core::Agent* agent = core::Agent::Convert(agent_handle);

  IS_VALID(agent);

  core::Signal* signal = core::Signal::Convert(hsa_signal);

  IS_VALID(signal);

  if (agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(agent);

  // Translate timestamp from GPU to system domain.
  gpu_agent->TranslateTime(signal, *time);

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_profiling_get_async_copy_time(
    hsa_signal_t hsa_signal, hsa_amd_profiling_async_copy_time_t* time) {
  TRY;
  IS_OPEN();

  IS_BAD_PTR(time);

  core::Signal* signal = core::Signal::Convert(hsa_signal);

  IS_VALID(signal);

  core::Agent* agent = signal->async_copy_agent();

  if (agent == nullptr) {
    return HSA_STATUS_ERROR;
  }

  if (agent->device_type() == core::Agent::DeviceType::kAmdGpuDevice) {
    // Translate timestamp from GPU to system domain.
    static_cast<AMD::GpuAgentInt*>(agent)->TranslateTime(signal, *time);
    return HSA_STATUS_SUCCESS;
  }

  // The timestamp is already in system domain.
  time->start = signal->signal_.start_ts;
  time->end = signal->signal_.end_ts;
  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_profiling_convert_tick_to_system_domain(hsa_agent_t agent_handle,
                                                             uint64_t agent_tick,
                                                             uint64_t* system_tick) {
  TRY;
  IS_OPEN();

  IS_BAD_PTR(system_tick);

  core::Agent* agent = core::Agent::Convert(agent_handle);

  IS_VALID(agent);

  if (agent->device_type() != core::Agent::kAmdGpuDevice) {
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(agent);

  *system_tick = gpu_agent->TranslateTime(agent_tick);

  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_signal_create(hsa_signal_value_t initial_value, uint32_t num_consumers,
                                   const hsa_agent_t* consumers, uint64_t attributes,
                                   hsa_signal_t* hsa_signal) {
  struct AgentHandleCompare {
    bool operator()(const hsa_agent_t& lhs, const hsa_agent_t& rhs) const {
      return lhs.handle < rhs.handle;
    }
  };

  TRY;
  IS_OPEN();
  IS_BAD_PTR(hsa_signal);

  core::Signal* ret;

  bool enable_ipc = attributes & HSA_AMD_SIGNAL_IPC;
  bool use_default =
      enable_ipc || (attributes & HSA_AMD_SIGNAL_AMD_GPU_ONLY) || (!core::g_use_interrupt_wait);

  if ((!use_default) && (num_consumers != 0)) {
    IS_BAD_PTR(consumers);

    // Check for duplicates in consumers.
    std::set<hsa_agent_t, AgentHandleCompare> consumer_set(consumers, consumers + num_consumers);
    if (consumer_set.size() != num_consumers) {
      return HSA_STATUS_ERROR_INVALID_ARGUMENT;
    }

    use_default = true;
    for (const core::Agent* cpu_agent : core::Runtime::runtime_singleton_->cpu_agents()) {
      use_default &= (consumer_set.find(cpu_agent->public_handle()) == consumer_set.end());
    }
  }

  if (use_default) {
    ret = new core::DefaultSignal(initial_value, enable_ipc);
  } else {
    ret = new core::InterruptSignal(initial_value);
  }

  *hsa_signal = core::Signal::Convert(ret);
  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_signal_value_pointer(hsa_signal_t hsa_signal,
                                          volatile hsa_signal_value_t** value_ptr) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(value_ptr);
  core::Signal* signal = core::Signal::Convert(hsa_signal);
  IS_VALID(signal);

  if(!core::BusyWaitSignal::IsType(signal))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  *value_ptr = (volatile hsa_signal_value_t*)&signal->signal_.value;
  return HSA_STATUS_SUCCESS;

  CATCH;
}

uint32_t hsa_amd_signal_wait_all(uint32_t signal_count, hsa_signal_t* hsa_signals,
                                 hsa_signal_condition_t* conds, hsa_signal_value_t* values,
                                 uint64_t timeout_hint, hsa_wait_state_t wait_hint,
                                 hsa_signal_value_t* satisfying_values) {
  TRY;
  if (!core::Runtime::runtime_singleton_->IsOpen()) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_NOT_INITIALIZED, "hsa_amd_signal_wait_all called while not initialized");
  }

  // Treat NULL and invalid signals as already satisfied their condition and skip them
  std::vector<hsa_signal_t> valid_signals;
  std::vector<uint32_t> valid_signal_ids;
  for (uint32_t i = 0; i < signal_count; i++){
    if (hsa_signals[i].handle != 0 && core::SharedSignal::Convert(hsa_signals[i])->IsValid()){
      valid_signals.emplace_back(hsa_signals[i]);
      valid_signal_ids.emplace_back(i);
    }
  }

  // Return if there's no valid signal to wait on
  if (valid_signals.empty()){
    if (satisfying_values) {
      // Set 0 as satisfying value for NULL and invalid signals
      std::fill(satisfying_values, satisfying_values + signal_count, 0);
    }
    return uint32_t(0);
  }

  uint32_t valid_signal_count = valid_signals.size();

  std::vector<hsa_signal_value_t> satisfying_values_vec(valid_signal_count);
  uint32_t first_satysifying_signal_idx =
      core::Signal::WaitMultiple(valid_signal_count, valid_signals.data(), conds, values, timeout_hint, wait_hint,
                                 satisfying_values_vec, true);

  if (satisfying_values) {
    // Set 0 as satisfying value for NULL and invalid signals
    std::vector<hsa_signal_value_t> satisfying_values_vec_result(signal_count, 0);
    for (uint32_t i = 0; i < valid_signal_count; i++){
      satisfying_values_vec_result[valid_signal_ids[i]] = satisfying_values_vec[i];
    }
    std::copy(satisfying_values_vec_result.begin(), satisfying_values_vec_result.end(), satisfying_values);
  }

  return first_satysifying_signal_idx;
  CATCHRET(uint32_t);
}

uint32_t hsa_amd_signal_wait_any(uint32_t signal_count, hsa_signal_t* hsa_signals,
                                 hsa_signal_condition_t* conds, hsa_signal_value_t* values,
                                 uint64_t timeout_hint, hsa_wait_state_t wait_hint,
                                 hsa_signal_value_t* satisfying_value) {
  TRY;
  if (!core::Runtime::runtime_singleton_->IsOpen()) {
    throw AMD::hsa_exception(HSA_STATUS_ERROR_NOT_INITIALIZED, "hsa_amd_signal_wait_any called while not initialized");
  }

  // Ignore NULL and invalid signals
  std::vector<hsa_signal_t> valid_signals;
  std::vector<uint32_t> valid_signal_ids;
  for (uint32_t i = 0; i < signal_count; i++){
    if (hsa_signals[i].handle != 0 && core::SharedSignal::Convert(hsa_signals[i])->IsValid()){
      valid_signals.emplace_back(hsa_signals[i]);
      valid_signal_ids.emplace_back(i);
    }
  }

  // Return if there's no valid signal to wait on
  // satisfying_value is ignored
  if (valid_signals.empty()){
    return std::numeric_limits<uint32_t>::max();
  }

  std::vector<hsa_signal_value_t> satisfying_value_vec(1);
  uint32_t satisfying_signal_idx =
      core::Signal::WaitMultiple(valid_signals.size(), valid_signals.data(), conds, values, timeout_hint, wait_hint,
                                 satisfying_value_vec, false);

  //  Map back the index
  satisfying_signal_idx = valid_signal_ids[satisfying_signal_idx];

  if (satisfying_value) *satisfying_value = satisfying_value_vec.at(0);

  return satisfying_signal_idx;
  CATCHRET(uint32_t);
}

hsa_status_t hsa_amd_signal_async_handler(hsa_signal_t hsa_signal, hsa_signal_condition_t cond,
                                          hsa_signal_value_t value, hsa_amd_signal_handler handler,
                                          void* arg) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(handler);

  core::Signal* signal = core::Signal::Convert(hsa_signal);
  IS_VALID(signal);

  if ((core::g_use_interrupt_wait && (!core::InterruptSignal::IsType(signal)) &&
      !core::IPCSignal::IsType(signal)))
    return HSA_STATUS_ERROR_INVALID_SIGNAL;
  return core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
      hsa_signal, cond, value, handler, arg);
  CATCH;
}

hsa_status_t hsa_amd_async_function(void (*callback)(void* arg), void* arg) {
  TRY;
  IS_OPEN();

  IS_BAD_PTR(callback);
  static const hsa_signal_t null_signal = {0};
  return core::Runtime::runtime_singleton_->SetAsyncSignalHandler(
      null_signal, HSA_SIGNAL_CONDITION_EQ, 0, (hsa_amd_signal_handler)callback,
      arg);
  CATCH;
}

hsa_status_t hsa_amd_queue_cu_set_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                       const uint32_t* cu_mask) {
  TRY;
  IS_OPEN();

  core::Queue* cmd_queue = core::Queue::Convert(queue);
  IS_VALID(cmd_queue);
  if (num_cu_mask_count != 0) IS_BAD_PTR(cu_mask);
  if (num_cu_mask_count % 32 != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return cmd_queue->SetCUMasking(num_cu_mask_count, cu_mask);
  CATCH;
}

hsa_status_t hsa_amd_queue_cu_get_mask(const hsa_queue_t* queue, uint32_t num_cu_mask_count,
                                       uint32_t* cu_mask) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(cu_mask);

  core::Queue* cmd_queue = core::Queue::Convert(queue);
  IS_VALID(cmd_queue);
  if ((num_cu_mask_count == 0) || (num_cu_mask_count % 32 != 0))
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  return cmd_queue->GetCUMasking(num_cu_mask_count, cu_mask);
  CATCH;
}

hsa_status_t hsa_amd_memory_lock(void* host_ptr, size_t size,
                                 hsa_agent_t* agents, int num_agent,
                                 void** agent_ptr) {
  TRY;
  IS_OPEN();

  if (size == 0 || host_ptr == nullptr || agent_ptr == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *agent_ptr = nullptr;

  if ((agents != nullptr && num_agent == 0) || (agents == nullptr && num_agent != 0)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  // Check for APU
  if (core::Runtime::runtime_singleton_->system_regions_coarse().size() == 0) {
    assert(core::Runtime::runtime_singleton_->system_regions_fine()[0]->full_profile() &&
           "Missing coarse grain host memory on dGPU system.");
    *agent_ptr = host_ptr;
    return HSA_STATUS_SUCCESS;
  }

  const AMD::MemoryRegion* system_region = static_cast<const AMD::MemoryRegion*>(
      core::Runtime::runtime_singleton_->system_regions_coarse()[0]);

  return system_region->Lock(num_agent, agents, host_ptr, size, agent_ptr);
  CATCH;
}

hsa_status_t hsa_amd_memory_lock_to_pool(void* host_ptr, size_t size, hsa_agent_t* agents,
                                         int num_agent, hsa_amd_memory_pool_t pool, uint32_t flags,
                                         void** agent_ptr) {
  TRY;
  IS_OPEN();

  if (size == 0 || host_ptr == nullptr || agent_ptr == nullptr || flags != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  *agent_ptr = nullptr;

  if ((agents != nullptr && num_agent == 0) || (agents == nullptr && num_agent != 0)) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_t region = {pool.handle};
  const AMD::MemoryRegion* mem_region = AMD::MemoryRegion::Convert(region);
  if (mem_region == nullptr) {
    return (hsa_status_t)HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  }
  if (mem_region->owner()->device_type() != core::Agent::kAmdCpuDevice)
    return (hsa_status_t)HSA_STATUS_ERROR_INVALID_MEMORY_POOL;

  return mem_region->Lock(num_agent, agents, host_ptr, size, agent_ptr);
  CATCH;
}

hsa_status_t hsa_amd_memory_unlock(void* host_ptr) {
  TRY;
  IS_OPEN();

  const AMD::MemoryRegion* system_region =
      reinterpret_cast<const AMD::MemoryRegion*>(
          core::Runtime::runtime_singleton_->system_regions_fine()[0]);

  return system_region->Unlock(host_ptr);
  CATCH;
}

hsa_status_t hsa_amd_memory_pool_get_info(hsa_amd_memory_pool_t memory_pool,
                                          hsa_amd_memory_pool_info_t attribute, void* value) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(value);

  hsa_region_t region = {memory_pool.handle};
  const AMD::MemoryRegion* mem_region = AMD::MemoryRegion::Convert(region);
  if (mem_region == NULL) {
    return (hsa_status_t)HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  }

  return mem_region->GetPoolInfo(attribute, value);
  CATCH;
}

hsa_status_t hsa_amd_agent_iterate_memory_pools(
    hsa_agent_t agent_handle,
    hsa_status_t (*callback)(hsa_amd_memory_pool_t memory_pool, void* data),
    void* data) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(callback);
  const core::Agent* agent = core::Agent::Convert(agent_handle);
  IS_VALID(agent);

  switch (agent->device_type()) {
  case core::Agent::kAmdCpuDevice:
    return reinterpret_cast<const AMD::CpuAgent *>(agent)->VisitRegion(
        false,
        reinterpret_cast<hsa_status_t (*)(hsa_region_t memory_pool,
                                          void *data)>(callback),
        data);
  case core::Agent::kAmdAieDevice:
    return reinterpret_cast<const AMD::AieAgent *>(agent)->VisitRegion(
        false,
        reinterpret_cast<hsa_status_t (*)(hsa_region_t memory_pool,
                                          void *data)>(callback),
        data);
  case core::Agent::kAmdGpuDevice:
    return reinterpret_cast<const AMD::GpuAgentInt *>(agent)->VisitRegion(
        false,
        reinterpret_cast<hsa_status_t (*)(hsa_region_t memory_pool,
                                          void *data)>(callback),
        data);
  default:
    return HSA_STATUS_ERROR_INVALID_AGENT;
  }

  CATCH;
}

hsa_status_t hsa_amd_memory_pool_allocate(hsa_amd_memory_pool_t memory_pool, size_t size,
                                          uint32_t flags, void** ptr) {
  TRY;
  IS_OPEN();

  if (size == 0 || ptr == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_t region = {memory_pool.handle};
  const core::MemoryRegion* mem_region = core::MemoryRegion::Convert(region);

  if (mem_region == NULL || !mem_region->IsValid()) {
    return (hsa_status_t)HSA_STATUS_ERROR_INVALID_MEMORY_POOL;
  }

  MemoryRegion::AllocateFlags alloc_flag = core::MemoryRegion::AllocateRestrict;

  if (flags & HSA_AMD_MEMORY_POOL_PCIE_FLAG)
    alloc_flag |= core::MemoryRegion::AllocatePCIeRW;

  if (flags & HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG)
    alloc_flag |= core::MemoryRegion::AllocateContiguous;

  if (flags & HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG)
    alloc_flag |= core::MemoryRegion::AllocateExecutable;

#ifdef SANITIZER_AMDGPU
  if (mem_region->owner()->device_type() == core::Agent::kAmdGpuDevice)
    alloc_flag |= core::MemoryRegion::AllocateAsan;
#endif

  return core::Runtime::runtime_singleton_->AllocateMemory(mem_region, size, alloc_flag, ptr);
  CATCH;
}

hsa_status_t hsa_amd_memory_pool_free(void* ptr) {
  return HSA::hsa_memory_free(ptr);
}

hsa_status_t hsa_amd_agents_allow_access(uint32_t num_agents, const hsa_agent_t* agents,
                                         const uint32_t* flags, const void* ptr) {
  TRY;
  IS_OPEN();

  if (num_agents == 0 || agents == NULL || flags != NULL || ptr == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return core::Runtime::runtime_singleton_->AllowAccess(num_agents, agents,
                                                        ptr);
  CATCH;
}

hsa_status_t hsa_amd_memory_pool_can_migrate(hsa_amd_memory_pool_t src_memory_pool,
                                             hsa_amd_memory_pool_t dst_memory_pool, bool* result) {
  TRY;
  IS_OPEN();

  if (result == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_t src_region_handle = {src_memory_pool.handle};
  const AMD::MemoryRegion* src_mem_region =
      AMD::MemoryRegion::Convert(src_region_handle);

  if (src_mem_region == NULL || !src_mem_region->IsValid()) {
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  }

  hsa_region_t dst_region_handle = {dst_memory_pool.handle};
  const AMD::MemoryRegion* dst_mem_region =
      AMD::MemoryRegion::Convert(dst_region_handle);

  if (dst_mem_region == NULL || !dst_mem_region->IsValid()) {
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  }

  return src_mem_region->CanMigrate(*dst_mem_region, *result);
  CATCH;
}

hsa_status_t hsa_amd_memory_migrate(const void* ptr,
                                    hsa_amd_memory_pool_t memory_pool,
                                    uint32_t flags) {
  TRY;
  IS_OPEN();

  if (ptr == NULL || flags != 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_region_t dst_region_handle = {memory_pool.handle};
  const AMD::MemoryRegion* dst_mem_region =
      AMD::MemoryRegion::Convert(dst_region_handle);

  if (dst_mem_region == NULL || !dst_mem_region->IsValid()) {
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  }

  return dst_mem_region->Migrate(flags, ptr);
  CATCH;
}

hsa_status_t hsa_amd_agent_memory_pool_get_info(
    hsa_agent_t agent_handle, hsa_amd_memory_pool_t memory_pool,
    hsa_amd_agent_memory_pool_info_t attribute, void* value) {
  TRY;
  IS_OPEN();

  if (value == NULL) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  const core::Agent* agent = core::Agent::Convert(agent_handle);
  IS_VALID(agent);

  hsa_region_t region_handle = {memory_pool.handle};
  const AMD::MemoryRegion* mem_region =
      AMD::MemoryRegion::Convert(region_handle);

  if (mem_region == NULL || !mem_region->IsValid()) {
    return static_cast<hsa_status_t>(HSA_STATUS_ERROR_INVALID_MEMORY_POOL);
  }

  return mem_region->GetAgentPoolInfo(*agent, attribute, value);
  CATCH;
}

hsa_status_t hsa_amd_interop_map_buffer(uint32_t num_agents,
                                        hsa_agent_t* agents, int interop_handle,
                                        uint32_t flags, size_t* size,
                                        void** ptr, size_t* metadata_size,
                                        const void** metadata) {
  static const int tinyArraySize=8;
  TRY;
  IS_OPEN();
  IS_BAD_PTR(agents);
  IS_BAD_PTR(size);
  IS_BAD_PTR(ptr);
  if (flags != 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (num_agents == 0) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  core::Agent* short_agents[tinyArraySize];
  core::Agent** core_agents = short_agents;
  if (num_agents > tinyArraySize) {
    core_agents = new core::Agent* [num_agents];
    if (core_agents == nullptr) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  }

  MAKE_SCOPE_GUARD([&]() {
    if (num_agents > tinyArraySize) delete[] core_agents;
  });

  for (uint32_t i = 0; i < num_agents; i++) {
    core::Agent* device = core::Agent::Convert(agents[i]);
    IS_VALID(device);
    core_agents[i] = device;
  }

  auto ret = core::Runtime::runtime_singleton_->InteropMap(
      num_agents, core_agents, interop_handle, flags, size, ptr, metadata_size,
      metadata);

  return ret;
  CATCH;
}

hsa_status_t hsa_amd_interop_unmap_buffer(void* ptr) {
  TRY;
  IS_OPEN();
  if (ptr != NULL) core::Runtime::runtime_singleton_->InteropUnmap(ptr);
  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_pointer_info(const void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
                                  uint32_t* num_accessible, hsa_agent_t** accessible) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  IS_BAD_PTR(info);
  return core::Runtime::runtime_singleton_->PtrInfo(ptr, info, alloc, num_accessible, accessible);
  CATCH;
}

hsa_status_t hsa_amd_pointer_info_set_userdata(const void* ptr, void* userdata) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  return core::Runtime::runtime_singleton_->SetPtrInfoData(ptr, userdata);
  CATCH;
}

hsa_status_t hsa_amd_ipc_memory_create(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  IS_BAD_PTR(handle);
  return core::Runtime::runtime_singleton_->IPCCreate(ptr, len, handle);
  CATCH;
}

hsa_status_t hsa_amd_ipc_memory_attach(const hsa_amd_ipc_memory_t* ipc, size_t len,
                                       uint32_t num_agents, const hsa_agent_t* mapping_agents,
                                       void** mapped_ptr) {
  static const int tinyArraySize = 8;
  TRY;
  IS_OPEN();
  IS_BAD_PTR(mapped_ptr);
  if (num_agents != 0) IS_BAD_PTR(mapping_agents);

  core::Agent** core_agents = nullptr;
  if (num_agents > tinyArraySize)
    core_agents = new core::Agent*[num_agents];
  else
    core_agents = (core::Agent**)alloca(sizeof(core::Agent*) * num_agents);
  if (core_agents == NULL) return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
  MAKE_SCOPE_GUARD([&]() {
    if (num_agents > tinyArraySize) delete[] core_agents;
  });

  for (uint32_t i = 0; i < num_agents; i++) {
    core::Agent* device = core::Agent::Convert(mapping_agents[i]);
    IS_VALID(device);
    core_agents[i] = device;
  }

  return core::Runtime::runtime_singleton_->IPCAttach(ipc, len, num_agents, core_agents,
                                                      mapped_ptr);
  CATCH;
}

hsa_status_t hsa_amd_ipc_memory_detach(void* mapped_ptr) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(mapped_ptr);
  return core::Runtime::runtime_singleton_->IPCDetach(mapped_ptr);
  CATCH;
}

hsa_status_t hsa_amd_ipc_signal_create(hsa_signal_t hsa_signal, hsa_amd_ipc_signal_t* handle) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(handle);
  core::Signal* signal = core::Signal::Convert(hsa_signal);
  IS_VALID(signal);
  core::IPCSignal::CreateHandle(signal, handle);
  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_ipc_signal_attach(const hsa_amd_ipc_signal_t* handle,
                                       hsa_signal_t* hsa_signal) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(handle);
  IS_BAD_PTR(hsa_signal);
  core::Signal* signal = core::IPCSignal::Attach(handle);
  *hsa_signal = core::Signal::Convert(signal);
  return HSA_STATUS_SUCCESS;
  CATCH;
}

// For use by tools only - not in library export table.
hsa_status_t hsa_amd_queue_intercept_create(
    hsa_agent_t agent_handle, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data), void* data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t** queue) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(queue);

  // A wrapped queue for the intercept queue must have at least 3 slots so
  // there is space for a packet, a new retry barrier packet, and an existing
  // retry packet that is in the process of being processed.
  if (size < 3) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_queue_t* lower_queue;
  hsa_status_t err = HSA::hsa_queue_create(agent_handle, size, type, callback, data,
                                           private_segment_size, group_segment_size, &lower_queue);
  if (err != HSA_STATUS_SUCCESS) return err;
  std::unique_ptr<core::Queue> lowerQueue(core::Queue::Convert(lower_queue));

  std::unique_ptr<core::InterceptQueue> upperQueue(new core::InterceptQueue(std::move(lowerQueue)));

  *queue = core::Queue::Convert(upperQueue.release());
  return HSA_STATUS_SUCCESS;
  CATCH;
}

// For use by tools only - not in library export table.
hsa_status_t hsa_amd_queue_intercept_register(hsa_queue_t* queue,
                                              hsa_amd_queue_intercept_handler callback,
                                              void* user_data) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(callback);
  core::Queue* cmd_queue = core::Queue::Convert(queue);
  IS_VALID(cmd_queue);
  if (!core::InterceptQueue::IsType(cmd_queue)) return HSA_STATUS_ERROR_INVALID_QUEUE;
  core::InterceptQueue* iQueue = static_cast<core::InterceptQueue*>(cmd_queue);
  iQueue->AddInterceptor(callback, user_data);
  return HSA_STATUS_SUCCESS;
  CATCH;
}

hsa_status_t hsa_amd_register_system_event_handler(hsa_amd_system_event_callback_t callback,
                                                   void* data) {
  TRY;
  IS_OPEN();
  return core::Runtime::runtime_singleton_->SetCustomSystemEventHandler(callback, data);
  CATCH;
}

hsa_status_t hsa_amd_queue_set_priority(hsa_queue_t* queue,
                                                hsa_amd_queue_priority_t priority) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(queue);
  core::Queue* cmd_queue = core::Queue::Convert(queue);
  IS_VALID(cmd_queue);

  // Highest queue priority allowed for HSA user is HSA_QUEUE_PRIORITY_HIGH
  // HSA_QUEUE_PRIORITY_MAXIMUM is reserved for PC Sampling and can only be allocated internally
  // in ROCR
  static std::map<hsa_amd_queue_priority_t, HSA_QUEUE_PRIORITY> ext_kmt_priomap = {
      {HSA_AMD_QUEUE_PRIORITY_LOW, HSA_QUEUE_PRIORITY_MINIMUM},
      {HSA_AMD_QUEUE_PRIORITY_NORMAL, HSA_QUEUE_PRIORITY_NORMAL},
      {HSA_AMD_QUEUE_PRIORITY_HIGH, HSA_QUEUE_PRIORITY_HIGH},
  };

  auto priority_it = ext_kmt_priomap.find(priority);

  if (priority_it == ext_kmt_priomap.end()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  return cmd_queue->SetPriority(priority_it->second);
  CATCH;
}

hsa_status_t hsa_amd_register_deallocation_callback(void* ptr,
                                                    hsa_amd_deallocation_callback_t callback,
                                                    void* user_data) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  IS_BAD_PTR(callback);

  return core::Runtime::runtime_singleton_->RegisterReleaseNotifier(ptr, callback, user_data);

  CATCH;
}

hsa_status_t hsa_amd_deregister_deallocation_callback(void* ptr,
                                                      hsa_amd_deallocation_callback_t callback) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  IS_BAD_PTR(callback);

  return core::Runtime::runtime_singleton_->DeregisterReleaseNotifier(ptr, callback);

  CATCH;
}

// For use by tools only - not in library export table.
hsa_status_t hsa_amd_runtime_queue_create_register(hsa_amd_runtime_queue_notifier callback,
                                                   void* user_data) {
  TRY;
  IS_OPEN();
  return core::Runtime::runtime_singleton_->SetInternalQueueCreateNotifier(callback, user_data);
  CATCH;
}

hsa_status_t hsa_amd_svm_attributes_set(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count) {
  TRY;
  IS_OPEN();
  return core::Runtime::runtime_singleton_->SetSvmAttrib(ptr, size, attribute_list,
                                                         attribute_count);
  CATCH;
}

hsa_status_t hsa_amd_svm_attributes_get(void* ptr, size_t size,
                                        hsa_amd_svm_attribute_pair_t* attribute_list,
                                        size_t attribute_count) {
  TRY;
  IS_OPEN();
  return core::Runtime::runtime_singleton_->GetSvmAttrib(ptr, size, attribute_list,
                                                         attribute_count);
  CATCH;
}

hsa_status_t hsa_amd_svm_prefetch_async(void* ptr, size_t size, hsa_agent_t agent,
                                        uint32_t num_dep_signals, const hsa_signal_t* dep_signals,
                                        hsa_signal_t completion_signal) {
  TRY;
  IS_OPEN();
  // Validate inputs.
  // if (core::g_use_interrupt_wait && (!core::InterruptSignal::IsType(signal)))
  return core::Runtime::runtime_singleton_->SvmPrefetch(ptr, size, agent, num_dep_signals,
                                                        dep_signals, completion_signal);
  CATCH;
}

hsa_status_t hsa_amd_spm_acquire(hsa_agent_t preferred_agent) {
  TRY;
  IS_OPEN();
  const core::Agent* agent = core::Agent::Convert(preferred_agent);
  // Currently, the SPM API is only supported for GPU agents.
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return agent->driver().SPMAcquire(agent->node_id());

  CATCH;
}

hsa_status_t hsa_amd_spm_release(hsa_agent_t preferred_agent) {
  TRY;
  IS_OPEN();

  const core::Agent* agent = core::Agent::Convert(preferred_agent);
  // Currently, the SPM API is only supported for GPU agents.
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return agent->driver().SPMRelease(agent->node_id());

  CATCH;
}

hsa_status_t hsa_amd_spm_set_dest_buffer(hsa_agent_t preferred_agent, size_t size_in_bytes,
                                         uint32_t* timeout, uint32_t* size_copied, void* dest,
                                         bool* is_data_loss) {
  TRY;
  IS_OPEN();

  const core::Agent* agent = core::Agent::Convert(preferred_agent);
  // Currently, the SPM API is only supported for GPU agents.
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  return agent->driver().SPMSetDestBuffer(agent->node_id(), size_in_bytes, timeout, size_copied,
                                          dest, is_data_loss);
  CATCH;
}

hsa_status_t hsa_amd_portable_export_dmabuf(const void* ptr, size_t size, int* dmabuf,
                                            uint64_t* offset) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(ptr);
  IS_BAD_PTR(dmabuf);
  IS_BAD_PTR(offset);
  IS_ZERO(size);
  return core::Runtime::runtime_singleton_->DmaBufExport(ptr, size, dmabuf, offset);
  CATCH;
}

hsa_status_t hsa_amd_portable_close_dmabuf(int dmabuf) {
  TRY;
  return core::Runtime::runtime_singleton_->DmaBufClose(dmabuf);
  CATCH;
}

hsa_status_t hsa_amd_vmem_address_reserve(void** va, size_t size, uint64_t address,
                                          uint64_t flags) {
  TRY;
  IS_OPEN();
  IS_ZERO(size);
  IS_TRUE(core::Runtime::runtime_singleton_->VirtualMemApiSupported());
  return core::Runtime::runtime_singleton_->VMemoryAddressReserve(va, size, address, 0, flags);
  CATCH;
}

hsa_status_t hsa_amd_vmem_address_reserve_align(void** va, size_t size, uint64_t address,
                                          uint64_t alignment, uint64_t flags) {
  TRY;
  IS_OPEN();
  IS_ZERO(size);
  IS_TRUE(core::Runtime::runtime_singleton_->VirtualMemApiSupported());
  return core::Runtime::runtime_singleton_->VMemoryAddressReserve(va, size, address, alignment, flags);
  CATCH;
}


hsa_status_t hsa_amd_vmem_address_free(void* va, size_t size) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(va);
  IS_ZERO(size);
  return core::Runtime::runtime_singleton_->VMemoryAddressFree(va, size);
  CATCH;
}

hsa_status_t hsa_amd_vmem_handle_create(hsa_amd_memory_pool_t memory_pool, size_t size,
                                        hsa_amd_memory_type_t type, uint64_t flags,
                                        hsa_amd_vmem_alloc_handle_t* memory_handle) {
  TRY;
  IS_OPEN();
  IS_ZERO(size);
  IS_TRUE(core::Runtime::runtime_singleton_->VirtualMemApiSupported());

  if (type != MEMORY_TYPE_NONE && type != MEMORY_TYPE_PINNED)
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  hsa_region_t region = {memory_pool.handle};
  const core::MemoryRegion* mem_region = core::MemoryRegion::Convert(region);

  if (mem_region == NULL || !mem_region->IsValid()) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  MemoryRegion::AllocateFlags alloc_flag = core::MemoryRegion::AllocateMemoryOnly;
  if (type == MEMORY_TYPE_PINNED) alloc_flag |= core::MemoryRegion::AllocatePinned;

  return core::Runtime::runtime_singleton_->VMemoryHandleCreate(mem_region, size, alloc_flag, flags,
                                                                memory_handle);
  CATCH;
}

hsa_status_t hsa_amd_vmem_handle_release(hsa_amd_vmem_alloc_handle_t memory_handle) {
  TRY;
  IS_OPEN();
  return core::Runtime::runtime_singleton_->VMemoryHandleRelease(memory_handle);
  CATCH;
}

hsa_status_t hsa_amd_vmem_map(void* va, size_t size, size_t in_offset,
                              hsa_amd_vmem_alloc_handle_t memory_handle, uint64_t flags) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(va);
  IS_ZERO(size);

  return core::Runtime::runtime_singleton_->VMemoryHandleMap(va, size, in_offset, memory_handle,
                                                             flags);
  CATCH;
}

hsa_status_t hsa_amd_vmem_unmap(void* va, size_t size) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(va);
  IS_ZERO(size);

  return core::Runtime::runtime_singleton_->VMemoryHandleUnmap(va, size);
  CATCH;
}

hsa_status_t hsa_amd_vmem_set_access(void* va, size_t size,
                                     const hsa_amd_memory_access_desc_t* desc,
                                     size_t desc_cnt) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(va);
  IS_ZERO(size);
  IS_BAD_PTR(desc);
  IS_ZERO(desc_cnt);

  return core::Runtime::runtime_singleton_->VMemorySetAccess(va, size, desc, desc_cnt);
  CATCH;
}

hsa_status_t hsa_amd_vmem_get_access(void* va, hsa_access_permission_t* perms,
                                     hsa_agent_t agent_handle) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(va);
  IS_BAD_PTR(perms);

  return core::Runtime::runtime_singleton_->VMemoryGetAccess(va, perms, agent_handle);
  CATCH;
}

hsa_status_t hsa_amd_vmem_export_shareable_handle(int* dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t handle,
                                                  uint64_t flags) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(dmabuf_fd);

  return core::Runtime::runtime_singleton_->VMemoryExportShareableHandle(dmabuf_fd, handle, flags);
  CATCH;
}

hsa_status_t hsa_amd_vmem_import_shareable_handle(int dmabuf_fd,
                                                  hsa_amd_vmem_alloc_handle_t* handle) {
  TRY;
  IS_BAD_PTR(handle);
  IS_VALID_FD(dmabuf_fd);

  return core::Runtime::runtime_singleton_->VMemoryImportShareableHandle(dmabuf_fd, handle);
  CATCH;
}

hsa_status_t hsa_amd_vmem_retain_alloc_handle(hsa_amd_vmem_alloc_handle_t* allocHandle,
                                              void* addr) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(addr);

  return core::Runtime::runtime_singleton_->VMemoryRetainAllocHandle(allocHandle, addr);
  CATCH;
}

hsa_status_t hsa_amd_vmem_get_alloc_properties_from_handle(hsa_amd_vmem_alloc_handle_t allocHandle,
                                                           hsa_amd_memory_pool_t* pool,
                                                           hsa_amd_memory_type_t* type) {
  TRY;
  IS_OPEN();
  IS_BAD_PTR(pool);
  IS_BAD_PTR(type);

  const core::MemoryRegion* mem_region = NULL;
  hsa_status_t ret = core::Runtime::runtime_singleton_->VMemoryGetAllocPropertiesFromHandle(
      allocHandle, &mem_region, type);
  if (ret == HSA_STATUS_SUCCESS) {
    hsa_region_t region = core::MemoryRegion::Convert(mem_region);
    pool->handle = region.handle;
  }

  return ret;
  CATCH;
}

hsa_status_t HSA_API hsa_amd_agent_set_async_scratch_limit(hsa_agent_t _agent, size_t threshold) {
  TRY;
  IS_OPEN();

  core::Agent* agent = core::Agent::Convert(_agent);
  if (agent == NULL || !agent->IsValid() || agent->device_type() != core::Agent::kAmdGpuDevice)
    return HSA_STATUS_ERROR_INVALID_AGENT;

  AMD::GpuAgentInt* gpu_agent = static_cast<AMD::GpuAgentInt*>(agent);

  if (!core::Runtime::runtime_singleton_->flag().enable_scratch_async_reclaim() ||
      !gpu_agent->AsyncScratchReclaimEnabled())
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;

  return gpu_agent->SetAsyncScratchThresholds(threshold);
  CATCH;
}

hsa_status_t HSA_API hsa_amd_queue_get_info(hsa_queue_t* _queue,
                                            hsa_queue_info_attribute_t attribute, void* value) {
  TRY;
  IS_OPEN();

  core::Queue* queue = core::Queue::Convert(_queue);
  IS_VALID(queue);

  return queue->GetInfo(attribute, value);
  CATCH;
}

hsa_status_t hsa_amd_enable_logging(uint8_t* flags, void *file) {
  TRY;
  return core::Runtime::runtime_singleton_->EnableLogging(flags, file);
  CATCH;
}

}   //  namespace amd
}   //  namespace rocr
