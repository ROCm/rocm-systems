#ifndef HSA_RUNTIME_INC_HSA_TOOL_HOOK_IMPL_H
#define HSA_RUNTIME_INC_HSA_TOOL_HOOK_IMPL_H

#include "inc/hsa_amd_tool.h"
#include "runtime.h"

// namespace rocr::AMD::tool {  // C++17
namespace rocr { namespace AMD { namespace tool {

using scratch_alloc_flag = hsa_amd_event_scratch_alloc_flag_t;

__forceinline void notify_event_scratch_alloc_start(const hsa_queue_t* queue,
                                                    scratch_alloc_flag flag, uint64_t dispatch_id);

__forceinline void notify_event_scratch_alloc_end(const hsa_queue_t* queue, scratch_alloc_flag flag,
                                                  uint64_t dispatch_id, size_t size,
                                                  size_t num_slots);

__forceinline void notify_event_scratch_free_start(const hsa_queue_t* queue,
                                                   scratch_alloc_flag flag);

__forceinline void notify_event_scratch_free_end(const hsa_queue_t* queue, scratch_alloc_flag flag);

__forceinline void notify_event_scratch_async_reclaim_start(const hsa_queue_t* queue,
                                                            scratch_alloc_flag flag);

__forceinline void notify_event_scratch_async_reclaim_end(const hsa_queue_t* queue,
                                                          scratch_alloc_flag flag);


// Impl

__forceinline void notify_event_scratch_alloc_start(const hsa_queue_t* queue,
                                                    scratch_alloc_flag flags,
                                                    uint64_t dispatch_id) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_alloc_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_alloc_start_t{.kind = HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_START,
                                                   .queue = queue,
                                                   .flags = flags,
                                                   .dispatch_id = dispatch_id};

  tool_table.hsa_amd_tool_scratch_event_alloc_start_fn(
      hsa_amd_tool_event_t{.scratch_alloc_start = &event});
}

__forceinline void notify_event_scratch_alloc_end(const hsa_queue_t* queue,
                                                  scratch_alloc_flag flags, uint64_t dispatch_id,
                                                  size_t size, size_t num_slots) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_alloc_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_alloc_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ALLOC_END,
      .queue = queue,
      .flags = flags,
      .dispatch_id = dispatch_id,
      .size = size,
      .num_slots = num_slots,
  };

  tool_table.hsa_amd_tool_scratch_event_alloc_end_fn(
      hsa_amd_tool_event_t{.scratch_alloc_end = &event});
}

__forceinline void notify_event_scratch_free_start(const hsa_queue_t* queue,
                                                   scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_free_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_free_start_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_FREE_START,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_free_start_fn(
      hsa_amd_tool_event_t{.scratch_free_start = &event});
}

__forceinline void notify_event_scratch_free_end(const hsa_queue_t* queue,
                                                 scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_free_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_free_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_FREE_END,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_free_end_fn(
      hsa_amd_tool_event_t{.scratch_free_end = &event});
}

__forceinline void notify_event_scratch_async_reclaim_start(const hsa_queue_t* queue,
                                                            scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_async_reclaim_start_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_async_reclaim_start_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_START,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_async_reclaim_start_fn(
      hsa_amd_tool_event_t{.scratch_async_reclaim_start = &event});
}

__forceinline void notify_event_scratch_async_reclaim_end(const hsa_queue_t* queue,
                                                          scratch_alloc_flag flags) {
  const auto& tool_table = core::hsa_api_table().tools_api;
  if (!tool_table.hsa_amd_tool_scratch_event_async_reclaim_end_fn) {
    return;
  }

  auto event = hsa_amd_event_scratch_async_reclaim_end_t{
      .kind = HSA_AMD_TOOL_EVENT_SCRATCH_ASYNC_RECLAIM_END,
      .queue = queue,
      .flags = flags,
  };

  tool_table.hsa_amd_tool_scratch_event_async_reclaim_end_fn(
      hsa_amd_tool_event_t{.scratch_async_reclaim_end = &event});
}

// }  // namespace rocr::AMD::tool
}  // namespace rocr
}  // namespace AMD
}  // namespace tool

#endif