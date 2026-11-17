// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa/hsa_api_trace.h"

#include <unordered_set>

#define RJ_TEST_EXPORT __attribute__((visibility("default")))

namespace {
decltype(hsa_queue_create) *g_create = nullptr;
decltype(hsa_queue_destroy) *g_destroy = nullptr;
decltype(AmdExtTable::hsa_amd_queue_create_fn) g_amd_create = nullptr;
std::unordered_set<hsa_queue_t *> g_owned;
} // namespace

extern "C" RJ_TEST_EXPORT void
rj_waitcheck_test_prior_queue_configure(decltype(hsa_queue_create) *create,
                                        decltype(hsa_queue_destroy) *destroy,
                                        decltype(AmdExtTable::hsa_amd_queue_create_fn) amd_create) {
  g_create = create;
  g_destroy = destroy;
  g_amd_create = amd_create;
  g_owned.clear();
}

extern "C" RJ_TEST_EXPORT size_t rj_waitcheck_test_prior_queue_count() { return g_owned.size(); }

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API rj_waitcheck_test_prior_queue_create(
    hsa_agent_t agent, uint32_t size, hsa_queue_type32_t type,
    void (*callback)(hsa_status_t, hsa_queue_t *, void *), void *data,
    uint32_t private_segment_size, uint32_t group_segment_size, hsa_queue_t **queue) {
  if (g_create == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status =
      g_create(agent, size, type, callback, data, private_segment_size, group_segment_size, queue);
  if (status == HSA_STATUS_SUCCESS && queue != nullptr && *queue != nullptr)
    g_owned.insert(*queue);
  return status;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API rj_waitcheck_test_prior_amd_queue_create(
    hsa_agent_t agent, hsa_amd_queue_create_desc_t *descs, uint32_t num_descs) {
  if (g_amd_create == nullptr)
    return HSA_STATUS_ERROR_NOT_INITIALIZED;
  const hsa_status_t status = g_amd_create(agent, descs, num_descs);
  for (uint32_t index = 0; descs != nullptr && index < num_descs; ++index)
    if (descs[index].queue != nullptr)
      g_owned.insert(descs[index].queue);
  return status;
}

extern "C" RJ_TEST_EXPORT hsa_status_t HSA_API
rj_waitcheck_test_prior_queue_destroy(hsa_queue_t *queue) {
  if (g_owned.erase(queue) == 0)
    return HSA_STATUS_SUCCESS;
  return g_destroy == nullptr ? HSA_STATUS_ERROR_NOT_INITIALIZED : g_destroy(queue);
}
