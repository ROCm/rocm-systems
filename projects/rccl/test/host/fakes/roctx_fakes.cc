/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// See roctx_fakes.h.

#include "roctx_fakes.h"

#include <mutex>

#include "roctx.h"

namespace {
std::mutex g_roctxMutex;  // ranges open on production worker threads too
std::vector<RoctxRange> g_roctxRanges;
int g_roctxRangesOpen = 0;

void Open(const char* name, const nvtxPayloadData_t* data) {
  RoctxRange range{name ? name : "", {}};
  if (data && data[0].payload) {
    const auto* bytes = static_cast<const unsigned char*>(data[0].payload);
    range.payload.assign(bytes, bytes + data[0].size);
  }
  std::lock_guard<std::mutex> lock(g_roctxMutex);
  g_roctxRanges.push_back(std::move(range));
  ++g_roctxRangesOpen;
}
}  // namespace

std::vector<RoctxRange> RoctxRanges() {
  std::lock_guard<std::mutex> lock(g_roctxMutex);
  return g_roctxRanges;
}

int RoctxRangesOpen() {
  std::lock_guard<std::mutex> lock(g_roctxMutex);
  return g_roctxRangesOpen;
}

void ResetRoctxFakes() {
  std::lock_guard<std::mutex> lock(g_roctxMutex);
  g_roctxRanges.clear();
  g_roctxRangesOpen = 0;
}

roctx_scoped_range_in::roctx_scoped_range_in(const nvtxPayloadSchemaEntry_t* /*schema*/,
                                             const nvtxPayloadData_t* data, const size_t /*numEntries*/,
                                             const char* schemaName) noexcept {
  Open(schemaName, data);
}
roctx_scoped_range_in::roctx_scoped_range_in(const char* message) noexcept { Open(message, nullptr); }
roctx_scoped_range_in::~roctx_scoped_range_in() noexcept {
  std::lock_guard<std::mutex> lock(g_roctxMutex);
  --g_roctxRangesOpen;
}
