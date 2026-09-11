// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once
#include <atomic>
#include <cstdint>

namespace amd::roc {
// Shared by ordinary PQ submission and resident fixup/root publication.
inline void packet_store_release(uint32_t* packet, uint16_t header, uint16_t rest) {
#if IS_WINDOWS
  std::atomic_ref<uint32_t> atomic_header(*packet);
  atomic_header.store(header | (uint32_t(rest) << 16), std::memory_order_release);
#else
  __atomic_store_n(packet, header | (uint32_t(rest) << 16), __ATOMIC_RELEASE);
#endif
}
}  // namespace amd::roc
