// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vfu/dma_mapper.h"
#include "rocjitsu/vfu/vfu_server.h"
#include "rocjitsu/kmd/linux/simulated_driver.h"

#include <libvfio-user.h>

#include <cstdio>

namespace rocjitsu::vfu {

DmaMapper::DmaMapper(SimulatedDriver &driver, uint32_t process_id)
    : driver_(driver), process_id_(process_id) {}

DmaMapper::~DmaMapper() {
  // Unmap all registered guest memory ranges on teardown.
  for (auto &m : mappings_) {
    if (m.vaddr)
      driver_.register_guest_dma(process_id_, m.iova, nullptr, m.length, false);
  }
}

void DmaMapper::on_register(vfu_dma_info_t *info) {
  if (!info)
    return;

  void *vaddr = info->vaddr; // Host VA if the region is accessible; nullptr otherwise.
  uint64_t iova = reinterpret_cast<uint64_t>(info->iova.iov_base);
  size_t length  = info->iova.iov_len;

  mappings_.push_back({iova, length, vaddr});

  if (vaddr) {
    // Map the guest memory range into rocjitsu's GPU VA space so the CP can
    // fetch AQL ring buffers and kernel descriptors from guest memory.
    driver_.register_guest_dma(process_id_, iova, vaddr, length, true);
  }
}

void DmaMapper::on_unregister(vfu_dma_info_t *info) {
  if (!info)
    return;

  uint64_t iova = reinterpret_cast<uint64_t>(info->iova.iov_base);
  size_t length  = info->iova.iov_len;

  driver_.register_guest_dma(process_id_, iova, nullptr, length, false);

  mappings_.erase(
      std::remove_if(mappings_.begin(), mappings_.end(),
                     [iova, length](const Mapping &m) {
                       return m.iova == iova && m.length == length;
                     }),
      mappings_.end());
}

void DmaMapper::dma_register(vfu_ctx_t *ctx, vfu_dma_info_t *info) {
  auto *srv = reinterpret_cast<VfuServer *>(vfu_get_private(ctx));
  if (srv && srv->dma())
    srv->dma()->on_register(info);
}

void DmaMapper::dma_unregister(vfu_ctx_t *ctx, vfu_dma_info_t *info) {
  auto *srv = reinterpret_cast<VfuServer *>(vfu_get_private(ctx));
  if (srv && srv->dma())
    srv->dma()->on_unregister(info);
}

} // namespace rocjitsu::vfu
