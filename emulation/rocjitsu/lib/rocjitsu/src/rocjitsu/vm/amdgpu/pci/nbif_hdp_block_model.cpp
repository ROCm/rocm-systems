// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/nbif_hdp_block_model.h"

#include "rocjitsu/vm/soc.h"

#include "util/log.h"

#include <format>

namespace rocjitsu {
namespace {

struct HdpRemapLayout {
  uint16_t major;
  uint16_t minor;
  uint16_t revision;
  uint32_t flush_dword;
};

// nbio_v7_11_set_reg_remap directs the selected driver's HDP flush to BAR
// byte 0x44000. The driver selects this remap for NBIF 7.11.0 through 7.11.3.
constexpr HdpRemapLayout kLayouts[] = {
    {7, 11, 0, 0x44000 / 4},
    {7, 11, 1, 0x44000 / 4},
    {7, 11, 2, 0x44000 / 4},
    {7, 11, 3, 0x44000 / 4},
};

} // namespace

NbifHdpBlockModel::NbifHdpBlockModel(IpRegisterWindow registers, uint32_t flush_dword)
    : IpBlockModel("the NBIF HDP remap", std::move(registers)), flush_dword_(flush_dword) {}

std::unique_ptr<NbifHdpBlockModel> NbifHdpBlockModel::create(const IpBlock &block,
                                                             IpRegisterWindow registers) {
  for (const HdpRemapLayout &layout : kLayouts) {
    if (block.major == layout.major && block.minor == layout.minor &&
        block.revision == layout.revision)
      return std::unique_ptr<NbifHdpBlockModel>(
          new NbifHdpBlockModel(registers.absolute_remap(), layout.flush_dword));
  }
  util::Logger::warn(std::format("{}: no HDP remap for NBIF {}.{}.{}", registers.owner(),
                                 block.major, block.minor, block.revision));
  return nullptr;
}

std::vector<RegisterClaim> NbifHdpBlockModel::claims() const {
  return {{.segment_index = 0, .first_dword = flush_dword_, .count = 1}};
}

bool NbifHdpBlockModel::reset() {
  completed_flushes_.store(0, std::memory_order_release);
  return registers_.define(0, flush_dword_, 0);
}

void NbifHdpBlockModel::observe_register_write(uint64_t byte_offset, PciMemoryAccess &) {
  if (registers_.resolve(0, flush_dword_) != byte_offset)
    return;
  std::atomic_thread_fence(std::memory_order_seq_cst);
  if (soc_ != nullptr)
    soc_->cache_coherence()->publish_local_backing_change();
  completed_flushes_.fetch_add(1, std::memory_order_release);
}

} // namespace rocjitsu
