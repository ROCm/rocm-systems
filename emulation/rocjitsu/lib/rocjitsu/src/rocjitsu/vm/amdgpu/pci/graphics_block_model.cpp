// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/graphics_block_model.h"

#include "util/log.h"

#include <bit>
#include <format>
#include <utility>

namespace rocjitsu {
namespace {

constexpr uint32_t kGcSegment = 0;
constexpr uint32_t kGcControlSegment = 1;
constexpr uint32_t kGbAddrConfigRead = 0x13e2;
constexpr uint32_t kScratchRegister0 = 0x2040;
constexpr uint32_t kGrbmGfxCntl = 0x0900;
constexpr uint32_t kShMemBases = 0x09e3;
constexpr uint32_t kShMemConfig = 0x09e4;
constexpr uint32_t kGrbmVmidShift = 4;
constexpr uint32_t kGrbmVmidMask = 0xf;
constexpr uint32_t kPrivateBaseShift = 10;
constexpr uint32_t kPrivateBaseMask = 0x3f;
constexpr uint32_t kSharedBaseShift = 16;
constexpr uint64_t kSharedApertureBytes = uint64_t{1} << 32;
constexpr uint64_t kPrivateApertureBytes = uint64_t{1} << 57;
constexpr uint32_t kComputeDataCacheOperation = 0x290c;
constexpr uint32_t kComputeInstructionCacheOperation = 0x297a;
constexpr uint32_t kCacheInvalidationComplete = 0x2;

std::optional<uint32_t> gb_addr_config(const GraphicsDiscoveryInfo &graphics) {
  if (!std::has_single_bit(graphics.shader_engines) ||
      !std::has_single_bit(graphics.render_backends_per_engine)) {
    return std::nullopt;
  }
  const uint32_t shader_engine_encoding = std::countr_zero(graphics.shader_engines);
  const uint32_t render_backend_encoding = std::countr_zero(graphics.render_backends_per_engine);
  if (shader_engine_encoding > 0xf || render_backend_encoding > 0x3) {
    return std::nullopt;
  }

  // Unspecified fields encode one pipe, 256-byte interleave, one compressed
  // fragment and one packer. The two topology fields agree with the GC-info
  // table, which is the consistency the driver depends on during setup.
  return (shader_engine_encoding << 19) | (render_backend_encoding << 26);
}

} // namespace

GraphicsBlockModel::GraphicsBlockModel(IpRegisterWindow registers, uint32_t gb_addr_config)
    : IpBlockModel("the graphics core", std::move(registers)), gb_addr_config_(gb_addr_config) {}

std::unique_ptr<GraphicsBlockModel>
GraphicsBlockModel::create(const IpBlock &block, const GraphicsDiscoveryInfo &graphics,
                           IpRegisterWindow registers) {
  if (block.hardware_id != IpHardwareId::Gc || block.major != 12 || block.minor != 1 ||
      block.revision != 0) {
    util::Logger::warn(std::format(
        "{}: no graphics-core register layout is known for version {}.{}.{}, so its startup "
        "registers cannot be answered",
        registers.owner(), block.major, block.minor, block.revision));
    return nullptr;
  }
  const std::optional<uint32_t> encoded = gb_addr_config(graphics);
  if (!encoded) {
    util::Logger::warn(std::format(
        "{}: graphics topology has {} shader engines and {} render backends per engine, which "
        "cannot be encoded in GB_ADDR_CONFIG_READ",
        registers.owner(), graphics.shader_engines, graphics.render_backends_per_engine));
    return nullptr;
  }
  return std::unique_ptr<GraphicsBlockModel>(
      new GraphicsBlockModel(std::move(registers), *encoded));
}

std::vector<RegisterClaim> GraphicsBlockModel::claims() const {
  return {
      {.segment_index = kGcSegment, .first_dword = kGbAddrConfigRead, .count = 1},
      {.segment_index = kGcControlSegment, .first_dword = kGrbmGfxCntl, .count = 1},
      {.segment_index = kGcControlSegment, .first_dword = kShMemBases, .count = 2},
      {.segment_index = kGcControlSegment, .first_dword = kScratchRegister0, .count = 1},
      {.segment_index = kGcControlSegment, .first_dword = kComputeDataCacheOperation, .count = 1},
      {.segment_index = kGcControlSegment,
       .first_dword = kComputeInstructionCacheOperation,
       .count = 1},
  };
}

bool GraphicsBlockModel::reset() {
  selected_vmid_ = 0;
  sh_mem_bases_.fill(0);
  sh_mem_config_.fill(0);
  bool defined = registers_.define_read_only(kGcSegment, kGbAddrConfigRead, gb_addr_config_);
  defined = registers_.define(kGcControlSegment, kGrbmGfxCntl, 0) && defined;
  defined = registers_.define(kGcControlSegment, kShMemBases, 0) && defined;
  defined = registers_.define(kGcControlSegment, kShMemConfig, 0) && defined;
  defined = registers_.define(kGcControlSegment, kScratchRegister0, 0) && defined;
  // No firmware processor or instruction/data cache executes in this device.
  // The driver's startup invalidations are therefore complete before it asks;
  // keeping these status registers read-only also preserves completion across
  // the trigger write in its read-modify-write sequence.
  defined = registers_.define_read_only(kGcControlSegment, kComputeDataCacheOperation,
                                        kCacheInvalidationComplete) &&
            defined;
  defined = registers_.define_read_only(kGcControlSegment, kComputeInstructionCacheOperation,
                                        kCacheInvalidationComplete) &&
            defined;
  return defined;
}

void GraphicsBlockModel::observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) {
  if (registers_.resolve(kGcControlSegment, kGrbmGfxCntl) == byte_offset) {
    selected_vmid_ =
        (registers_.read(kGcControlSegment, kGrbmGfxCntl) >> kGrbmVmidShift) & kGrbmVmidMask;
    (void)registers_.write(kGcControlSegment, kShMemBases, sh_mem_bases_[selected_vmid_]);
    (void)registers_.write(kGcControlSegment, kShMemConfig, sh_mem_config_[selected_vmid_]);
    return;
  }
  if (registers_.resolve(kGcControlSegment, kShMemConfig) == byte_offset) {
    sh_mem_config_[selected_vmid_] = registers_.read(kGcControlSegment, kShMemConfig);
    publish_selected_apertures(memory);
    return;
  }
  if (registers_.resolve(kGcControlSegment, kShMemBases) != byte_offset)
    return;

  sh_mem_bases_[selected_vmid_] = registers_.read(kGcControlSegment, kShMemBases);
  publish_selected_apertures(memory);
}

void GraphicsBlockModel::publish_selected_apertures(PciMemoryAccess &memory) {
  if (selected_vmid_ == 0)
    return;
  if ((sh_mem_config_[selected_vmid_] & 1u) != 0) {
    (void)memory.program_shader_apertures(selected_vmid_, 0, 0, 0, 0);
    return;
  }
  // GC 12.1 encodes the private base in units of 2^58 and the shared base
  // in units of 2^48. The aperture sizes are fixed by this shader generation.
  const uint32_t bases = sh_mem_bases_[selected_vmid_];
  const uint64_t private_base = uint64_t{(bases >> kPrivateBaseShift) & kPrivateBaseMask} << 58;
  const uint64_t shared_base = uint64_t{bases >> kSharedBaseShift} << 48;
  if (!memory.program_shader_apertures(selected_vmid_, shared_base,
                                       shared_base + kSharedApertureBytes - 1, private_base,
                                       private_base + kPrivateApertureBytes - 1))
    util::Logger::warn("graphics core: shader aperture register write could not reach its SoC");
}

} // namespace rocjitsu
