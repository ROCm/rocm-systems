// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/memory_hub_block_model.h"

#include "rocjitsu/vm/amdgpu/pci/physical_memory_access.h"
#include "rocjitsu/vm/soc.h"

#include "util/log.h"

#include <array>
#include <format>
#include <utility>

namespace rocjitsu {
namespace {

/// @brief The segment every offset below is measured from, its `_BASE_IDX`.
constexpr uint32_t kHubSegment = 0;

/// @brief Invalidation engines each memory hub has.
///
/// @details The driver picks one by index and reaches it by stride, so the
/// device answers all of them rather than guessing which. Engine 17 is the one
/// the GART flush uses, but that is a driver convention rather than a property
/// of the hardware.
constexpr uint32_t kInvalidationEngines = 18;

/// @brief Registers between one invalidation engine and the next.
constexpr uint32_t kInvalidationEngineStride = 1;

/// @brief Value a semaphore reads as when the acquire has been granted.
///
/// @details The GC 12.0 flush path acquires an engine's semaphore before
/// writing its request and releases it by writing zero, so this has to answer
/// reads and ignore writes: one that stored the release would grant the first
/// acquire and stall every flush after it. That path takes the semaphore for
/// MMHUB only; both hubs are answered anyway, because eighteen more registers
/// cost nothing and a hub that answered half a handshake would be the harder
/// thing to explain.
///
/// The GC 12.1 path this device publishes today does not take the semaphore at
/// all -- it writes the request and polls the acknowledge -- so these registers
/// are answered for the profile rather than for the boot, and are untouched on
/// gfx1250. They are still worth answering: this model is deliberately not
/// specific to one part, and every other GC 12.x takes the path that does.
///
/// Granting unconditionally is right only while nothing else acquires. The
/// device does not execute rings, so the driver's own lock is all that
/// serializes flushes; a device that ran packets would have a second acquirer
/// and this register would stop meaning what it means on hardware.
constexpr uint32_t kInvalidationSemaphoreHeld = 0x1;

/// @brief VMID bits carried in an invalidation request and acknowledgement.
///
/// @details VMID-0 is completed only after its pending translator has been
/// published. Other VMIDs have no hub-local cached state in this model and are
/// acknowledged synchronously.
constexpr uint32_t kVmidRequestMask = 0xffff;

/// @details GC from `regGCVM_INVALIDATE_ENG0_{SEM,REQ,ACK}` of
/// `gc_12_1_0_offset.h`, MMHUB from `regMMVM_INVALIDATE_ENG0_{SEM,REQ,ACK}` of
/// `mmhub_4_1_0_offset.h`; all `_BASE_IDX 0`, so all relative to the block's
/// first register segment.
///
/// One row per layout that has been read out of a header and checked. A block
/// whose version matches no row is refused rather than guessed at: there is no
/// safe default, because a wrong offset is indistinguishable from hardware that
/// never completes a flush.
constexpr HubInvalidationLayout kHubInvalidationLayouts[] = {
    {.id = IpHardwareId::Gc,
     .major = 12,
     .minor = 1,
     .revision = 0,
     .semaphore = 0x1645,
     .request = 0x1657,
     .acknowledge = 0x1669,
     .page_table_base_low = 0x169f,
     .page_table_base_high = 0x16a0,
     .page_table_start_low = 0x16bf,
     .page_table_start_high = 0x16c0,
     .page_table_end_low = 0x16df,
     .page_table_end_high = 0x16e0},
    {.id = IpHardwareId::MmHub,
     .major = 4,
     .minor = 1,
     .revision = 0,
     .semaphore = 0x0575,
     .request = 0x0587,
     .acknowledge = 0x0599,
     .page_table_base_low = 0x05cf,
     .page_table_base_high = 0x05d0,
     .page_table_start_low = 0x05ef,
     .page_table_start_high = 0x05f0,
     .page_table_end_low = 0x060f,
     .page_table_end_high = 0x0610},
};

/// @brief One of the three register blocks an engine's handshake spans.
struct EngineBlock {
  uint32_t first_dword = 0; ///< Engine 0's register, from the block's first segment.
  uint32_t value = 0;       ///< What every engine's register starts at.
  bool read_only = false;   ///< Whether it answers reads and ignores writes.
};

/// @brief The three blocks of @p layout, in the order an engine uses them.
[[nodiscard]] std::array<EngineBlock, 3> engine_blocks(const HubInvalidationLayout &layout) {
  return {EngineBlock{layout.semaphore, kInvalidationSemaphoreHeld, true},
          EngineBlock{layout.request, 0, false}, EngineBlock{layout.acknowledge, 0, true}};
}

uint64_t join(uint32_t low, uint32_t high) {
  return static_cast<uint64_t>(low) | (static_cast<uint64_t>(high) << 32);
}

} // namespace

MemoryHubBlockModel::MemoryHubBlockModel(std::string what, IpRegisterWindow registers,
                                         const HubInvalidationLayout &layout)
    : IpBlockModel(std::move(what), std::move(registers)), layout_(&layout) {}

std::unique_ptr<MemoryHubBlockModel> MemoryHubBlockModel::create(const IpBlock &block,
                                                                 IpRegisterWindow registers) {
  for (const HubInvalidationLayout &candidate : kHubInvalidationLayouts) {
    if (candidate.id == block.hardware_id && candidate.major == block.major &&
        candidate.minor == block.minor && candidate.revision == block.revision) {
      // Named by hardware id rather than by a friendly word, because the two
      // hubs are the same model and a diagnostic naming one of them has to say
      // which record it came from.
      return std::unique_ptr<MemoryHubBlockModel>(new MemoryHubBlockModel(
          std::format("the memory hub of hardware id {}", static_cast<uint16_t>(block.hardware_id)),
          std::move(registers), candidate));
    }
  }
  util::Logger::warn(std::format(
      "{}: no invalidation-register layout is known for hardware id {} version {}.{}.{}, and "
      "answering with another version's addresses would leave the driver polling registers "
      "this device never defined",
      registers.owner(), static_cast<uint16_t>(block.hardware_id), block.major, block.minor,
      block.revision));
  return nullptr;
}

std::vector<RegisterClaim> MemoryHubBlockModel::claims() const {
  std::vector<RegisterClaim> claimed;
  for (const EngineBlock &block : engine_blocks(*layout_)) {
    claimed.push_back({.segment_index = kHubSegment,
                       .first_dword = block.first_dword,
                       .count = kInvalidationEngines * kInvalidationEngineStride});
  }
  for (const uint32_t config : {layout_->page_table_base_low, layout_->page_table_base_high,
                                layout_->page_table_start_low, layout_->page_table_start_high,
                                layout_->page_table_end_low, layout_->page_table_end_high}) {
    claimed.push_back({.segment_index = kHubSegment, .first_dword = config, .count = 1});
  }
  return claimed;
}

bool MemoryHubBlockModel::reset() {
  // The blocks sit back to back, so a wrong engine count does not overrun the
  // aperture -- it silently writes one block's registers over the next one's,
  // and the flush the overwritten register served stalls again.
  const uint32_t span = kInvalidationEngines * kInvalidationEngineStride;
  if (layout_->semaphore + span > layout_->request ||
      layout_->request + span > layout_->acknowledge) {
    util::Logger::warn(
        std::format("{}: {} invalidation engines do not fit between the semaphore, request and "
                    "acknowledge blocks of this hub's registers",
                    registers_.owner(), kInvalidationEngines));
    return false;
  }

  const std::array<EngineBlock, 3> blocks = engine_blocks(*layout_);
  for (uint32_t engine = 0; engine < kInvalidationEngines; ++engine) {
    const uint32_t offset = engine * kInvalidationEngineStride;
    // A hub whose registers fall outside the aperture is reached through an
    // indirect window this device does not model, so there is nothing to define
    // and defining it would corrupt an unrelated register. Addresses rise with
    // the engine, so no later engine is in range once one is not.
    //
    // Every block is tested rather than the last one alone. Every layout known
    // today puts the acknowledge highest, but that is a property of those
    // layouts and not of the hardware, and a layout ordered otherwise would pass
    // a check on the acknowledge while its semaphore sat outside the aperture.
    for (const EngineBlock &block : blocks) {
      if (!registers_.resolve(kHubSegment, block.first_dword + offset).has_value()) {
        util::Logger::warn(std::format(
            "{}: invalidation engine {} of this hub is outside the register aperture, so its "
            "flushes cannot be answered",
            registers_.owner(), engine));
        return false;
      }
    }
    for (const EngineBlock &block : blocks) {
      const bool defined =
          block.read_only
              ? registers_.define_read_only(kHubSegment, block.first_dword + offset, block.value)
              : registers_.define(kHubSegment, block.first_dword + offset, block.value);
      if (!defined) {
        return false;
      }
    }
  }
  bool defined = true;
  for (const uint32_t config : {layout_->page_table_base_low, layout_->page_table_base_high,
                                layout_->page_table_start_low, layout_->page_table_start_high,
                                layout_->page_table_end_low, layout_->page_table_end_high}) {
    defined = registers_.define(kHubSegment, config, 0) && defined;
  }
  return defined;
}

void MemoryHubBlockModel::observe_register_write(uint64_t byte_offset, PciMemoryAccess &memory) {
  for (uint32_t engine = 0; engine < kInvalidationEngines; ++engine) {
    const uint32_t request_dword = layout_->request + engine * kInvalidationEngineStride;
    const std::optional<uint64_t> request = registers_.resolve(kHubSegment, request_dword);
    if (!request || *request != byte_offset)
      continue;

    const uint32_t requested = registers_.read(kHubSegment, request_dword) & kVmidRequestMask;
    const uint32_t acknowledge_dword = layout_->acknowledge + engine * kInvalidationEngineStride;
    uint32_t completed = requested;
    if ((requested & 1u) != 0) {
      completed &= ~1u;
      if (soc_ != nullptr) {
        const amdgpu::GartConfig config{
            .page_table_base = join(registers_.read(kHubSegment, layout_->page_table_base_low),
                                    registers_.read(kHubSegment, layout_->page_table_base_high)),
            .aperture_start = join(registers_.read(kHubSegment, layout_->page_table_start_low),
                                   registers_.read(kHubSegment, layout_->page_table_start_high))
                              << 12,
            .aperture_end = join(registers_.read(kHubSegment, layout_->page_table_end_low),
                                 registers_.read(kHubSegment, layout_->page_table_end_high))
                            << 12,
        };
        auto physical = std::make_shared<PciPhysicalMemoryAccess>(memory);
        if (soc_->gpu_vm().publish_gart(config, std::move(physical))) {
          completed |= 1u;
          util::Logger::vm([&](auto &os) {
            os << std::format("{}: published VMID-0 root={:#x} aperture=[{:#x}, {:#x}]",
                              registers_.owner(), config.page_table_base, config.aperture_start,
                              config.aperture_end);
          });
        }
      }
    }

    // Publication above is synchronous and protected by GpuVm's registry
    // lock. Only expose completion after the new immutable translator and its
    // epoch are visible to every MES/SDMA reader.
    const uint32_t prior = registers_.read(kHubSegment, acknowledge_dword);
    (void)registers_.write(kHubSegment, acknowledge_dword, prior | completed);
    return;
  }
}

} // namespace rocjitsu
