// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access.h
/// @brief Instruction-facing register access facade for observed VGPR regions.

#ifndef ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
#define ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include <cassert>
#include <cstdint>
#include <span>

namespace rocjitsu::amdgpu {

/// @brief Facade for instruction-visible register reads and writes.
///
/// @details This is the first step toward making register access non-social:
/// instruction emulators should acquire values or lane spans through this class
/// instead of pairing raw storage access with separate plugin notifications.
/// Read and read-write VGPR region acquisition fires the plugin read hook once
/// per physical register in the region. Write-only views deliberately do not
/// report reads.
///
/// The view objects are intentionally lightweight and instruction-scoped. They
/// expose spans over the underlying VGPR lane storage so hot paths can keep the
/// current zero-copy behavior while the observation contract remains localized
/// here.
class RegisterAccess {
public:
  class VgprReadRegion {
  public:
    VgprReadRegion() = default;

    uint32_t base() const { return base_; }
    uint32_t reg_count() const { return reg_count_; }
    uint32_t wf_size() const { return wf_size_; }
    bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

    std::span<const uint32_t> lanes(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprReadRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      return {reg_data(relative_reg), wf_size_};
    }

    const uint32_t *reg_data(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprReadRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      return reinterpret_cast<const uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
    }

    uint32_t lane(uint32_t relative_reg, uint32_t lane) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      return lanes(relative_reg)[lane];
    }

    uint64_t lane64(uint32_t relative_reg, uint32_t lane) const {
      assert(relative_reg + 1 < reg_count_ && "64-bit lane read needs two VGPRs");
      uint64_t lo = this->lane(relative_reg, lane);
      uint64_t hi = this->lane(relative_reg + 1, lane);
      return lo | (hi << 32);
    }

  private:
    friend class RegisterAccess;

    VgprReadRegion(const ComputeUnitCore &cu, uint32_t base, uint32_t reg_count)
        : cu_(&cu), base_(base), reg_count_(reg_count), wf_size_(cu.wf_size()) {}

    const ComputeUnitCore *cu_ = nullptr;
    uint32_t base_ = 0;
    uint32_t reg_count_ = 0;
    uint32_t wf_size_ = 0;
  };

  class VgprWriteRegion {
  public:
    VgprWriteRegion() = default;

    uint32_t base() const { return base_; }
    uint32_t reg_count() const { return reg_count_; }
    uint32_t wf_size() const { return wf_size_; }
    uint64_t lane_mask() const { return lane_mask_; }
    bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

    std::span<uint32_t> lanes(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprWriteRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside write region");
      return {reg_data(relative_reg), wf_size_};
    }

    uint32_t *reg_data(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprWriteRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside write region");
      return reinterpret_cast<uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
    }

    void set_lane(uint32_t relative_reg, uint32_t lane, uint32_t value) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      if ((lane_mask_ & (uint64_t{1} << lane)) != 0)
        lanes(relative_reg)[lane] = value;
    }

    void set_lane64(uint32_t relative_reg, uint32_t lane, uint64_t value) const {
      assert(relative_reg + 1 < reg_count_ && "64-bit lane write needs two VGPRs");
      set_lane(relative_reg, lane, static_cast<uint32_t>(value));
      set_lane(relative_reg + 1, lane, static_cast<uint32_t>(value >> 32));
    }

  private:
    friend class RegisterAccess;

    VgprWriteRegion(ComputeUnitCore &cu, uint32_t base, uint32_t reg_count, uint64_t lane_mask)
        : cu_(&cu), base_(base), reg_count_(reg_count), wf_size_(cu.wf_size()),
          lane_mask_(lane_mask) {}

    ComputeUnitCore *cu_ = nullptr;
    uint32_t base_ = 0;
    uint32_t reg_count_ = 0;
    uint32_t wf_size_ = 0;
    uint64_t lane_mask_ = 0;
  };

  class VgprReadWriteRegion {
  public:
    VgprReadWriteRegion() = default;

    const VgprReadRegion &read() const { return read_; }
    const VgprWriteRegion &write() const { return write_; }

    std::span<const uint32_t> read_lanes(uint32_t relative_reg = 0) const {
      return read_.lanes(relative_reg);
    }
    std::span<uint32_t> write_lanes(uint32_t relative_reg = 0) const {
      return write_.lanes(relative_reg);
    }

  private:
    friend class RegisterAccess;

    VgprReadWriteRegion(VgprReadRegion read, VgprWriteRegion write) : read_(read), write_(write) {}

    VgprReadRegion read_;
    VgprWriteRegion write_;
  };

  explicit RegisterAccess(ComputeUnitCore &cu) : cu_(cu) {}

  uint32_t read_vgpr(uint32_t physical_reg, uint32_t lane, uint8_t byte_mask = 0xF) const {
    return read_vgpr_region(physical_reg, 1, uint64_t{1} << lane, byte_mask).lane(0, lane);
  }

  uint64_t read_vgpr64(uint32_t physical_reg, uint32_t lane, uint8_t byte_mask = 0xF) const {
    return read_vgpr_region(physical_reg, 2, uint64_t{1} << lane, byte_mask).lane64(0, lane);
  }

  void write_vgpr(uint32_t physical_reg, uint32_t lane, uint32_t value) const {
    write_vgpr_region(physical_reg, 1, uint64_t{1} << lane).set_lane(0, lane, value);
  }

  void write_vgpr64(uint32_t physical_reg, uint32_t lane, uint64_t value) const {
    write_vgpr_region(physical_reg, 2, uint64_t{1} << lane).set_lane64(0, lane, value);
  }

  VgprReadRegion read_vgpr_region(uint32_t physical_base, uint32_t reg_count, uint64_t lane_mask,
                                  uint8_t byte_mask = 0xF) const {
    observe_vgpr_region(physical_base, reg_count, lane_mask, byte_mask);
    return VgprReadRegion(cu_, physical_base, reg_count);
  }

  VgprWriteRegion write_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                    uint64_t lane_mask) const {
    return VgprWriteRegion(cu_, physical_base, reg_count, lane_mask);
  }

  VgprReadWriteRegion readwrite_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                            uint64_t lane_mask, uint8_t byte_mask = 0xF) const {
    return VgprReadWriteRegion(read_vgpr_region(physical_base, reg_count, lane_mask, byte_mask),
                               write_vgpr_region(physical_base, reg_count, lane_mask));
  }

private:
  void observe_vgpr_region(uint32_t physical_base, uint32_t reg_count, uint64_t lane_mask,
                           uint8_t byte_mask) const {
    if (lane_mask == 0)
      return;
    for (uint32_t reg = 0; reg < reg_count; ++reg)
      cu_.notify_vgpr_read_by_reg(physical_base + reg, lane_mask, byte_mask);
  }

  ComputeUnitCore &cu_;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
