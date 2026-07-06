// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access.h
/// @brief Instruction-facing register access facade for observed VGPR regions.

#ifndef ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
#define ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_

#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "simdojo/components/vector_reg.h"
#include "util/simd.h"

#include <bit>
#include <cassert>
#include <cstddef>
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
  class OperandReadView {
  public:
    OperandReadView() = default;

    bool has_storage() const { return storage_ != nullptr; }

    uint32_t lane(uint32_t lane) const {
      assert(op_ && "OperandReadView is empty");
      return storage_ ? (*storage_)[lane] : scalar_;
    }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert(op_ && "OperandReadView is empty");
      return storage_ ? storage_->template simd_load<T>(lane_base) : util::broadcast<T>(scalar_);
    }

    template <typename T> util::narrow32<T> load_narrow(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_narrow expects 32-bit lanes");
      assert(op_ && "OperandReadView is empty");
      return storage_ ? storage_->template simd_load_narrow<T>(lane_base)
                      : util::broadcast_narrow<T>(scalar_);
    }

  private:
    friend class RegisterAccess;

    OperandReadView(const Operand &op, const Wavefront &wf, const VgprStorage *storage)
        : op_(&op), storage_(storage), scalar_(storage ? 0u : op.read_scalar(wf)) {}

    const Operand *op_ = nullptr;
    const VgprStorage *storage_ = nullptr;
    uint32_t scalar_ = 0;
  };

  class OperandWriteView {
  public:
    OperandWriteView() = default;

    bool has_storage() const { return storage_ != nullptr; }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWriteView is empty");
      if (storage_) {
        storage_->template simd_store<T>(lane_base, value, lane_mask);
        return;
      }
      constexpr std::size_t W = util::native_width_v<T>;
      alignas(util::native<T>) uint32_t buf[W];
      util::blit_to_buffer<T>(buf, value);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

    template <typename T>
    void store_narrow(uint32_t lane_base, util::narrow32<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_narrow expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWriteView is empty");
      if (storage_) {
        storage_->template simd_store_narrow<T>(lane_base, value, lane_mask);
        return;
      }
      constexpr std::size_t W = util::native_width64;
      alignas(util::narrow32<T>) T vals[W];
      value.copy_to(vals, util::stdx::vector_aligned);
      uint32_t buf[W];
      for (std::size_t i = 0; i < W; ++i)
        buf[i] = std::bit_cast<uint32_t>(vals[i]);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

  private:
    friend class RegisterAccess;

    OperandWriteView(const Operand &op, Wavefront &wf, VgprStorage *storage)
        : op_(&op), wf_(&wf), storage_(storage) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStorage *storage_ = nullptr;
  };

  class OperandRead64View {
  public:
    OperandRead64View() = default;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "load_native expects 64-bit lanes");
      assert(op_ && "OperandRead64View is empty");
      return storage_.lo ? storage_.lo->template simd_load64<T>(*storage_.hi, lane_base)
                         : util::broadcast64<T>(scalar_);
    }

  private:
    friend class RegisterAccess;

    OperandRead64View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage)
        : op_(&op), storage_(storage), scalar_(storage.lo ? 0u : op.read_scalar64(wf)) {}

    const Operand *op_ = nullptr;
    ConstVgprStoragePair64 storage_{};
    uint64_t scalar_ = 0;
  };

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

  OperandReadView read_operand(const Operand &op, const Wavefront &wf, uint64_t lane_mask,
                               uint8_t byte_mask = 0xF) const {
    const VgprStorage *storage = SimdAccess::vgpr_storage(op, wf);
    if (storage)
      SimdAccess::notify_read(op, wf, lane_mask, byte_mask);
    return OperandReadView(op, wf, storage);
  }

  OperandRead64View read_operand64(const Operand &op, const Wavefront &wf, uint64_t lane_mask,
                                   uint8_t byte_mask = 0xF) const {
    ConstVgprStoragePair64 storage = SimdAccess::vgpr_storage64(op, wf);
    if (storage.lo)
      SimdAccess::notify_read64(op, wf, lane_mask, byte_mask);
    return OperandRead64View(op, wf, storage);
  }

  OperandWriteView write_operand(const Operand &op, Wavefront &wf, uint64_t /*lane_mask*/) const {
    return OperandWriteView(op, wf, SimdAccess::vgpr_storage_mut(op, wf));
  }

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
