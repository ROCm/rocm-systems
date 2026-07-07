// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access.h
/// @brief Instruction-facing register access facade for observed VGPR regions.
///
/// @details AMDGPU instruction emulation has two competing needs. Most code
/// should read and write operands through the ISA operand API, but hot SIMD and
/// matrix paths also need direct lane spans over physical VGPR storage. This
/// file provides the boundary between those instruction-visible accesses and
/// the lower-level register files owned by ComputeUnitCore.
///
/// Reads acquired through RegisterAccess notify the execution plugin before
/// exposing scalar values, SIMD storage, or physical VGPR regions. Write-only
/// accessors do not report reads. Read-write accessors report the read part at
/// acquisition time and then allow the caller to update the same instruction-
/// scoped storage. VM/storage code may still use raw register storage for tasks
/// such as memory completion, but instruction emulators should use this facade
/// for physical VGPR access.

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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace rocjitsu::amdgpu {

/// @brief Facade for instruction-visible VGPR reads and writes.
///
/// @details This class centralizes the observation contract for
/// instruction-visible VGPR access. Callers should not pair raw storage access
/// with ad hoc plugin notifications. Instead, they acquire one of two forms of
/// access here:
///
/// - Operand views, used by SIMD helpers that start from ISA operands and may
///   need either scalar fallback values or contiguous VGPR lane storage.
/// - Physical VGPR regions, used by matrix, memory-address, and other helpers
///   that already operate on physical register indices.
///
/// API selection guide:
/// - Use read_operand() for logical source operands.
/// - Use write_operand() for destinations whose old value is not read.
/// - Use readwrite_operand() when a destination is also an input.
/// - Use read_vgpr_region() when a helper already has physical VGPR indices.
/// - Use write_vgpr_region() for physical writes that do not read old values.
/// - Use readwrite_vgpr_region() for physical read-modify-write operations.
///
/// Read and read-write acquisition fires the plugin read hook before any lane
/// storage is exposed. Region reads notify once per physical register in the
/// requested range with the caller-provided lane and byte masks. Write-only
/// views deliberately do not report reads.
///
/// Operand read views may be VGPR-backed or scalar-backed. Scalar-backed views
/// represent SGPR, inline literal, immediate, and special-register operands as
/// lane-broadcast values; they do not imply a missing VGPR read.
///
/// The view objects are intentionally lightweight and instruction-scoped. They
/// expose spans over the underlying VGPR lane storage so hot paths can keep the
/// current zero-copy behavior while the observation contract remains localized
/// here. They should be acquired during a single instruction's emulation and
/// not cached across instructions.
///
/// A RegisterAccess constructed from a const ComputeUnitCore supports read-only
/// physical access. Write access requires construction from a mutable
/// ComputeUnitCore.
class RegisterAccess {
  template <typename T>
  static T require_scalar_fallback(const std::optional<T> &fallback, const char *view_name) {
    if (!fallback)
      throw std::logic_error(std::string(view_name) + " has no scalar fallback");
    return *fallback;
  }

public:
  class OperandReadView {
  public:
    OperandReadView() = delete;

    bool has_storage() const { return storage_ != nullptr; }

    uint32_t lane(uint32_t lane) const {
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return storage_ ? (*storage_)[lane] : scalar_fallback();
    }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return storage_ ? storage_->template simd_load<T>(lane_base)
                      : util::broadcast<T>(scalar_fallback());
    }

    template <typename T> util::narrow32<T> load_narrow(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_narrow expects 32-bit lanes");
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return storage_ ? storage_->template simd_load_narrow<T>(lane_base)
                      : util::broadcast_narrow<T>(scalar_fallback());
    }

  private:
    friend class RegisterAccess;

    OperandReadView(const Operand &op, const Wavefront &wf, const VgprStorage *storage)
        : storage_(storage) {
      if (!storage_)
        scalar_fallback_.emplace(op.read_scalar(wf));
    }

    uint32_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadView");
    }

    const VgprStorage *storage_ = nullptr;
    std::optional<uint32_t> scalar_fallback_;
  };

  class OperandWriteView {
  public:
    OperandWriteView() = delete;

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

  class OperandWrite64View {
  public:
    OperandWrite64View() = delete;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "store_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandWrite64View is empty");
      if (storage_.lo) {
        storage_.lo->template simd_store64<T>(*storage_.hi, lane_base, value, lane_mask);
        return;
      }
      constexpr std::size_t W = util::native_width64;
      alignas(util::native<T>) uint64_t buf[W];
      util::stdx::native_simd<uint64_t> bits = [&] {
        if constexpr (std::is_same_v<T, uint64_t>)
          return value;
        else
          return std::bit_cast<util::stdx::native_simd<uint64_t>>(value);
      }();
      bits.copy_to(buf, util::stdx::vector_aligned);
      for (std::size_t i = 0; i < W; ++i)
        if (lane_mask & (1ULL << i))
          op_->write_lane64(*wf_, lane_base + static_cast<uint32_t>(i), buf[i]);
    }

  private:
    friend class RegisterAccess;

    OperandWrite64View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage)
        : op_(&op), wf_(&wf), storage_(storage) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
  };

  class OperandReadWriteView {
  public:
    OperandReadWriteView() = delete;

    bool has_storage() const { return storage_ != nullptr; }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      return storage_ ? storage_->template simd_load<T>(lane_base)
                      : util::broadcast<T>(scalar_fallback());
    }

    template <typename T> util::narrow32<T> load_narrow(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_narrow expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      return storage_ ? storage_->template simd_load_narrow<T>(lane_base)
                      : util::broadcast_narrow<T>(scalar_fallback());
    }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
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
      assert(op_ && wf_ && "OperandReadWriteView is empty");
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

    OperandReadWriteView(const Operand &op, Wavefront &wf, VgprStorage *storage)
        : op_(&op), wf_(&wf), storage_(storage) {
      if (!storage_)
        scalar_fallback_.emplace(op.read_scalar(wf));
    }

    uint32_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadWriteView");
    }

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStorage *storage_ = nullptr;
    std::optional<uint32_t> scalar_fallback_;
  };

  class OperandReadWrite64View {
  public:
    OperandReadWrite64View() = delete;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "load_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandReadWrite64View is empty");
      return storage_.lo ? storage_.lo->template simd_load64<T>(*storage_.hi, lane_base)
                         : util::broadcast64<T>(scalar_fallback());
    }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "store_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandReadWrite64View is empty");
      if (storage_.lo) {
        storage_.lo->template simd_store64<T>(*storage_.hi, lane_base, value, lane_mask);
        return;
      }
      constexpr std::size_t W = util::native_width64;
      alignas(util::native<T>) uint64_t buf[W];
      util::stdx::native_simd<uint64_t> bits = [&] {
        if constexpr (std::is_same_v<T, uint64_t>)
          return value;
        else
          return std::bit_cast<util::stdx::native_simd<uint64_t>>(value);
      }();
      bits.copy_to(buf, util::stdx::vector_aligned);
      for (std::size_t i = 0; i < W; ++i)
        if (lane_mask & (1ULL << i))
          op_->write_lane64(*wf_, lane_base + static_cast<uint32_t>(i), buf[i]);
    }

  private:
    friend class RegisterAccess;

    OperandReadWrite64View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage)
        : op_(&op), wf_(&wf), storage_(storage) {
      if (!storage_.lo)
        scalar_fallback_.emplace(op.read_scalar64(wf));
    }

    uint64_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadWrite64View");
    }

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
    std::optional<uint64_t> scalar_fallback_;
  };

  class OperandRead64View {
  public:
    OperandRead64View() = delete;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "load_native expects 64-bit lanes");
      assert((storage_.lo || scalar_fallback_) && "OperandRead64View has no source");
      return storage_.lo ? storage_.lo->template simd_load64<T>(*storage_.hi, lane_base)
                         : util::broadcast64<T>(scalar_fallback());
    }

  private:
    friend class RegisterAccess;

    OperandRead64View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage)
        : storage_(storage) {
      if (!storage_.lo)
        scalar_fallback_.emplace(op.read_scalar64(wf));
    }

    uint64_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandRead64View");
    }

    ConstVgprStoragePair64 storage_{};
    std::optional<uint64_t> scalar_fallback_;
  };

  class OperandReadPair32View {
  public:
    OperandReadPair32View() = delete;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> util::native<T> load_lo_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_lo_native expects 32-bit lanes");
      assert((storage_.lo || scalar_fallback_) && "OperandReadPair32View has no source");
      return storage_.lo ? storage_.lo->template simd_load<T>(lane_base)
                         : util::broadcast<T>(scalar_fallback());
    }

    template <typename T> util::native<T> load_hi_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_hi_native expects 32-bit lanes");
      assert((storage_.hi || scalar_fallback_) && "OperandReadPair32View has no source");
      return storage_.hi ? storage_.hi->template simd_load<T>(lane_base)
                         : util::broadcast<T>(scalar_fallback());
    }

  private:
    friend class RegisterAccess;

    OperandReadPair32View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage)
        : storage_(storage) {
      if (!storage_.lo)
        scalar_fallback_.emplace(op.read_scalar(wf));
    }

    uint32_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadPair32View");
    }

    ConstVgprStoragePair64 storage_{};
    std::optional<uint32_t> scalar_fallback_;
  };

  class OperandWritePair32View {
  public:
    OperandWritePair32View() = delete;

    bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T>
    void store_native_pair(uint32_t lane_base, util::native<T> lo, util::native<T> hi,
                           uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native_pair expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWritePair32View is empty");
      if (storage_.lo) {
        storage_.lo->template simd_store<T>(lane_base, lo, lane_mask);
        storage_.hi->template simd_store<T>(lane_base, hi, lane_mask);
        return;
      }
      constexpr std::size_t W = util::native_width_v<T>;
      alignas(util::native<T>) uint32_t lo_buf[W];
      alignas(util::native<T>) uint32_t hi_buf[W];
      util::blit_to_buffer<T>(lo_buf, lo);
      util::blit_to_buffer<T>(hi_buf, hi);
      for (std::size_t i = 0; i < W; ++i) {
        if ((lane_mask & (1ULL << i)) == 0)
          continue;
        const uint64_t value =
            static_cast<uint64_t>(lo_buf[i]) | (static_cast<uint64_t>(hi_buf[i]) << 32);
        op_->write_lane64(*wf_, lane_base + static_cast<uint32_t>(i), value);
      }
    }

  private:
    friend class RegisterAccess;

    OperandWritePair32View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage)
        : op_(&op), wf_(&wf), storage_(storage) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
  };

  class VgprReadRegion {
  public:
    VgprReadRegion() = delete;

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
    VgprWriteRegion() = delete;

    uint32_t base() const { return base_; }
    uint32_t reg_count() const { return reg_count_; }
    uint32_t wf_size() const { return wf_size_; }
    uint64_t lane_mask() const { return lane_mask_; }
    bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

    void set_lane(uint32_t relative_reg, uint32_t lane, uint32_t value) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      if ((lane_mask_ & (uint64_t{1} << lane)) != 0)
        reg_data(relative_reg)[lane] = value;
    }

    void set_lane64(uint32_t relative_reg, uint32_t lane, uint64_t value) const {
      assert(relative_reg + 1 < reg_count_ && "64-bit lane write needs two VGPRs");
      set_lane(relative_reg, lane, static_cast<uint32_t>(value));
      set_lane(relative_reg + 1, lane, static_cast<uint32_t>(value >> 32));
    }

    void set_linear_word(uint32_t linear_index, uint32_t value) const {
      assert(wf_size_ != 0 && "VgprWriteRegion is empty");
      set_lane(linear_index / wf_size_, linear_index % wf_size_, value);
    }

  private:
    friend class RegisterAccess;

    VgprWriteRegion(ComputeUnitCore &cu, uint32_t base, uint32_t reg_count, uint64_t lane_mask)
        : cu_(&cu), base_(base), reg_count_(reg_count), wf_size_(cu.wf_size()),
          lane_mask_(lane_mask) {}

    uint32_t *reg_data(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprWriteRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside write region");
      return reinterpret_cast<uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
    }

    ComputeUnitCore *cu_ = nullptr;
    uint32_t base_ = 0;
    uint32_t reg_count_ = 0;
    uint32_t wf_size_ = 0;
    uint64_t lane_mask_ = 0;
  };

  class VgprReadWriteRegion {
  public:
    VgprReadWriteRegion() = delete;

    const VgprReadRegion &read() const { return read_; }
    const VgprWriteRegion &write() const { return write_; }

    std::span<const uint32_t> read_lanes(uint32_t relative_reg = 0) const {
      return read_.lanes(relative_reg);
    }

    uint32_t linear_word(uint32_t linear_index) const {
      assert(read_.wf_size() != 0 && "VgprReadWriteRegion is empty");
      return read_.lane(linear_index / read_.wf_size(), linear_index % read_.wf_size());
    }

    void set_linear_word(uint32_t linear_index, uint32_t value) const {
      write_.set_linear_word(linear_index, value);
    }

  private:
    friend class RegisterAccess;

    VgprReadWriteRegion(VgprReadRegion read, VgprWriteRegion write) : read_(read), write_(write) {}

    VgprReadRegion read_;
    VgprWriteRegion write_;
  };

  explicit RegisterAccess(ComputeUnitCore &cu) : cu_(&cu), mutable_cu_(&cu) {}
  explicit RegisterAccess(const ComputeUnitCore &cu) : cu_(&cu) {}

  // Logical operand access. These APIs are for instruction helpers that still
  // want operand semantics for scalar fallback, literals, delegates, and
  // 32/64-bit VGPR pairing, but need SIMD-friendly storage when available.
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

  OperandReadPair32View read_operand_pair32(const Operand &op, const Wavefront &wf,
                                            uint64_t lane_mask, uint8_t byte_mask = 0xF) const {
    ConstVgprStoragePair64 storage = SimdAccess::vgpr_storage64(op, wf);
    if (storage.lo)
      SimdAccess::notify_read64(op, wf, lane_mask, byte_mask);
    return OperandReadPair32View(op, wf, storage);
  }

  OperandWriteView write_operand(const Operand &op, Wavefront &wf, uint64_t /*lane_mask*/) const {
    return OperandWriteView(op, wf, SimdAccess::vgpr_storage_mut(op, wf));
  }

  OperandWrite64View write_operand64(const Operand &op, Wavefront &wf,
                                     uint64_t /*lane_mask*/) const {
    return OperandWrite64View(op, wf, SimdAccess::vgpr_storage64_mut(op, wf));
  }

  OperandWritePair32View write_operand_pair32(const Operand &op, Wavefront &wf,
                                              uint64_t /*lane_mask*/) const {
    return OperandWritePair32View(op, wf, SimdAccess::vgpr_storage64_mut(op, wf));
  }

  OperandReadWriteView readwrite_operand(const Operand &op, Wavefront &wf, uint64_t lane_mask,
                                         uint8_t byte_mask = 0xF) const {
    VgprStorage *storage = SimdAccess::vgpr_storage_mut(op, wf);
    if (storage)
      SimdAccess::notify_read_mut(op, wf, lane_mask, byte_mask);
    return OperandReadWriteView(op, wf, storage);
  }

  OperandReadWrite64View readwrite_operand64(const Operand &op, Wavefront &wf, uint64_t lane_mask,
                                             uint8_t byte_mask = 0xF) const {
    VgprStoragePair64 storage = SimdAccess::vgpr_storage64_mut(op, wf);
    if (storage.lo)
      SimdAccess::notify_read64_mut(op, wf, lane_mask, byte_mask);
    return OperandReadWrite64View(op, wf, storage);
  }

  // Physical VGPR access. These APIs are for helpers that already know the
  // physical register index, such as matrix layout code and generated memory
  // address/data collection. Reads observe the supplied register/lane range
  // before returning views over the raw storage.
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
    return VgprReadRegion(*cu_, physical_base, reg_count);
  }

  VgprWriteRegion write_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                    uint64_t lane_mask) const {
    return VgprWriteRegion(mutable_cu(), physical_base, reg_count, lane_mask);
  }

  VgprReadWriteRegion readwrite_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                            uint64_t lane_mask, uint8_t byte_mask = 0xF) const {
    return VgprReadWriteRegion(read_vgpr_region(physical_base, reg_count, lane_mask, byte_mask),
                               write_vgpr_region(physical_base, reg_count, lane_mask));
  }

private:
  ComputeUnitCore &mutable_cu() const {
    if (!mutable_cu_)
      throw std::logic_error(
          "RegisterAccess constructed from const CU cannot write physical VGPRs");
    return *mutable_cu_;
  }

  void observe_vgpr_region(uint32_t physical_base, uint32_t reg_count, uint64_t lane_mask,
                           uint8_t byte_mask) const {
    if (lane_mask == 0)
      return;
    for (uint32_t reg = 0; reg < reg_count; ++reg)
      cu_->notify_vgpr_read_by_reg(physical_base + reg, lane_mask, byte_mask);
  }

  const ComputeUnitCore *cu_ = nullptr;
  ComputeUnitCore *mutable_cu_ = nullptr;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
