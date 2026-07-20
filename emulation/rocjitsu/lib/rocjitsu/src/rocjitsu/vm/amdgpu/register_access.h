// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file register_access.h
/// @brief Instruction-facing facade for observed AMDGPU register access.
///
/// @details AMDGPU instruction emulation has two competing needs. Most code
/// reads and writes logical operands, while hot SIMD and matrix paths also need
/// direct lane spans over VGPR storage. This file provides the boundary between
/// those instruction-visible accesses and the lower-level register files owned
/// by ComputeUnitCore.
///
/// Reads acquired through RegisterAccess notify the execution plugin before
/// exposing scalar values, SIMD storage, or physical register regions. Write-only
/// accessors do not report reads. Read-write accessors report the read part at
/// acquisition time and then allow the caller to update the same instruction-
/// scoped storage. VM/storage code may still use raw register storage for tasks
/// such as memory completion, but instruction emulators should use this facade
/// for operand and physical register access.

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

struct OperandPair32 {
  uint32_t lo;
  uint32_t hi;
};

/// @brief Facade for instruction-visible register reads and writes.
///
/// @details This class centralizes the observation contract for
/// instruction-visible register access. Callers should not pair raw storage access
/// with ad hoc plugin notifications. Instead, they acquire one of these forms of
/// access here:
///
/// - Logical operand access, including scalar/lane/chunk reads and SIMD views
///   that start from ISA operands and may need either scalar fallback values or
///   contiguous VGPR lane storage.
/// - Physical VGPR regions, used by matrix, memory-address, and other helpers
///   that already operate on physical register indices.
/// - Physical SGPR access, used by address, descriptor, and generated
///   SGPR-relative helpers.
///
/// API selection guide:
/// - Use read_scalar(), read_lane(), and the 64-bit variants for
///   value-semantic logical operand reads.
/// - Use write_scalar(), write_lane(), and the 64-bit variants for
///   value-semantic logical operand writes.
/// - Use read_chunk() / write_chunk() for logical operand lane chunks.
/// - Use read_operand() for SIMD logical source operand views.
/// - Use write_operand() for SIMD destination views whose old value is not read.
/// - Use readwrite_operand() when a SIMD destination is also an input.
/// - Use read_vgpr_region() when a helper already has physical VGPR indices.
/// - Use write_vgpr_region() for physical writes that do not read old values.
/// - Use readwrite_vgpr_region() for physical read-modify-write operations.
/// - Use read_sgpr() / write_sgpr() when a helper already has physical SGPR
///   indices, such as address calculation or generated SGPR-relative forms.
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
/// Logical operand APIs require construction from a Wavefront. Physical register
/// APIs may be constructed from a ComputeUnitCore; write access requires
/// a mutable ComputeUnitCore.
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

    [[nodiscard]] bool has_storage() const { return storage_ != nullptr; }

    [[nodiscard]] uint32_t lane(uint32_t lane) const {
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return storage_ ? (*storage_)[lane] : scalar_fallback();
    }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return storage_ ? storage_->template simd_load<T>(lane_base)
                      : util::broadcast<T>(scalar_fallback());
    }

    template <typename T> [[nodiscard]] util::narrow32<T> load_narrow(uint32_t lane_base) const {
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
        scalar_fallback_.emplace(op.read_lane(wf, 0));
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

    [[nodiscard]] bool has_storage() const { return storage_ != nullptr; }

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

    [[nodiscard]] bool has_storage() const { return storage_.lo != nullptr; }

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

    [[nodiscard]] bool has_storage() const { return storage_ != nullptr; }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      return storage_ ? storage_->template simd_load<T>(lane_base)
                      : util::broadcast<T>(scalar_fallback());
    }

    template <typename T> [[nodiscard]] util::narrow32<T> load_narrow(uint32_t lane_base) const {
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
        scalar_fallback_.emplace(op.read_lane(wf, 0));
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

    [[nodiscard]] bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
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
        scalar_fallback_.emplace(op.read_lane64(wf, 0));
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

    [[nodiscard]] bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
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
        scalar_fallback_.emplace(op.read_lane64(wf, 0));
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

    [[nodiscard]] bool has_storage() const { return storage_.lo != nullptr; }

    template <typename T> [[nodiscard]] util::native<T> load_lo_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_lo_native expects 32-bit lanes");
      assert((storage_.lo || scalar_fallback_) && "OperandReadPair32View has no source");
      return storage_.lo ? storage_.lo->template simd_load<T>(lane_base)
                         : util::broadcast<T>(scalar_fallback(/*high=*/false));
    }

    template <typename T> [[nodiscard]] util::native<T> load_hi_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_hi_native expects 32-bit lanes");
      assert((storage_.hi || scalar_fallback_) && "OperandReadPair32View has no source");
      return storage_.hi ? storage_.hi->template simd_load<T>(lane_base)
                         : util::broadcast<T>(scalar_fallback(/*high=*/true));
    }

  private:
    friend class RegisterAccess;

    OperandReadPair32View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage)
        : storage_(storage) {
      if (!storage_.lo)
        scalar_fallback_.emplace(RegisterAccess(wf).read_lane_pair32(op, 0));
    }

    uint32_t scalar_fallback(bool high) const {
      const OperandPair32 pair =
          RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadPair32View");
      return high ? pair.hi : pair.lo;
    }

    ConstVgprStoragePair64 storage_{};
    std::optional<OperandPair32> scalar_fallback_;
  };

  class OperandWritePair32View {
  public:
    OperandWritePair32View() = delete;

    [[nodiscard]] bool has_storage() const { return storage_.lo != nullptr; }

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

    [[nodiscard]] uint32_t base() const { return base_; }
    [[nodiscard]] uint32_t reg_count() const { return reg_count_; }
    [[nodiscard]] uint32_t wf_size() const { return wf_size_; }
    [[nodiscard]] bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

    [[nodiscard]] std::span<const uint32_t> lanes(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprReadRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      return {reg_data(relative_reg), wf_size_};
    }

    [[nodiscard]] const uint32_t *reg_data(uint32_t relative_reg = 0) const {
      assert(cu_ && "VgprReadRegion is empty");
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      return reinterpret_cast<const uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
    }

    [[nodiscard]] uint32_t lane(uint32_t relative_reg, uint32_t lane) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      return lanes(relative_reg)[lane];
    }

    [[nodiscard]] uint64_t lane64(uint32_t relative_reg, uint32_t lane) const {
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

    [[nodiscard]] uint32_t base() const { return base_; }
    [[nodiscard]] uint32_t reg_count() const { return reg_count_; }
    [[nodiscard]] uint32_t wf_size() const { return wf_size_; }
    [[nodiscard]] uint64_t lane_mask() const { return lane_mask_; }
    [[nodiscard]] bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

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

    [[nodiscard]] const VgprReadRegion &read() const { return read_; }
    [[nodiscard]] const VgprWriteRegion &write() const { return write_; }

    [[nodiscard]] std::span<const uint32_t> read_lanes(uint32_t relative_reg = 0) const {
      return read_.lanes(relative_reg);
    }

    [[nodiscard]] uint32_t linear_word(uint32_t linear_index) const {
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
  explicit RegisterAccess(InstructionComputeUnitView &cu)
      : cu_(&cu.raw_cu()), mutable_cu_(&cu.raw_cu()) {}
  explicit RegisterAccess(const InstructionComputeUnitView &cu) : cu_(&cu.raw_cu()) {}
  explicit RegisterAccess(Wavefront &wf) : RegisterAccess(wf.cu()) {
    wf_ = &wf;
    mutable_wf_ = &wf;
  }
  explicit RegisterAccess(const Wavefront &wf) : RegisterAccess(wf.cu()) { wf_ = &wf; }

  // Scalar and per-lane operand access. Instruction implementations use these
  // for value-semantic operand reads and writes; Operand remains the
  // ISA-specific resolver/backend.
  [[nodiscard]] uint32_t read_scalar(const Operand &op) const {
    return op.read_scalar(wavefront());
  }
  [[nodiscard]] uint64_t read_scalar64(const Operand &op) const {
    return op.read_scalar64(wavefront());
  }
  [[nodiscard]] uint32_t read_lane(const Operand &op, uint32_t lane) const {
    return op.read_lane(wavefront(), lane);
  }
  [[nodiscard]] uint64_t read_lane64(const Operand &op, uint32_t lane) const {
    return op.read_lane64(wavefront(), lane);
  }
  [[nodiscard]] OperandPair32 read_lane_pair32(const Operand &op, uint32_t lane) const {
    if (const auto literal = op.literal64_value())
      return {static_cast<uint32_t>(*literal), static_cast<uint32_t>(*literal >> 32)};

    if (const auto reg = op.to_register_ref(); reg && reg->width >= 2) {
      const uint64_t pair = op.read_lane64(wavefront(), lane);
      return {static_cast<uint32_t>(pair), static_cast<uint32_t>(pair >> 32)};
    }

    const uint32_t value = op.read_lane(wavefront(), lane);
    return {value, value};
  }
  void write_scalar(const Operand &op, uint32_t value) const {
    op.write_scalar(mutable_wavefront(), value);
  }
  void write_scalar64(const Operand &op, uint64_t value) const {
    op.write_scalar64(mutable_wavefront(), value);
  }
  void write_lane(const Operand &op, uint32_t lane, uint32_t value) const {
    op.write_lane(mutable_wavefront(), lane, value);
  }
  void write_lane64(const Operand &op, uint32_t lane, uint64_t value) const {
    op.write_lane64(mutable_wavefront(), lane, value);
  }
  void read_chunk(const Operand &op, uint32_t lane_base, uint32_t count, uint32_t *out) const {
    op.read_lane_chunk(wavefront(), lane_base, count, out);
  }
  void write_chunk(const Operand &op, uint32_t lane_base, uint32_t count, const uint32_t *values,
                   uint64_t lane_mask) const {
    op.write_lane_chunk(mutable_wavefront(), lane_base, count, values, lane_mask);
  }

  // Logical operand access. These APIs are for instruction helpers that still
  // want operand semantics for scalar fallback, literals, delegates, and
  // 32/64-bit VGPR pairing, but need SIMD-friendly storage when available.
  // The default byte mask is deliberately conservative: operand width alone
  // does not identify high-vs-low sub-dword selections such as true16 op_sel.
  // Callers with precise byte windows can pass a narrower mask explicitly.
  [[nodiscard]] OperandReadView read_operand(const Operand &op, uint64_t lane_mask,
                                             uint8_t byte_mask = 0xF) const {
    const Wavefront &wf = wavefront();
    const VgprStorage *storage = op.simd_vgpr_storage(wf);
    if (storage)
      op.simd_notify_read(wf, lane_mask, byte_mask);
    return OperandReadView(op, wf, storage);
  }

  [[nodiscard]] OperandRead64View read_operand64(const Operand &op, uint64_t lane_mask,
                                                 uint8_t byte_mask = 0xF) const {
    const Wavefront &wf = wavefront();
    ConstVgprStoragePair64 storage = op.simd_vgpr_storage64(wf);
    if (storage.lo)
      op.simd_notify_read64(wf, lane_mask, byte_mask);
    return OperandRead64View(op, wf, storage);
  }

  [[nodiscard]] OperandReadPair32View read_operand_pair32(const Operand &op, uint64_t lane_mask,
                                                          uint8_t byte_mask = 0xF) const {
    const Wavefront &wf = wavefront();
    ConstVgprStoragePair64 storage = op.simd_vgpr_storage64(wf);
    if (storage.lo)
      op.simd_notify_read64(wf, lane_mask, byte_mask);
    return OperandReadPair32View(op, wf, storage);
  }

  [[nodiscard]] OperandWriteView write_operand(const Operand &op, uint64_t lane_mask) const {
    // Kept at acquisition time so future write-observation plugins can validate
    // the instruction's destination lane set before writable storage is exposed.
    (void)lane_mask;
    Wavefront &wf = mutable_wavefront();
    return OperandWriteView(op, wf, op.simd_vgpr_storage_mut(wf));
  }

  [[nodiscard]] OperandWrite64View write_operand64(const Operand &op, uint64_t lane_mask) const {
    // See write_operand().
    (void)lane_mask;
    Wavefront &wf = mutable_wavefront();
    return OperandWrite64View(op, wf, op.simd_vgpr_storage64_mut(wf));
  }

  [[nodiscard]] OperandWritePair32View write_operand_pair32(const Operand &op,
                                                            uint64_t lane_mask) const {
    // See write_operand().
    (void)lane_mask;
    Wavefront &wf = mutable_wavefront();
    return OperandWritePair32View(op, wf, op.simd_vgpr_storage64_mut(wf));
  }

  [[nodiscard]] OperandReadWriteView readwrite_operand(const Operand &op, uint64_t lane_mask,
                                                       uint8_t byte_mask = 0xF) const {
    Wavefront &wf = mutable_wavefront();
    VgprStorage *storage = op.simd_vgpr_storage_mut(wf);
    if (storage)
      op.simd_notify_read_mut(wf, lane_mask, byte_mask);
    return OperandReadWriteView(op, wf, storage);
  }

  [[nodiscard]] OperandReadWrite64View readwrite_operand64(const Operand &op, uint64_t lane_mask,
                                                           uint8_t byte_mask = 0xF) const {
    Wavefront &wf = mutable_wavefront();
    VgprStoragePair64 storage = op.simd_vgpr_storage64_mut(wf);
    if (storage.lo)
      op.simd_notify_read64_mut(wf, lane_mask, byte_mask);
    return OperandReadWrite64View(op, wf, storage);
  }

  // Physical SGPR access. These APIs are for helpers that already know the
  // physical scalar register index. ComputeUnitCore owns SGPR read observation,
  // so reads delegate to the CU accessor rather than exposing raw SGPR storage.
  [[nodiscard]] uint32_t read_sgpr(uint32_t physical_reg) const {
    return cu_->read_sgpr(physical_reg);
  }

  [[nodiscard]] uint64_t read_sgpr64(uint32_t physical_reg) const {
    uint64_t lo = read_sgpr(physical_reg);
    uint64_t hi = read_sgpr(physical_reg + 1);
    return lo | (hi << 32);
  }

  void write_sgpr(uint32_t physical_reg, uint32_t value) const {
    mutable_cu().write_sgpr(physical_reg, value);
  }

  void write_sgpr64(uint32_t physical_reg, uint64_t value) const {
    write_sgpr(physical_reg, static_cast<uint32_t>(value));
    write_sgpr(physical_reg + 1, static_cast<uint32_t>(value >> 32));
  }

  // Physical VGPR access. These APIs are for helpers that already know the
  // physical register index, such as matrix layout code and generated memory
  // address/data collection. Reads observe the supplied register/lane range
  // before returning views over the raw storage.
  [[nodiscard]] uint32_t read_vgpr(uint32_t physical_reg, uint32_t lane,
                                   uint8_t byte_mask = 0xF) const {
    return read_vgpr_region(physical_reg, 1, uint64_t{1} << lane, byte_mask).lane(0, lane);
  }

  [[nodiscard]] uint64_t read_vgpr64(uint32_t physical_reg, uint32_t lane,
                                     uint8_t byte_mask = 0xF) const {
    return read_vgpr_region(physical_reg, 2, uint64_t{1} << lane, byte_mask).lane64(0, lane);
  }

  void write_vgpr(uint32_t physical_reg, uint32_t lane, uint32_t value) const {
    write_vgpr_region(physical_reg, 1, uint64_t{1} << lane).set_lane(0, lane, value);
  }

  void write_vgpr64(uint32_t physical_reg, uint32_t lane, uint64_t value) const {
    write_vgpr_region(physical_reg, 2, uint64_t{1} << lane).set_lane64(0, lane, value);
  }

  [[nodiscard]] VgprReadRegion read_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                                uint64_t lane_mask, uint8_t byte_mask = 0xF) const {
    observe_vgpr_region(physical_base, reg_count, lane_mask, byte_mask);
    return VgprReadRegion(*cu_, physical_base, reg_count);
  }

  [[nodiscard]] VgprWriteRegion write_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                                  uint64_t lane_mask) const {
    return VgprWriteRegion(mutable_cu(), physical_base, reg_count, lane_mask);
  }

  [[nodiscard]] VgprReadWriteRegion readwrite_vgpr_region(uint32_t physical_base,
                                                          uint32_t reg_count, uint64_t lane_mask,
                                                          uint8_t byte_mask = 0xF) const {
    return VgprReadWriteRegion(read_vgpr_region(physical_base, reg_count, lane_mask, byte_mask),
                               write_vgpr_region(physical_base, reg_count, lane_mask));
  }

private:
  const Wavefront &wavefront() const {
    if (!wf_)
      throw std::logic_error("RegisterAccess was not constructed from a Wavefront");
    return *wf_;
  }

  Wavefront &mutable_wavefront() const {
    if (!mutable_wf_)
      throw std::logic_error("RegisterAccess was not constructed from mutable Wavefront");
    return *mutable_wf_;
  }

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
  const Wavefront *wf_ = nullptr;
  Wavefront *mutable_wf_ = nullptr;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
