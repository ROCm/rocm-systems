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
/// exposing scalar values, SIMD storage, or physical register regions. Write
/// views and regions notify at acquisition, before writable storage is exposed.
/// Every later view store is constrained to the lane mask covered by that
/// notification. Write-only accessors do not report reads. Read-write accessors
/// report the read part at acquisition time and then allow the caller to update
/// the same instruction-scoped storage. VM/storage code may still use raw
/// register storage for tasks such as memory completion, but instruction
/// emulators should use this facade for operand and physical register access.

#ifndef ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
#define ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_

#include "rocjitsu/isa/arch/amdgpu/shared/scalar_operand_selectors.h"
#include "rocjitsu/isa/operand.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/plugins/execution_plugin.h"
#include "simdojo/components/vector_reg.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

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
/// - Use read_lane_pair32() when a packed operation treats a 64-bit register
///   source as two 32-bit words but splats single-word sources.
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
/// storage is exposed. VGPR region reads notify once per physical register with
/// the caller-provided lane and byte masks; SGPR region reads carry one typed
/// architectural range. Write and read-write acquisition similarly fires the
/// plugin write hook before mutable storage is exposed. A write view remembers
/// the observed lane mask and rejects caller stores outside its acquisition
/// window. Write-only views deliberately do not report reads.
///
/// Operand read views may be VGPR-backed or scalar-backed. Scalar-backed views
/// represent SGPR, inline literal, immediate, and special-register operands as
/// lane-broadcast values; pair views preserve distinct words for register
/// pairs. They do not imply a missing VGPR read.
///
/// The view objects are intentionally lightweight and instruction-scoped. They
/// expose spans over the underlying VGPR lane storage so hot paths can keep the
/// current zero-copy behavior while the observation contract remains localized
/// here. They should be acquired during a single instruction's emulation and
/// not cached across instructions.
///
/// Logical operand APIs require construction from a Wavefront. Physical VGPR
/// APIs may use a CU-bound compatibility path, but a complete region must map
/// to one live wave before any callback or storage view is produced. Physical
/// SGPR writes require a mutable Wavefront; raw VM/runtime writes belong on
/// ComputeUnitCore and deliberately bypass this instruction-facing facade.
class RegisterAccess {
  static uint32_t byte_bit_mask(uint8_t byte_mask) {
    uint32_t bit_mask = 0;
    for (uint32_t byte = 0; byte < sizeof(uint32_t); ++byte)
      if (byte_mask & (uint8_t{1} << byte))
        bit_mask |= uint32_t{0xFF} << (byte * 8);
    return bit_mask;
  }

  template <typename T>
  static T require_scalar_fallback(const std::optional<T> &fallback, const char *view_name) {
    if (!fallback)
      throw std::logic_error(std::string(view_name) + " has no scalar fallback");
    return *fallback;
  }

  static uint64_t checked_store_lane_mask(uint64_t observed_lane_mask, uint32_t lane_base,
                                          std::size_t lane_count, uint64_t lane_mask,
                                          const char *view_name) {
    if (lane_base >= 64 || lane_count > 64 - lane_base)
      throw std::logic_error(std::string(view_name) + " store lane window exceeds wave size");
    const uint64_t width_mask =
        lane_count == 64 ? ~uint64_t{0} : (uint64_t{1} << lane_count) - uint64_t{1};
    const uint64_t effective_lane_mask = lane_mask & width_mask;
    const uint64_t global_lane_mask = effective_lane_mask << lane_base;
    if ((global_lane_mask & ~observed_lane_mask) != 0)
      throw std::logic_error(std::string(view_name) +
                             " store lane mask exceeds plugin-observed lanes");
    return effective_lane_mask;
  }

public:
  /// @brief Transform a merged destination dword using current architectural state.
  using PostWriteTransform = uint32_t (*)(uint32_t, const Wavefront &);

  class OperandReadView {
  public:
    OperandReadView() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_); }

    [[nodiscard]] uint32_t lane(uint32_t lane) const {
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      return (storage_ ? storage_[lane] : scalar_fallback()) &
             RegisterAccess::byte_bit_mask(byte_mask_);
    }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return storage_ ? storage_.template simd_load<T>(lane_base)
                        : util::broadcast<T>(scalar_fallback());
      auto bits = storage_ ? storage_.template simd_load<uint32_t>(lane_base)
                           : util::broadcast<uint32_t>(scalar_fallback());
      bits &= util::broadcast<uint32_t>(RegisterAccess::byte_bit_mask(byte_mask_));
      return std::bit_cast<util::native<T>>(bits);
    }

    template <typename T> [[nodiscard]] util::narrow32<T> load_narrow(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_narrow expects 32-bit lanes");
      assert((storage_ || scalar_fallback_) && "OperandReadView has no source");
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return storage_ ? storage_.template simd_load_narrow<T>(lane_base)
                        : util::broadcast_narrow<T>(scalar_fallback());
      auto bits = storage_ ? storage_.template simd_load_narrow<uint32_t>(lane_base)
                           : util::broadcast_narrow<uint32_t>(scalar_fallback());
      bits &= util::broadcast_narrow<uint32_t>(RegisterAccess::byte_bit_mask(byte_mask_));
      alignas(util::narrow32<uint32_t>) uint32_t buffer[util::native_width64];
      bits.copy_to(buffer, util::stdx::vector_aligned);
      return util::load_narrow<T>(buffer);
    }

  private:
    friend class RegisterAccess;

    OperandReadView(const Operand &op, const Wavefront &wf, ConstVgprStorage storage,
                    uint8_t byte_mask, bool denied)
        : storage_(storage), byte_mask_(byte_mask) {
      if (!storage_)
        scalar_fallback_.emplace(denied ? 0u : op.read_lane(wf, 0));
    }

    uint32_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadView");
    }

    ConstVgprStorage storage_{};
    std::optional<uint32_t> scalar_fallback_;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
  };

  class OperandWriteView {
  public:
    OperandWriteView() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_); }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWriteView is empty");
      constexpr std::size_t W = util::native_width_v<T>;
      if (storage_)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandWriteView");
      if (storage_) {
        if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask) {
          storage_.template simd_store<T>(lane_base, value, lane_mask);
          return;
        }
        alignas(util::native<T>) uint32_t values[W];
        util::blit_to_buffer<T>(values, value);
        const uint32_t bit_mask = RegisterAccess::byte_bit_mask(byte_mask_);
        for (std::size_t lane = 0; lane < W; ++lane) {
          if ((lane_mask & (uint64_t{1} << lane)) == 0)
            continue;
          uint32_t &destination = storage_[lane_base + static_cast<uint32_t>(lane)];
          destination = (destination & ~bit_mask) | (values[lane] & bit_mask);
        }
        return;
      }
      if (denied_)
        return;
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte OperandWriteView requires VGPR storage");
      alignas(util::native<T>) uint32_t buf[W];
      util::blit_to_buffer<T>(buf, value);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

    template <typename T>
    void store_narrow(uint32_t lane_base, util::narrow32<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_narrow expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWriteView is empty");
      constexpr std::size_t W = util::native_width64;
      if (storage_)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandWriteView");
      if (storage_) {
        if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask) {
          storage_.template simd_store_narrow<T>(lane_base, value, lane_mask);
          return;
        }
        alignas(util::narrow32<T>) T values[W];
        value.copy_to(values, util::stdx::vector_aligned);
        const uint32_t bit_mask = RegisterAccess::byte_bit_mask(byte_mask_);
        for (std::size_t lane = 0; lane < W; ++lane) {
          if ((lane_mask & (uint64_t{1} << lane)) == 0)
            continue;
          uint32_t bits = std::bit_cast<uint32_t>(values[lane]);
          uint32_t &destination = storage_[lane_base + static_cast<uint32_t>(lane)];
          destination = (destination & ~bit_mask) | (bits & bit_mask);
        }
        return;
      }
      if (denied_)
        return;
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte OperandWriteView requires VGPR storage");
      alignas(util::narrow32<T>) T vals[W];
      value.copy_to(vals, util::stdx::vector_aligned);
      uint32_t buf[W];
      for (std::size_t i = 0; i < W; ++i)
        buf[i] = std::bit_cast<uint32_t>(vals[i]);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

  private:
    friend class RegisterAccess;

    OperandWriteView(const Operand &op, Wavefront &wf, VgprStorage storage,
                     uint64_t permitted_write_lane_mask, uint8_t byte_mask, bool denied)
        : op_(&op), wf_(&wf), storage_(storage),
          permitted_write_lane_mask_(permitted_write_lane_mask), byte_mask_(byte_mask),
          denied_(denied) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStorage storage_{};
    uint64_t permitted_write_lane_mask_ = 0;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
    bool denied_ = false;
  };

  class OperandWrite64View {
  public:
    OperandWrite64View() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_.lo); }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "store_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandWrite64View is empty");
      constexpr std::size_t W = util::native_width64;
      if (storage_.lo)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandWrite64View");
      if (storage_.lo) {
        storage_.lo.template simd_store64<T>(storage_.hi, lane_base, value, lane_mask);
        return;
      }
      if (denied_)
        return;
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

    OperandWrite64View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage,
                       uint64_t permitted_write_lane_mask, bool denied)
        : op_(&op), wf_(&wf), storage_(storage),
          permitted_write_lane_mask_(permitted_write_lane_mask), denied_(denied) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
    uint64_t permitted_write_lane_mask_ = 0;
    bool denied_ = false;
  };

  class OperandReadWriteView {
  public:
    OperandReadWriteView() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_); }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      if (read_byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return storage_ ? storage_.template simd_load<T>(lane_base)
                        : util::broadcast<T>(scalar_fallback());
      auto bits = storage_ ? storage_.template simd_load<uint32_t>(lane_base)
                           : util::broadcast<uint32_t>(scalar_fallback());
      bits &= util::broadcast<uint32_t>(RegisterAccess::byte_bit_mask(read_byte_mask_));
      return std::bit_cast<util::native<T>>(bits);
    }

    template <typename T> [[nodiscard]] util::narrow32<T> load_narrow(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_narrow expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      if (read_byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return storage_ ? storage_.template simd_load_narrow<T>(lane_base)
                        : util::broadcast_narrow<T>(scalar_fallback());
      auto bits = storage_ ? storage_.template simd_load_narrow<uint32_t>(lane_base)
                           : util::broadcast_narrow<uint32_t>(scalar_fallback());
      bits &= util::broadcast_narrow<uint32_t>(RegisterAccess::byte_bit_mask(read_byte_mask_));
      alignas(util::narrow32<uint32_t>) uint32_t buffer[util::native_width64];
      bits.copy_to(buffer, util::stdx::vector_aligned);
      return util::load_narrow<T>(buffer);
    }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      constexpr std::size_t W = util::native_width_v<T>;
      if (storage_)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandReadWriteView");
      if (storage_) {
        if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask) {
          storage_.template simd_store<T>(lane_base, value, lane_mask);
          return;
        }
        alignas(util::native<T>) uint32_t values[W];
        util::blit_to_buffer<T>(values, value);
        const uint32_t bit_mask = RegisterAccess::byte_bit_mask(byte_mask_);
        for (std::size_t lane = 0; lane < W; ++lane) {
          if ((lane_mask & (uint64_t{1} << lane)) == 0)
            continue;
          uint32_t &destination = storage_[lane_base + static_cast<uint32_t>(lane)];
          destination = (destination & ~bit_mask) | (values[lane] & bit_mask);
        }
        return;
      }
      if (denied_)
        return;
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte OperandReadWriteView requires VGPR storage");
      alignas(util::native<T>) uint32_t buf[W];
      util::blit_to_buffer<T>(buf, value);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

    template <typename T>
    void store_narrow(uint32_t lane_base, util::narrow32<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_narrow expects 32-bit lanes");
      assert(op_ && wf_ && "OperandReadWriteView is empty");
      constexpr std::size_t W = util::native_width64;
      if (storage_)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandReadWriteView");
      if (storage_) {
        if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask) {
          storage_.template simd_store_narrow<T>(lane_base, value, lane_mask);
          return;
        }
        alignas(util::narrow32<T>) T values[W];
        value.copy_to(values, util::stdx::vector_aligned);
        const uint32_t bit_mask = RegisterAccess::byte_bit_mask(byte_mask_);
        for (std::size_t lane = 0; lane < W; ++lane) {
          if ((lane_mask & (uint64_t{1} << lane)) == 0)
            continue;
          uint32_t bits = std::bit_cast<uint32_t>(values[lane]);
          uint32_t &destination = storage_[lane_base + static_cast<uint32_t>(lane)];
          destination = (destination & ~bit_mask) | (bits & bit_mask);
        }
        return;
      }
      if (denied_)
        return;
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte OperandReadWriteView requires VGPR storage");
      alignas(util::narrow32<T>) T vals[W];
      value.copy_to(vals, util::stdx::vector_aligned);
      uint32_t buf[W];
      for (std::size_t i = 0; i < W; ++i)
        buf[i] = std::bit_cast<uint32_t>(vals[i]);
      op_->write_lane_chunk(*wf_, lane_base, static_cast<uint32_t>(W), buf, lane_mask);
    }

  private:
    friend class RegisterAccess;

    OperandReadWriteView(const Operand &op, Wavefront &wf, VgprStorage storage,
                         uint64_t permitted_write_lane_mask, uint8_t read_byte_mask,
                         uint8_t write_byte_mask, bool denied)
        : op_(&op), wf_(&wf), storage_(storage),
          permitted_write_lane_mask_(permitted_write_lane_mask), read_byte_mask_(read_byte_mask),
          byte_mask_(write_byte_mask), denied_(denied) {
      if (!storage_)
        scalar_fallback_.emplace(denied ? 0u : op.read_lane(wf, 0));
    }

    uint32_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadWriteView");
    }

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStorage storage_{};
    std::optional<uint32_t> scalar_fallback_;
    uint64_t permitted_write_lane_mask_ = 0;
    uint8_t read_byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
    bool denied_ = false;
  };

  class OperandReadWrite64View {
  public:
    OperandReadWrite64View() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_.lo); }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "load_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandReadWrite64View is empty");
      auto value = storage_.lo ? storage_.lo.template simd_load64<T>(storage_.hi, lane_base)
                               : util::broadcast64<T>(scalar_fallback());
      if (read_byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return value;
      auto bits = std::bit_cast<util::native<uint64_t>>(value);
      uint64_t dword_mask = RegisterAccess::byte_bit_mask(read_byte_mask_);
      bits &= util::broadcast64<uint64_t>(dword_mask | (dword_mask << 32));
      return std::bit_cast<util::native<T>>(bits);
    }

    template <typename T>
    void store_native(uint32_t lane_base, util::native<T> value, uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "store_native expects 64-bit lanes");
      assert(op_ && wf_ && "OperandReadWrite64View is empty");
      constexpr std::size_t W = util::native_width64;
      if (storage_.lo)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandReadWrite64View");
      if (storage_.lo) {
        storage_.lo.template simd_store64<T>(storage_.hi, lane_base, value, lane_mask);
        return;
      }
      if (denied_)
        return;
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

    OperandReadWrite64View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage,
                           uint64_t permitted_write_lane_mask, uint8_t read_byte_mask, bool denied)
        : op_(&op), wf_(&wf), storage_(storage),
          permitted_write_lane_mask_(permitted_write_lane_mask), read_byte_mask_(read_byte_mask),
          denied_(denied) {
      if (!storage_.lo)
        scalar_fallback_.emplace(denied ? 0u : op.read_lane64(wf, 0));
    }

    uint64_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadWrite64View");
    }

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
    std::optional<uint64_t> scalar_fallback_;
    uint64_t permitted_write_lane_mask_ = 0;
    uint8_t read_byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
    bool denied_ = false;
  };

  class OperandRead64View {
  public:
    OperandRead64View() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_.lo); }

    [[nodiscard]] uint64_t lane(uint32_t lane) const {
      assert((storage_.lo || scalar_fallback_) && "OperandRead64View has no source");
      uint64_t value = storage_.lo
                           ? uint64_t{storage_.lo[lane]} | (uint64_t{storage_.hi[lane]} << 32)
                           : scalar_fallback();
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return value;
      uint64_t dword_mask = RegisterAccess::byte_bit_mask(byte_mask_);
      return value & (dword_mask | (dword_mask << 32));
    }

    template <typename T> [[nodiscard]] util::native<T> load_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint64_t), "load_native expects 64-bit lanes");
      assert((storage_.lo || scalar_fallback_) && "OperandRead64View has no source");
      auto value = storage_.lo ? storage_.lo.template simd_load64<T>(storage_.hi, lane_base)
                               : util::broadcast64<T>(scalar_fallback());
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return value;
      auto bits = std::bit_cast<util::native<uint64_t>>(value);
      uint64_t dword_mask = RegisterAccess::byte_bit_mask(byte_mask_);
      bits &= util::broadcast64<uint64_t>(dword_mask | (dword_mask << 32));
      return std::bit_cast<util::native<T>>(bits);
    }

  private:
    friend class RegisterAccess;

    OperandRead64View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage,
                      uint8_t byte_mask, bool denied)
        : storage_(storage), byte_mask_(byte_mask) {
      if (!storage_.lo)
        scalar_fallback_.emplace(denied ? 0u : op.read_lane64(wf, 0));
    }

    uint64_t scalar_fallback() const {
      return RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandRead64View");
    }

    ConstVgprStoragePair64 storage_{};
    std::optional<uint64_t> scalar_fallback_;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
  };

  class OperandReadPair32View {
  public:
    OperandReadPair32View() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_.lo); }

    template <typename T> [[nodiscard]] util::native<T> load_lo_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_lo_native expects 32-bit lanes");
      assert((storage_.lo || scalar_fallback_) && "OperandReadPair32View has no source");
      auto value = storage_.lo ? storage_.lo.template simd_load<T>(lane_base)
                               : util::broadcast<T>(scalar_fallback(/*high=*/false));
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return value;
      auto bits = std::bit_cast<util::native<uint32_t>>(value);
      bits &= util::broadcast<uint32_t>(RegisterAccess::byte_bit_mask(byte_mask_));
      return std::bit_cast<util::native<T>>(bits);
    }

    template <typename T> [[nodiscard]] util::native<T> load_hi_native(uint32_t lane_base) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "load_hi_native expects 32-bit lanes");
      assert((storage_.hi || scalar_fallback_) && "OperandReadPair32View has no source");
      auto value = storage_.hi ? storage_.hi.template simd_load<T>(lane_base)
                               : util::broadcast<T>(scalar_fallback(/*high=*/true));
      if (byte_mask_ == rocjitsu::ExecutionPlugin::kFullByteMask)
        return value;
      auto bits = std::bit_cast<util::native<uint32_t>>(value);
      bits &= util::broadcast<uint32_t>(RegisterAccess::byte_bit_mask(byte_mask_));
      return std::bit_cast<util::native<T>>(bits);
    }

  private:
    friend class RegisterAccess;

    OperandReadPair32View(const Operand &op, const Wavefront &wf, ConstVgprStoragePair64 storage,
                          uint8_t byte_mask, bool denied)
        : storage_(storage), byte_mask_(byte_mask) {
      if (!storage_.lo)
        scalar_fallback_.emplace(denied ? OperandPair32{}
                                        : RegisterAccess(wf).read_lane_pair32(op, 0));
    }

    uint32_t scalar_fallback(bool high) const {
      const OperandPair32 pair =
          RegisterAccess::require_scalar_fallback(scalar_fallback_, "OperandReadPair32View");
      return high ? pair.hi : pair.lo;
    }

    ConstVgprStoragePair64 storage_{};
    std::optional<OperandPair32> scalar_fallback_;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
  };

  class OperandWritePair32View {
  public:
    OperandWritePair32View() = delete;

    [[nodiscard]] bool has_storage() const { return static_cast<bool>(storage_.lo); }

    template <typename T>
    void store_native_pair(uint32_t lane_base, util::native<T> lo, util::native<T> hi,
                           uint64_t lane_mask) const {
      static_assert(sizeof(T) == sizeof(uint32_t), "store_native_pair expects 32-bit lanes");
      assert(op_ && wf_ && "OperandWritePair32View is empty");
      constexpr std::size_t W = util::native_width_v<T>;
      if (storage_.lo)
        lane_mask &= wf_->vgpr_write_mask() >> lane_base;
      lane_mask = RegisterAccess::checked_store_lane_mask(permitted_write_lane_mask_, lane_base, W,
                                                          lane_mask, "OperandWritePair32View");
      if (storage_.lo) {
        storage_.lo.template simd_store<T>(lane_base, lo, lane_mask);
        storage_.hi.template simd_store<T>(lane_base, hi, lane_mask);
        return;
      }
      if (denied_)
        return;
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

    OperandWritePair32View(const Operand &op, Wavefront &wf, VgprStoragePair64 storage,
                           uint64_t permitted_write_lane_mask, bool denied)
        : op_(&op), wf_(&wf), storage_(storage),
          permitted_write_lane_mask_(permitted_write_lane_mask), denied_(denied) {}

    const Operand *op_ = nullptr;
    Wavefront *wf_ = nullptr;
    VgprStoragePair64 storage_{};
    uint64_t permitted_write_lane_mask_ = 0;
    bool denied_ = false;
  };

  class VgprReadRegion {
  public:
    VgprReadRegion() = delete;

    [[nodiscard]] uint32_t base() const { return base_; }
    [[nodiscard]] uint32_t reg_count() const { return reg_count_; }
    [[nodiscard]] uint32_t wf_size() const { return wf_size_; }
    [[nodiscard]] bool valid() const { return valid_; }
    [[nodiscard]] bool empty() const { return reg_count_ == 0 || !observed_; }

    /// @brief Visit each logical VGPR as one lane span, in register order.
    template <typename Function> void for_each(Function &&function) const {
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte VgprReadRegion does not expose raw lane spans");
      if (!valid_ || !observed_) {
        for (uint32_t reg = 0; reg < reg_count_; ++reg)
          function(std::span<const uint32_t>(zero_lanes_.data(), wf_size_));
        return;
      }
      cu_->for_each_raw_vgpr(base_, reg_count_, [&](std::span<const uint32_t> physical_lanes) {
        assert(physical_lanes.size() >= wf_size_ && "physical VGPR is narrower than wave");
        function(physical_lanes.first(wf_size_));
      });
    }

    /// @brief Gather wave-width lane words from this logical VGPR range.
    void copy_to(std::span<uint32_t> destination) const {
      assert(destination.size() <= static_cast<size_t>(reg_count_) * wf_size_ &&
             "destination exceeds read region");
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte VgprReadRegion cannot copy raw words");
      if (!valid_ || !observed_) {
        std::fill(destination.begin(), destination.end(), 0);
        return;
      }
      size_t copied = 0;
      for_each([&](std::span<const uint32_t> lanes) {
        const size_t count = std::min(lanes.size(), destination.size() - copied);
        std::copy_n(lanes.begin(), count, destination.begin() + copied);
        copied += count;
      });
    }

    [[nodiscard]] std::span<const uint32_t> lanes(uint32_t relative_reg = 0) const {
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte VgprReadRegion does not expose raw lane spans");
      if (!valid_ || !observed_)
        return {zero_lanes_.data(), wf_size_};
      return {reg_data(relative_reg), wf_size_};
    }

    [[nodiscard]] uint32_t lane(uint32_t relative_reg, uint32_t lane) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      if (!valid_ || !observed_)
        return 0;
      const auto *data =
          reinterpret_cast<const uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
      return data[lane] & RegisterAccess::byte_bit_mask(byte_mask_);
    }

    [[nodiscard]] uint64_t lane64(uint32_t relative_reg, uint32_t lane) const {
      assert(relative_reg + 1 < reg_count_ && "64-bit lane read needs two VGPRs");
      uint64_t lo = this->lane(relative_reg, lane);
      uint64_t hi = this->lane(relative_reg + 1, lane);
      return lo | (hi << 32);
    }

    /// @brief Snapshot dword registers into lane-major memory.
    /// @details The region acquisition has already reported one plugin read per
    /// register. This copies only selected lanes without repeating ownership
    /// lookup and plugin dispatch for every individual lane.
    void copy_dwords_lane_major(uint8_t *destination, uint64_t lane_mask) const {
      assert(destination != nullptr && "VGPR snapshot destination must not be null");
      const size_t lane_stride = static_cast<size_t>(reg_count_) * sizeof(uint32_t);
      for (uint32_t reg = 0; reg < reg_count_; ++reg) {
        const auto values = lanes(reg);
        uint64_t active_lanes = lane_mask;
        while (active_lanes) {
          const uint32_t lane = std::countr_zero(active_lanes);
          active_lanes &= active_lanes - 1;
          std::memcpy(destination + static_cast<size_t>(lane) * lane_stride +
                          static_cast<size_t>(reg) * sizeof(uint32_t),
                      &values[lane], sizeof(uint32_t));
        }
      }
    }

  private:
    friend class RegisterAccess;

    /// @brief Return the lane storage for one VGPR in this region.
    /// @details The returned pointer spans exactly @c wf_size() lanes. Adjacent
    /// VGPRs are not required to have contiguous backing, and this pointer must
    /// not be incremented to traverse adjacent registers. For an unmaterialized
    /// register, the pointer may refer to shared immutable zero backing: treat
    /// it as an ephemeral value snapshot that need not observe a later write
    /// through another handle. Pointers must not be retained after the owning
    /// wavefront retires.
    [[nodiscard]] const uint32_t *reg_data(uint32_t relative_reg) const {
      assert(relative_reg < reg_count_ && "relative VGPR outside read region");
      if (byte_mask_ != rocjitsu::ExecutionPlugin::kFullByteMask)
        throw std::logic_error("partial-byte VgprReadRegion does not expose raw register data");
      if (!valid_ || !observed_)
        return zero_lanes_.data();
      return reinterpret_cast<const uint32_t *>(cu_->raw_vgpr_data(base_ + relative_reg));
    }

    VgprReadRegion(const ComputeUnitCore &cu, uint32_t base, uint32_t reg_count, uint32_t wave_size,
                   uint8_t byte_mask, bool valid = true, bool observed = true)
        : cu_(&cu), base_(base), reg_count_(reg_count), wf_size_(wave_size), byte_mask_(byte_mask),
          valid_(valid), observed_(observed) {}

    const ComputeUnitCore *cu_ = nullptr;
    uint32_t base_ = 0;
    uint32_t reg_count_ = 0;
    uint32_t wf_size_ = 0;
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
    bool valid_ = false;
    bool observed_ = false;
    inline static constexpr std::array<uint32_t, 64> zero_lanes_{};
  };

  class VgprWriteRegion {
  public:
    VgprWriteRegion() = delete;

    [[nodiscard]] uint32_t base() const { return base_; }
    [[nodiscard]] uint32_t reg_count() const { return reg_count_; }
    [[nodiscard]] uint32_t wf_size() const { return wf_size_; }
    [[nodiscard]] uint64_t lane_mask() const { return lane_mask_; }
    [[nodiscard]] bool valid() const { return cu_ != nullptr; }
    [[nodiscard]] bool empty() const { return cu_ == nullptr || reg_count_ == 0; }

    void set_lane(uint32_t relative_reg, uint32_t lane, uint32_t value) const {
      assert(lane < wf_size_ && "lane outside wavefront");
      if (!cu_ || (lane_mask_ & (uint64_t{1} << lane)) == 0)
        return;
      uint32_t &destination = reg_data(relative_reg)[lane];
      const uint32_t bit_mask = RegisterAccess::byte_bit_mask(byte_mask_);
      destination = (destination & ~bit_mask) | (value & bit_mask);
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

    VgprWriteRegion(ComputeUnitCore &cu, uint32_t base, uint32_t reg_count, uint32_t wave_size,
                    uint64_t lane_mask, uint8_t byte_mask, bool valid = true)
        : cu_(valid ? &cu : nullptr), base_(base), reg_count_(reg_count), wf_size_(wave_size),
          lane_mask_(lane_mask), byte_mask_(byte_mask) {}

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
    uint8_t byte_mask_ = rocjitsu::ExecutionPlugin::kFullByteMask;
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

  /// @brief Read-only observed range of physical SGPRs.
  ///
  /// The region is valid only when its complete range belongs to one live
  /// wavefront. Acquisition validates the complete range and reports every
  /// dword before this object exposes a value. A denied region returns zero
  /// values for defensive callers, but callers that need to distinguish denial
  /// from architectural zero must check valid().
  class SgprReadRegion {
  public:
    SgprReadRegion() = delete;

    [[nodiscard]] uint32_t base() const { return base_; }
    [[nodiscard]] uint32_t reg_count() const { return reg_count_; }
    [[nodiscard]] bool valid() const { return valid_; }

    [[nodiscard]] uint32_t dword(uint32_t relative_reg = 0) const {
      assert(relative_reg < reg_count_ && "relative SGPR outside read region");
      return valid_ ? cu_->read_sgpr_storage(base_ + relative_reg) : 0;
    }

    [[nodiscard]] uint64_t qword(uint32_t relative_reg = 0) const {
      assert(relative_reg + 1 < reg_count_ && "SGPR pair outside read region");
      return static_cast<uint64_t>(dword(relative_reg)) |
             (static_cast<uint64_t>(dword(relative_reg + 1)) << 32);
    }

  private:
    friend class RegisterAccess;

    SgprReadRegion(const ComputeUnitCore &cu, uint32_t base, uint32_t reg_count, bool valid)
        : cu_(&cu), base_(base), reg_count_(reg_count), valid_(valid) {}

    const ComputeUnitCore *cu_ = nullptr;
    uint32_t base_ = 0;
    uint32_t reg_count_ = 0;
    bool valid_ = false;
  };

  explicit RegisterAccess(ComputeUnitCore &cu) : cu_(&cu), mutable_cu_(&cu) {}
  explicit RegisterAccess(const ComputeUnitCore &cu) : cu_(&cu) {}
  explicit RegisterAccess(InstructionComputeUnitView &cu)
      : cu_(&cu.raw_cu()), mutable_cu_(&cu.raw_cu()), wf_(&cu.raw_wavefront()),
        mutable_wf_(&cu.raw_wavefront()) {}
  explicit RegisterAccess(const InstructionComputeUnitView &cu)
      : cu_(&cu.raw_cu()), wf_(&cu.raw_wavefront()) {}
  explicit RegisterAccess(Wavefront &wf) : RegisterAccess(wf.cu()) {
    wf_ = &wf;
    mutable_wf_ = &wf;
  }
  explicit RegisterAccess(const Wavefront &wf) : RegisterAccess(wf.cu()) { wf_ = &wf; }

  // Scalar and per-lane operand access. Instruction implementations use these
  // for value-semantic operand reads and writes; Operand remains the
  // ISA-specific resolver/backend.
  [[nodiscard]] uint32_t read_scalar(const Operand &op) const {
    const Wavefront &wf = wavefront();
    if (auto base = op.simd_vgpr_base(wf); base && !cu_->owns_vgpr_range(wf, *base, 1))
      return 0;
    return op.read_scalar(wf);
  }
  [[nodiscard]] uint64_t read_scalar64(const Operand &op) const {
    const Wavefront &wf = wavefront();
    if (auto base = op.simd_vgpr_base(wf); base && !cu_->owns_vgpr_range(wf, *base, 2))
      return 0;
    return op.read_scalar64(wf);
  }
  [[nodiscard]] uint32_t read_lane(const Operand &op, uint32_t lane) const {
    const Wavefront &wf = wavefront();
    if (auto base = op.simd_vgpr_base(wf); base && !cu_->owns_vgpr_range(wf, *base, 1))
      return 0;
    return op.read_lane(wf, lane);
  }

  /// @brief Read the lane selected by a scalar-lane instruction.
  [[nodiscard]] uint32_t read_scalar_selected_lane(const Operand &op, uint32_t lane) const {
    const Wavefront &wf = wavefront();
    assert((wf.wf_size() == 32 || wf.wf_size() == 64) &&
           "scalar lane selection requires Wave32 or Wave64");
    lane &= wf.wf_size() - 1;
    auto physical_reg = op.simd_vgpr_base(wf);
    if (!physical_reg)
      throw std::logic_error("scalar lane read requires a VGPR source");
    if (!cu_->owns_vgpr_range(wf, *physical_reg, 1))
      return 0;
    return read_vgpr(*physical_reg, lane);
  }
  [[nodiscard]] uint64_t read_lane64(const Operand &op, uint32_t lane) const {
    const Wavefront &wf = wavefront();
    if (auto base = op.simd_vgpr_base(wf); base && !cu_->owns_vgpr_range(wf, *base, 2))
      return 0;
    return op.read_lane64(wf, lane);
  }

  /// @brief Read a packed pair of 32-bit words from one logical source.
  /// @details Literal64 operands split into low and high words. AMDGPU VGPR,
  /// SGPR, and special scalar pair selectors use their 64-bit execution
  /// resolver. Inline constants, literal32 operands, M0, and other single-word
  /// scalar sources are read as 32-bit values and splatted. Pair classification
  /// uses the same execution predicate as resolve_src_scalar64(), independently
  /// of the narrower analysis-liveness mapping in to_register_ref().
  [[nodiscard]] OperandPair32 read_lane_pair32(const Operand &op, uint32_t lane) const {
    if (const auto literal = op.literal64_value())
      return {static_cast<uint32_t>(*literal), static_cast<uint32_t>(*literal >> 32)};

    if (const auto constant = op.const_value()) {
      const uint32_t value = static_cast<uint32_t>(*constant);
      return {value, value};
    }

    const Wavefront &wf = wavefront();
    const bool is_register_pair =
        op.size_bits() >= 64 &&
        (op.simd_vgpr_base(wf).has_value() || is_src_scalar_register_pair(op.encoding_value()));
    if (is_register_pair) {
      const uint64_t pair = op.read_lane64(wf, lane);
      return {static_cast<uint32_t>(pair), static_cast<uint32_t>(pair >> 32)};
    }

    const uint32_t value = op.read_lane(wf, lane);
    return {value, value};
  }
  void write_scalar(const Operand &op, uint32_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (auto base = op.simd_vgpr_base_mut(wf); base && !mutable_cu().owns_vgpr_range(wf, *base, 1))
      return;
    op.write_scalar(wf, value);
  }
  void write_scalar64(const Operand &op, uint64_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (auto base = op.simd_vgpr_base_mut(wf); base && !mutable_cu().owns_vgpr_range(wf, *base, 2))
      return;
    op.write_scalar64(wf, value);
  }
  void write_lane(const Operand &op, uint32_t lane, uint32_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (auto base = op.simd_vgpr_base_mut(wf); base && !mutable_cu().owns_vgpr_range(wf, *base, 1))
      return;
    if (op.simd_vgpr_storage_mut(wf) && !(wf.vgpr_write_mask() & (uint64_t{1} << lane)))
      return;
    op.write_lane(wf, lane, value);
  }

  /// @brief Write the lane selected by a scalar-lane instruction.
  /// @details V_WRITELANE is independent of EXEC and of execution-local DPP
  /// destination masks. Selector high bits are ignored according to the
  /// dispatched architectural wave width.
  void write_scalar_selected_lane(const Operand &op, uint32_t lane, uint32_t value) const {
    Wavefront &wf = mutable_wavefront();
    assert((wf.wf_size() == 32 || wf.wf_size() == 64) &&
           "scalar lane selection requires Wave32 or Wave64");
    lane &= wf.wf_size() - 1;
    auto physical_reg = op.simd_vgpr_base_mut(wf);
    if (!physical_reg)
      throw std::logic_error("scalar lane write requires a VGPR destination");
    if (!mutable_cu().owns_vgpr_range(wf, *physical_reg, 1))
      return;
    VgprStorage storage = op.simd_vgpr_storage_mut(wf);
    if (!storage)
      throw std::logic_error("scalar lane write requires VGPR storage");
    mutable_cu().notify_scalar_lane_vgpr_write(&wf, *physical_reg, uint64_t{1} << lane);
    storage[lane] = value;
  }
  /// @brief Write selected destination bytes in one VGPR lane.
  ///
  /// `update_byte_mask` controls which stored bytes receive `value`.
  /// `observed_byte_mask` reports the complete architectural write effect,
  /// which may be wider when a transform uses the merged dword.
  void write_lane_masked(const Operand &op, uint32_t lane, uint32_t value, uint8_t update_byte_mask,
                         uint8_t observed_byte_mask, PostWriteTransform post_transform) const {
    Wavefront &wf = mutable_wavefront();
    auto physical_reg = op.simd_vgpr_base_mut(wf);
    if (!physical_reg)
      throw std::logic_error("partial-byte operand write requires a VGPR destination");
    if (!mutable_cu().owns_vgpr_range(wf, *physical_reg, 1))
      return;
    VgprStorage storage = op.simd_vgpr_storage_mut(wf);
    if (!storage)
      throw std::logic_error("partial-byte operand write requires VGPR storage");
    if (!(wf.vgpr_write_mask() & (uint64_t{1} << lane)))
      return;
    mutable_cu().notify_vgpr_write(&wf, *physical_reg, uint64_t{1} << lane, observed_byte_mask);
    const uint32_t bit_mask = byte_bit_mask(update_byte_mask);
    uint32_t merged = (storage[lane] & ~bit_mask) | (value & bit_mask);
    if (post_transform)
      merged = post_transform(merged, wf);
    storage[lane] = merged;
  }

  void write_lane64(const Operand &op, uint32_t lane, uint64_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (auto base = op.simd_vgpr_base_mut(wf); base && !mutable_cu().owns_vgpr_range(wf, *base, 2))
      return;
    if (op.simd_vgpr_storage64_mut(wf).lo && !(wf.vgpr_write_mask() & (uint64_t{1} << lane)))
      return;
    op.write_lane64(wf, lane, value);
  }

  void read_chunk(const Operand &op, uint32_t lane_base, uint32_t count, uint32_t *out) const {
    const Wavefront &wf = wavefront();
    if (auto base = op.simd_vgpr_base(wf); base && !cu_->owns_vgpr_range(wf, *base, 1)) {
      std::fill_n(out, count, 0u);
      return;
    }
    op.read_lane_chunk(wf, lane_base, count, out);
  }
  void write_chunk(const Operand &op, uint32_t lane_base, uint32_t count, const uint32_t *values,
                   uint64_t lane_mask) const {
    Wavefront &wf = mutable_wavefront();
    uint64_t full_mask = util::mask<uint64_t>(static_cast<int>(count));
    if (op.simd_capable()) {
      auto base = op.simd_vgpr_base_mut(wf);
      if (base && !mutable_cu().owns_vgpr_range(wf, *base, 1))
        return;
      if (op.simd_vgpr_storage_mut(wf))
        lane_mask &= wf.vgpr_write_mask() >> lane_base;
      op.simd_notify_write_mut(wf, (lane_mask & full_mask) << lane_base,
                               rocjitsu::ExecutionPlugin::kFullByteMask);
    }
    op.write_lane_chunk(wf, lane_base, count, values, lane_mask);
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
    auto base = op.simd_vgpr_base(wf);
    const bool valid = !base || cu_->owns_vgpr_range(wf, *base, 1);
    ConstVgprStorage storage = valid ? op.simd_vgpr_storage(wf) : ConstVgprStorage{};
    if (storage)
      op.simd_notify_read(wf, lane_mask, byte_mask);
    return OperandReadView(op, wf, storage, byte_mask, base && !valid);
  }

  [[nodiscard]] OperandRead64View read_operand64(const Operand &op, uint64_t lane_mask,
                                                 uint8_t byte_mask = 0xF) const {
    const Wavefront &wf = wavefront();
    auto base = op.simd_vgpr_base(wf);
    const bool valid = !base || cu_->owns_vgpr_range(wf, *base, 2);
    ConstVgprStoragePair64 storage = valid ? op.simd_vgpr_storage64(wf) : ConstVgprStoragePair64{};
    if (storage.lo)
      op.simd_notify_read64(wf, lane_mask, byte_mask);
    return OperandRead64View(op, wf, storage, byte_mask, base && !valid);
  }

  [[nodiscard]] OperandReadPair32View read_operand_pair32(const Operand &op, uint64_t lane_mask,
                                                          uint8_t byte_mask = 0xF) const {
    const Wavefront &wf = wavefront();
    auto base = op.simd_vgpr_base(wf);
    const bool valid = !base || cu_->owns_vgpr_range(wf, *base, 2);
    ConstVgprStoragePair64 storage = valid ? op.simd_vgpr_storage64(wf) : ConstVgprStoragePair64{};
    if (storage.lo)
      op.simd_notify_read64(wf, lane_mask, byte_mask);
    return OperandReadPair32View(op, wf, storage, byte_mask, base && !valid);
  }

  [[nodiscard]] OperandWriteView
  write_operand(const Operand &op, uint64_t lane_mask,
                uint8_t byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const {
    Wavefront &wf = mutable_wavefront();
    auto base = op.simd_vgpr_base_mut(wf);
    const bool valid = !base || mutable_cu().owns_vgpr_range(wf, *base, 1);
    VgprStorage storage = valid ? op.simd_vgpr_storage_mut(wf) : VgprStorage{};
    const uint64_t permitted_write_lane_mask =
        storage ? lane_mask & wf.vgpr_write_mask() : lane_mask;
    if (storage)
      op.simd_notify_write_mut(wf, permitted_write_lane_mask, byte_mask);
    return OperandWriteView(op, wf, storage, permitted_write_lane_mask, byte_mask, base && !valid);
  }

  [[nodiscard]] OperandWrite64View write_operand64(const Operand &op, uint64_t lane_mask) const {
    Wavefront &wf = mutable_wavefront();
    auto base = op.simd_vgpr_base_mut(wf);
    const bool valid = !base || mutable_cu().owns_vgpr_range(wf, *base, 2);
    VgprStoragePair64 storage = valid ? op.simd_vgpr_storage64_mut(wf) : VgprStoragePair64{};
    const uint64_t permitted_write_lane_mask =
        storage.lo ? lane_mask & wf.vgpr_write_mask() : lane_mask;
    if (storage.lo)
      op.simd_notify_write64_mut(wf, permitted_write_lane_mask,
                                 rocjitsu::ExecutionPlugin::kFullByteMask);
    return OperandWrite64View(op, wf, storage, permitted_write_lane_mask, base && !valid);
  }

  [[nodiscard]] OperandWritePair32View write_operand_pair32(const Operand &op,
                                                            uint64_t lane_mask) const {
    Wavefront &wf = mutable_wavefront();
    auto base = op.simd_vgpr_base_mut(wf);
    const bool valid = !base || mutable_cu().owns_vgpr_range(wf, *base, 2);
    VgprStoragePair64 storage = valid ? op.simd_vgpr_storage64_mut(wf) : VgprStoragePair64{};
    const uint64_t permitted_write_lane_mask =
        storage.lo ? lane_mask & wf.vgpr_write_mask() : lane_mask;
    if (storage.lo)
      op.simd_notify_write64_mut(wf, permitted_write_lane_mask,
                                 rocjitsu::ExecutionPlugin::kFullByteMask);
    return OperandWritePair32View(op, wf, storage, permitted_write_lane_mask, base && !valid);
  }

  [[nodiscard]] OperandReadWriteView
  readwrite_operand(const Operand &op, uint64_t lane_mask, uint8_t byte_mask = 0xF,
                    uint8_t write_byte_mask = rocjitsu::ExecutionPlugin::kFullByteMask) const {
    Wavefront &wf = mutable_wavefront();
    auto base = op.simd_vgpr_base_mut(wf);
    const bool valid = !base || mutable_cu().owns_vgpr_range(wf, *base, 1);
    VgprStorage storage = valid ? op.simd_vgpr_storage_mut(wf) : VgprStorage{};
    const uint64_t permitted_write_lane_mask =
        storage ? lane_mask & wf.vgpr_write_mask() : lane_mask;
    if (storage) {
      op.simd_notify_read_mut(wf, lane_mask, byte_mask);
      op.simd_notify_write_mut(wf, permitted_write_lane_mask, write_byte_mask);
    }
    return OperandReadWriteView(op, wf, storage, permitted_write_lane_mask, byte_mask,
                                write_byte_mask, base && !valid);
  }

  [[nodiscard]] OperandReadWrite64View readwrite_operand64(const Operand &op, uint64_t lane_mask,
                                                           uint8_t byte_mask = 0xF) const {
    Wavefront &wf = mutable_wavefront();
    auto base = op.simd_vgpr_base_mut(wf);
    const bool valid = !base || mutable_cu().owns_vgpr_range(wf, *base, 2);
    VgprStoragePair64 storage = valid ? op.simd_vgpr_storage64_mut(wf) : VgprStoragePair64{};
    const uint64_t permitted_write_lane_mask =
        storage.lo ? lane_mask & wf.vgpr_write_mask() : lane_mask;
    if (storage.lo) {
      op.simd_notify_read64_mut(wf, lane_mask, byte_mask);
      op.simd_notify_write64_mut(wf, permitted_write_lane_mask,
                                 rocjitsu::ExecutionPlugin::kFullByteMask);
    }
    return OperandReadWrite64View(op, wf, storage, permitted_write_lane_mask, byte_mask,
                                  base && !valid);
  }

  // Logical TTMP access. TTMPs are wave-private scalar registers and do not
  // occupy slots in the CU's physical SGPR file.
  [[nodiscard]] uint32_t read_ttmp(uint32_t index) const {
    const Wavefront &wf = wavefront();
    if (index >= kTtmpRegisterCount)
      return 0;
    cu_->notify_scalar_register_read(wf,
                                     RegisterRef{RegClass::TTMP, static_cast<uint16_t>(index), 1});
    return wf.ttmp(index);
  }

  [[nodiscard]] uint64_t read_ttmp64(uint32_t index) const {
    const Wavefront &wf = wavefront();
    if (index >= kTtmpRegisterCount || kTtmpRegisterCount - index < 2)
      return 0;
    cu_->notify_scalar_register_read(wf,
                                     RegisterRef{RegClass::TTMP, static_cast<uint16_t>(index), 2});
    return static_cast<uint64_t>(wf.ttmp(index)) |
           (static_cast<uint64_t>(wf.ttmp(index + 1)) << 32);
  }

  void write_ttmp(uint32_t index, uint32_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (index >= kTtmpRegisterCount)
      return;
    mutable_cu().notify_scalar_register_write(
        wf, RegisterRef{RegClass::TTMP, static_cast<uint16_t>(index), 1});
    wf.set_ttmp(index, value);
  }

  void write_ttmp64(uint32_t index, uint64_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (index >= kTtmpRegisterCount || kTtmpRegisterCount - index < 2)
      return;
    mutable_cu().notify_scalar_register_write(
        wf, RegisterRef{RegClass::TTMP, static_cast<uint16_t>(index), 2});
    wf.set_ttmp(index, static_cast<uint32_t>(value));
    wf.set_ttmp(index + 1, static_cast<uint32_t>(value >> 32));
  }

  /// @brief Write one dword to validated scalar storage without reporting an
  /// instruction-visible register write.
  /// @details Memory completion and runtime initialization use this path
  /// because they are state transitions, not instruction destination writes.
  void write_scalar_unobserved(const ScalarRegisterRange &range, uint32_t offset,
                               uint32_t value) const {
    Wavefront &wf = mutable_wavefront();
    if (offset >= range.width)
      return;
    const uint32_t index = range.index + offset;
    switch (range.storage) {
    case ScalarRegisterStorage::SGPR:
      if (mutable_cu().owns_sgpr_range(wf, wf.sgpr_alloc().base + range.index, range.width))
        mutable_cu().write_sgpr(wf.sgpr_alloc().base + index, value);
      break;
    case ScalarRegisterStorage::FLAT_SCRATCH: {
      const uint64_t old = wf.scratch_base();
      wf.set_scratch_base(index == 0 ? (old & 0xffffffff00000000ULL) | value
                                     : (old & 0x00000000ffffffffULL) |
                                           (static_cast<uint64_t>(value) << 32));
      break;
    }
    case ScalarRegisterStorage::VCC: {
      const uint64_t old = wf.vcc();
      wf.set_vcc_raw(index == 0
                         ? (old & 0xffffffff00000000ULL) | value
                         : (old & 0x00000000ffffffffULL) | (static_cast<uint64_t>(value) << 32));
      break;
    }
    case ScalarRegisterStorage::TTMP:
      wf.set_ttmp(index, value);
      break;
    case ScalarRegisterStorage::M0:
      wf.set_m0(value);
      break;
    case ScalarRegisterStorage::EXEC: {
      const uint64_t old = wf.exec_raw();
      wf.set_exec_raw(index == 0
                          ? (old & 0xffffffff00000000ULL) | value
                          : (old & 0x00000000ffffffffULL) | (static_cast<uint64_t>(value) << 32));
      break;
    }
    case ScalarRegisterStorage::DISCARD:
      break;
    }
  }

  // Physical SGPR access. These APIs are for helpers that already know the
  // physical scalar register index. Reads may use a CU-bound compatibility path
  // that resolves one owner for the complete range. Writes require an explicit
  // mutable wavefront so instruction code cannot silently reach raw SGPR
  // storage; VM/runtime code uses ComputeUnitCore's raw write API directly.
  [[nodiscard]] bool owns_sgpr_range(uint32_t physical_base, uint32_t physical_count) const {
    if (wf_)
      return cu_->owns_sgpr_range(*wf_, physical_base, physical_count);
    return cu_->sgpr_owner_for_range(physical_base, physical_count) != nullptr;
  }

  [[nodiscard]] bool owns_vgpr_range(uint32_t physical_base, uint32_t physical_count) const {
    return vgpr_owner_for_range(physical_base, physical_count) != nullptr;
  }

  [[nodiscard]] SgprReadRegion read_sgpr_region(uint32_t physical_base, uint32_t reg_count) const {
    const Wavefront *owner = sgpr_owner_for_range(physical_base, reg_count);
    if (!owner)
      return SgprReadRegion(*cu_, physical_base, reg_count, false);
    observe_sgpr_region(*owner, physical_base, reg_count);
    return SgprReadRegion(*cu_, physical_base, reg_count, true);
  }

  [[nodiscard]] uint32_t read_sgpr(uint32_t physical_reg) const {
    return read_sgpr_region(physical_reg, 1).dword();
  }

  [[nodiscard]] uint64_t read_sgpr64(uint32_t physical_reg) const {
    return read_sgpr_region(physical_reg, 2).qword();
  }

  void write_sgpr(uint32_t physical_reg, uint32_t value) const {
    mutable_cu().write_sgpr(mutable_wavefront(), physical_reg, value);
  }

  void write_sgpr64(uint32_t physical_reg, uint64_t value) const {
    mutable_cu().write_sgpr64(mutable_wavefront(), physical_reg, value);
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

  void write_vgpr(uint32_t physical_reg, uint32_t lane, uint32_t value,
                  uint8_t byte_mask = 0xF) const {
    write_vgpr_region(physical_reg, 1, uint64_t{1} << lane, byte_mask).set_lane(0, lane, value);
  }

  void write_vgpr64(uint32_t physical_reg, uint32_t lane, uint64_t value) const {
    write_vgpr_region(physical_reg, 2, uint64_t{1} << lane).set_lane64(0, lane, value);
  }

  [[nodiscard]] VgprReadRegion read_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                                uint64_t lane_mask, uint8_t byte_mask = 0xF) const {
    const uint32_t wave_size = vgpr_region_wave_size(physical_base, reg_count);
    const Wavefront *owner = vgpr_owner_for_range(physical_base, reg_count);
    if (!owner)
      return VgprReadRegion(*cu_, physical_base, reg_count, wave_size, byte_mask, false, false);
    if (lane_mask == 0)
      return VgprReadRegion(*cu_, physical_base, reg_count, wave_size, byte_mask, true, false);
    observe_vgpr_region(*owner, physical_base, reg_count, lane_mask, byte_mask);
    return VgprReadRegion(*cu_, physical_base, reg_count, wave_size, byte_mask);
  }

  [[nodiscard]] VgprWriteRegion write_vgpr_region(uint32_t physical_base, uint32_t reg_count,
                                                  uint64_t lane_mask,
                                                  uint8_t byte_mask = 0xF) const {
    if (mutable_wf_)
      lane_mask &= mutable_wf_->vgpr_write_mask();
    const uint32_t wave_size = vgpr_region_wave_size(physical_base, reg_count);
    const Wavefront *owner = vgpr_owner_for_range(physical_base, reg_count);
    if (!owner || lane_mask == 0 || byte_mask == 0)
      return VgprWriteRegion(mutable_cu(), physical_base, reg_count, wave_size, lane_mask,
                             byte_mask, false);
    observe_vgpr_write_region(*owner, physical_base, reg_count, lane_mask, byte_mask);
    return VgprWriteRegion(mutable_cu(), physical_base, reg_count, wave_size, lane_mask, byte_mask);
  }

  [[nodiscard]] VgprReadWriteRegion readwrite_vgpr_region(uint32_t physical_base,
                                                          uint32_t reg_count, uint64_t lane_mask,
                                                          uint8_t byte_mask = 0xF) const {
    return readwrite_vgpr_region(physical_base, reg_count, lane_mask, byte_mask, byte_mask);
  }

  [[nodiscard]] VgprReadWriteRegion readwrite_vgpr_region(uint32_t physical_base,
                                                          uint32_t reg_count, uint64_t lane_mask,
                                                          uint8_t read_byte_mask,
                                                          uint8_t write_byte_mask) const {
    return VgprReadWriteRegion(
        read_vgpr_region(physical_base, reg_count, lane_mask, read_byte_mask),
        write_vgpr_region(physical_base, reg_count, lane_mask, write_byte_mask));
  }

private:
  [[nodiscard]] uint32_t vgpr_region_wave_size(uint32_t physical_base, uint32_t reg_count) const {
    if (wf_)
      return wf_->wf_size();
    if (const Wavefront *owner = cu_->vgpr_owner_for_range(physical_base, reg_count))
      return owner->wf_size();
    return cu_->vgpr_storage_lane_count();
  }

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

  [[nodiscard]] const Wavefront *vgpr_owner_for_range(uint32_t physical_base,
                                                      uint32_t reg_count) const {
    if (wf_)
      return cu_->owns_vgpr_range(*wf_, physical_base, reg_count) ? wf_ : nullptr;
    return cu_->vgpr_owner_for_range(physical_base, reg_count);
  }

  [[nodiscard]] const Wavefront *sgpr_owner_for_range(uint32_t physical_base,
                                                      uint32_t reg_count) const {
    if (wf_)
      return cu_->owns_sgpr_range(*wf_, physical_base, reg_count) ? wf_ : nullptr;
    return cu_->sgpr_owner_for_range(physical_base, reg_count);
  }

  void observe_sgpr_region(const Wavefront &owner, uint32_t physical_base,
                           uint32_t reg_count) const {
    assert(reg_count <= std::numeric_limits<uint8_t>::max());
    cu_->notify_scalar_register_read(
        owner,
        RegisterRef{RegClass::SGPR, static_cast<uint16_t>(physical_base - owner.sgpr_alloc().base),
                    static_cast<uint8_t>(reg_count)});
  }

  void observe_vgpr_region(const Wavefront &owner, uint32_t physical_base, uint32_t reg_count,
                           uint64_t lane_mask, uint8_t byte_mask) const {
    for (uint32_t reg = 0; reg < reg_count; ++reg)
      cu_->notify_vgpr_read(&owner, physical_base + reg, lane_mask, byte_mask);
  }

  void observe_vgpr_write_region(const Wavefront &owner, uint32_t physical_base, uint32_t reg_count,
                                 uint64_t lane_mask, uint8_t byte_mask) const {
    for (uint32_t reg = 0; reg < reg_count; ++reg)
      cu_->notify_vgpr_write(&owner, physical_base + reg, lane_mask, byte_mask);
  }

  const ComputeUnitCore *cu_ = nullptr;
  ComputeUnitCore *mutable_cu_ = nullptr;
  const Wavefront *wf_ = nullptr;
  Wavefront *mutable_wf_ = nullptr;
};

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_VM_AMDGPU_REGISTER_ACCESS_H_
