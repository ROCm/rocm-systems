// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file operand.h
/// @brief Instruction operand base class with register read/write interface.

#ifndef ROCJITSU_ISA_OPERAND_H_
#define ROCJITSU_ISA_OPERAND_H_

#include "rocjitsu/isa/arch/amdgpu/shared/vgpr_msb.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/result.h"
#include "util/diagnostic.h"
#include "util/simd.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace simdojo {
template <size_t NUM_ELEMS, typename VecElem> class VectorReg;
}

namespace rocjitsu {
template <typename Isa> class AmdgpuIsaOperand;

namespace amdgpu {
class ComputeUnitCore;
class InstructionComputeUnitView;
class RegisterAccess;
class Wavefront;

/// @brief Non-owning mutable view of one physical VGPR.
/// @details The view keeps SIMD execution independent of the ISA-specific
/// physical register width. gfx1250 stores 32 lanes while RDNA and CDNA store
/// up to 64; both use the same native SIMD load/store path.
class VgprStorage {
public:
  VgprStorage() = default;
  VgprStorage(uint32_t *data, uint32_t lane_count) : data_(data), lane_count_(lane_count) {}

  explicit operator bool() const { return data_ != nullptr; }
  uint32_t lane_count() const { return lane_count_; }
  uint32_t &operator[](size_t lane) const {
    assert(lane < lane_count_);
    return data_[lane];
  }

  template <typename T> util::native<T> simd_load(size_t lane_base) const {
    assert(lane_base + util::native_width_v<T> <= lane_count_);
    return util::load<T>(data_ + lane_base);
  }
  template <typename T>
  void simd_store(size_t lane_base, util::native<T> value, uint64_t mask) const {
    assert(lane_base + util::native_width_v<T> <= lane_count_);
    util::masked_store<T>(data_ + lane_base, value, mask);
  }
  template <typename T> util::narrow32<T> simd_load_narrow(size_t lane_base) const {
    assert(lane_base + util::native_width64 <= lane_count_);
    return util::load_narrow<T>(data_ + lane_base);
  }
  template <typename T>
  void simd_store_narrow(size_t lane_base, util::narrow32<T> value, uint64_t mask) const {
    assert(lane_base + util::native_width64 <= lane_count_);
    util::masked_store_narrow<T>(data_ + lane_base, value, mask);
  }
  template <typename T> util::native<T> simd_load64(const VgprStorage &hi, size_t lane_base) const {
    assert(lane_count_ == hi.lane_count_);
    assert(lane_base + util::native_width64 <= lane_count_);
    return util::load64<T>(data_ + lane_base, hi.data_ + lane_base);
  }
  template <typename T>
  void simd_store64(const VgprStorage &hi, size_t lane_base, util::native<T> value,
                    uint64_t mask) const {
    assert(lane_count_ == hi.lane_count_);
    assert(lane_base + util::native_width64 <= lane_count_);
    util::masked_store64<T>(data_ + lane_base, hi.data_ + lane_base, value, mask);
  }

private:
  uint32_t *data_ = nullptr;
  uint32_t lane_count_ = 0;
};

/// @brief Non-owning read-only view of one physical VGPR.
class ConstVgprStorage {
public:
  ConstVgprStorage() = default;
  ConstVgprStorage(const uint32_t *data, uint32_t lane_count)
      : data_(data), lane_count_(lane_count) {}

  explicit operator bool() const { return data_ != nullptr; }
  uint32_t lane_count() const { return lane_count_; }
  uint32_t operator[](size_t lane) const {
    assert(lane < lane_count_);
    return data_[lane];
  }
  template <typename T> util::native<T> simd_load(size_t lane_base) const {
    assert(lane_base + util::native_width_v<T> <= lane_count_);
    return util::load<T>(data_ + lane_base);
  }
  template <typename T> util::narrow32<T> simd_load_narrow(size_t lane_base) const {
    assert(lane_base + util::native_width64 <= lane_count_);
    return util::load_narrow<T>(data_ + lane_base);
  }
  template <typename T>
  util::native<T> simd_load64(const ConstVgprStorage &hi, size_t lane_base) const {
    assert(lane_count_ == hi.lane_count_);
    assert(lane_base + util::native_width64 <= lane_count_);
    return util::load64<T>(data_ + lane_base, hi.data_ + lane_base);
  }

private:
  const uint32_t *data_ = nullptr;
  uint32_t lane_count_ = 0;
};

/// @brief A `{lo, hi}` pair of typed per-register VGPR storage views for a
/// 64-bit-lane operand. `lo` is the lower-numbered VGPR (reg N, bits [31:0]);
/// `hi` is reg N+1 (bits [63:32]). Either both are valid or both are nullptr.
struct VgprStoragePair64 {
  VgprStorage lo;
  VgprStorage hi;
};

/// @brief Read-only counterpart of `VgprStoragePair64`.
struct ConstVgprStoragePair64 {
  ConstVgprStorage lo;
  ConstVgprStorage hi;
};
} // namespace amdgpu

namespace detail {
template <typename Isa>
void amdgpu_isa_read_lane_chunk_base(const AmdgpuIsaOperand<Isa> &op, const amdgpu::Wavefront &wf,
                                     uint32_t lane_base, uint32_t count, uint32_t *out);
template <typename Isa>
void amdgpu_isa_write_lane_chunk_base(const AmdgpuIsaOperand<Isa> &op, amdgpu::Wavefront &wf,
                                      uint32_t lane_base, uint32_t count, const uint32_t *vals,
                                      uint64_t mask);
} // namespace detail

class ScopedOperandDelegate;

/// @brief Base class for an instruction operand with value resolution.
///
/// @details Instruction execution code treats Operand as a descriptor: it can
/// query names, register references, encoding values, widths, and SIMD
/// capability. It must not read or write operand values directly. Value access
/// is private backend API used by amdgpu::RegisterAccess so all VGPR reads pass
/// through the observed register-access facade.
class Operand {
public:
  // RegisterAccess is the only instruction-facing facade allowed to enter the
  // operand value-access backend. Keep these hooks private so generated
  // instruction bodies cannot bypass read observation by calling them directly.
  friend class amdgpu::RegisterAccess;
  friend class ScopedOperandDelegate;
  template <typename Isa> friend class AmdgpuIsaOperand;

  Operand() = default;

  /// @brief Construct an operand with the given size and encoding value.
  /// @param size_bits Operand width in bits.
  /// @param encoding_value ISA-specific encoding value identifying the register or literal.
  Operand(int size_bits, int encoding_value)
      : size_bits_(size_bits), encoding_value_(encoding_value) {}
  virtual ~Operand() = default;

  /// Validate encoding constraints deferred until the complete instruction is
  /// decoded. Literal sentinels may replace their provisional operands first.
  Result validate_encoding(const util::DiagnosticEmitter &emit_error = {}) const {
    if (encoding_error_ == EncodingError::None) [[likely]]
      return Result::success();
    return emit_encoding_error(emit_error);
  }

  /// @brief Human-readable name for this operand (e.g. "v0", "s4", or a literal).
  virtual std::string name() const { return std::to_string(encoding_value_); }

  /// @brief Map this operand to an analysis register reference.
  ///
  /// @details Returns nullopt for literals, labels, waitcnt immediates, message
  /// IDs, and other non-register operands. ISA-specific subclasses override
  /// this using generated OperandType selector ranges so analysis never has to
  /// parse the display string returned by name().
  [[nodiscard]] virtual std::optional<RegisterRef> to_register_ref() const;

  /// @brief Raw encoding value from the instruction binary.
  int encoding_value() const { return encoding_value_; }

  /// @brief Full 64-bit literal value when this operand came from a literal64 encoding.
  [[nodiscard]] virtual std::optional<uint64_t> literal64_value() const { return std::nullopt; }

  /// @brief Compile-time constant value of this operand, resolved without any
  /// register/wavefront state, or nullopt for registers and other non-constant
  /// operands.
  ///
  /// @details Unlike `literal64_value()` (which only reports the literal64
  /// encoding), this also resolves inline constants — small integers and the
  /// inline float constants whose value is implied by the encoding. The base
  /// default covers only the literal case; ISA subclasses override it to add
  /// inline-constant resolution. Useful for static analysis (e.g. detecting
  /// `s_mov exec, -1`) where no wavefront is available.
  [[nodiscard]] virtual std::optional<uint64_t> const_value() const { return literal64_value(); }

  /// @brief Operand width in bits.
  int size_bits() const { return size_bits_; }

  /// @brief Whether this operand references a VGPR or AccVGPR.
  /// @details A construction-time capability flag. Field-bearing operands are
  /// classified by the ISA-specific subclass constructor using the generated
  /// is_vgpr_operand_type(); fieldless operands get their value from
  /// apply_fieldless_caps(). Accessors query the stored flag directly.
  [[nodiscard]] bool is_vgpr() const { return is_vgpr_; }

  /// @brief Whether this operand yields a real value through the normal read /
  /// SIMD accessors. A construction-time capability flag: true for field-bearing
  /// operands and read-enabled fieldless operands. False for inert operands.
  [[nodiscard]] bool reads_value() const { return reads_value_; }

  /// @brief Whether this operand is a valid target for the normal write
  /// accessors. A construction-time capability flag: true for field-bearing
  /// operands and write-enabled fieldless operands. False for inert operands.
  [[nodiscard]] bool is_writable() const { return writable_; }

  /// @brief Whether this operand is fieldless
  /// @details Fieldless operands (has no encoding field in MR ISA) are
  /// constructed from a fixed canonical encoding value rather than a decoded
  /// field. This stays a structural marker: it drives disassembly suppression
  /// and ordinary to_register_ref() suppression, while runtime read/write/SIMD
  /// behavior is driven by the capability flags above.
  [[nodiscard]] bool is_fieldless() const { return fieldless_; }

  /// @brief Mark this operand fieldless and apply its runtime capability
  /// policy. Emitted by generated constructors in place of a bare fieldless
  /// marker; the (reads_value, writable, is_vgpr) triple comes from the shared
  /// fieldless operand policy table.
  ///
  /// @warning Construction-only. Call exactly once, from a constructor,
  /// before the operand is observable by any reader. The capability flags are
  /// read locklessly on the CU thread and are assumed immutable after
  /// construction; mutating them on a live operand is a data race.
  void apply_fieldless_caps(bool reads_value, bool writable, bool is_vgpr) {
    assert(!fieldless_ && "apply_fieldless_caps must be called once, at construction");
    // Mirror the FieldlessCaps.__post_init__ invariant: writable/is_vgpr imply
    // reads_value. Otherwise the SIMD fast path (gated on reads_value via
    // simd_capable/resolved_vgpr_offset) and the scalar write path would
    // disagree for a writable-but-!reads_value operand.
    assert((reads_value || (!writable && !is_vgpr)) &&
           "fieldless caps: writable/is_vgpr require reads_value");
    fieldless_ = true;
    reads_value_ = reads_value;
    writable_ = writable;
    is_vgpr_ = is_vgpr;
  }

  /// @brief Assign the GFX12 VGPR high-bank role for this operand.
  void set_vgpr_msb_role(amdgpu::VgprMsbRole role) { vgpr_msb_role_ = role; }

  /// @brief Return the GFX12 VGPR high-bank role for this operand.
  [[nodiscard]] amdgpu::VgprMsbRole vgpr_msb_role() const { return vgpr_msb_role_; }

  /// @brief Unified VGPR index for this operand (0-511).
  /// @details Maps AMDGPU encoding ranges to a unified index space:
  ///   VGPRs 0-255, AccVGPRs 256-511. Only valid when is_vgpr() is true.
  [[nodiscard]] uint16_t unified_vgpr_index() const {
    if (encoding_value_ >= 768)
      return static_cast<uint16_t>(encoding_value_ - 512);
    if (encoding_value_ >= 512)
      return static_cast<uint16_t>(encoding_value_ - 256);
    if (encoding_value_ >= 256)
      return static_cast<uint16_t>(encoding_value_ - 256);
    return static_cast<uint16_t>(encoding_value_);
  }

  /// @brief Number of consecutive VGPRs this operand spans.
  [[nodiscard]] uint16_t vgpr_count() const {
    return static_cast<uint16_t>(std::max(1, (size_bits_ + 31) / 32));
  }

private:
  Result emit_encoding_error(const util::DiagnosticEmitter &emit_error) const;

  // Value access is intentionally private. Instruction implementations use
  // RegisterAccess; Operand remains the ISA-specific resolver/backend.

  /// @brief Read this operand as a scalar 32-bit value.
  /// @param wf Wavefront providing register state.
  /// @returns The 32-bit scalar value.
  virtual uint32_t read_scalar(const amdgpu::Wavefront &wf) const;

  /// @brief Read this operand's value for a specific SIMD lane.
  ///
  /// @details For scalar operands, broadcasts the scalar value to all lanes.
  /// For vector operands, reads the lane from the vector register.
  /// @param wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @returns The 32-bit lane value.
  virtual uint32_t read_lane(const amdgpu::Wavefront &wf, uint32_t lane) const;

  /// @brief Write a scalar 32-bit value to this operand's destination.
  /// @param[in,out] wf Wavefront providing register state.
  /// @param val Value to write.
  virtual void write_scalar(amdgpu::Wavefront &wf, uint32_t val) const;

  /// @brief Write a 32-bit value to a specific SIMD lane of this operand.
  /// @param[in,out] wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @param val Value to write.
  virtual void write_lane(amdgpu::Wavefront &wf, uint32_t lane, uint32_t val) const;

  /// @brief Read a 64-bit value from a SIMD lane (VGPR pair).
  /// @param wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @returns The 64-bit lane value.
  virtual uint64_t read_lane64(const amdgpu::Wavefront &wf, uint32_t lane) const;

  /// @brief Write a 64-bit value to a SIMD lane (VGPR pair).
  /// @param[in,out] wf Wavefront providing register state.
  /// @param lane SIMD lane index.
  /// @param val Value to write.
  virtual void write_lane64(amdgpu::Wavefront &wf, uint32_t lane, uint64_t val) const;

  /// @brief Read this operand as a 64-bit scalar (e.g., SGPR pair, VCC, EXEC).
  /// @param wf Wavefront providing register state.
  /// @returns The 64-bit scalar value.
  virtual uint64_t read_scalar64(const amdgpu::Wavefront &wf) const;

  /// @brief Write a 64-bit scalar value (e.g., SGPR pair, VCC, EXEC).
  /// @param[in,out] wf Wavefront providing register state.
  /// @param val Value to write.
  virtual void write_scalar64(amdgpu::Wavefront &wf, uint64_t val) const;

public:
  /// @brief Return the active read delegate, if any.
  ///
  /// Delegate mutation is restricted to ScopedOperandDelegate so restoration
  /// cannot be skipped on exceptions or early returns.
  Operand *delegate() const { return delegate_; }

  /// @brief Whether `read_lane_chunk` / `write_lane_chunk` produce correct,
  /// SIMD-friendly results for this operand.
  ///
  /// @details Default is false. Arch subclasses override to return true for
  /// operands whose per-lane values can be read or written as a contiguous
  /// uint32_t buffer (VGPRs, SGPR/immediate/inline-const broadcasts, DPP/SDWA
  /// delegated operands). Kernels gate SIMD fast paths on this predicate; if
  /// any source/dest reports false, the kernel falls back to its scalar loop.
  virtual bool simd_capable() const {
    if (delegate_)
      return delegate_->simd_capable();
    return false;
  }

private:
  void set_delegate(Operand *delegate) { delegate_ = delegate; }

  /// @brief Fill `out[0..count)` with operand values for lanes
  /// `[lane_base, lane_base + count)`.
  ///
  /// @details Default implementation calls `read_lane` per element so any
  /// operand stays correct without an override. Arch subclasses override with
  /// memcpy-based VGPR reads or scalar broadcasts.
  virtual void read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                               uint32_t *out) const {
    if (delegate_) {
      delegate_->read_lane_chunk(wf, lane_base, count, out);
      return;
    }
    for (uint32_t i = 0; i < count; ++i)
      out[i] = read_lane(wf, lane_base + i);
  }

  /// @brief Apply masked write of `vals[0..count)` to lanes
  /// `[lane_base, lane_base + count)`. Bit `i` of `mask` enables lane `i`.
  virtual void write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                                const uint32_t *vals, uint64_t mask) const {
    for (uint32_t i = 0; i < count; ++i)
      if (mask & (1ULL << i))
        write_lane(wf, lane_base + i, vals[i]);
  }

public:
  // These stay public: subclass constructors and decode/disassembly paths read
  // and set size_bits_/encoding_value_/vgpr_msb_role_ directly. The capability
  // flags below are protected instead because they are construction-only and
  // read locklessly on the hot path, so only the class hierarchy may set them.
  int size_bits_ = 0;
  int encoding_value_ = 0;
  amdgpu::VgprMsbRole vgpr_msb_role_ = amdgpu::VgprMsbRole::None;

protected:
  enum class EncodingError : uint8_t {
    None,
    InvalidSelector,
    InvalidScalarRegisterSelector,
    InvalidLaneSelector,
    InvalidExecSelector,
    InvalidVgprSourceSelector,
    InvalidScalarSourceSelector,
  };

  void defer_encoding_error(EncodingError error) { encoding_error_ = error; }

  /// @brief Accept a selector already checked by a narrower instruction-level
  /// validator.
  ///
  /// Some encoded operand types cover several width-specific LLVM register
  /// classes. Their generated constructor can only apply the common selector
  /// set, while an instruction factory can validate the exact class. Keep this
  /// protected so only an ISA Operand factory can bridge that validation.
  void accept_instruction_validated_selector() {
    if (encoding_error_ == EncodingError::InvalidSelector)
      encoding_error_ = EncodingError::None;
  }

  /// @brief Capability/role flags, set once at construction and never
  /// mutated afterward. Subclass constructors set is_vgpr_; fieldless
  /// operands get their (reads_value, writable, is_vgpr) triple from
  /// apply_fieldless_caps(). Kept out of the public interface so only
  /// construction can flip them: readers do lockless bool loads on the CU
  /// thread and rely on the flags being immutable post-construction. Query
  /// through is_vgpr() / reads_value() / is_writable() / is_fieldless().
  ///
  /// Defaults describe a normal field-bearing operand (readable, writable,
  /// not fieldless).
  bool is_vgpr_ = false;
  bool reads_value_ = true;
  bool writable_ = true;
  bool fieldless_ = false;
  EncodingError encoding_error_ = EncodingError::None;

private:
  // Private SIMD fast-path backend for RegisterAccess.
  //
  // Instruction emulators should not call these hooks directly. They acquire
  // read/write/read-write views from `amdgpu::RegisterAccess`, which centralizes
  // plugin read observation and destination write resolution. These hooks remain
  // as the operand-specific storage/notification backend for that facade.

  std::optional<uint32_t> simd_vgpr_base(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_base(wf);
    return simd_vgpr_base_impl(wf);
  }

  std::optional<uint32_t> simd_vgpr_base_mut(amdgpu::Wavefront &wf) const {
    return simd_vgpr_base_mut_impl(wf);
  }

  amdgpu::ConstVgprStorage simd_vgpr_storage(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_storage(wf);
    return simd_vgpr_storage_impl(wf);
  }

  amdgpu::VgprStorage simd_vgpr_storage_mut(amdgpu::Wavefront &wf) const {
    return simd_vgpr_storage_mut_impl(wf);
  }

  void simd_notify_read(const amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const {
    if (delegate_) {
      delegate_->simd_notify_read(wf, lane_mask, byte_mask);
      return;
    }
    simd_notify_read_impl(wf, lane_mask, byte_mask);
  }

  void simd_notify_read_mut(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const {
    simd_notify_read_mut_impl(wf, lane_mask, byte_mask);
  }

  void simd_notify_read64(const amdgpu::Wavefront &wf, uint64_t lane_mask,
                          uint8_t byte_mask) const {
    if (delegate_) {
      delegate_->simd_notify_read64(wf, lane_mask, byte_mask);
      return;
    }
    simd_notify_read64_impl(wf, lane_mask, byte_mask);
  }

  void simd_notify_read64_mut(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const {
    simd_notify_read64_mut_impl(wf, lane_mask, byte_mask);
  }

  void simd_notify_write_mut(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const {
    simd_notify_write_mut_impl(wf, lane_mask, byte_mask);
  }

  void simd_notify_write64_mut(amdgpu::Wavefront &wf, uint64_t lane_mask, uint8_t byte_mask) const {
    simd_notify_write64_mut_impl(wf, lane_mask, byte_mask);
  }

  amdgpu::ConstVgprStoragePair64 simd_vgpr_storage64(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_storage64(wf);
    return simd_vgpr_storage64_impl(wf);
  }

  amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut(amdgpu::Wavefront &wf) const {
    return simd_vgpr_storage64_mut_impl(wf);
  }

  /// @brief If this operand resolves to per-lane VGPR storage, return its
  /// physical register index (`wf.vgpr_alloc().base + offset`). Otherwise
  /// nullopt (SGPR/imm/inline-const/DPP) — the caller broadcasts a scalar. The
  /// RegisterAccess passes this index to the plugin read-notification hook with
  /// the full register extent. Internal SIMD fast-path hook, reachable only
  /// through `amdgpu::RegisterAccess`.
  virtual std::optional<uint32_t> simd_vgpr_base_impl(const amdgpu::Wavefront &wf) const {
    (void)wf;
    return std::nullopt;
  }

  /// @brief Mutable-destination counterpart of `simd_vgpr_base_impl`.
  ///
  /// Write-side resolution is separate because GPR indexing can select a
  /// different physical register for destination operands than for sources.
  virtual std::optional<uint32_t> simd_vgpr_base_mut_impl(amdgpu::Wavefront &wf) const {
    (void)wf;
    return std::nullopt;
  }

  /// @brief If this operand resolves to per-lane VGPR storage, return a typed
  /// const view of that register. Otherwise an empty view — the caller falls
  /// back to a scalar broadcast via `read_scalar`. Resolves the storage in a
  /// SINGLE virtual dispatch — the SIMD hot path reads through this without a
  /// raw pointer crossing the instruction-facing RegisterAccess API.
  virtual amdgpu::ConstVgprStorage simd_vgpr_storage_impl(const amdgpu::Wavefront &wf) const {
    (void)wf;
    return {};
  }

  /// @brief Mutable counterpart of `simd_vgpr_storage` for the dst write path
  /// (no delegate — a dst is never DPP/SDWA).
  virtual amdgpu::VgprStorage simd_vgpr_storage_mut_impl(amdgpu::Wavefront &wf) const {
    (void)wf;
    return {};
  }

  /// @brief Notify the plugin system that this operand's VGPR was read
  /// by lanes in `lane_mask`. No-op for non-VGPR operands.
  virtual void simd_notify_read_impl(const amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                     uint8_t /*byte_mask*/) const {}

  /// @brief Notify a read through a mutable destination operand, used by
  /// dst-accumulate forms where vdst is both source and destination.
  virtual void simd_notify_read_mut_impl(amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                         uint8_t /*byte_mask*/) const {}

  /// @brief 64-bit counterpart of `simd_notify_read`; a per-lane f64/i64 read
  /// consumes two consecutive VGPRs, so VGPR operands notify both halves.
  virtual void simd_notify_read64_impl(const amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                       uint8_t /*byte_mask*/) const {}

  /// @brief 64-bit counterpart of `simd_notify_read_mut`.
  virtual void simd_notify_read64_mut_impl(amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                           uint8_t /*byte_mask*/) const {}

  /// @brief Notify that this mutable destination operand's VGPR was written.
  /// No-op for non-VGPR operands.
  virtual void simd_notify_write_mut_impl(amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                          uint8_t /*byte_mask*/) const {}

  /// @brief 64-bit counterpart of `simd_notify_write_mut`.
  virtual void simd_notify_write64_mut_impl(amdgpu::Wavefront & /*wf*/, uint64_t /*lane_mask*/,
                                            uint8_t /*byte_mask*/) const {}

  /// @brief 64-bit-lane counterpart of `simd_vgpr_storage`. A per-lane f64/i64
  /// value occupies two consecutive VGPRs (reg N + reg N+1), so this returns a
  /// `{lo, hi}` pair of typed register views (lo = reg N, hi = reg N+1) in a
  /// SINGLE virtual dispatch. Returns empty views when the operand is
  /// not contiguous VGPR storage — the caller broadcasts via `read_scalar64`.
  virtual amdgpu::ConstVgprStoragePair64
  simd_vgpr_storage64_impl(const amdgpu::Wavefront &wf) const {
    (void)wf;
    return {};
  }

  /// @brief Mutable counterpart of `simd_vgpr_storage64` for the 64-bit dst
  /// write path; returns writable `{lo, hi}` register views or
  /// empty views.
  virtual amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut_impl(amdgpu::Wavefront &wf) const {
    (void)wf;
    return {};
  }

private:
  Operand *delegate_ = nullptr;
};

/// @brief ISA-parameterized operand that adds an ISA-specific operand type tag.
/// @tparam Isa ISA traits type providing an OperandType enum or type alias.
template <typename Isa> class IsaOperand : public Operand {
public:
  IsaOperand() = default;

  /// @brief Construct an ISA operand with size, type, and encoding value.
  /// @param size_bits Operand width in bits.
  /// @param opr_type ISA-specific operand type (e.g. SGPR, VGPR, literal).
  /// @param encoding_value ISA-specific encoding value identifying the register or literal.
  IsaOperand(int size_bits, typename Isa::OperandType opr_type, int encoding_value = 0)
      : Operand(size_bits, encoding_value), opr_type_(opr_type) {}

  /// @brief ISA-specific operand type tag.
  typename Isa::OperandType opr_type_{};
};

/// @brief AMDGPU-flavored `IsaOperand` that owns the SIMD fast-path
/// overrides (`simd_capable`, `read_lane_chunk`, `write_lane_chunk`,
/// `simd_vgpr_storage`, `simd_vgpr_storage_mut`, the 64-bit pair forms, and
/// `simd_vgpr_base`) so per-arch `Operand` subclasses do
/// not duplicate the same body across 9 ISAs. The implementations live
/// in `isa_operand_simd_inl.h` and call into the per-arch `Isa::`
/// traits struct (`resolved_vgpr_offset`, `is_immediate_type`,
/// `can_resolve_src_scalar`, `resolve_src_scalar`). Non-AMDGPU arches
/// (e.g. RISC-V) inherit directly from `IsaOperand` and use the base
/// `Operand` defaults.
///
/// This remains as the generator fallback for AMDGPU profiles that opt out of
/// split execution sources. Built-in AMDGPU targets use `IsaOperand` plus a
/// per-target execution table instead. Execution-only fallback definitions live
/// in `isa_operand_simd_inl.h`.
///
/// @tparam Isa AMDGPU arch ISA traits providing the SIMD helpers above.
template <typename Isa> class AmdgpuIsaOperand : public IsaOperand<Isa> {
public:
  using IsaOperand<Isa>::IsaOperand;

  bool simd_capable() const override;

private:
  template <typename OtherIsa>
  friend void detail::amdgpu_isa_read_lane_chunk_base(const AmdgpuIsaOperand<OtherIsa> &op,
                                                      const amdgpu::Wavefront &wf,
                                                      uint32_t lane_base, uint32_t count,
                                                      uint32_t *out);
  template <typename OtherIsa>
  friend void detail::amdgpu_isa_write_lane_chunk_base(const AmdgpuIsaOperand<OtherIsa> &op,
                                                       amdgpu::Wavefront &wf, uint32_t lane_base,
                                                       uint32_t count, const uint32_t *vals,
                                                       uint64_t mask);

  void read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                       uint32_t *out) const override;
  void write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                        const uint32_t *vals, uint64_t mask) const override;

  std::optional<uint32_t> simd_vgpr_base_impl(const amdgpu::Wavefront &wf) const override;
  std::optional<uint32_t> simd_vgpr_base_mut_impl(amdgpu::Wavefront &wf) const override;
  amdgpu::ConstVgprStorage simd_vgpr_storage_impl(const amdgpu::Wavefront &wf) const override;
  amdgpu::VgprStorage simd_vgpr_storage_mut_impl(amdgpu::Wavefront &wf) const override;
  amdgpu::ConstVgprStoragePair64
  simd_vgpr_storage64_impl(const amdgpu::Wavefront &wf) const override;
  amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut_impl(amdgpu::Wavefront &wf) const override;
  void simd_notify_read_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask,
                             uint8_t byte_mask) const override;
  void simd_notify_read_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                 uint8_t byte_mask) const override;
  void simd_notify_read64_impl(const amdgpu::Wavefront &wf, uint64_t lane_mask,
                               uint8_t byte_mask) const override;
  void simd_notify_read64_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                   uint8_t byte_mask) const override;
  void simd_notify_write_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                  uint8_t byte_mask) const override;
  void simd_notify_write64_mut_impl(amdgpu::Wavefront &wf, uint64_t lane_mask,
                                    uint8_t byte_mask) const override;
};

/// @brief Temporarily redirect operand reads through another operand.
///
/// Restores the previous delegate on scope exit, including exception and early
/// return paths. A null delegate leaves the operand unchanged.
class ScopedOperandDelegate {
public:
  ScopedOperandDelegate(Operand &operand, Operand *delegate) noexcept {
    if (!delegate)
      return;
    operand_ = &operand;
    previous_ = operand.delegate();
    operand.set_delegate(delegate);
  }

  ~ScopedOperandDelegate() noexcept { restore(); }

  ScopedOperandDelegate(const ScopedOperandDelegate &) = delete;
  ScopedOperandDelegate &operator=(const ScopedOperandDelegate &) = delete;
  ScopedOperandDelegate(ScopedOperandDelegate &&) = delete;
  ScopedOperandDelegate &operator=(ScopedOperandDelegate &&) = delete;

private:
  void restore() noexcept {
    if (!operand_)
      return;
    operand_->set_delegate(previous_);
    operand_ = nullptr;
  }

  Operand *operand_ = nullptr;
  Operand *previous_ = nullptr;
};

} // namespace rocjitsu

#endif // ROCJITSU_ISA_OPERAND_H_
