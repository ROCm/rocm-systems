// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file operand.h
/// @brief Instruction operand base class with register read/write interface.

#ifndef ROCJITSU_ISA_OPERAND_H_
#define ROCJITSU_ISA_OPERAND_H_

#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/vm/amdgpu/vgpr_msb.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace simdojo {
template <size_t NUM_ELEMS, typename VecElem> class VectorReg;
}

namespace rocjitsu {
namespace amdgpu {
class Wavefront;
struct SimdAccess;

/// @brief Wave64 VGPR storage type — the register file element. Operands hand
/// the SIMD glue a typed view of this (a `simdojo::VectorReg<64,uint32_t>`, no
/// raw pointer) for the read/write fast path.
using VgprStorage = simdojo::VectorReg<64, uint32_t>;

/// @brief A `{lo, hi}` pair of typed per-register VGPR storage views for a
/// 64-bit-lane operand. `lo` is the lower-numbered VGPR (reg N, bits [31:0]);
/// `hi` is reg N+1 (bits [63:32]). Either both are valid or both are nullptr.
struct VgprStoragePair64 {
  VgprStorage *lo;
  VgprStorage *hi;
};

/// @brief Read-only counterpart of `VgprStoragePair64`.
struct ConstVgprStoragePair64 {
  const VgprStorage *lo;
  const VgprStorage *hi;
};
} // namespace amdgpu

/// @brief Base class for an instruction operand with value resolution.
///
/// Subclasses implement read/write methods using ISA-specific encoding value
/// ranges and the wavefront's register state.
class Operand {
public:
  friend struct amdgpu::SimdAccess;

  Operand() = default;

  /// @brief Construct an operand with the given size and encoding value.
  /// @param size_bits Operand width in bits.
  /// @param encoding_value ISA-specific encoding value identifying the register or literal.
  Operand(int size_bits, int encoding_value)
      : size_bits_(size_bits), encoding_value_(encoding_value) {}
  virtual ~Operand() = default;

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

  /// @brief Operand width in bits.
  int size_bits() const { return size_bits_; }

  /// @brief Whether this operand references a VGPR or AccVGPR.
  /// @details Classified at construction time by ISA-specific subclasses using
  /// the auto-generated is_vgpr_operand_type() from operand_types.h.
  [[nodiscard]] bool is_vgpr() const { return is_vgpr_; }

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
    return static_cast<uint16_t>(std::max(1, size_bits_ / 32));
  }

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

  /// @brief Set a delegate operand that overrides read methods.
  ///
  /// @details Used by DPP/SDWA substitution to redirect reads through a
  /// DppOperand without changing the member variable's type.
  void set_delegate(Operand *d) { delegate_ = d; }
  void clear_delegate() { delegate_ = nullptr; }
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

  int size_bits_ = 0;
  int encoding_value_ = 0;
  bool is_vgpr_ = false;
  amdgpu::VgprMsbRole vgpr_msb_role_ = amdgpu::VgprMsbRole::None;

private:
  /// @brief If this operand resolves to per-lane VGPR storage, return its
  /// physical register index (`wf.vgpr_alloc().base + offset`). Otherwise
  /// nullopt (SGPR/imm/inline-const/DPP) — the caller broadcasts a scalar. The
  /// glue passes this index to the plugin read-notification hook with the full
  /// register extent. Internal SIMD fast-path hook, reachable only through
  /// `amdgpu::SimdAccess`; plugins observe register reads via the public
  /// `read_lane` / `read_lane_chunk` surface.
  virtual std::optional<uint32_t> simd_vgpr_base(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_base(wf);
    (void)wf;
    return std::nullopt;
  }

  /// @brief If this operand resolves to per-lane VGPR storage, return a typed
  /// const view of that register (the `VgprStorage` the file holds, a
  /// `simdojo::VectorReg<64,uint32_t>&`). Otherwise nullptr — the caller falls
  /// back to a scalar broadcast via `read_scalar`. Resolves the storage in a
  /// SINGLE virtual dispatch — the SIMD hot path reads through this without a
  /// raw pointer crossing the operand/glue API.
  virtual const amdgpu::VgprStorage *simd_vgpr_storage(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_storage(wf);
    (void)wf;
    return nullptr;
  }

  /// @brief Mutable counterpart of `simd_vgpr_storage` for the dst write path
  /// (no delegate — a dst is never DPP/SDWA).
  virtual amdgpu::VgprStorage *simd_vgpr_storage_mut(amdgpu::Wavefront &wf) const {
    (void)wf;
    return nullptr;
  }

  /// @brief Notify the plugin system that this operand's VGPR was read
  /// for lanes [lane_begin, lane_end). No-op for non-VGPR operands.
  virtual void simd_notify_read(const amdgpu::Wavefront & /*wf*/, uint32_t /*lane_begin*/,
                                uint32_t /*lane_end*/, uint8_t /*byte_mask*/) const {}

  /// @brief 64-bit-lane counterpart of `simd_vgpr_storage`. A per-lane f64/i64
  /// value occupies two consecutive VGPRs (reg N + reg N+1), so this returns a
  /// `{lo, hi}` pair of typed register views (lo = reg N, hi = reg N+1) in a
  /// SINGLE virtual dispatch. Returns `{nullptr, nullptr}` when the operand is
  /// not contiguous VGPR storage — the caller broadcasts via `read_scalar64`.
  virtual amdgpu::ConstVgprStoragePair64 simd_vgpr_storage64(const amdgpu::Wavefront &wf) const {
    if (delegate_)
      return delegate_->simd_vgpr_storage64(wf);
    (void)wf;
    return {nullptr, nullptr};
  }

  /// @brief Mutable counterpart of `simd_vgpr_storage64` for the 64-bit dst
  /// write path; returns writable `{lo, hi}` register views or
  /// `{nullptr, nullptr}`.
  virtual amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut(amdgpu::Wavefront &wf) const {
    (void)wf;
    return {nullptr, nullptr};
  }

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
/// TODO: this AMDGPU-specific operand machinery could move under the
/// `isa/arch/amdgpu/shared` directory alongside the other per-arch shared
/// code; left here for now to keep the SIMD change self-contained.
///
/// @tparam Isa AMDGPU arch ISA traits providing the SIMD helpers above.
template <typename Isa> class AmdgpuIsaOperand : public IsaOperand<Isa> {
public:
  friend struct amdgpu::SimdAccess;

  using IsaOperand<Isa>::IsaOperand;

  bool simd_capable() const override;
  void read_lane_chunk(const amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                       uint32_t *out) const override;
  void write_lane_chunk(amdgpu::Wavefront &wf, uint32_t lane_base, uint32_t count,
                        const uint32_t *vals, uint64_t mask) const override;

private:
  std::optional<uint32_t> simd_vgpr_base(const amdgpu::Wavefront &wf) const override;
  const amdgpu::VgprStorage *simd_vgpr_storage(const amdgpu::Wavefront &wf) const override;
  amdgpu::VgprStorage *simd_vgpr_storage_mut(amdgpu::Wavefront &wf) const override;
  amdgpu::ConstVgprStoragePair64 simd_vgpr_storage64(const amdgpu::Wavefront &wf) const override;
  amdgpu::VgprStoragePair64 simd_vgpr_storage64_mut(amdgpu::Wavefront &wf) const override;
  void simd_notify_read(const amdgpu::Wavefront &wf, uint32_t lane_begin, uint32_t lane_end,
                        uint8_t byte_mask) const override;
};

/// @brief DPP-aware operand proxy that applies lane permutation on read.
///
/// Wraps a regular VGPR operand and overrides read_lane() to return
/// pre-permuted values. Constructed by the VOP1/VOP2 encoding base when
/// DPP is detected (src0 == 250). The pre-permuted values are computed
/// once at construction time.
class DppOperand : public Operand {
public:
  friend struct amdgpu::SimdAccess;

  static constexpr int MAX_LANES = 64;

  DppOperand() = default;

  /// @brief Construct from a source operand + pre-permuted data.
  /// @param base The underlying operand (for name/size/scalar reads).
  /// @param data Pre-permuted lane values (one per lane).
  /// @param lane_count Number of valid lanes.
  DppOperand(const Operand &base, const uint32_t *data, int lane_count)
      : Operand(base.size_bits_, base.encoding_value_), lane_count_(lane_count) {
    for (int i = 0; i < lane_count && i < MAX_LANES; ++i)
      data_[i] = data[i];
  }

  uint32_t read_lane(const amdgpu::Wavefront & /*wf*/, uint32_t lane) const override {
    return (lane < static_cast<uint32_t>(lane_count_)) ? data_[lane] : 0;
  }

  uint32_t read_scalar(const amdgpu::Wavefront & /*wf*/) const override { return data_[0]; }

  std::string name() const override { return "dpp_src"; }

  bool simd_capable() const override { return true; }

  void read_lane_chunk(const amdgpu::Wavefront & /*wf*/, uint32_t lane_base, uint32_t count,
                       uint32_t *out) const override {
    uint32_t lanes = static_cast<uint32_t>(lane_count_);
    for (uint32_t i = 0; i < count; ++i) {
      uint32_t l = lane_base + i;
      out[i] = (l < lanes) ? data_[l] : 0u;
    }
  }

private:
  /// The pre-permuted lane data is held in a `MAX_LANES`-wide `uint32_t` array
  /// that is bit-layout-identical to `VgprStorage` (`simdojo::VectorReg<64,
  /// uint32_t>` — a `std::array<uint32_t,64>` with the layout `static_assert`
  /// enforced in `ComputeUnitCore::vgpr_reg`). Unused lanes are zero (the
  /// EXEC mask gates them off in the glue), so the whole array is a valid
  /// read-only register view. The cast targets the forward-declared
  /// `VgprStorage`; the glue dereferences it where the full type is visible.
  const amdgpu::VgprStorage *simd_vgpr_storage(const amdgpu::Wavefront & /*wf*/) const override {
    static_assert(sizeof(data_) == MAX_LANES * sizeof(uint32_t),
                  "DppOperand data_ must be layout-compatible with VgprStorage");
    return reinterpret_cast<const amdgpu::VgprStorage *>(&data_);
  }

  uint32_t data_[MAX_LANES]{};
  int lane_count_ = 0;
};

namespace amdgpu {
/// @brief Privileged accessor for the operand SIMD fast-path hooks.
///
/// Only the SIMD glue in `arch/amdgpu/shared/simd_glue.h` (and arch operand
/// implementations that need to forward a delegate dispatch) reaches the
/// private `simd_vgpr_storage` / `simd_vgpr_base` virtuals through this struct.
/// Plugin-visible register I/O stays on the public `read_lane` /
/// `read_lane_chunk` / `write_lane` / `write_lane_chunk` surface.
struct SimdAccess {
  /// Physical register index for plugin read-notification (nullopt for
  /// non-VGPR / DPP operands), resolved in one virtual dispatch.
  template <typename Op>
  static std::optional<uint32_t> vgpr_base(const Op &op, const Wavefront &wf) {
    return op.simd_vgpr_base(wf);
  }
  /// Typed const register view for the 32-bit read path (null for non-VGPR).
  template <typename Op> static const VgprStorage *vgpr_storage(const Op &op, const Wavefront &wf) {
    return op.simd_vgpr_storage(wf);
  }
  /// Typed mutable register view for the 32-bit write path (null for non-VGPR).
  template <typename Op> static VgprStorage *vgpr_storage_mut(const Op &op, Wavefront &wf) {
    return op.simd_vgpr_storage_mut(wf);
  }
  /// Typed `{lo, hi}` const register-view pair for the 64-bit read path.
  template <typename Op>
  static ConstVgprStoragePair64 vgpr_storage64(const Op &op, const Wavefront &wf) {
    return op.simd_vgpr_storage64(wf);
  }
  /// Typed `{lo, hi}` mutable register-view pair for the 64-bit write path.
  template <typename Op> static VgprStoragePair64 vgpr_storage64_mut(const Op &op, Wavefront &wf) {
    return op.simd_vgpr_storage64_mut(wf);
  }
  template <typename Op>
  static void notify_read(const Op &op, const Wavefront &wf, uint32_t lane_begin, uint32_t lane_end,
                          uint8_t byte_mask) {
    op.simd_notify_read(wf, lane_begin, lane_end, byte_mask);
  }
};
} // namespace amdgpu

} // namespace rocjitsu

#endif // ROCJITSU_ISA_OPERAND_H_
