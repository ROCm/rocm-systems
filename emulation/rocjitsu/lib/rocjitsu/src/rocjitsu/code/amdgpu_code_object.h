// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amdgpu_code_object.h
/// @brief AMD GPU HSA ELF code object representation.

#ifndef ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_
#define ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_

#include "rocjitsu/code/code_object.h"
#include "rocjitsu/code/rj_code.h"

#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {

inline constexpr uint64_t kAmdGpuCodeObjectRetainedDerivedStateImageUnits = 3;

/// Conservative image-sized ownership units retained by one parsed object.
///
/// The bound covers the backing image, up to two units for section objects and
/// their vector slots, one copied-payload unit, one copied-section-name unit,
/// and up to three bounded symbol- and metadata-derived units for names and
/// fixed entry state.
inline constexpr uint64_t kAmdGpuCodeObjectRetainedMajorImageUnits =
    5 + kAmdGpuCodeObjectRetainedDerivedStateImageUnits;

struct AmdGpuKernelInfo {
  std::string name;
  uint64_t descriptor_file_offset = 0;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t text_size = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
  bool has_text_range = false;
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
  std::optional<uint16_t> sgpr_count;
  std::optional<std::array<uint32_t, 3>> required_workgroup_size;
};

struct AmdGpuFunctionInfo {
  std::string name;
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t text_size = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
};

namespace amdgpu_code_object_detail {

/// Parser-private layout models and checked ownership-charge arithmetic.
///
/// The private container charges remain expressible as constant expressions in
/// this header and are pinned against the production types in
/// amdgpu_code_object.cpp.
struct KernelMetadata {
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
  std::optional<uint16_t> sgpr_count;
  std::optional<std::array<uint32_t, 3>> required_workgroup_size;
};

[[nodiscard]] inline constexpr uint64_t aligned_charge(uint64_t bytes, uint64_t alignment) {
  return ((bytes + alignment - 1u) / alignment) * alignment;
}

inline constexpr uint64_t kAssociativeEntryBookkeepingBytes = 4 * sizeof(void *);
inline constexpr uint64_t kViewedBooleanEntryBytes =
    sizeof(std::pair<const std::string_view, bool>);
inline constexpr uint64_t kFunctionSymbolEntryBytes = aligned_charge(
    sizeof(std::string_view) + 4 * sizeof(uint64_t) + sizeof(bool), alignof(uint64_t));
inline constexpr uint64_t kFunctionEvidenceEntryBytes =
    aligned_charge(3 * sizeof(uint64_t) + sizeof(bool), alignof(uint64_t));

[[nodiscard]] inline constexpr std::optional<uint64_t>
checked_allocation_charge(uint64_t accumulated, uint64_t count, uint64_t element_bytes) {
  if (count != 0 && element_bytes > std::numeric_limits<uint64_t>::max() / count)
    return std::nullopt;
  const uint64_t bytes = count * element_bytes;
  if (bytes > std::numeric_limits<uint64_t>::max() - accumulated)
    return std::nullopt;
  return accumulated + bytes;
}

[[nodiscard]] inline constexpr std::optional<uint64_t>
excess_vector_charge(uint64_t capacity, uint64_t requested, uint64_t element_bytes) {
  if (capacity < requested)
    return std::nullopt;
  return checked_allocation_charge(0, capacity - requested, element_bytes);
}

} // namespace amdgpu_code_object_detail

/// Conservative fixed charges for parser-derived state.
///
/// Aggregate accounting combines symbol roles that share a logical name. These
/// charges are built from retained public records, transient entries, metadata
/// entries, and associative container nodes. Copied name bytes and excess
/// vector capacity are charged separately. Allocator bookkeeping remains
/// outside the major-image model.
inline constexpr uint64_t kAmdGpuCodeObjectTransientSymbolEntryChargeBytes =
    amdgpu_code_object_detail::kViewedBooleanEntryBytes +
    amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes;
inline constexpr uint64_t kAmdGpuCodeObjectKernelEntryChargeBytes =
    sizeof(AmdGpuKernelInfo) + sizeof(std::pair<const std::string, uint64_t>) +
    2 * sizeof(std::pair<const std::string_view, uint64_t>) +
    3 * amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes;
inline constexpr uint64_t kAmdGpuCodeObjectKernelAndTransientEntryChargeBytes =
    kAmdGpuCodeObjectKernelEntryChargeBytes + kAmdGpuCodeObjectTransientSymbolEntryChargeBytes;
inline constexpr uint64_t kAmdGpuCodeObjectFunctionEntryChargeBytes =
    sizeof(AmdGpuFunctionInfo) + amdgpu_code_object_detail::kFunctionSymbolEntryBytes +
    amdgpu_code_object_detail::kFunctionEvidenceEntryBytes +
    2 * amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes;
inline constexpr uint64_t kAmdGpuCodeObjectFunctionAndTransientEntryChargeBytes =
    kAmdGpuCodeObjectFunctionEntryChargeBytes + kAmdGpuCodeObjectTransientSymbolEntryChargeBytes;
inline constexpr uint64_t kAmdGpuCodeObjectKernelMetadataEntryChargeBytes =
    sizeof(std::pair<const std::string_view, amdgpu_code_object_detail::KernelMetadata>) +
    amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes;

namespace amdgpu_code_object_detail {

[[nodiscard]] inline constexpr uint64_t
retained_symbol_role_charge(bool has_kernel, bool has_function, bool has_dynamic_stack) {
  return (has_kernel ? kAmdGpuCodeObjectKernelEntryChargeBytes : 0u) +
         (has_function ? kAmdGpuCodeObjectFunctionEntryChargeBytes : 0u) +
         (has_dynamic_stack ? kAmdGpuCodeObjectTransientSymbolEntryChargeBytes : 0u);
}

} // namespace amdgpu_code_object_detail

/// Decode `GRANULATED_WAVEFRONT_SGPR_COUNT` for one kernel descriptor.
///
/// A zero field denotes an eight-register allocation on CDNA and the fixed
/// per-wave SGPR pool on RDNA. Nonzero fields encode `(granulated + 1) * 8`
/// on both families.
[[nodiscard]] uint32_t amdgpu_kernel_descriptor_sgpr_count(uint32_t granulated,
                                                           rj_code_arch_t arch);

/// @brief Represents a single AMD GPU HSA ELF code object.
///
/// A code object is a device ELF containing GPU machine code (.text sections),
/// read-only data (.rodata), and metadata. It may be loaded from a standalone
/// file or extracted from a HIP fat binary by Executable.
///
/// @see [AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
class AmdGpuCodeObject : public CodeObject {
public:
  AmdGpuCodeObject() = default;
  AmdGpuCodeObject(const AmdGpuCodeObject &) = delete;
  AmdGpuCodeObject &operator=(const AmdGpuCodeObject &) = delete;
  AmdGpuCodeObject(AmdGpuCodeObject &&) noexcept;
  AmdGpuCodeObject &operator=(AmdGpuCodeObject &&) = delete;
  ~AmdGpuCodeObject();

  /// @brief Load a standalone device ELF from @p elf_path.
  /// @param[in] elf_path Path to a standalone device ELF file.
  explicit AmdGpuCodeObject(const std::string &elf_path);

  /// @brief Construct from raw ELF bytes in memory.
  /// @param[in] elf_bytes Pointer to the ELF image.
  /// @param[in] elf_size Size of the ELF image in bytes.
  AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size);

  /// @brief Construct from an embedded ELF image copied out of a HIP fat binary.
  /// @param[in] size Size of the embedded ELF in bytes.
  /// @param[in] elf_bytes Pointer to the embedded ELF image.
  /// @param[in] offload_kind Clang offload bundle kind string (e.g. "hip", "hipv4").
  /// @param[in] target_triple GPU target triple string (e.g. "gfx942", "gfx950", "gfx1250").
  AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size, std::string offload_kind,
                   std::string target_triple);

  /// @brief The target ID for this code object (e.g. ROCJITSU_CODE_TARGET_GFX942).
  /// @returns Target ID enum value.
  rj_code_target_id_t target_id() const { return target_id_; }

  /// @brief The GPU target triple string (e.g. "gfx942").
  /// @returns Reference to the target triple string.
  const std::string &target_triple() const { return target_triple_; }

  /// @brief Kernel descriptor/function symbols discovered in this code object.
  const std::vector<AmdGpuKernelInfo> &kernels() const { return kernels_; }

  /// @brief All `.text` function symbols discovered in this code object.
  const std::vector<AmdGpuFunctionInfo> &functions() const { return functions_; }

  uint64_t kernel_descriptor_offset(const std::string &kernel_name) const override;

  /// @brief Smallest per-wavefront SGPR allocation across this object's kernels.
  ///
  /// @details Decodes each kernel descriptor's `GRANULATED_WAVEFRONT_SGPR_COUNT`
  /// and returns the minimum, or nullopt when no readable kernel descriptor is
  /// present.
  ///
  /// Mirrors the command processor's decode: when the descriptor encodes the
  /// count (granulated != 0, or a CDNA target) the value is `(granulated+1)*8`;
  /// otherwise the granulated field is an RDNA-style sentinel and the wave owns
  /// the fixed per-wave SGPR pool. @p arch selects that interpretation.
  [[nodiscard]] std::optional<uint32_t> min_kernel_sgpr_count(rj_code_arch_t arch) const;

private:
  void load_sections();

  rj_code_target_id_t target_id_ = ROCJITSU_CODE_TARGET_INVALID;
  std::string offload_kind_;
  std::string target_triple_;
  std::unordered_map<std::string, uint64_t> kd_offsets_; ///< kernel_name -> .kd symbol offset
  std::vector<AmdGpuKernelInfo> kernels_;
  std::vector<AmdGpuFunctionInfo> functions_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_AMDGPU_CODE_OBJECT_H_
