//===-- aegisbit/DescriptorUpdater.h - Kernel Descriptor Update --*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernel descriptor parsing and modification for AMD GPU kernels.
/// The kernel descriptor is a 64-byte structure in .rodata that defines
/// register usage, scratch size, and other kernel properties.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_DESCRIPTOR_UPDATER_H
#define AEGISBIT_DESCRIPTOR_UPDATER_H

#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <vector>

namespace aegisbit {

/// AMD Kernel Descriptor layout for GFX9+ architectures
/// See AMD GPU ISA documentation for full details.
///
/// The descriptor is 64 bytes and contains register counts encoded in
/// COMPUTE_PGM_RSRC1/2 registers with granularity.
///
/// VGPR encoding: granulated_vgpr_count = (vgpr_count / granularity) - 1
///   gfx90a+: granularity = 8, e.g., 128 VGPRs: (128/8) - 1 = 15
///   older:   granularity = 4, e.g., 128 VGPRs: (128/4) - 1 = 31
/// SGPR encoding: granulated_sgpr_count = (sgpr_count / 8) - 1
///   For example, 48 SGPRs: (48/8) - 1 = 5 (encoded value)
class DescriptorUpdater {
public:
  /// SGPR granularity (allocation unit is 8 SGPRs for all architectures)
  static constexpr uint32_t SGPR_GRANULARITY = 8;

  /// Maximum VGPRs encodable in COMPUTE_PGM_RSRC1 bits [5:0].
  /// Actual limit depends on VGPR granularity:
  ///   pre-gfx90a (granularity 4): (63+1)*4 = 256
  ///   gfx90a+    (granularity 8): (63+1)*8 = 512
  static constexpr uint32_t MAX_VGPRS_GRAN4 = 256;
  static constexpr uint32_t MAX_VGPRS_GRAN8 = 512;

  /// Return the hardware VGPR limit for a given granularity.
  static constexpr uint32_t maxVGPRs(uint32_t Granularity) {
    return (VGPR_COUNT_MASK + 1) * Granularity;  // (63+1) * gran
  }

  /// Maximum SGPRs (104 for GFX9+)
  static constexpr uint32_t MAX_SGPRS = 104;

  /// Kernel descriptor size in bytes
  static constexpr size_t DESCRIPTOR_SIZE = 64;

  /// Bit masks for COMPUTE_PGM_RSRC1 register fields
  static constexpr uint32_t VGPR_COUNT_MASK = 0x3Fu;      ///< Bits [5:0]
  static constexpr uint32_t SGPR_COUNT_MASK = 0x3C0u;     ///< Bits [9:6]
  static constexpr uint32_t SGPR_COUNT_SHIFT = 6;         ///< Bit position for SGPR field

  /// Bit masks for COMPUTE_PGM_RSRC3 register fields (gfx90a+)
  static constexpr uint32_t ACCUM_OFFSET_MASK = 0x3Fu;    ///< Bits [5:0]
  static constexpr uint32_t ACCUM_OFFSET_GRANULARITY = 4; ///< Always 4

  /// Get VGPR granularity for a given GPU architecture
  /// \param GPUArch GPU architecture name (e.g., "gfx90a", "gfx942", "gfx950")
  /// \return VGPR granularity (8 for gfx90a+, 4 for older)
  static uint32_t getVGPRGranularity(llvm::StringRef GPUArch);

  /// Parse a 64-byte kernel descriptor
  /// \param Bytes Raw descriptor bytes (must be exactly 64 bytes)
  /// \param GPUArch Optional GPU architecture (for correct VGPR granularity)
  /// \return Parsed KernelDescriptor or error
  static llvm::Expected<KernelDescriptor> parse(llvm::ArrayRef<uint8_t> Bytes,
                                                 llvm::StringRef GPUArch = "");

  /// Update VGPR count in descriptor
  /// Handles granulation automatically (rounds up to nearest 4).
  /// \param KD Kernel descriptor to modify
  /// \param NewCount New total VGPR count
  /// \return Error if count exceeds maximum
  static llvm::Error updateVGPRCount(KernelDescriptor& KD, uint32_t NewCount);

  /// Update SGPR count in descriptor
  /// Handles granulation automatically (rounds up to nearest 8).
  /// \param KD Kernel descriptor to modify
  /// \param NewCount New total SGPR count
  /// \return Error if count exceeds maximum
  static llvm::Error updateSGPRCount(KernelDescriptor& KD, uint32_t NewCount);

  /// Update ACCUM_OFFSET in COMPUTE_PGM_RSRC3 (gfx90a/gfx942).
  /// \param KD Kernel descriptor to modify
  /// \param NewOffset New AccumOffset value (VGPR index, must be multiple of 4)
  static void updateAccumOffset(KernelDescriptor& KD, uint32_t NewOffset);

  /// Check if this descriptor uses AccVGPRs (ACCUM_OFFSET > 0).
  static bool hasAccumVGPRs(const KernelDescriptor& KD) {
    return KD.AccumOffset > 0;
  }

  /// Update scratch (private segment) size
  /// \param KD Kernel descriptor to modify
  /// \param NewSize New scratch size in bytes per work-item
  static void updateScratchSize(KernelDescriptor& KD, uint32_t NewSize);

  /// Enable private segment (scratch memory) in COMPUTE_PGM_RSRC2.
  /// Must be called when adding scratch to a kernel that didn't use it.
  /// \param KD Kernel descriptor to modify
  static void enablePrivateSegment(KernelDescriptor& KD);

  /// Update LDS (group segment) size
  /// \param KD Kernel descriptor to modify
  /// \param NewSize New LDS size in bytes
  static void updateLDSSize(KernelDescriptor& KD, uint32_t NewSize);

  /// Update kernarg size
  /// \param KD Kernel descriptor to modify
  /// \param NewSize New kernel argument size in bytes
  static void updateKernargSize(KernelDescriptor& KD, uint32_t NewSize);

  /// Serialize kernel descriptor back to bytes
  /// \param KD Kernel descriptor to serialize
  /// \return 64-byte serialized descriptor
  static std::vector<uint8_t> serialize(const KernelDescriptor& KD);

  /// Parse KERNEL_CODE_PROPERTIES from descriptor bytes
  /// \param DescriptorBytes Full 64-byte kernel descriptor
  /// \return 16-bit KERNEL_CODE_PROPERTIES value (0 if invalid descriptor)
  static uint16_t extractKernelCodeProperties(llvm::ArrayRef<uint8_t> DescriptorBytes);

  /// Compute the SGPR index where KERNARG_SEGMENT_PTR is located
  /// \param KernelCodeProperties 16-bit properties value from descriptor
  /// \return SGPR index (0-based) or -1 if KERNARG_SEGMENT_PTR not enabled
  static int computeKernargBaseSGPR(uint16_t KernelCodeProperties);

  /// Round VGPR count up to granularity
  static uint32_t roundUpVGPR(uint32_t Count, uint32_t Granularity) {
    return ((Count + Granularity - 1) / Granularity) * Granularity;
  }

  /// Round SGPR count up to granularity
  static constexpr uint32_t roundUpSGPR(uint32_t Count) {
    return ((Count + SGPR_GRANULARITY - 1) / SGPR_GRANULARITY) * SGPR_GRANULARITY;
  }

private:
  /// Extract VGPR count from COMPUTE_PGM_RSRC1
  /// \param PgmRsrc1 The COMPUTE_PGM_RSRC1 register value
  /// \param Granularity VGPR granularity (4 or 8)
  static uint32_t extractVGPRCount(uint32_t PgmRsrc1, uint32_t Granularity);

  /// Extract SGPR count from COMPUTE_PGM_RSRC1
  static uint32_t extractSGPRCount(uint32_t PgmRsrc1);

  /// Encode VGPR count into COMPUTE_PGM_RSRC1
  /// \param PgmRsrc1 The current COMPUTE_PGM_RSRC1 value
  /// \param VGPRCount Actual VGPR count to encode
  /// \param Granularity VGPR granularity (4 or 8)
  static uint32_t encodeVGPRCount(uint32_t PgmRsrc1, uint32_t VGPRCount, uint32_t Granularity);

  /// Encode SGPR count into COMPUTE_PGM_RSRC1
  static uint32_t encodeSGPRCount(uint32_t PgmRsrc1, uint32_t SGPRCount);
};

} // namespace aegisbit

#endif // AEGISBIT_DESCRIPTOR_UPDATER_H
