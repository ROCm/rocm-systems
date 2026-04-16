//===-- DescriptorUpdater.cpp - Kernel Descriptor Update --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of kernel descriptor parsing and modification.
///
/// AMD Kernel Descriptor layout (64 bytes) for AMDHSA.
/// Source: llvm/include/llvm/Support/AMDHSAKernelDescriptor.h
///
/// Offset  Size  Field
/// ------  ----  -----
///   0      4    group_segment_fixed_size (LDS size in bytes)
///   4      4    private_segment_fixed_size (scratch size per work-item)
///   8      4    kernarg_size
///  12      4    reserved0
///  16      8    kernel_code_entry_byte_offset
///  24     20    reserved1
///  44      4    compute_pgm_rsrc3 (GFX10+ and GFX90A+)
///  48      4    compute_pgm_rsrc1
///  52      4    compute_pgm_rsrc2
///  56      2    kernel_code_properties (KERNEL_CODE_PROPERTY_* bits)
///  58      2    kernarg_preload
///  60      4    reserved3
///
/// COMPUTE_PGM_RSRC1 layout:
///   [5:0]   GRANULATED_WORKITEM_VGPR_COUNT
///   [9:6]   GRANULATED_WAVEFRONT_SGPR_COUNT
///   [11:10] PRIORITY
///   [13:12] FLOAT_ROUND_MODE_32
///   [15:14] FLOAT_ROUND_MODE_16_64
///   [17:16] FLOAT_DENORM_MODE_32
///   [19:18] FLOAT_DENORM_MODE_16_64
///   [20]    PRIV
///   [21]    ENABLE_DX10_CLAMP
///   [22]    DEBUG_MODE
///   [23]    ENABLE_IEEE_MODE
///   [24]    BULKY
///   [25]    CDBG_USER
///   [26]    FP16_OVFL
///   [27]    Reserved
///   [29:28] RESERVED (should be 0)
///   [31:30] WGP_MODE (GFX10+) / Reserved (GFX9)
///
/// COMPUTE_PGM_RSRC2 layout:
///   [0]     ENABLE_PRIVATE_SEGMENT
///   [5:1]   USER_SGPR_COUNT
///   [6]     TRAP_HANDLER
///   [7]     TGID_X_EN
///   [8]     TGID_Y_EN
///   [9]     TGID_Z_EN
///   [10]    TG_SIZE_EN
///   [15:11] TIDIG_COMP_CNT (0-3)
///   [16]    EXCP_EN_MSB
///   [22:17] LDS_SIZE
///   [31:23] EXCP_EN
///
//===----------------------------------------------------------------------===//

#include "aegisbit/DescriptorUpdater.h"
#include "aegisbit/Endian.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Error.h"

using namespace llvm;

namespace aegisbit {

uint32_t DescriptorUpdater::getVGPRGranularity(StringRef GPUArch) {
  // gfx90a and later (MI210, MI300, MI350) use granularity 8
  // Older architectures (gfx900, gfx906, etc.) use granularity 4
  if (GPUArch.starts_with("gfx90a") ||   // MI210
      GPUArch.starts_with("gfx940") ||   // MI300A
      GPUArch.starts_with("gfx942") ||   // MI300X
      GPUArch.starts_with("gfx950") ||   // MI350X (CDNA4)
      GPUArch.starts_with("gfx1250")) {  // MI450 (CDNA5)
    return 8;
  }
  return 4;  // Default for older architectures
}

uint32_t DescriptorUpdater::extractVGPRCount(uint32_t PgmRsrc1, uint32_t Granularity) {
  // GRANULATED_WORKITEM_VGPR_COUNT is in bits [5:0]
  uint32_t Granulated = PgmRsrc1 & VGPR_COUNT_MASK;
  // Actual count = (granulated + 1) * granularity
  return (Granulated + 1) * Granularity;
}

uint32_t DescriptorUpdater::extractSGPRCount(uint32_t PgmRsrc1) {
  // GRANULATED_WAVEFRONT_SGPR_COUNT is in bits [9:6]
  uint32_t Granulated = (PgmRsrc1 >> SGPR_COUNT_SHIFT) & (SGPR_COUNT_MASK >> SGPR_COUNT_SHIFT);
  // Actual count = (granulated + 1) * 8
  return (Granulated + 1) * SGPR_GRANULARITY;
}

uint32_t DescriptorUpdater::encodeVGPRCount(uint32_t PgmRsrc1, uint32_t VGPRCount, uint32_t Granularity) {
  // Round up to granularity
  uint32_t Rounded = roundUpVGPR(VGPRCount, Granularity);
  // Granulated = (count / granularity) - 1
  uint32_t Granulated = (Rounded / Granularity) - 1;
  // Clamp to 6-bit field (max 63)
  if (Granulated > VGPR_COUNT_MASK)
    Granulated = VGPR_COUNT_MASK;
  // Clear VGPR bits and set new value
  return (PgmRsrc1 & ~VGPR_COUNT_MASK) | Granulated;
}

uint32_t DescriptorUpdater::encodeSGPRCount(uint32_t PgmRsrc1, uint32_t SGPRCount) {
  // Round up to granularity
  uint32_t Rounded = roundUpSGPR(SGPRCount);
  // Granulated = (count / 8) - 1
  uint32_t Granulated = (Rounded / SGPR_GRANULARITY) - 1;
  // Clamp to 4-bit field (max 15)
  if (Granulated > (SGPR_COUNT_MASK >> SGPR_COUNT_SHIFT))
    Granulated = SGPR_COUNT_MASK >> SGPR_COUNT_SHIFT;
  // Clear SGPR bits and set new value
  return (PgmRsrc1 & ~SGPR_COUNT_MASK) | (Granulated << SGPR_COUNT_SHIFT);
}

Expected<KernelDescriptor> DescriptorUpdater::parse(ArrayRef<uint8_t> Bytes, StringRef GPUArch) {
  if (Bytes.size() < DESCRIPTOR_SIZE) {
    return createStringError(
        inconvertibleErrorCode(),
        "Kernel descriptor too small: expected " + Twine(DESCRIPTOR_SIZE) +
            " bytes, got " + Twine(Bytes.size()));
  }

  KernelDescriptor KD;

  // Parse fields at known offsets per LLVM's AMDHSAKernelDescriptor.h
  KD.GroupSegmentFixedSize = readLE32(&Bytes[0]);    // Offset 0: LDS size
  KD.PrivateSegmentFixedSize = readLE32(&Bytes[4]);  // Offset 4: scratch size
  KD.KernargSize = readLE32(&Bytes[8]);              // Offset 8: kernarg size
  KD.KernelCodeEntryByteOffset = readLE64(&Bytes[16]); // Offset 16: code offset

  // Offset 44: COMPUTE_PGM_RSRC3
  KD.ComputePgmRsrc3 = readLE32(&Bytes[44]);

  // Offset 48: COMPUTE_PGM_RSRC1 (contains VGPR/SGPR counts)
  KD.ComputePgmRsrc1 = readLE32(&Bytes[48]);

  // Offset 52: COMPUTE_PGM_RSRC2
  KD.ComputePgmRsrc2 = readLE32(&Bytes[52]);

  // Offset 56: kernel_code_properties (16-bit)
  KD.KernelCodeProperties = readLE16(&Bytes[56]);

  // Offset 58: kernarg_preload (16-bit)
  KD.KernargPreload = readLE16(&Bytes[58]);

  // Determine VGPR granularity based on architecture
  uint32_t VGPRGranularity = getVGPRGranularity(GPUArch);

  // Extract VGPR and SGPR counts
  KD.VGPRCount = extractVGPRCount(KD.ComputePgmRsrc1, VGPRGranularity);
  KD.SGPRCount = extractSGPRCount(KD.ComputePgmRsrc1);

  // Store granularity in descriptor for later use
  KD.VGPRGranularity = VGPRGranularity;

  // gfx940+/CDNA3+ have ArchitectedFlatScratch: 6 implicit SGPRs
  // (VCC=2, FLAT_SCRATCH=2, XNACK_MASK=2). Earlier GFX9 targets have 4.
  if (GPUArch.starts_with("gfx94") || GPUArch.starts_with("gfx95"))
    KD.ImplicitSGPRs = 6;
  else
    KD.ImplicitSGPRs = 4;

  // Extract ACCUM_OFFSET from COMPUTE_PGM_RSRC3 (gfx90a/gfx942/gfx950)
  // Bits [5:0], granularity 4: actual offset = (encoded + 1) * 4
  uint32_t AccumEncoded = KD.ComputePgmRsrc3 & ACCUM_OFFSET_MASK;
  if (KD.ComputePgmRsrc3 != 0) {
    KD.AccumOffset = (AccumEncoded + 1) * ACCUM_OFFSET_GRANULARITY;
  } else {
    KD.AccumOffset = 0;
  }

  // Default wavefront size for GFX9+
  KD.WavefrontSize = 64;

  return KD;
}

Error DescriptorUpdater::updateVGPRCount(KernelDescriptor& KD, uint32_t NewCount) {
  // Use the granularity stored in the descriptor (set during parse)
  uint32_t Granularity = KD.VGPRGranularity > 0 ? KD.VGPRGranularity : 4;
  uint32_t HWMax = maxVGPRs(Granularity);
  if (NewCount > HWMax) {
    return createStringError(
        inconvertibleErrorCode(),
        "VGPR count " + Twine(NewCount) + " exceeds maximum " +
            Twine(HWMax) + " (granularity " + Twine(Granularity) + ")");
  }

  // Round up to granularity
  uint32_t Rounded = roundUpVGPR(NewCount, Granularity);

  // Update the extracted count
  KD.VGPRCount = Rounded;

  // Update COMPUTE_PGM_RSRC1
  KD.ComputePgmRsrc1 = encodeVGPRCount(KD.ComputePgmRsrc1, Rounded, Granularity);

  return Error::success();
}

Error DescriptorUpdater::updateSGPRCount(KernelDescriptor& KD, uint32_t NewCount) {
  if (NewCount > MAX_SGPRS) {
    return createStringError(
        inconvertibleErrorCode(),
        "SGPR count " + Twine(NewCount) + " exceeds maximum " + Twine(MAX_SGPRS));
  }

  // Round up to granularity
  uint32_t Rounded = roundUpSGPR(NewCount);

  // Update the extracted count
  KD.SGPRCount = Rounded;

  // Update COMPUTE_PGM_RSRC1
  KD.ComputePgmRsrc1 = encodeSGPRCount(KD.ComputePgmRsrc1, Rounded);

  return Error::success();
}

void DescriptorUpdater::updateAccumOffset(KernelDescriptor& KD, uint32_t NewOffset) {
  // Round up to ACCUM_OFFSET_GRANULARITY (4)
  uint32_t Rounded = ((NewOffset + ACCUM_OFFSET_GRANULARITY - 1) /
                       ACCUM_OFFSET_GRANULARITY) * ACCUM_OFFSET_GRANULARITY;
  KD.AccumOffset = Rounded;

  // Encode: encoded = (offset / 4) - 1
  uint32_t Encoded = (Rounded / ACCUM_OFFSET_GRANULARITY) - 1;
  if (Encoded > ACCUM_OFFSET_MASK)
    Encoded = ACCUM_OFFSET_MASK;

  KD.ComputePgmRsrc3 = (KD.ComputePgmRsrc3 & ~ACCUM_OFFSET_MASK) | Encoded;
}

void DescriptorUpdater::updateScratchSize(KernelDescriptor& KD, uint32_t NewSize) {
  KD.PrivateSegmentFixedSize = NewSize;
}

void DescriptorUpdater::enablePrivateSegment(KernelDescriptor& KD) {
  // COMPUTE_PGM_RSRC2 bit 0 is ENABLE_PRIVATE_SEGMENT
  // This must be set when the kernel uses scratch memory so that
  // flat_scratch is initialized by the runtime.
  constexpr uint32_t ENABLE_PRIVATE_SEGMENT_BIT = 0x1;
  KD.ComputePgmRsrc2 |= ENABLE_PRIVATE_SEGMENT_BIT;
}

void DescriptorUpdater::updateLDSSize(KernelDescriptor& KD, uint32_t NewSize) {
  KD.GroupSegmentFixedSize = NewSize;
}

void DescriptorUpdater::updateKernargSize(KernelDescriptor& KD, uint32_t NewSize) {
  KD.KernargSize = NewSize;
}

std::vector<uint8_t> DescriptorUpdater::serialize(const KernelDescriptor& KD) {
  std::vector<uint8_t> Bytes(DESCRIPTOR_SIZE, 0);

  // Write fields at known offsets per LLVM's AMDHSAKernelDescriptor.h
  writeLE32(&Bytes[0], KD.GroupSegmentFixedSize);      // Offset 0
  writeLE32(&Bytes[4], KD.PrivateSegmentFixedSize);    // Offset 4
  writeLE32(&Bytes[8], KD.KernargSize);                // Offset 8
  // Offset 12: reserved0 (4 bytes, zeros)
  writeLE64(&Bytes[16], KD.KernelCodeEntryByteOffset); // Offset 16
  // Offset 24-43: reserved1 (20 bytes, zeros)
  writeLE32(&Bytes[44], KD.ComputePgmRsrc3);           // Offset 44
  writeLE32(&Bytes[48], KD.ComputePgmRsrc1);           // Offset 48
  writeLE32(&Bytes[52], KD.ComputePgmRsrc2);           // Offset 52
  writeLE16(&Bytes[56], KD.KernelCodeProperties);      // Offset 56
  writeLE16(&Bytes[58], KD.KernargPreload);            // Offset 58
  // Offset 60-63: reserved3 (4 bytes, zeros)

  return Bytes;
}

uint16_t DescriptorUpdater::extractKernelCodeProperties(
    ArrayRef<uint8_t> DescriptorBytes) {
  if (DescriptorBytes.size() < DESCRIPTOR_SIZE) {
    return 0;  // Invalid descriptor
  }
  // KERNEL_CODE_PROPERTIES at offset 56 (per LLVM's AMDHSAKernelDescriptor.h)
  // This is a 16-bit field containing KERNEL_CODE_PROPERTY_* bits
  return readLE16(&DescriptorBytes[56]);
}

int DescriptorUpdater::computeKernargBaseSGPR(uint16_t KernelCodeProperties) {
  // Bit definitions from AMD AMDHSA ABI specification
  const uint16_t ENABLE_PRIVATE_SEGMENT_BUFFER = (1 << 0);
  const uint16_t ENABLE_DISPATCH_PTR            = (1 << 1);
  const uint16_t ENABLE_QUEUE_PTR               = (1 << 2);
  const uint16_t ENABLE_KERNARG_SEGMENT_PTR     = (1 << 3);

  // Check if KERNARG_SEGMENT_PTR is enabled
  if (!(KernelCodeProperties & ENABLE_KERNARG_SEGMENT_PTR)) {
    return -1;  // Not enabled - cannot use kernarg-based trace
  }

  // Count enabled USER SGPRs before KERNARG_SEGMENT_PTR
  // Each enabled feature uses a specific number of SGPRs
  int SGPRIndex = 0;
  if (KernelCodeProperties & ENABLE_PRIVATE_SEGMENT_BUFFER) SGPRIndex += 4;
  if (KernelCodeProperties & ENABLE_DISPATCH_PTR) SGPRIndex += 2;
  if (KernelCodeProperties & ENABLE_QUEUE_PTR) SGPRIndex += 2;

  return SGPRIndex;  // KERNARG_SEGMENT_PTR location
}

} // namespace aegisbit
