//===-- DescriptorFixture.h - Kernel Descriptor Test Fixture ---*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Test fixture for kernel descriptor parsing and modification tests.
/// Provides sample descriptors and validation helpers.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TEST_DESCRIPTOR_FIXTURE_H
#define AEGISBIT_TEST_DESCRIPTOR_FIXTURE_H

#include "aegisbit/DescriptorUpdater.h"
#include <gtest/gtest.h>
#include <vector>

namespace aegisbit {
namespace test {

//===----------------------------------------------------------------------===//
// LLVM Error/Expected Helpers for GoogleTest
//
// LLVM's Error and Expected<T> have non-const operator bool(), which doesn't
// work directly with GoogleTest's ASSERT_TRUE/EXPECT_TRUE. Use these macros.
//===----------------------------------------------------------------------===//

/// Check that an llvm::Expected<T> contains a value (not an error).
/// Usage: AEGIS_ASSERT_SUCCESS(MyExpected);
#define AEGIS_ASSERT_SUCCESS(expr) \
  ASSERT_TRUE(static_cast<bool>(expr)) << llvm::toString((expr).takeError())

/// Check that an llvm::Expected<T> contains a value (non-fatal).
#define AEGIS_EXPECT_SUCCESS(expr) \
  EXPECT_TRUE(static_cast<bool>(expr)) << llvm::toString((expr).takeError())

/// Check that an llvm::Error is success (no error).
/// Usage: AEGIS_EXPECT_NO_ERROR(MyError);
#define AEGIS_EXPECT_NO_ERROR(err) \
  do { \
    auto _err = std::move(err); \
    EXPECT_FALSE(static_cast<bool>(_err)) << llvm::toString(std::move(_err)); \
  } while (0)

/// Check that an llvm::Error contains an error.
#define AEGIS_EXPECT_ERROR(err) \
  do { \
    llvm::Error _err = (err); \
    EXPECT_TRUE(static_cast<bool>(_err)); \
    if (_err) llvm::consumeError(std::move(_err)); \
  } while (0)

/// Fixture for kernel descriptor tests.
/// Provides sample descriptors and validation utilities.
class DescriptorFixture : public ::testing::Test {
protected:
  /// Create a minimal valid 64-byte kernel descriptor.
  /// Uses reasonable defaults for a simple kernel.
  static std::vector<uint8_t> makeDescriptor(
      uint32_t VGPRCount = 32,
      uint32_t SGPRCount = 16,
      uint32_t LDSSize = 0,
      uint32_t ScratchSize = 0,
      uint16_t KernelCodeProperties = 0,
      uint16_t KernargPreload = 0) {

    std::vector<uint8_t> Desc(DescriptorUpdater::DESCRIPTOR_SIZE, 0);

    // Write fields per LLVM's AMDHSAKernelDescriptor.h
    // Offset 0: GROUP_SEGMENT_FIXED_SIZE (LDS)
    writeLE32(&Desc[0], LDSSize);

    // Offset 4: PRIVATE_SEGMENT_FIXED_SIZE (scratch)
    writeLE32(&Desc[4], ScratchSize);

    // Offset 8: KERNARG_SIZE (default 32 bytes)
    writeLE32(&Desc[8], 32);

    // Offset 12: reserved0 (already zeros)

    // Offset 16: KERNEL_CODE_ENTRY_BYTE_OFFSET (256 = typical offset)
    writeLE64(&Desc[16], 256);

    // Offset 24-43: reserved1 (already zeros)

    // Offset 44: COMPUTE_PGM_RSRC3 (leave as zero for now)
    writeLE32(&Desc[44], 0);

    // Offset 48: COMPUTE_PGM_RSRC1 with VGPR/SGPR counts
    uint32_t PgmRsrc1 = encodeVGPRCount(VGPRCount) | encodeSGPRCount(SGPRCount);
    writeLE32(&Desc[48], PgmRsrc1);

    // Offset 52: COMPUTE_PGM_RSRC2 (minimal settings)
    writeLE32(&Desc[52], 0);

    // Offset 56: KERNEL_CODE_PROPERTIES (16-bit)
    writeLE16(&Desc[56], KernelCodeProperties);

    // Offset 58: KERNARG_PRELOAD (16-bit)
    writeLE16(&Desc[58], KernargPreload);

    // Offset 60-63: reserved3 (already zeros)

    return Desc;
  }

  /// Parse descriptor and verify success.
  /// Defaults to older architecture (granularity 4) for backwards compatibility.
  llvm::Expected<KernelDescriptor> parse(llvm::ArrayRef<uint8_t> Bytes,
                                          llvm::StringRef GPUArch = "") {
    return DescriptorUpdater::parse(Bytes, GPUArch);
  }

  /// Serialize descriptor back to bytes.
  static std::vector<uint8_t> serialize(const KernelDescriptor& KD) {
    return DescriptorUpdater::serialize(KD);
  }

  /// Verify descriptor round-trips correctly.
  void expectRoundTrip(const KernelDescriptor& KD) {
    auto Bytes = serialize(KD);
    ASSERT_EQ(Bytes.size(), DescriptorUpdater::DESCRIPTOR_SIZE);

    auto ParsedOrErr = parse(Bytes);
    ASSERT_TRUE(static_cast<bool>(ParsedOrErr))
        << "Failed to parse serialized descriptor";

    const auto& Parsed = *ParsedOrErr;
    EXPECT_EQ(Parsed.VGPRCount, KD.VGPRCount);
    EXPECT_EQ(Parsed.SGPRCount, KD.SGPRCount);
    EXPECT_EQ(Parsed.GroupSegmentFixedSize, KD.GroupSegmentFixedSize);
    EXPECT_EQ(Parsed.PrivateSegmentFixedSize, KD.PrivateSegmentFixedSize);
    EXPECT_EQ(Parsed.KernargSize, KD.KernargSize);
    EXPECT_EQ(Parsed.KernelCodeEntryByteOffset, KD.KernelCodeEntryByteOffset);
    EXPECT_EQ(Parsed.KernelCodeProperties, KD.KernelCodeProperties);
    EXPECT_EQ(Parsed.KernargPreload, KD.KernargPreload);
  }

  /// Verify VGPR count is correctly encoded and decoded.
  void expectVGPRCount(const KernelDescriptor& KD, uint32_t Expected) {
    // Round up to granularity (use descriptor's stored granularity)
    uint32_t Granularity = KD.VGPRGranularity > 0 ? KD.VGPRGranularity : 4;
    uint32_t Rounded = DescriptorUpdater::roundUpVGPR(Expected, Granularity);
    EXPECT_EQ(KD.VGPRCount, Rounded)
        << "VGPR count " << Expected << " should round to " << Rounded;
  }

  /// Verify SGPR count is correctly encoded and decoded.
  void expectSGPRCount(const KernelDescriptor& KD, uint32_t Expected) {
    // Round up to granularity
    uint32_t Rounded = DescriptorUpdater::roundUpSGPR(Expected);
    EXPECT_EQ(KD.SGPRCount, Rounded)
        << "SGPR count " << Expected << " should round to " << Rounded;
  }

private:
  /// Write little-endian uint16_t to buffer.
  static void writeLE16(uint8_t* Bytes, uint16_t Value) {
    Bytes[0] = static_cast<uint8_t>(Value);
    Bytes[1] = static_cast<uint8_t>(Value >> 8);
  }

  /// Write little-endian uint32_t to buffer.
  static void writeLE32(uint8_t* Bytes, uint32_t Value) {
    Bytes[0] = static_cast<uint8_t>(Value);
    Bytes[1] = static_cast<uint8_t>(Value >> 8);
    Bytes[2] = static_cast<uint8_t>(Value >> 16);
    Bytes[3] = static_cast<uint8_t>(Value >> 24);
  }

  /// Write little-endian uint64_t to buffer.
  static void writeLE64(uint8_t* Bytes, uint64_t Value) {
    writeLE32(Bytes, static_cast<uint32_t>(Value));
    writeLE32(Bytes + 4, static_cast<uint32_t>(Value >> 32));
  }

  /// Encode VGPR count to granulated field value.
  /// Default granularity is 4 for backwards compatibility.
  static uint32_t encodeVGPRCount(uint32_t Count, uint32_t Granularity = 4) {
    uint32_t Rounded = DescriptorUpdater::roundUpVGPR(Count, Granularity);
    uint32_t Granulated = (Rounded / Granularity) - 1;
    return Granulated & DescriptorUpdater::VGPR_COUNT_MASK;
  }

  /// Encode SGPR count to granulated field value.
  static uint32_t encodeSGPRCount(uint32_t Count) {
    uint32_t Rounded = DescriptorUpdater::roundUpSGPR(Count);
    uint32_t Granulated = (Rounded / DescriptorUpdater::SGPR_GRANULARITY) - 1;
    return (Granulated << DescriptorUpdater::SGPR_COUNT_SHIFT) &
           DescriptorUpdater::SGPR_COUNT_MASK;
  }
};

} // namespace test
} // namespace aegisbit

#endif // AEGISBIT_TEST_DESCRIPTOR_FIXTURE_H
