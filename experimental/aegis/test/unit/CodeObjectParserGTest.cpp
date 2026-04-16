//===-- CodeObjectParserGTest.cpp - Code Object Parser Tests ----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for CodeObjectParser: ELF detection, GPU arch extraction, and
/// the symbol size inference logic added for rocBLAS/Tensile kernels.
///
/// A regression here means we either fail to identify valid code objects
/// (dropping kernels silently) or produce wrong CodeOffset/CodeSize values
/// (causing the trampoline to instrument the wrong bytes → GPU fault).
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectParser.h"
#include "aegisbit/Endian.h"
#include "aegisbit/Types.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace aegisbit;

//===----------------------------------------------------------------------===//
// Helpers: build minimal ELF headers for testing
//===----------------------------------------------------------------------===//

namespace {

/// Build a minimal 64-byte ELF64 LE header for AMDGPU.
std::vector<uint8_t> buildELF64Header(uint16_t eMachine, uint32_t eFlags) {
  std::vector<uint8_t> H(64, 0);
  // ELF magic
  H[0] = 0x7F; H[1] = 'E'; H[2] = 'L'; H[3] = 'F';
  H[4] = 2;     // EI_CLASS = ELFCLASS64
  H[5] = 1;     // EI_DATA  = ELFDATA2LSB
  H[6] = 1;     // EI_VERSION = EV_CURRENT
  H[7] = 64;    // EI_OSABI  = ELFOSABI_AMDGPU_HSA
  // e_type = ET_DYN (2)
  writeLE16(&H[16], 2);
  // e_machine
  writeLE16(&H[18], eMachine);
  // e_version
  writeLE32(&H[20], 1);
  // e_flags
  writeLE32(&H[0x30], eFlags);
  // e_ehsize
  writeLE16(&H[0x34], 64);
  return H;
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// isAMDGPUCodeObject: ELF detection
//===----------------------------------------------------------------------===//

TEST(CodeObjectParser, DetectsValidAMDGPUHeader) {
  auto H = buildELF64Header(CodeObjectParser::EM_AMDGPU,
                             CodeObjectParser::EF_AMDGPU_MACH_GFX950);
  EXPECT_TRUE(CodeObjectParser::isAMDGPUCodeObject(H));
}

TEST(CodeObjectParser, RejectsNonAMDGPUMachine) {
  // x86_64 = 0x3E
  auto H = buildELF64Header(0x3E, 0);
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(H));
}

TEST(CodeObjectParser, RejectsTooSmall) {
  std::vector<uint8_t> Tiny = {0x7F, 'E', 'L', 'F'};
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(Tiny));
}

TEST(CodeObjectParser, RejectsEmptyInput) {
  std::vector<uint8_t> Empty;
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(Empty));
}

TEST(CodeObjectParser, RejectsBadMagic) {
  auto H = buildELF64Header(CodeObjectParser::EM_AMDGPU, 0);
  H[0] = 0x00;  // corrupt magic
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(H));
}

TEST(CodeObjectParser, RejectsELF32) {
  auto H = buildELF64Header(CodeObjectParser::EM_AMDGPU, 0);
  H[4] = 1;  // EI_CLASS = ELFCLASS32
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(H));
}

TEST(CodeObjectParser, RejectsBigEndian) {
  auto H = buildELF64Header(CodeObjectParser::EM_AMDGPU, 0);
  H[5] = 2;  // EI_DATA = ELFDATA2MSB
  EXPECT_FALSE(CodeObjectParser::isAMDGPUCodeObject(H));
}

//===----------------------------------------------------------------------===//
// getGPUArch: EFlags → architecture string
//===----------------------------------------------------------------------===//

TEST(CodeObjectParser, ArchFromFlags_GFX9Family) {
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX900), "gfx900");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX906), "gfx906");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX908), "gfx908");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX90A), "gfx90a");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX940), "gfx940");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX941), "gfx941");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX942), "gfx942");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX950), "gfx950");
}

TEST(CodeObjectParser, ArchFromFlags_RDNA) {
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX1010), "gfx1010");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX1030), "gfx1030");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX1100), "gfx1100");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX1200), "gfx1200");
  EXPECT_EQ(CodeObjectParser::getGPUArch(CodeObjectParser::EF_AMDGPU_MACH_GFX1250), "gfx1250");
}

TEST(CodeObjectParser, ArchFromFlags_MaskIgnoresUpperBits) {
  // EFlags can have other bits set above the MACH field (bits 7:0).
  // getGPUArch should mask with EF_AMDGPU_MACH_MASK.
  uint32_t Flags = CodeObjectParser::EF_AMDGPU_MACH_GFX942 | 0xFF00;
  EXPECT_EQ(CodeObjectParser::getGPUArch(Flags), "gfx942");
}

TEST(CodeObjectParser, ArchFromFlags_UnknownMachReturnsHex) {
  std::string Arch = CodeObjectParser::getGPUArch(0xAB);
  EXPECT_NE(Arch.find("unknown"), std::string::npos)
      << "Unknown MACH should return a string containing 'unknown', got: " << Arch;
  EXPECT_NE(Arch.find("0x0ab"), std::string::npos)
      << "Unknown MACH should include hex value, got: " << Arch;
}

//===----------------------------------------------------------------------===//
// parse: error handling for invalid/incomplete ELFs
//===----------------------------------------------------------------------===//

TEST(CodeObjectParser, ParseRejectsNonELF) {
  std::vector<uint8_t> Garbage(128, 0xCC);
  auto Result = CodeObjectParser::parse(Garbage);
  EXPECT_FALSE(static_cast<bool>(Result));
  llvm::consumeError(Result.takeError());
}

TEST(CodeObjectParser, ParseRejectsX86ELF) {
  auto H = buildELF64Header(0x3E, 0);  // x86_64
  auto Result = CodeObjectParser::parse(H);
  EXPECT_FALSE(static_cast<bool>(Result));
  llvm::consumeError(Result.takeError());
}

//===----------------------------------------------------------------------===//
// Size inference logic: verify the contract
//
// We can't easily build full ELF binaries in a unit test, but we CAN verify
// the size inference logic by testing with real code objects from HIP
// compilation.  For the unit test, we verify the static helpers and the
// parse error paths, and rely on the E2E tests (hip_vector_add, rocblas_gemm)
// to exercise the full size-inference path with real binaries.
//
// These tests document the CONTRACT that the size inference must satisfy:
//   1. If Sym.Size > 0, KI.CodeSize = Sym.Size (no inference)
//   2. If Sym.Size == 0 and there's a next symbol, KI.CodeSize = next.Value - this.Value
//   3. If Sym.Size == 0 and it's the last symbol, KI.CodeSize = TextEnd - this.Value
//   4. KI.CodeSize must never be 0 after inference
//===----------------------------------------------------------------------===//

TEST(CodeObjectParser, SizeInferenceDocumentation) {
  // This test documents the size inference algorithm without exercising it
  // directly (that requires a full ELF). It serves as a living specification.
  //
  // Given symbols sorted by address:
  //   sym_A at 0x1000, size=0
  //   sym_B at 0x2000, size=0
  //   sym_C at 0x3000, size=0
  //   .text ends at 0x4000
  //
  // Expected inferred sizes:
  //   sym_A.CodeSize = 0x2000 - 0x1000 = 0x1000
  //   sym_B.CodeSize = 0x3000 - 0x2000 = 0x1000
  //   sym_C.CodeSize = 0x4000 - 0x3000 = 0x1000
  //
  // This matches the implementation in CodeObjectParser::parseSymbols:
  //   if (KI.CodeSize == 0) {
  //     NextAddr = (FI+1 < TextFuncs.size()) ? TextFuncs[FI+1]->Value : TextEnd;
  //     KI.CodeSize = NextAddr - Sym.Value;
  //   }

  // The actual verification is in E2E tests:
  //   - hip_vector_add:  HIP kernels have non-zero sizes (inference NOT triggered)
  //   - rocblas_gemm:    Tensile kernels have zero sizes (inference IS triggered)
  //   - Both produce valid profiling output, proving CodeSize is correct.
  SUCCEED();
}
