//===-- PatchedELFLayoutGTest.cpp - Patched ELF layout invariants -*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Verifies structural invariants of the patched ELF produced by KernelPatcher:
/// - Trampoline islands never overlap the kernel code region
/// - The kernel descriptor is consistent with the patched code
/// - The .text section is large enough to hold all components
///
/// Uses the mega_gather fixture (110 KB kernel with 2080 memory sites) which
/// forces the SwapPC shared-body code path.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/KernelPatcher.h"
#include "aegisbit/Types.h"
#include <cstring>
#include <fstream>
#include <gtest/gtest.h>

using namespace aegisbit;
using namespace llvm;

#ifndef MEGA_GATHER_FIXTURE_PATH
#error "MEGA_GATHER_FIXTURE_PATH must be defined at compile time"
#endif

namespace {

std::vector<uint8_t> loadFixture(const char *Path) {
  std::ifstream F(Path, std::ios::binary | std::ios::ate);
  if (!F)
    return {};
  auto Size = F.tellg();
  F.seekg(0);
  std::vector<uint8_t> Bytes(Size);
  F.read(reinterpret_cast<char *>(Bytes.data()), Size);
  return Bytes;
}

class PatchedELFLayoutTest : public ::testing::Test {
protected:
  void SetUp() override {
    FixtureBytes = loadFixture(MEGA_GATHER_FIXTURE_PATH);
    ASSERT_FALSE(FixtureBytes.empty())
        << "Cannot load fixture: " << MEGA_GATHER_FIXTURE_PATH;
    ASSERT_GE(FixtureBytes.size(), 4u);
    ASSERT_EQ(FixtureBytes[0], 0x7f);
    ASSERT_EQ(FixtureBytes[1], 'E');
  }

  std::vector<uint8_t> FixtureBytes;
};

TEST_F(PatchedELFLayoutTest, IslandDoesNotOverlapKernelCode) {
  // Strategy: compare original and patched kernel code byte-by-byte.
  // Every difference must be at a known 8-byte patch site (s_movk_i32 +
  // s_swappc_b64 replacing the original memory instruction). If any byte
  // outside a patch site differs, island code has leaked into the kernel.

  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);
  ASSERT_GT(KI->CodeSize, 0u);

  auto OrigText = OrigHandler->getTextSection();

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 1;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 1;
  Symbol.CodeObjectId = 1;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);
  ASSERT_FALSE(Result->PatchedELF.empty());

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto PatchedText = PatchedHandler->getTextSection();

  // The patched .text has a prologue prepended, shifting the kernel forward.
  // The symbol size stays the same, but all code is shifted by the prologue
  // size. We detect the shift by finding where the original code appears in
  // the patched text (the first non-patch bytes should match).
  uint64_t OrigStart = KI->CodeOffset;
  uint64_t CodeSize = KI->CodeSize;
  ASSERT_EQ(PKI->CodeSize, CodeSize) << "Kernel code size changed after patching";

  // The prologue prepend shifts original text content forward.
  // PatchedText = [zeros/pad | prologue | original_text_content_with_patches | island]
  // The original text was memcpy'd at offset PadSize in the new text buffer.
  // PadSize = AlignedPrologue for SwapPC. Detect it:
  uint64_t PrologueShift = PatchedText.size() - OrigText.size() -
                            (PatchedText.size() - PKI->CodeOffset - CodeSize);
  // Simpler: find the shift by scanning for first matching 32-byte block.
  uint64_t Shift = 0;
  for (uint64_t Try = 0; Try <= 256; Try += 4) {
    uint64_t PO = PKI->CodeOffset + Try;
    if (PO + 32 > PatchedText.size())
      break;
    if (std::memcmp(OrigText.data() + OrigStart,
                    PatchedText.data() + PO, 32) == 0) {
      Shift = Try;
      break;
    }
  }

  uint64_t PatchedStart = PKI->CodeOffset + Shift;

  std::cout << "Original code at .text offset: " << OrigStart << std::endl;
  std::cout << "Patched code at .text offset:  " << PatchedStart
            << " (shift=" << Shift << ")" << std::endl;
  std::cout << "Kernel code size:              " << CodeSize << std::endl;

  ASSERT_LE(OrigStart + CodeSize, OrigText.size());
  ASSERT_LE(PatchedStart + CodeSize, PatchedText.size());

  // Build a set of patch site offsets (relative to kernel code start).
  // Each patch site is 8 bytes (s_movk_i32 + s_swappc_b64).
  constexpr uint32_t SOPK_MOVK_MASK = 0xF8000000;
  constexpr uint32_t SOPK_MOVK_VAL  = 0xB0000000;
  constexpr uint32_t SDST_MASK      = 0x007F0000;
  uint32_t ScratchSGPRBase = 20;
  uint32_t ScratchSGPREnd = 28;
  constexpr uint32_t PatchSlotSize = 8;

  std::vector<bool> IsPatchByte(CodeSize, false);
  uint32_t PatchCount = 0;
  for (uint64_t Off = 0; Off + 4 <= CodeSize; Off += 4) {
    uint32_t Word;
    if (PatchedStart + Off + 4 > PatchedText.size())
      break;
    std::memcpy(&Word, PatchedText.data() + PatchedStart + Off, 4);
    if ((Word & SOPK_MOVK_MASK) == SOPK_MOVK_VAL) {
      uint32_t Sdst = (Word & SDST_MASK) >> 16;
      if (Sdst >= ScratchSGPRBase && Sdst < ScratchSGPREnd) {
        for (uint64_t B = Off; B < Off + PatchSlotSize && B < CodeSize; ++B)
          IsPatchByte[B] = true;
        ++PatchCount;
      }
    }
  }

  std::cout << "Patch sites found:             " << PatchCount << std::endl;

  // Compare every non-patch byte between original and patched.
  uint32_t CorruptedBytes = 0;
  uint64_t FirstCorruptedOff = 0;
  for (uint64_t Off = 0; Off < CodeSize; ++Off) {
    if (IsPatchByte[Off])
      continue;
    uint8_t OrigByte = OrigText[OrigStart + Off];
    uint8_t PatchedByte = PatchedText[PatchedStart + Off];
    if (OrigByte != PatchedByte) {
      if (CorruptedBytes == 0)
        FirstCorruptedOff = Off;
      ++CorruptedBytes;
    }
  }

  std::cout << "Non-patch bytes that differ:    " << CorruptedBytes << std::endl;
  if (CorruptedBytes > 0) {
    std::cout << "First corrupted at code offset: 0x" << std::hex
              << FirstCorruptedOff << std::dec << std::endl;
  }

  EXPECT_EQ(CorruptedBytes, 0u)
      << CorruptedBytes << " bytes in the kernel code region were modified "
         "outside of known patch sites — island data may have overwritten "
         "kernel code";

  // Also verify the island area exists and has content.
  uint64_t KernelEndInPatched = PatchedStart + CodeSize;
  if (KernelEndInPatched < PatchedText.size()) {
    uint64_t PostKernelSpace = PatchedText.size() - KernelEndInPatched;
    uint64_t MinExpectedIsland = Result->NumMemorySites * 8;
    std::cout << "Post-kernel space:             " << PostKernelSpace << std::endl;
    EXPECT_GE(PostKernelSpace, MinExpectedIsland)
        << "Post-kernel space too small for island";
  }
}

TEST_F(PatchedELFLayoutTest, ProloguePointsToIslandStart) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto OrigText = OrigHandler->getTextSection();

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 5;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 5;
  Symbol.CodeObjectId = 5;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto PatchedText = PatchedHandler->getTextSection();
  uint64_t CodeSize = KI->CodeSize;

  // Find the prologue: it's the 16 bytes right before the kernel code body.
  // The kernel code body starts where the original code started + shift.
  uint64_t Shift = 0;
  for (uint64_t Try = 0; Try <= 256; Try += 4) {
    uint64_t PO = PKI->CodeOffset + Try;
    if (PO + 32 > PatchedText.size())
      break;
    if (std::memcmp(OrigText.data() + KI->CodeOffset,
                    PatchedText.data() + PO, 32) == 0) {
      Shift = Try;
      break;
    }
  }

  uint64_t PatchedCodeStart = PKI->CodeOffset + Shift;
  ASSERT_GE(PatchedCodeStart, Shift);
  uint64_t PrologueStart = PatchedCodeStart - Shift;

  std::cout << "Prologue at .text offset:  " << PrologueStart << std::endl;
  std::cout << "Kernel code at .text off:  " << PatchedCodeStart << std::endl;
  std::cout << "Shift (prologue size):     " << Shift << std::endl;

  // Dump prologue bytes for debugging
  std::cout << "Prologue bytes: ";
  for (uint64_t I = PrologueStart; I < PatchedCodeStart && I < PatchedText.size(); ++I) {
    char Buf[4];
    snprintf(Buf, sizeof(Buf), "%02x ", PatchedText[I]);
    std::cout << Buf;
  }
  std::cout << std::endl;

  // Dump as 32-bit words
  for (uint64_t Off = PrologueStart; Off < PatchedCodeStart && Off + 4 <= PatchedText.size(); Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    std::cout << "  .text[" << Off << "] = 0x" << std::hex << Word << std::dec << std::endl;
  }

  // The prologue for the SwapTargetSGPR is 3 instructions (12 bytes):
  //   s_getpc_b64 s[swap:swap+1]            (4 bytes)
  //   s_add_u32   s[swap], s[swap], <imm>   (4 or 8 bytes)
  //   s_addc_u32  s[swap+1], s[swap+1], <imm> (4 or 8 bytes)
  //
  // The prologue is placed at .text[PrologueStart..PrologueStart+Shift].
  ASSERT_GE(Shift, 4u) << "No room for prologue";

  // Parse instructions from the prologue region.
  // s_getpc_b64 s[N:N+1] encoding: SOP1, opcode=0x47, bits:
  //   [31:23]=101111101, [22:16]=SDST, [15:8]=opcode=0x47, [7:0]=SSRC0
  // For s_getpc_b64: SSRC0 is ignored (set to 0 usually).
  // SwapTargetSGPR = s26, so SDST = 26 (but it's a pair, so SDST = 26/2 = 13 in pair encoding)
  // Actually for SOP1: SDST is 7-bit raw SGPR index for the lo half.

  // Rather than decode instruction-by-instruction, use the simpler approach:
  // the prologue computes SwapTargetSGPR = getpc_result + immediate_offset.
  // getpc_result = address of the instruction after s_getpc_b64.
  // So: SwapTargetSGPR = (prologue_getpc_addr + 4) + immediate_offset
  //
  // The island starts after the kernel code + shift for alignment.
  // Island start in .text = PatchedCodeStart + CodeSize (approximately).
  // Let's find the island start: it's the first non-kernel, non-zero content
  // after the kernel code.

  uint64_t IslandStart = PatchedCodeStart + CodeSize;
  // Round up to 256-byte alignment (standard island alignment)
  uint64_t AlignedIsland = (IslandStart + 255) & ~255ULL;

  // The island may not be 256-aligned in all cases; scan for non-zero content.
  uint64_t ActualIslandStart = IslandStart;
  // Skip zero padding between kernel end and island start.
  while (ActualIslandStart < PatchedText.size() &&
         PatchedText[ActualIslandStart] == 0)
    ++ActualIslandStart;

  // The virtual address of the island start:
  // .text section VA + ActualIslandStart
  // The prologue's getpc returns the VA of the instruction after s_getpc_b64.
  // getpc is at some offset in the prologue; the prologue is at VA = text_va + PrologueStart.

  // We need the .text section's virtual address to compute absolute addresses.
  // From the patched ELF: .text vaddr. Parse it.
  // Actually we know from the section headers that .text vaddr = 0x1b00 (from earlier).
  // But let's not hardcode; read it.
  //
  // Simpler approach: since all addresses are relative to .text, we can work
  // entirely in .text-relative offsets.
  //
  // The prologue does:
  //   s_getpc_b64 s[26:27]
  //     -> s26 = low32(getpc_addr + 4 + text_vaddr)
  //     -> s27 = high32(getpc_addr + 4 + text_vaddr)
  //   s_add_u32 s26, s26, <offset_lo>
  //   s_addc_u32 s27, s27, <offset_hi>
  //     -> s[26:27] = text_vaddr + getpc_offset_in_text + 4 + offset
  //
  // For the result to point to the island start:
  //   text_vaddr + getpc_offset_in_text + 4 + offset = text_vaddr + IslandStart
  //   => offset = IslandStart - getpc_offset_in_text - 4
  //
  // So we can verify: extract offset from the immediate fields and check
  //   getpc_offset_in_text + 4 + offset == IslandStart (in .text coordinates)

  // Find s_getpc_b64 in the prologue: scan for opcode 0x47 in SOP1 format.
  // SOP1: bits[31:23]=10111110_1, bits[22:16]=SDST, bits[15:8]=opcode, bits[7:0]=SSRC0
  int64_t GetpcTextOffset = -1;
  for (uint64_t Off = PrologueStart; Off < PatchedCodeStart && Off + 4 <= PatchedText.size(); Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    uint32_t Top9 = (Word >> 23) & 0x1FF;
    uint32_t Opcode = (Word >> 8) & 0xFF;
    if (Top9 == 0x17D && (Opcode == 0x1C || Opcode == 0x47)) {  // SOP1 + S_GETPC_B64 (0x1C on GFX9, 0x47 on other gens)
      GetpcTextOffset = static_cast<int64_t>(Off);
      std::cout << "s_getpc_b64 at .text offset:  " << GetpcTextOffset << std::endl;
      break;
    }
  }
  ASSERT_GE(GetpcTextOffset, 0) << "s_getpc_b64 not found in prologue";

  // The next instruction(s) are s_add_u32 and s_addc_u32.
  // SOP2 format: bits[31:30]=10, bits[29:23]=opcode
  // s_add_u32 opcode = 0x00, s_addc_u32 opcode = 0x04
  // We need the immediate field. SOP2 with a literal constant is 8 bytes.
  // SOP2 with inline constant is 4 bytes.
  // bits[7:0] = SSRC0, bits[15:8] = SSRC1
  // If SSRC0 or SSRC1 = 0xFF, a 32-bit literal follows.

  uint64_t NextOff = static_cast<uint64_t>(GetpcTextOffset) + 4;
  int32_t OffsetLo = 0;
  int32_t OffsetHi = 0;

  // AMDGPU inline constant decoding (SSRC field).
  // 0..103:     SGPR registers (not a constant)
  // 128:        integer 0
  // 129..192:   integer 1..64
  // 193..208:   integer -1..-16
  // 240:        0.5f, 241: -0.5f, etc. (float constants, not relevant here)
  // 255 (0xFF): literal constant in next dword
  auto decodeInlineOrLiteral = [&PatchedText](uint32_t SSRC, uint64_t &Offset,
                                               int32_t &Result) -> bool {
    if (SSRC == 0xFF) {
      if (Offset + 8 > PatchedText.size())
        return false;
      std::memcpy(&Result, PatchedText.data() + Offset + 4, 4);
      Offset += 8;
      return true;
    }
    if (SSRC == 128) { Result = 0; }
    else if (SSRC >= 129 && SSRC <= 192) { Result = static_cast<int32_t>(SSRC - 128); }
    else if (SSRC >= 193 && SSRC <= 208) { Result = -static_cast<int32_t>(SSRC - 192); }
    else { Result = 0; }
    Offset += 4;
    return true;
  };

  // Parse s_add_u32 (the immediate that encodes the lo offset)
  if (NextOff + 4 <= PatchedText.size()) {
    uint32_t AddWord;
    std::memcpy(&AddWord, PatchedText.data() + NextOff, 4);
    uint32_t SSRC1 = (AddWord >> 8) & 0xFF;
    decodeInlineOrLiteral(SSRC1, NextOff, OffsetLo);
  }

  // Parse s_addc_u32 (the hi offset, usually 0 or -1)
  if (NextOff + 4 <= PatchedText.size()) {
    uint32_t AdcWord;
    std::memcpy(&AdcWord, PatchedText.data() + NextOff, 4);
    uint32_t SSRC1 = (AdcWord >> 8) & 0xFF;
    decodeInlineOrLiteral(SSRC1, NextOff, OffsetHi);
  }

  int64_t TotalOffset = (static_cast<int64_t>(OffsetHi) << 32) |
                         (static_cast<uint32_t>(OffsetLo));
  uint64_t ComputedTarget = static_cast<uint64_t>(GetpcTextOffset + 4) +
                            static_cast<uint64_t>(TotalOffset);

  std::cout << "Prologue offset_lo:        " << OffsetLo << std::endl;
  std::cout << "Prologue offset_hi:        " << OffsetHi << std::endl;
  std::cout << "getpc .text offset + 4:    " << (GetpcTextOffset + 4) << std::endl;
  std::cout << "Computed target (.text):   " << ComputedTarget << std::endl;
  std::cout << "Actual island start:       " << ActualIslandStart << std::endl;

  // NOTE: the computed target is in virtual address space (text_vaddr + offset),
  // but since getpc returns VA and we're adding to it, ComputedTarget is a VA.
  // We computed it as .text offset, so it's actually:
  //   target_VA = text_vaddr + getpc_text_offset + 4 + TotalOffset
  // And the island's VA = text_vaddr + ActualIslandStart
  // So target_text_offset = getpc_text_offset + 4 + TotalOffset should equal
  // ActualIslandStart.

  EXPECT_EQ(ComputedTarget, ActualIslandStart)
      << "Prologue SwapTargetSGPR points to .text offset " << ComputedTarget
      << " but island starts at " << ActualIslandStart
      << " (delta=" << (static_cast<int64_t>(ComputedTarget) -
                        static_cast<int64_t>(ActualIslandStart)) << ")";
}

TEST_F(PatchedELFLayoutTest, DescriptorFieldsAreConsistent) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 2;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 2;
  Symbol.CodeObjectId = 2;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  const auto &Orig = KI->Descriptor;
  const auto &Patched = PKI->Descriptor;

  std::cout << "Original:  VGPRs=" << Orig.VGPRCount
            << " SGPRs=" << Orig.SGPRCount
            << " AccumOffset=" << Orig.AccumOffset
            << " PrivateSeg=" << Orig.PrivateSegmentFixedSize << std::endl;
  std::cout << "Patched:   VGPRs=" << Patched.VGPRCount
            << " SGPRs=" << Patched.SGPRCount
            << " AccumOffset=" << Patched.AccumOffset
            << " PrivateSeg=" << Patched.PrivateSegmentFixedSize << std::endl;

  // VGPRs should not decrease.
  EXPECT_GE(Patched.VGPRCount, Orig.VGPRCount)
      << "Patched kernel has fewer VGPRs than original";

  // SGPRs should not decrease.
  EXPECT_GE(Patched.SGPRCount, Orig.SGPRCount)
      << "Patched kernel has fewer SGPRs than original";

  // Private segment (scratch) should not decrease.
  EXPECT_GE(Patched.PrivateSegmentFixedSize, Orig.PrivateSegmentFixedSize)
      << "Patched kernel has less scratch than original";

  // AccumOffset should not decrease — shrinking it would cause the hardware
  // to misinterpret regular VGPRs as AccVGPRs.
  EXPECT_GE(Patched.AccumOffset, Orig.AccumOffset)
      << "Patched AccumOffset is smaller than original — AccVGPR boundary "
         "corruption";
}

TEST_F(PatchedELFLayoutTest, ScratchSpillSizeIsCorrect) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 3;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 3;
  Symbol.CodeObjectId = 3;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  uint32_t OrigScratch = KI->Descriptor.PrivateSegmentFixedSize;
  uint32_t PatchedScratch = PKI->Descriptor.PrivateSegmentFixedSize;
  uint32_t AdditionalScratch = Result->AdditionalScratchBytes;

  std::cout << "Original scratch:     " << OrigScratch << " bytes" << std::endl;
  std::cout << "Additional scratch:   " << AdditionalScratch << " bytes" << std::endl;
  std::cout << "Patched scratch:      " << PatchedScratch << " bytes" << std::endl;

  // Patched scratch = original + additional (no more, no less).
  EXPECT_EQ(PatchedScratch, OrigScratch + AdditionalScratch)
      << "Scratch size mismatch: expected " << (OrigScratch + AdditionalScratch)
      << " but got " << PatchedScratch;

  // For scratch spill mode (3 VGPRs), additional must be exactly 12 bytes
  // (3 dwords, one per spilled VGPR, per work-item).
  if (AdditionalScratch > 0) {
    EXPECT_EQ(AdditionalScratch, 12u)
        << "Scratch spill should be exactly 12 bytes (3 VGPRs × 4 bytes)";
  }

  // Spill offset must not exceed patched scratch size.
  // Spill occupies [OrigScratch, OrigScratch + AdditionalScratch).
  EXPECT_LE(OrigScratch + AdditionalScratch, PatchedScratch)
      << "Scratch spill region extends beyond allocated scratch";
}

TEST_F(PatchedELFLayoutTest, DispatchAndReturnTableStructure) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 4;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 4;
  Symbol.CodeObjectId = 4;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto Text = PatchedHandler->getTextSection();
  uint64_t KernelStart = PKI->CodeOffset;
  uint64_t KernelEnd = PKI->CodeOffset + PKI->CodeSize;

  // Count s_movk_i32 patch sites in and near the kernel code region.
  // SOPK encoding: bits[31:28]=1011, bits[27:23]=00000 (s_movk_i32),
  //                bits[22:16]=SDST, bits[15:0]=SIMM16
  // Our patches target a specific ScratchSGPR. For the mega_gather kernel
  // with SGPRCount=24 and SwapPC mode: ScratchBase = 24-4 = 20,
  // ScratchSGPR = s22, so top 16 bits = 0xB016.
  // We match any s_movk_i32 with SDST in the scratch SGPR range [s20..s27].
  // We extend the search a few bytes past the kernel symbol boundary because
  // the last patch slot may spill slightly past it.
  constexpr uint32_t SOPK_MOVK_MASK = 0xF8000000;  // bits[31:27]
  constexpr uint32_t SOPK_MOVK_VAL  = 0xB0000000;  // 10110 << 27
  constexpr uint32_t SDST_MASK      = 0x007F0000;   // bits[22:16]
  uint32_t ScratchSGPRBase = 20;
  uint32_t ScratchSGPREnd = 28;
  uint64_t SearchEnd = std::min(KernelEnd + 16, static_cast<uint64_t>(Text.size()));

  uint32_t PatchSiteCount = 0;
  for (uint64_t Off = KernelStart; Off + 4 <= SearchEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, Text.data() + Off, 4);
    if ((Word & SOPK_MOVK_MASK) == SOPK_MOVK_VAL) {
      uint32_t Sdst = (Word & SDST_MASK) >> 16;
      if (Sdst >= ScratchSGPRBase && Sdst < ScratchSGPREnd)
        ++PatchSiteCount;
    }
  }

  std::cout << "Patch sites (s_movk_i32): " << PatchSiteCount << std::endl;
  std::cout << "Expected sites:           " << Result->NumMemorySites
            << std::endl;

  // Every instrumented site should have an s_movk_i32 in the kernel body.
  EXPECT_EQ(PatchSiteCount, Result->NumMemorySites)
      << "Mismatch between s_movk_i32 count and reported instrumented sites";

  // The island starts after the kernel code. It should contain:
  //   Preamble (variable size, but typically 16-20 bytes)
  //   Dispatch table: NumSites × 12 bytes
  //   Shared body VMEM (variable)
  //   Shared body LDS (variable, may be 0)
  //   Return table: NumSites × 24 bytes
  //
  // Verify the island is large enough for dispatch + return tables.
  uint64_t IslandSize = Text.size() - KernelEnd;
  uint64_t MinDispatchTable = Result->NumMemorySites * 12;
  uint64_t MinReturnTable = Result->NumMemorySites * 24;
  uint64_t MinIslandSize = MinDispatchTable + MinReturnTable;

  std::cout << "Island size:              " << IslandSize << std::endl;
  std::cout << "Min dispatch table:       " << MinDispatchTable << std::endl;
  std::cout << "Min return table:         " << MinReturnTable << std::endl;
  std::cout << "Min island (tables only): " << MinIslandSize << std::endl;

  EXPECT_GE(IslandSize, MinIslandSize)
      << "Island too small to hold dispatch + return tables for "
      << Result->NumMemorySites << " sites";

  // Verify s_movk_i32 site IDs are sequential [0, NumSites).
  // The immediate field (bits [15:0]) encodes the site index.
  std::vector<uint16_t> SiteIDs;
  for (uint64_t Off = KernelStart; Off + 4 <= SearchEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, Text.data() + Off, 4);
    if ((Word & SOPK_MOVK_MASK) == SOPK_MOVK_VAL) {
      uint32_t Sdst = (Word & SDST_MASK) >> 16;
      if (Sdst >= ScratchSGPRBase && Sdst < ScratchSGPREnd) {
        uint16_t SiteID = Word & 0xFFFF;
        SiteIDs.push_back(SiteID);
      }
    }
  }

  // Site IDs should cover [0, NumSites) — though not necessarily in order
  // if sites were sorted by offset.
  std::sort(SiteIDs.begin(), SiteIDs.end());
  SiteIDs.erase(std::unique(SiteIDs.begin(), SiteIDs.end()), SiteIDs.end());

  EXPECT_EQ(SiteIDs.size(), Result->NumMemorySites)
      << "Not all site IDs are unique";

  if (!SiteIDs.empty()) {
    EXPECT_EQ(SiteIDs.front(), 0u)
        << "Site IDs don't start at 0";
    EXPECT_EQ(SiteIDs.back(), Result->NumMemorySites - 1)
        << "Site IDs don't end at NumSites-1";
  }
}

TEST_F(PatchedELFLayoutTest, PreambleDispatchTableOffset) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto OrigText = OrigHandler->getTextSection();

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 6;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 6;
  Symbol.CodeObjectId = 6;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto PatchedText = PatchedHandler->getTextSection();

  // Find the island start (same logic as ProloguePointsToIslandStart)
  uint64_t Shift = 0;
  for (uint64_t Try = 0; Try <= 256; Try += 4) {
    uint64_t PO = PKI->CodeOffset + Try;
    if (PO + 32 > PatchedText.size())
      break;
    if (std::memcmp(OrigText.data() + KI->CodeOffset,
                    PatchedText.data() + PO, 32) == 0) {
      Shift = Try;
      break;
    }
  }
  uint64_t PatchedCodeStart = PKI->CodeOffset + Shift;
  uint64_t IslandStart = PatchedCodeStart + KI->CodeSize;
  while (IslandStart < PatchedText.size() && PatchedText[IslandStart] == 0)
    ++IslandStart;

  ASSERT_LT(IslandStart, PatchedText.size()) << "Island not found";
  std::cout << "Island starts at .text offset: " << IslandStart << std::endl;

  // The preamble is at the start of the island. It consists of:
  //   s_lshl_b32 ExecSaveSGPRLo, ScratchSGPR, 3       (4 bytes)
  //   s_lshl_b32 ScratchSGPR, ScratchSGPR, 2           (4 bytes)
  //   s_add_u32  ExecSaveSGPRHi, ScratchSGPR, ExecSaveSGPRLo  (4 bytes)
  //   s_getpc_b64 s[ScratchSGPR:ScratchSGPR+1]         (4 bytes)
  //   s_add_u32  ScratchSGPR, ScratchSGPR, <offset_lo> (4 or 8 bytes)
  //   s_addc_u32 ExecSaveSGPRLo, ExecSaveSGPRLo, <offset_hi> (4 bytes)
  //   s_add_u32  ScratchSGPR, ScratchSGPR, ExecSaveSGPRHi (4 bytes)
  //   s_addc_u32 ExecSaveSGPRLo, ExecSaveSGPRLo, 0    (4 bytes)
  //   s_setpc_b64 s[ScratchSGPR:ScratchSGPR+1]         (4 bytes)
  //
  // The s_getpc_b64 is at offset 12 within the preamble. The dispatch
  // table starts immediately after the preamble.

  // Find s_getpc_b64 in the preamble (scan first 64 bytes of island)
  int64_t GetpcOff = -1;
  uint64_t ScanEnd = std::min(IslandStart + 64, (uint64_t)PatchedText.size());
  for (uint64_t Off = IslandStart; Off + 4 <= ScanEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    uint32_t Top9 = (Word >> 23) & 0x1FF;
    uint32_t Opcode = (Word >> 8) & 0xFF;
    if (Top9 == 0x17D && (Opcode == 0x1C || Opcode == 0x47)) {
      GetpcOff = static_cast<int64_t>(Off);
      break;
    }
  }
  ASSERT_GE(GetpcOff, 0) << "s_getpc_b64 not found in preamble";
  std::cout << "Preamble s_getpc_b64 at .text offset: " << GetpcOff << std::endl;
  std::cout << "Offset within preamble: " << (GetpcOff - (int64_t)IslandStart) << std::endl;

  // Parse the s_add_u32 after s_getpc_b64 to get GetpcToDispatchTable offset
  uint64_t NextOff = static_cast<uint64_t>(GetpcOff) + 4;
  int32_t DispatchOffsetLo = 0;
  int32_t DispatchOffsetHi = 0;

  auto decodeInlineOrLiteral = [&PatchedText](uint32_t SSRC, uint64_t &Offset,
                                               int32_t &Result) -> bool {
    if (SSRC == 0xFF) {
      if (Offset + 8 > PatchedText.size())
        return false;
      std::memcpy(&Result, PatchedText.data() + Offset + 4, 4);
      Offset += 8;
      return true;
    }
    if (SSRC == 128) { Result = 0; }
    else if (SSRC >= 129 && SSRC <= 192) { Result = static_cast<int32_t>(SSRC - 128); }
    else if (SSRC >= 193 && SSRC <= 208) { Result = -static_cast<int32_t>(SSRC - 192); }
    else { Result = 0; }
    Offset += 4;
    return true;
  };

  if (NextOff + 4 <= PatchedText.size()) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + NextOff, 4);
    uint32_t SSRC1 = (Word >> 8) & 0xFF;
    decodeInlineOrLiteral(SSRC1, NextOff, DispatchOffsetLo);
  }

  if (NextOff + 4 <= PatchedText.size()) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + NextOff, 4);
    uint32_t SSRC1 = (Word >> 8) & 0xFF;
    decodeInlineOrLiteral(SSRC1, NextOff, DispatchOffsetHi);
  }

  // The preamble computes: target = getpc_result + offset + site_index * 12
  // For site 0: target = getpc_result + offset
  // This should point to the first dispatch entry.
  //
  // But the preamble also has s_add_u32/s_addc_u32 that adds site_index * 12,
  // so the base target (without site offset) should point to dispatch entry 0.
  //
  // getpc_result = VA of (getpc instruction + 4) = text_va + getpc_text_off + 4
  // target = text_va + getpc_text_off + 4 + offset (for site 0)
  // In .text-relative terms: getpc_text_off + 4 + offset = dispatch_table_start

  int64_t TotalOffset = (static_cast<int64_t>(DispatchOffsetHi) << 32) |
                         (static_cast<uint32_t>(DispatchOffsetLo));
  uint64_t ComputedDispatchStart = static_cast<uint64_t>(GetpcOff + 4) +
                                    static_cast<uint64_t>(TotalOffset);

  std::cout << "Preamble offset to dispatch table: " << TotalOffset << std::endl;
  std::cout << "Computed dispatch table start (.text): " << ComputedDispatchStart << std::endl;

  // The dispatch table should start right after the preamble.
  // We know the preamble is at IslandStart, so dispatch table is at
  // IslandStart + PreambleSize. Let's find PreambleSize by looking at where
  // the first s_mov_b32 (dispatch entry) appears.
  //
  // Dispatch entries use s_mov_b32 with literal (SOP1 format, 8 bytes) + s_branch (4 bytes).
  // s_mov_b32 in SOP1: bits[31:23]=101111101, opcode=0x03 (S_MOV_B32)
  // Actually s_mov_b32 is SOP1 opcode 0x03 on GFX9.
  // Let's find the first 12-byte dispatch entry pattern.

  // Actually, a simpler approach: the preamble's s_setpc_b64 is the last
  // instruction. After that comes the dispatch table. s_setpc_b64 is SOP1
  // opcode 0x1D on GFX9.

  int64_t SetPCOff = -1;
  for (uint64_t Off = IslandStart; Off + 4 <= ScanEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    uint32_t Top9 = (Word >> 23) & 0x1FF;
    uint32_t Opcode = (Word >> 8) & 0xFF;
    if (Top9 == 0x17D && (Opcode == 0x1D || Opcode == 0x48)) {
      SetPCOff = static_cast<int64_t>(Off);
    }
  }
  ASSERT_GE(SetPCOff, 0) << "s_setpc_b64 not found in preamble";

  uint64_t ActualDispatchStart = static_cast<uint64_t>(SetPCOff) + 4;
  uint64_t PreambleSize = ActualDispatchStart - IslandStart;
  std::cout << "Preamble size: " << PreambleSize << " bytes" << std::endl;
  std::cout << "Actual dispatch table start (.text): " << ActualDispatchStart << std::endl;

  EXPECT_EQ(ComputedDispatchStart, ActualDispatchStart)
      << "Preamble getpc+offset targets .text offset " << ComputedDispatchStart
      << " but dispatch table starts at " << ActualDispatchStart
      << " (delta=" << (static_cast<int64_t>(ComputedDispatchStart) -
                        static_cast<int64_t>(ActualDispatchStart)) << ")";
}

TEST_F(PatchedELFLayoutTest, DispatchEntryBranchTargets) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto OrigText = OrigHandler->getTextSection();

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 8;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 8;
  Symbol.CodeObjectId = 8;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto PatchedText = PatchedHandler->getTextSection();

  // Find island start
  uint64_t Shift = 0;
  for (uint64_t Try = 0; Try <= 256; Try += 4) {
    uint64_t PO = PKI->CodeOffset + Try;
    if (PO + 32 > PatchedText.size())
      break;
    if (std::memcmp(OrigText.data() + KI->CodeOffset,
                    PatchedText.data() + PO, 32) == 0) {
      Shift = Try;
      break;
    }
  }
  uint64_t PatchedCodeStart = PKI->CodeOffset + Shift;
  uint64_t IslandStart = PatchedCodeStart + KI->CodeSize;
  while (IslandStart < PatchedText.size() && PatchedText[IslandStart] == 0)
    ++IslandStart;

  // Find preamble s_setpc_b64 to get dispatch table start
  uint64_t ScanEnd = std::min(IslandStart + 64, (uint64_t)PatchedText.size());
  int64_t SetPCOff = -1;
  for (uint64_t Off = IslandStart; Off + 4 <= ScanEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    uint32_t Top9 = (Word >> 23) & 0x1FF;
    uint32_t Opcode = (Word >> 8) & 0xFF;
    if (Top9 == 0x17D && (Opcode == 0x1D || Opcode == 0x48))
      SetPCOff = static_cast<int64_t>(Off);
  }
  ASSERT_GE(SetPCOff, 0);
  uint64_t DispatchTableStart = static_cast<uint64_t>(SetPCOff) + 4;

  // Each dispatch entry is 12 bytes: s_mov_b32 (8 bytes) + s_branch (4 bytes).
  // s_branch encoding (SOPP): bits[31:16] = 0xBF82, bits[15:0] = signed SIMM16
  // Target = PC + 4 + SIMM16 * 4  (PC = address of the s_branch itself)
  //
  // All VMEM dispatch entries should branch to the VMEM shared body.
  // All LDS dispatch entries should branch to the LDS shared body.
  // For mega_gather (all global_load), all should target the VMEM body.

  uint32_t NumSites = Result->NumMemorySites;
  uint32_t BadBranches = 0;
  int64_t FirstTarget = -1;

  for (uint32_t I = 0; I < NumSites && I < 2080; ++I) {
    uint64_t EntryOff = DispatchTableStart + static_cast<uint64_t>(I) * 12;
    uint64_t BranchOff = EntryOff + 8;
    if (BranchOff + 4 > PatchedText.size())
      break;

    uint32_t BrWord;
    std::memcpy(&BrWord, PatchedText.data() + BranchOff, 4);

    uint32_t BrTop = (BrWord >> 16) & 0xFFFF;
    // s_branch is SOPP opcode 2: top 16 bits = 0xBF82
    if (BrTop != 0xBF82) {
      if (BadBranches < 5) {
        std::cout << "  Entry " << I << " at .text " << BranchOff
                  << ": not s_branch (word=0x" << std::hex << BrWord
                  << std::dec << ")" << std::endl;
      }
      ++BadBranches;
      continue;
    }

    int16_t Simm16 = static_cast<int16_t>(BrWord & 0xFFFF);
    int64_t Target = static_cast<int64_t>(BranchOff) + 4 +
                     static_cast<int64_t>(Simm16) * 4;

    if (FirstTarget < 0)
      FirstTarget = Target;

    // All VMEM entries should branch to the same shared body
    if (Target != FirstTarget) {
      if (BadBranches < 5) {
        std::cout << "  Entry " << I << ": branch targets .text " << Target
                  << " but entry 0 targets " << FirstTarget << std::endl;
      }
      ++BadBranches;
    }
  }

  std::cout << "Dispatch table at .text: " << DispatchTableStart << std::endl;
  std::cout << "First dispatch entry branch target: " << FirstTarget << std::endl;
  std::cout << "Bad branches: " << BadBranches << " / " << NumSites << std::endl;

  // The target should be within the island, after the dispatch table.
  uint64_t DispatchTableEnd = DispatchTableStart + static_cast<uint64_t>(NumSites) * 12;
  EXPECT_GE(FirstTarget, static_cast<int64_t>(DispatchTableEnd))
      << "Shared body target is inside the dispatch table";
  EXPECT_LT(FirstTarget, static_cast<int64_t>(PatchedText.size()))
      << "Shared body target is beyond .text";

  EXPECT_EQ(BadBranches, 0u)
      << BadBranches << " dispatch entries have incorrect branch targets";
}

TEST_F(PatchedELFLayoutTest, ReturnTableDisplacedInstructionsMatchOriginal) {
  auto OrigHandler = CodeObjectHandler::loadFromBytes(FixtureBytes);
  ASSERT_TRUE(!!OrigHandler) << toString(OrigHandler.takeError());

  auto Names = OrigHandler->getKernelNames();
  ASSERT_FALSE(Names.empty());
  const KernelInfo *KI = OrigHandler->getKernel(Names[0]);
  ASSERT_NE(KI, nullptr);

  auto OrigText = OrigHandler->getTextSection();

  auto Patcher = KernelPatcher::create("gfx950");
  ASSERT_TRUE(!!Patcher) << toString(Patcher.takeError());

  CapturedCodeObject CodeObj;
  CodeObj.CodeObjectId = 7;
  CodeObj.Bytes = FixtureBytes;

  CapturedKernelSymbol Symbol;
  Symbol.KernelId = 7;
  Symbol.CodeObjectId = 7;
  Symbol.KernelName = KI->Name;
  Symbol.KernargSegmentSize = KI->Descriptor.KernargSize;
  Symbol.GroupSegmentSize = KI->Descriptor.GroupSegmentFixedSize;
  Symbol.PrivateSegmentSize = KI->Descriptor.PrivateSegmentFixedSize;
  Symbol.SGPRCount = KI->Descriptor.SGPRCount;
  Symbol.VGPRCount = KI->Descriptor.VGPRCount;

  TraceConfig Trace;
  Trace.BufferAddr = 0xDEAD000000000000ULL;
  Trace.CounterAddr = 0xDEAD000000001000ULL;
  Trace.BufferSize = 32768;
  Trace.Strategy = PayloadStrategy::OnGpuReduce;
  Trace.SupportsGPUAtomics = true;

  auto ResultOrErr = (*Patcher)->getOrPatch(CodeObj, Symbol,
                                            InstrumentationMode::MEMORY_ONLY,
                                            &Trace);
  ASSERT_TRUE(!!ResultOrErr) << toString(ResultOrErr.takeError());
  const PatchedKernel *Result = *ResultOrErr;
  ASSERT_NE(Result, nullptr);

  auto PatchedHandler =
      CodeObjectHandler::loadFromBytes(Result->PatchedELF);
  ASSERT_TRUE(!!PatchedHandler) << toString(PatchedHandler.takeError());

  auto PatchedNames = PatchedHandler->getKernelNames();
  ASSERT_FALSE(PatchedNames.empty());
  const KernelInfo *PKI = PatchedHandler->getKernel(PatchedNames[0]);
  ASSERT_NE(PKI, nullptr);

  auto PatchedText = PatchedHandler->getTextSection();

  // Find the original code in the patched text (accounting for prologue shift)
  uint64_t Shift = 0;
  for (uint64_t Try = 0; Try <= 256; Try += 4) {
    uint64_t PO = PKI->CodeOffset + Try;
    if (PO + 32 > PatchedText.size())
      break;
    if (std::memcmp(OrigText.data() + KI->CodeOffset,
                    PatchedText.data() + PO, 32) == 0) {
      Shift = Try;
      break;
    }
  }
  uint64_t PatchedCodeStart = PKI->CodeOffset + Shift;

  // Collect patch site offsets (relative to kernel code start) and their
  // original instructions from the original .text.
  constexpr uint32_t SOPK_MOVK_MASK = 0xF8000000;
  constexpr uint32_t SOPK_MOVK_VAL  = 0xB0000000;
  constexpr uint32_t SDST_MASK      = 0x007F0000;
  uint32_t ScratchSGPRBase = 20;
  uint32_t ScratchSGPREnd = 28;

  struct PatchSite {
    uint16_t SiteID;
    uint64_t KernelRelOffset;
    uint8_t OrigBytes[8];
  };

  std::vector<PatchSite> Sites;
  uint64_t SearchEnd = std::min(PatchedCodeStart + KI->CodeSize + 16,
                                 (uint64_t)PatchedText.size());
  for (uint64_t Off = PatchedCodeStart; Off + 4 <= SearchEnd; Off += 4) {
    uint32_t Word;
    std::memcpy(&Word, PatchedText.data() + Off, 4);
    if ((Word & SOPK_MOVK_MASK) == SOPK_MOVK_VAL) {
      uint32_t Sdst = (Word & SDST_MASK) >> 16;
      if (Sdst >= ScratchSGPRBase && Sdst < ScratchSGPREnd) {
        PatchSite PS;
        PS.SiteID = Word & 0xFFFF;
        PS.KernelRelOffset = Off - PatchedCodeStart;
        uint64_t OrigOff = KI->CodeOffset + PS.KernelRelOffset;
        if (OrigOff + 8 <= OrigText.size()) {
          std::memcpy(PS.OrigBytes, OrigText.data() + OrigOff, 8);
        } else {
          std::memset(PS.OrigBytes, 0, 8);
        }
        Sites.push_back(PS);
      }
    }
  }

  std::sort(Sites.begin(), Sites.end(),
            [](const PatchSite &A, const PatchSite &B) {
              return A.SiteID < B.SiteID;
            });

  ASSERT_EQ(Sites.size(), Result->NumMemorySites)
      << "Site count mismatch";

  // The return table is the last section of the island. Each entry is 24 bytes:
  //   12 bytes: SCC extraction + restore (s_and_b32, s_andn2_b32, s_cmp_lg_u32)
  //   4 or 8 bytes: displaced instruction
  //   0 or 4 bytes: s_nop padding (if displaced was 4 bytes)
  //   4 bytes: s_setpc_b64
  //
  // The displaced instruction is at bytes [12..19] of each 24-byte entry.
  // Find the return table: it's at the end of the island. The island layout is:
  //   [Preamble][DispatchTable][SharedBodyVMEM][SharedBodyLDS][ReturnTable]
  // ReturnTable size = NumSites * 24
  // So ReturnTable starts at: PatchedText.size() - NumSites * 24
  // (if there's no padding after the return table)

  uint64_t ReturnTableSize = Result->NumMemorySites * 24;
  ASSERT_LE(ReturnTableSize, PatchedText.size());

  uint64_t ReturnTableStart = PatchedText.size() - ReturnTableSize;
  std::cout << "Return table at .text offset: " << ReturnTableStart << std::endl;
  std::cout << "Return table entries: " << Result->NumMemorySites << std::endl;

  // Verify each return table entry contains the correct displaced instruction.
  uint32_t Mismatches = 0;
  uint32_t Checked = 0;
  for (uint32_t I = 0; I < Sites.size(); ++I) {
    uint64_t EntryStart = ReturnTableStart + static_cast<uint64_t>(I) * 24;
    uint64_t DisplacedStart = EntryStart + 12;
    if (DisplacedStart + 8 > PatchedText.size())
      break;

    if (std::memcmp(PatchedText.data() + DisplacedStart,
                    Sites[I].OrigBytes, 8) != 0) {
      if (Mismatches < 5) {
        std::cout << "  Site " << Sites[I].SiteID
                  << " displaced mismatch at return entry " << I
                  << " (kernel offset 0x" << std::hex
                  << Sites[I].KernelRelOffset << std::dec << ")" << std::endl;
        std::cout << "    Original: ";
        for (int B = 0; B < 8; ++B) {
          char Buf[4];
          snprintf(Buf, sizeof(Buf), "%02x ", Sites[I].OrigBytes[B]);
          std::cout << Buf;
        }
        std::cout << std::endl;
        std::cout << "    In return table: ";
        for (int B = 0; B < 8; ++B) {
          char Buf[4];
          snprintf(Buf, sizeof(Buf), "%02x ", PatchedText[DisplacedStart + B]);
          std::cout << Buf;
        }
        std::cout << std::endl;
      }
      ++Mismatches;
    }
    ++Checked;
  }

  std::cout << "Checked " << Checked << " return entries, "
            << Mismatches << " mismatches" << std::endl;

  EXPECT_EQ(Mismatches, 0u)
      << Mismatches << " return table entries have displaced instructions "
         "that don't match the original kernel code";
}

} // namespace
