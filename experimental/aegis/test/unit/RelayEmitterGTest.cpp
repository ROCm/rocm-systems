//===-- RelayEmitterGTest.cpp - Relay Emitter Tests ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//

#include "aegisbit/RelayEmitter.h"
#include "aegisbit/Disassembler.h"
#include "aegisbit/ISAEncoder.h"
#include "aegisbit/TrampolineBridge.h"

#include "gtest/gtest.h"
#include <cstring>

using namespace aegisbit;
using namespace llvm;

namespace {

class RelayEmitterTest : public ::testing::Test {
protected:
  void SetUp() override {
    auto DisasmOrErr =
        Disassembler::create("amdgcn-amd-amdhsa", "gfx942", "+wavefrontsize64");
    if (!DisasmOrErr) {
      GTEST_SKIP() << "Cannot create AMDGPU disassembler";
    }
    Disasm = std::move(*DisasmOrErr);

    auto EncOrErr = ISAEncoder::create("gfx942", *Disasm);
    if (!EncOrErr) {
      GTEST_SKIP() << "Cannot create ISAEncoder";
    }
    Enc = std::move(*EncOrErr);
  }

  InstrumentationSite makeSite(uint64_t Offset, uint64_t OrigSize = 4) {
    InstrumentationSite S;
    S.Address = Offset;
    S.Offset = Offset;
    S.OrigInstSize = OrigSize;
    S.IsLoad = true;
    S.IsGlobal = true;
    S.AddrVGPRIndex = 0;
    return S;
  }

  ScratchRegisters makeZeroSGPRScratch() {
    KernelDescriptor KD{};
    KD.VGPRCount = 8;
    KD.SGPRCount = 104;
    return ScratchRegisters::fromDescriptorZeroSGPR(KD);
  }

  // Build a code buffer of s_nop instructions
  std::vector<uint8_t> makeNopCode(uint64_t Size) {
    std::vector<uint8_t> Code(Size, 0x00);
    for (size_t i = 0; i + 3 < Code.size(); i += 4) {
      Code[i + 0] = 0x00; Code[i + 1] = 0x00;
      Code[i + 2] = 0x80; Code[i + 3] = 0xBF;
    }
    return Code;
  }

  std::unique_ptr<Disassembler> Disasm;
  std::unique_ptr<ISAEncoder> Enc;
};

TEST_F(RelayEmitterTest, EmptyStubsWhenReturnOutOfRange) {
  RelayEmitter Relay(*Enc);
  auto Scratch = makeZeroSGPRScratch();
  auto Code = makeNopCode(1024);
  auto Site = makeSite(0);

  // ReturnTargetAbs far from RetBranchPC so s_branch is out of range
  // RetBranchPC = island cursor position; set far from return target
  uint64_t ReturnTargetAbs = 4;
  uint64_t RetBranchPC = 500 * 1024; // 500KB away

  auto StubsOrErr = Relay.emitRelayStubs(Site, Scratch, Code,
                                           ReturnTargetAbs, RetBranchPC);
  ASSERT_TRUE(!!StubsOrErr) << toString(StubsOrErr.takeError());

  EXPECT_TRUE(StubsOrErr->ForwardStub.empty())
      << "Should return empty stubs when return s_branch is out of range";
  EXPECT_TRUE(StubsOrErr->ReturnStub.empty());
}

TEST_F(RelayEmitterTest, ValidStubsWithinRange) {
  RelayEmitter Relay(*Enc);
  auto Scratch = makeZeroSGPRScratch();
  auto Code = makeNopCode(1024);
  auto Site = makeSite(0);

  // Set up so return s_branch is within ±128KB
  uint64_t ReturnTargetAbs = 4;
  uint64_t RetBranchPC = 0; // Relay stubs placed at beginning

  auto StubsOrErr = Relay.emitRelayStubs(Site, Scratch, Code,
                                           ReturnTargetAbs, RetBranchPC);
  ASSERT_TRUE(!!StubsOrErr) << toString(StubsOrErr.takeError());

  auto &Stubs = *StubsOrErr;
  EXPECT_FALSE(Stubs.ForwardStub.empty())
      << "Forward stub should not be empty for reachable site";
  EXPECT_FALSE(Stubs.ReturnStub.empty())
      << "Return stub should not be empty for reachable site";

  // Forward stub should contain long-jump placeholder at FwdLongJumpOffset
  EXPECT_GT(Stubs.FwdLongJumpOffset, 0u);
  EXPECT_LE(Stubs.FwdLongJumpOffset + 16, Stubs.ForwardStub.size())
      << "Long-jump placeholder must fit in forward stub";

  // Return stub should end with s_branch (0xBF82xxxx)
  size_t RS = Stubs.ReturnStub.size();
  ASSERT_GE(RS, 4u);
  uint32_t LastWord = Stubs.ReturnStub[RS - 4]
                    | (Stubs.ReturnStub[RS - 3] << 8)
                    | (Stubs.ReturnStub[RS - 2] << 16)
                    | (Stubs.ReturnStub[RS - 1] << 24);
  EXPECT_EQ(LastWord >> 16, 0xBF82u)
      << "Return stub should end with s_branch";
}

TEST_F(RelayEmitterTest, FixupPatchesCorrectOffsets) {
  RelayEmitter Relay(*Enc);

  // Simulate a body entry
  std::vector<uint8_t> FakeBody(128, 0xAA);
  size_t BodyOff = Relay.addBodyEntry(FakeBody);
  EXPECT_EQ(BodyOff, 0u);

  auto RetLJOff = Relay.appendReturnLongJump();
  ASSERT_TRUE(!!RetLJOff) << toString(RetLJOff.takeError());

  // Create a fake stub island containing a long-jump placeholder
  auto PlaceholderLJ = Enc->encodeLongJumpVCC(RelayEmitter::LJ_PLACEHOLDER);
  ASSERT_TRUE(!!PlaceholderLJ) << toString(PlaceholderLJ.takeError());
  size_t LJSize = PlaceholderLJ->size();

  TrampolineIsland StubIsland;
  StubIsland.Offset = 1024; // stub island at absolute 1024
  // Put some padding before the long-jump
  StubIsland.Bytes.resize(64, 0x00);
  StubIsland.Bytes.insert(StubIsland.Bytes.end(),
                           PlaceholderLJ->begin(), PlaceholderLJ->end());

  RelayFixup Fix;
  Fix.FwdGetPCAbs = StubIsland.Offset + 64; // points to start of LJ in stub
  Fix.RelayReturnAbs = StubIsland.Offset + 64 + LJSize;
  Fix.BodyEntryOff = BodyOff;
  Fix.RetGetPCBodyOff = *RetLJOff;
  Relay.addFixup(Fix);

  uint64_t BodyIslandStart = 65536;
  std::vector<TrampolineIsland> Islands = {StubIsland};

  auto BodyIslOrErr = Relay.fixupRelays(BodyIslandStart, Islands);
  ASSERT_TRUE(!!BodyIslOrErr) << toString(BodyIslOrErr.takeError());

  // The stub island's long-jump bytes should now be different from the placeholder
  auto &PatchedStub = Islands[0];
  std::vector<uint8_t> PatchedLJBytes(
      PatchedStub.Bytes.begin() + 64,
      PatchedStub.Bytes.begin() + 64 + LJSize);
  EXPECT_NE(PatchedLJBytes, *PlaceholderLJ)
      << "Fixup should have patched the long-jump bytes";

  // Verify the patched long-jump encodes the correct offset
  uint64_t ExpectedBodyEntry = BodyIslandStart + BodyOff;
  int64_t ExpectedFwdOffset = static_cast<int64_t>(ExpectedBodyEntry) -
                              static_cast<int64_t>(Fix.FwdGetPCAbs);
  auto ExpectedLJ = Enc->encodeLongJumpVCC(ExpectedFwdOffset);
  ASSERT_TRUE(!!ExpectedLJ) << toString(ExpectedLJ.takeError());
  EXPECT_EQ(PatchedLJBytes, *ExpectedLJ)
      << "Forward long-jump should encode the correct distance to body entry";
}

TEST_F(RelayEmitterTest, MultipleFixupsAllPatched) {
  RelayEmitter Relay(*Enc);

  auto PlaceholderLJ = Enc->encodeLongJumpVCC(RelayEmitter::LJ_PLACEHOLDER);
  ASSERT_TRUE(!!PlaceholderLJ) << toString(PlaceholderLJ.takeError());
  size_t LJSize = PlaceholderLJ->size();

  TrampolineIsland StubIsland;
  StubIsland.Offset = 2048;

  constexpr int NumFixups = 3;
  for (int i = 0; i < NumFixups; ++i) {
    std::vector<uint8_t> FakeBody(64, static_cast<uint8_t>(0xB0 + i));
    size_t BodyOff = Relay.addBodyEntry(FakeBody);
    auto RetLJOff = Relay.appendReturnLongJump();
    ASSERT_TRUE(!!RetLJOff) << toString(RetLJOff.takeError());

    size_t StubOff = StubIsland.Bytes.size();
    StubIsland.Bytes.insert(StubIsland.Bytes.end(),
                             PlaceholderLJ->begin(), PlaceholderLJ->end());

    RelayFixup Fix;
    Fix.FwdGetPCAbs = StubIsland.Offset + StubOff;
    Fix.RelayReturnAbs = StubIsland.Offset + StubOff + LJSize;
    Fix.BodyEntryOff = BodyOff;
    Fix.RetGetPCBodyOff = *RetLJOff;
    Relay.addFixup(Fix);
  }

  uint64_t BodyIslandStart = 131072;
  std::vector<TrampolineIsland> Islands = {StubIsland};

  auto BodyIslOrErr = Relay.fixupRelays(BodyIslandStart, Islands);
  ASSERT_TRUE(!!BodyIslOrErr) << toString(BodyIslOrErr.takeError());

  // Verify all 3 long-jumps in the stub island were patched
  auto &Patched = Islands[0];
  for (int i = 0; i < NumFixups; ++i) {
    size_t Off = i * LJSize;
    std::vector<uint8_t> PatchedBytes(Patched.Bytes.begin() + Off,
                                       Patched.Bytes.begin() + Off + LJSize);
    EXPECT_NE(PatchedBytes, *PlaceholderLJ)
        << "Fixup " << i << " should have been patched";
  }

  // Body island should exist and be non-empty
  EXPECT_FALSE(BodyIslOrErr->Bytes.empty());
  EXPECT_EQ(BodyIslOrErr->Offset, BodyIslandStart);
}

} // namespace
