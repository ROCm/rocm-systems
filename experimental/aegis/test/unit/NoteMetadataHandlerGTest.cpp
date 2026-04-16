//===-- NoteMetadataHandlerGTest.cpp - Note Metadata Tests ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for NoteMetadataHandler: parse, update, and serialize kernel
/// metadata from .note sections.  A bug here causes wrong VGPR/SGPR/LDS
/// counts in the kernel descriptor, leading to GPU faults or silent
/// corruption during dispatch.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/NoteMetadataHandler.h"
#include "aegisbit/Endian.h"
#include "aegisbit/Types.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/Error.h"
#include <gtest/gtest.h>
#include <cstring>

using namespace aegisbit;

//===----------------------------------------------------------------------===//
// Helpers: build synthetic .note sections with msgpack payloads
//===----------------------------------------------------------------------===//

namespace {

/// Build a minimal msgpack blob representing AMDGPU kernel metadata.
/// Fields: amdhsa.target, amdhsa.kernels (array of kernel maps).
std::string buildMsgPack(
    const std::string &Target,
    const std::vector<KernelMetadata> &Kernels) {
  llvm::msgpack::Document Doc;
  auto &Root = Doc.getRoot();
  Root = Doc.getMapNode();
  auto &RootMap = Root.getMap();

  // ".target"
  RootMap[Doc.getNode(".target")] = Doc.getNode(Target);

  // "amdhsa.kernels"
  auto KernelsNode = Doc.getArrayNode();
  auto &KArr = KernelsNode.getArray();

  for (const auto &KM : Kernels) {
    auto KNode = Doc.getMapNode();
    auto &KMap = KNode.getMap();
    KMap[Doc.getNode(".name")] = Doc.getNode(KM.Name);
    KMap[Doc.getNode(".sgpr_count")] = Doc.getNode(static_cast<uint64_t>(KM.SGPRCount));
    KMap[Doc.getNode(".vgpr_count")] = Doc.getNode(static_cast<uint64_t>(KM.VGPRCount));
    KMap[Doc.getNode(".kernarg_segment_size")] = Doc.getNode(static_cast<uint64_t>(KM.KernargSize));
    KMap[Doc.getNode(".group_segment_fixed_size")] = Doc.getNode(static_cast<uint64_t>(KM.LDSSize));
    KMap[Doc.getNode(".private_segment_fixed_size")] = Doc.getNode(static_cast<uint64_t>(KM.ScratchSize));
    KArr.push_back(std::move(KNode));
  }

  RootMap[Doc.getNode("amdhsa.kernels")] = std::move(KernelsNode);

  std::string Blob;
  Doc.writeToBlob(Blob);
  return Blob;
}

/// Wrap a msgpack blob in a .note section with the AMDGPU note header.
std::vector<uint8_t> wrapInNoteSection(const std::string &MsgPackBlob) {
  const char *Name = "AMDGPU";
  size_t NameLen = strlen(Name) + 1;  // include null
  size_t AlignedNameLen = (NameLen + 3) & ~3;
  size_t DescLen = MsgPackBlob.size();
  size_t AlignedDescLen = (DescLen + 3) & ~3;
  size_t Total = 12 + AlignedNameLen + AlignedDescLen;

  std::vector<uint8_t> Note(Total, 0);
  writeLE32(&Note[0], static_cast<uint32_t>(NameLen));
  writeLE32(&Note[4], static_cast<uint32_t>(DescLen));
  writeLE32(&Note[8], NoteMetadataHandler::NT_AMDGPU_METADATA);
  std::memcpy(&Note[12], Name, NameLen);
  std::memcpy(&Note[12 + AlignedNameLen], MsgPackBlob.data(), DescLen);
  return Note;
}

/// Build a complete .note section from kernel metadata.
std::vector<uint8_t> buildNoteSection(
    const std::string &Target,
    const std::vector<KernelMetadata> &Kernels) {
  return wrapInNoteSection(buildMsgPack(Target, Kernels));
}

/// Shorthand: single-kernel note section.
std::vector<uint8_t> buildSingleKernelNote(
    const std::string &Name, uint32_t VGPRs, uint32_t SGPRs,
    uint32_t LDS = 0, uint32_t Scratch = 0, uint32_t Kernarg = 32) {
  KernelMetadata KM;
  KM.Name = Name;
  KM.VGPRCount = VGPRs;
  KM.SGPRCount = SGPRs;
  KM.LDSSize = LDS;
  KM.ScratchSize = Scratch;
  KM.KernargSize = Kernarg;
  return buildNoteSection("amdgcn-amd-amdhsa--gfx950", {KM});
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Parse tests: extract correct values from synthetic note sections
//===----------------------------------------------------------------------===//

TEST(NoteMetadataHandler, ParseSingleKernel) {
  auto Note = buildSingleKernelNote("my_kernel", 128, 48, 16384, 256, 64);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  auto &C = *ContentOrErr;
  EXPECT_EQ(C.GPUArch, "gfx950");
  ASSERT_EQ(C.Kernels.size(), 1u);

  auto &K = C.Kernels[0];
  EXPECT_EQ(K.Name, "my_kernel");
  EXPECT_EQ(K.VGPRCount, 128u);
  EXPECT_EQ(K.SGPRCount, 48u);
  EXPECT_EQ(K.LDSSize, 16384u);
  EXPECT_EQ(K.ScratchSize, 256u);
  EXPECT_EQ(K.KernargSize, 64u);
}

TEST(NoteMetadataHandler, ParseMultipleKernels) {
  KernelMetadata K1, K2, K3;
  K1.Name = "vector_add"; K1.VGPRCount = 16;  K1.SGPRCount = 8;
  K2.Name = "matmul";     K2.VGPRCount = 256; K2.SGPRCount = 104;
  K2.LDSSize = 65536; K2.ScratchSize = 512;
  K3.Name = "softmax";    K3.VGPRCount = 64;  K3.SGPRCount = 32;

  auto Note = buildNoteSection("amdgcn-amd-amdhsa--gfx942", {K1, K2, K3});
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  auto &C = *ContentOrErr;
  EXPECT_EQ(C.GPUArch, "gfx942");
  ASSERT_EQ(C.Kernels.size(), 3u);

  EXPECT_EQ(C.Kernels[0].Name, "vector_add");
  EXPECT_EQ(C.Kernels[0].VGPRCount, 16u);
  EXPECT_EQ(C.Kernels[0].SGPRCount, 8u);

  EXPECT_EQ(C.Kernels[1].Name, "matmul");
  EXPECT_EQ(C.Kernels[1].VGPRCount, 256u);
  EXPECT_EQ(C.Kernels[1].SGPRCount, 104u);
  EXPECT_EQ(C.Kernels[1].LDSSize, 65536u);
  EXPECT_EQ(C.Kernels[1].ScratchSize, 512u);

  EXPECT_EQ(C.Kernels[2].Name, "softmax");
  EXPECT_EQ(C.Kernels[2].VGPRCount, 64u);
}

TEST(NoteMetadataHandler, ParseExtractsArchFromTarget) {
  auto Note = buildNoteSection("amdgcn-amd-amdhsa--gfx90a", {});
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());
  EXPECT_EQ(ContentOrErr->GPUArch, "gfx90a");
}

TEST(NoteMetadataHandler, ParseZeroRegistersPreserved) {
  auto Note = buildSingleKernelNote("empty_kernel", 0, 0, 0, 0, 0);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  ASSERT_EQ(ContentOrErr->Kernels.size(), 1u);
  EXPECT_EQ(ContentOrErr->Kernels[0].VGPRCount, 0u);
  EXPECT_EQ(ContentOrErr->Kernels[0].SGPRCount, 0u);
  EXPECT_EQ(ContentOrErr->Kernels[0].LDSSize, 0u);
}

//===----------------------------------------------------------------------===//
// Error handling: malformed input
//===----------------------------------------------------------------------===//

TEST(NoteMetadataHandler, RejectsTooSmallNote) {
  std::vector<uint8_t> Tiny = {0x00, 0x01, 0x02};
  auto ContentOrErr = NoteMetadataHandler::parse(Tiny);
  EXPECT_FALSE(static_cast<bool>(ContentOrErr));
  llvm::consumeError(ContentOrErr.takeError());
}

TEST(NoteMetadataHandler, RejectsNoteWithWrongType) {
  // Build a note section with wrong type (not NT_AMDGPU_METADATA)
  const char *Name = "AMDGPU";
  size_t NameLen = strlen(Name) + 1;
  size_t AlignedNameLen = (NameLen + 3) & ~3;
  std::string FakeDesc = "not real metadata";
  size_t Total = 12 + AlignedNameLen + ((FakeDesc.size() + 3) & ~3);

  std::vector<uint8_t> Note(Total, 0);
  writeLE32(&Note[0], static_cast<uint32_t>(NameLen));
  writeLE32(&Note[4], static_cast<uint32_t>(FakeDesc.size()));
  writeLE32(&Note[8], 99u);  // wrong type
  std::memcpy(&Note[12], Name, NameLen);
  std::memcpy(&Note[12 + AlignedNameLen], FakeDesc.data(), FakeDesc.size());

  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  EXPECT_FALSE(static_cast<bool>(ContentOrErr));
  llvm::consumeError(ContentOrErr.takeError());
}

//===----------------------------------------------------------------------===//
// Update tests: modify kernel metadata and verify round-trip
//===----------------------------------------------------------------------===//

TEST(NoteMetadataHandler, UpdateVGPRCount) {
  auto Note = buildSingleKernelNote("test_kernel", 64, 32, 0, 0, 32);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  auto &Content = *ContentOrErr;
  ASSERT_EQ(Content.Kernels[0].VGPRCount, 64u);

  KernelDescriptor Desc;
  Desc.VGPRCount = 128;
  Desc.SGPRCount = 32;
  Desc.KernargSize = 32;
  Desc.GroupSegmentFixedSize = 0;
  Desc.PrivateSegmentFixedSize = 0;

  auto Err = NoteMetadataHandler::updateKernelMetadata(Content, "test_kernel", Desc);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  EXPECT_EQ(Content.Kernels[0].VGPRCount, 128u);

  // Re-parse the updated msgpack to verify the blob was actually modified
  auto Reserialized = NoteMetadataHandler::serialize(Content);
  auto RoundTripOrErr = NoteMetadataHandler::parse(Reserialized);
  ASSERT_TRUE(static_cast<bool>(RoundTripOrErr)) << toString(RoundTripOrErr.takeError());

  EXPECT_EQ(RoundTripOrErr->Kernels[0].VGPRCount, 128u);
  EXPECT_EQ(RoundTripOrErr->Kernels[0].SGPRCount, 32u);
}

TEST(NoteMetadataHandler, UpdateAllFields) {
  auto Note = buildSingleKernelNote("gemm_kernel", 32, 16, 0, 0, 64);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  KernelDescriptor Desc;
  Desc.VGPRCount = 256;
  Desc.SGPRCount = 104;
  Desc.KernargSize = 128;
  Desc.GroupSegmentFixedSize = 32768;
  Desc.PrivateSegmentFixedSize = 1024;

  auto Err = NoteMetadataHandler::updateKernelMetadata(
      *ContentOrErr, "gemm_kernel", Desc);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  auto &K = ContentOrErr->Kernels[0];
  EXPECT_EQ(K.VGPRCount, 256u);
  EXPECT_EQ(K.SGPRCount, 104u);
  EXPECT_EQ(K.KernargSize, 128u);
  EXPECT_EQ(K.LDSSize, 32768u);
  EXPECT_EQ(K.ScratchSize, 1024u);

  // Round-trip through serialize → parse
  auto Reserialized = NoteMetadataHandler::serialize(*ContentOrErr);
  auto RoundTripOrErr = NoteMetadataHandler::parse(Reserialized);
  ASSERT_TRUE(static_cast<bool>(RoundTripOrErr)) << toString(RoundTripOrErr.takeError());

  auto &RT = RoundTripOrErr->Kernels[0];
  EXPECT_EQ(RT.VGPRCount, 256u);
  EXPECT_EQ(RT.SGPRCount, 104u);
  EXPECT_EQ(RT.KernargSize, 128u);
  EXPECT_EQ(RT.LDSSize, 32768u);
  EXPECT_EQ(RT.ScratchSize, 1024u);
}

TEST(NoteMetadataHandler, UpdateWithKDSuffix) {
  auto Note = buildSingleKernelNote("flash_attention_kernel", 128, 64);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  KernelDescriptor Desc;
  Desc.VGPRCount = 192;
  Desc.SGPRCount = 80;
  Desc.KernargSize = 32;
  Desc.GroupSegmentFixedSize = 0;
  Desc.PrivateSegmentFixedSize = 0;

  // Pass name with .kd suffix — handler should strip it
  auto Err = NoteMetadataHandler::updateKernelMetadata(
      *ContentOrErr, "flash_attention_kernel.kd", Desc);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  EXPECT_EQ(ContentOrErr->Kernels[0].VGPRCount, 192u);
  EXPECT_EQ(ContentOrErr->Kernels[0].SGPRCount, 80u);
}

TEST(NoteMetadataHandler, UpdateMultiKernelTargetsCorrectOne) {
  KernelMetadata K1, K2;
  K1.Name = "kernel_a"; K1.VGPRCount = 32;  K1.SGPRCount = 16;
  K2.Name = "kernel_b"; K2.VGPRCount = 64;  K2.SGPRCount = 32;

  auto Note = buildNoteSection("amdgcn-amd-amdhsa--gfx950", {K1, K2});
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  KernelDescriptor Desc;
  Desc.VGPRCount = 128;
  Desc.SGPRCount = 48;
  Desc.KernargSize = 0;
  Desc.GroupSegmentFixedSize = 0;
  Desc.PrivateSegmentFixedSize = 0;

  auto Err = NoteMetadataHandler::updateKernelMetadata(
      *ContentOrErr, "kernel_b", Desc);
  ASSERT_FALSE(static_cast<bool>(Err)) << toString(std::move(Err));

  // kernel_a should be unchanged
  EXPECT_EQ(ContentOrErr->Kernels[0].VGPRCount, 32u);
  EXPECT_EQ(ContentOrErr->Kernels[0].SGPRCount, 16u);
  // kernel_b should be updated
  EXPECT_EQ(ContentOrErr->Kernels[1].VGPRCount, 128u);
  EXPECT_EQ(ContentOrErr->Kernels[1].SGPRCount, 48u);
}

TEST(NoteMetadataHandler, UpdateNonexistentKernelFails) {
  auto Note = buildSingleKernelNote("real_kernel", 64, 32);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  KernelDescriptor Desc;
  Desc.VGPRCount = 128;
  Desc.SGPRCount = 48;
  Desc.KernargSize = 0;
  Desc.GroupSegmentFixedSize = 0;
  Desc.PrivateSegmentFixedSize = 0;

  auto Err = NoteMetadataHandler::updateKernelMetadata(
      *ContentOrErr, "nonexistent_kernel", Desc);
  EXPECT_TRUE(static_cast<bool>(Err));
  llvm::consumeError(std::move(Err));
}

//===----------------------------------------------------------------------===//
// Serialize tests: verify output can be re-parsed identically
//===----------------------------------------------------------------------===//

TEST(NoteMetadataHandler, SerializeRoundTrip) {
  auto Note = buildSingleKernelNote("roundtrip_kernel", 96, 56, 8192, 128, 48);
  auto ContentOrErr = NoteMetadataHandler::parse(Note);
  ASSERT_TRUE(static_cast<bool>(ContentOrErr)) << toString(ContentOrErr.takeError());

  auto Serialized = NoteMetadataHandler::serialize(*ContentOrErr);
  auto ReparseOrErr = NoteMetadataHandler::parse(Serialized);
  ASSERT_TRUE(static_cast<bool>(ReparseOrErr)) << toString(ReparseOrErr.takeError());

  EXPECT_EQ(ReparseOrErr->GPUArch, "gfx950");
  ASSERT_EQ(ReparseOrErr->Kernels.size(), 1u);
  EXPECT_EQ(ReparseOrErr->Kernels[0].Name, "roundtrip_kernel");
  EXPECT_EQ(ReparseOrErr->Kernels[0].VGPRCount, 96u);
  EXPECT_EQ(ReparseOrErr->Kernels[0].SGPRCount, 56u);
  EXPECT_EQ(ReparseOrErr->Kernels[0].LDSSize, 8192u);
  EXPECT_EQ(ReparseOrErr->Kernels[0].ScratchSize, 128u);
  EXPECT_EQ(ReparseOrErr->Kernels[0].KernargSize, 48u);
}
