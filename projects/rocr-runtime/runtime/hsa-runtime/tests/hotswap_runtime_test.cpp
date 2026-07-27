/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/runtime/hotswap_aql_patch.h"
#include "core/inc/hotswap_env.hpp"
#include "loader/hotswap_kernel_registry.h"

#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace {

int Failures = 0;
bool PrepareCalled = false;
bool FenceCalled = false;
uint16_t HeaderDuringPrepare = 0;
uint64_t KernelObjectDuringFence = 0;
rocr::core::AqlPacket* PacketUnderPatch = nullptr;

void Expect(bool condition, const char* expression, int line) {
  if (condition) return;
  std::cerr << "hotswap-runtime-unit-tests:" << line
            << ": check failed: " << expression << "\n";
  ++Failures;
}

#define EXPECT_TRUE(expr) Expect((expr), #expr, __LINE__)
#define EXPECT_FALSE(expr) Expect(!(expr), "!(" #expr ")", __LINE__)
#define EXPECT_EQ(lhs, rhs) Expect(((lhs) == (rhs)), #lhs " == " #rhs, __LINE__)
#define EXPECT_NE(lhs, rhs) Expect(((lhs) != (rhs)), #lhs " != " #rhs, __LINE__)

uint16_t KernelDispatchHeader() {
  return (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
}

uint16_t ExtDispatchHeader() {
  return (HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
         (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
}

bool PreparePatchSuccess(uint64_t* kernel_object,
                         uint32_t* private_segment_size,
                         uint32_t* group_segment_size,
                         uint32_t* scaled_dispatch_factor) {
  PrepareCalled = true;
  if (PacketUnderPatch)
    HeaderDuringPrepare = PacketUnderPatch->packet.header;
  *kernel_object = 0x2000;
  *private_segment_size = 64;
  *group_segment_size = 128;
  *scaled_dispatch_factor = 1;
  return true;
}

bool PreparePatchFailure(uint64_t*, uint32_t*, uint32_t*, uint32_t*) {
  return false;
}

bool PrepareScaledPatchSuccess(uint64_t* kernel_object,
                               uint32_t* private_segment_size,
                               uint32_t* group_segment_size,
                               uint32_t* scaled_dispatch_factor) {
  *kernel_object = 0x2000;
  *private_segment_size = 64;
  *group_segment_size = 128;
  *scaled_dispatch_factor = 2;
  return true;
}

void PacketBodyFence() {
  FenceCalled = true;
  if (PacketUnderPatch) {
    KernelObjectDuringFence =
        PacketUnderPatch->ext_dispatch.amd_format ==
                HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH
            ? PacketUnderPatch->ext_dispatch.kernel_object
            : PacketUnderPatch->dispatch.kernel_object;
  }
}

rocr::core::AqlPacket MakeKernelDispatchPacket() {
  rocr::core::AqlPacket packet = {};
  packet.dispatch.header = KernelDispatchHeader();
  packet.dispatch.setup = 1;
  packet.dispatch.kernel_object = 0x1000;
  packet.dispatch.private_segment_size = 4;
  packet.dispatch.group_segment_size = 8;
  return packet;
}

rocr::core::AqlPacket MakeExtDispatchPacket() {
  rocr::core::AqlPacket packet = {};
  packet.ext_dispatch.header = ExtDispatchHeader();
  packet.ext_dispatch.amd_format = HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH;
  packet.ext_dispatch.setup = 1;
  packet.ext_dispatch.kernel_object = 0x1000;
  packet.ext_dispatch.private_segment_size = 4;
  packet.ext_dispatch.group_segment_size = 8;
  return packet;
}

void TestPatchPublishedKernelDispatchPacket() {
  PrepareCalled = false;
  FenceCalled = false;
  HeaderDuringPrepare = 0;
  KernelObjectDuringFence = 0;
  rocr::core::AqlPacket packet = MakeKernelDispatchPacket();
  PacketUnderPatch = &packet;

  const bool patched = rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
      packet, PreparePatchSuccess, PacketBodyFence);
  PacketUnderPatch = nullptr;

  EXPECT_TRUE(patched);
  EXPECT_TRUE(PrepareCalled);
  EXPECT_TRUE(FenceCalled);
  EXPECT_EQ(HeaderDuringPrepare,
            static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID
                                  << HSA_PACKET_HEADER_TYPE));
  EXPECT_EQ(KernelObjectDuringFence, 0x2000u);
  EXPECT_EQ(packet.dispatch.header, KernelDispatchHeader());
  EXPECT_EQ(packet.dispatch.kernel_object, 0x2000u);
  EXPECT_EQ(packet.dispatch.private_segment_size, 64u);
  EXPECT_EQ(packet.dispatch.group_segment_size, 128u);
}

void TestPatchPublishedExtKernelDispatchPacket() {
  PrepareCalled = false;
  FenceCalled = false;
  HeaderDuringPrepare = 0;
  KernelObjectDuringFence = 0;
  rocr::core::AqlPacket packet = MakeExtDispatchPacket();
  PacketUnderPatch = &packet;

  const bool patched = rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
      packet, PreparePatchSuccess, PacketBodyFence);
  PacketUnderPatch = nullptr;

  EXPECT_TRUE(patched);
  EXPECT_TRUE(PrepareCalled);
  EXPECT_TRUE(FenceCalled);
  EXPECT_EQ(HeaderDuringPrepare,
            static_cast<uint16_t>(HSA_PACKET_TYPE_INVALID
                                  << HSA_PACKET_HEADER_TYPE));
  EXPECT_EQ(KernelObjectDuringFence, 0x2000u);
  EXPECT_EQ(packet.ext_dispatch.header, ExtDispatchHeader());
  EXPECT_EQ(packet.ext_dispatch.kernel_object, 0x2000u);
  EXPECT_EQ(packet.ext_dispatch.private_segment_size, 64u);
  EXPECT_EQ(packet.ext_dispatch.group_segment_size, 128u);
}

void TestPatchScalesDispatchExtent() {
  rocr::core::AqlPacket packet = MakeKernelDispatchPacket();
  packet.dispatch.workgroup_size_x = 32;
  packet.dispatch.grid_size_x = 512;

  const bool patched = rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
      packet, PrepareScaledPatchSuccess, PacketBodyFence);

  EXPECT_TRUE(patched);
  EXPECT_EQ(packet.dispatch.workgroup_size_x, 64u);
  EXPECT_EQ(packet.dispatch.grid_size_x, 1024u);
}

void TestPatchLeavesUnscaledDispatchExtent() {
  rocr::core::AqlPacket packet = MakeKernelDispatchPacket();
  packet.dispatch.workgroup_size_x = 32;
  packet.dispatch.grid_size_x = 512;

  const bool patched = rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
      packet, PreparePatchSuccess, PacketBodyFence);

  EXPECT_TRUE(patched);
  EXPECT_EQ(packet.dispatch.workgroup_size_x, 32u);
  EXPECT_EQ(packet.dispatch.grid_size_x, 512u);
}

void TestPatchSkipsNonKernelPacket() {
  PrepareCalled = false;
  FenceCalled = false;
  PacketUnderPatch = nullptr;
  rocr::core::AqlPacket packet = {};
  packet.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND
                              << HSA_PACKET_HEADER_TYPE;

  const bool patched = rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
      packet, PreparePatchSuccess, PacketBodyFence);

  EXPECT_FALSE(patched);
  EXPECT_FALSE(PrepareCalled);
  EXPECT_FALSE(FenceCalled);
  EXPECT_EQ(packet.barrier_and.header,
            HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE);
}

void TestDoorbellPatchRangeMonotonic() {
  uint64_t first = 0;
  bool hasWork = false;
  const char* failure = nullptr;
  EXPECT_TRUE(rocr::AMD::hotswap::lazy::ComputeDoorbellPatchRange(
      UINT64_MAX, 3, 8, first, hasWork, failure));
  EXPECT_TRUE(hasWork);
  EXPECT_EQ(first, 0u);
  EXPECT_TRUE(failure == nullptr);

  EXPECT_TRUE(rocr::AMD::hotswap::lazy::ComputeDoorbellPatchRange(
      3, 5, 8, first, hasWork, failure));
  EXPECT_TRUE(hasWork);
  EXPECT_EQ(first, 4u);
  EXPECT_TRUE(failure == nullptr);
}

void TestDoorbellPatchRangeSkipsReplayedDoorbell() {
  uint64_t first = 42;
  bool hasWork = true;
  const char* failure = nullptr;
  EXPECT_TRUE(rocr::AMD::hotswap::lazy::ComputeDoorbellPatchRange(
      7, 7, 8, first, hasWork, failure));
  EXPECT_FALSE(hasWork);
  EXPECT_EQ(first, 8u);
  EXPECT_TRUE(failure == nullptr);
}

void TestDoorbellPatchRangeRefusesAliasingRange() {
  uint64_t first = 0;
  bool hasWork = false;
  const char* failure = nullptr;
  EXPECT_FALSE(rocr::AMD::hotswap::lazy::ComputeDoorbellPatchRange(
      50, 99, 8, first, hasWork, failure));
  EXPECT_FALSE(hasWork);
  EXPECT_TRUE(failure != nullptr);
}

void TestPatchFailureAborts() {
  const pid_t child = fork();
  if (child == 0) {
    rocr::core::AqlPacket packet = MakeKernelDispatchPacket();
    rocr::AMD::hotswap::lazy::PatchPublishedKernelDispatchPacket(
        packet, PreparePatchFailure, nullptr);
    _exit(0);
  }

  EXPECT_NE(child, static_cast<pid_t>(-1));
  int status = 0;
  EXPECT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFSIGNALED(status));
  EXPECT_EQ(WTERMSIG(status), SIGABRT);
}

void TestRocrBlitRegistryRangeAndLifetime() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto record = registry.RegisterRocrBlitTargetKernelObject(
      0x1000, 0x80, "gfx942", "embedded_rocr_target_blit_shader",
      "invalid");

  EXPECT_EQ(record->Kind.load(),
            rocr::amd::hsa::loader::HotSwapKernelKind::RuntimeTargetInternal);
  EXPECT_TRUE(record->IsRegisteredRocrBlit(0x1000));
  EXPECT_TRUE(record->IsRegisteredRocrBlit(0x107f));
  EXPECT_FALSE(record->IsRegisteredRocrBlit(0x0fff));
  EXPECT_FALSE(record->IsRegisteredRocrBlit(0x1080));
  EXPECT_TRUE(registry.Lookup(0x1000).get() == record.get());
  EXPECT_TRUE(registry.Lookup(0x107f).get() == record.get());

  EXPECT_TRUE(registry.UnregisterRocrBlitTargetKernelObject(0x1000));
  EXPECT_FALSE(registry.Lookup(0x1000));
  EXPECT_FALSE(registry.Lookup(0x107f));
}

void TestInvalidRocrBlitRegistrationIsNotAllowed() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto record = registry.RegisterRocrBlitTargetKernelObject(
      0, 0x80, "<unknown>", "embedded_rocr_target_blit_shader", "invalid");

  EXPECT_EQ(record->Kind.load(),
            rocr::amd::hsa::loader::HotSwapKernelKind::Untranslated);
  EXPECT_EQ(record->Failure, std::string("invalid"));
  EXPECT_FALSE(record->IsRegisteredRocrBlit(0));
  EXPECT_FALSE(registry.Lookup(0));
}

void TestHotSwapKernelRecordNameDerivation() {
  std::string recordName;
  std::string failure;
  EXPECT_TRUE(rocr::amd::hsa::loader::DeriveHotSwapKernelRecordName(
      "kernel.kd", true, recordName, failure));
  EXPECT_EQ(recordName, std::string("kernel"));
  EXPECT_TRUE(failure.empty());

  EXPECT_TRUE(rocr::amd::hsa::loader::DeriveHotSwapKernelRecordName(
      "legacy_kernel", false, recordName, failure));
  EXPECT_EQ(recordName, std::string("legacy_kernel"));
  EXPECT_TRUE(failure.empty());

  EXPECT_FALSE(rocr::amd::hsa::loader::DeriveHotSwapKernelRecordName(
      "bad", true, recordName, failure));
  EXPECT_TRUE(failure.find(".kd") != std::string::npos);
}

void TestDuplicateKernelObjectRegistrationAborts() {
  const pid_t child = fork();
  if (child == 0) {
    rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
    registry.RegisterKernelObject(
        0x1800, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
        nullptr);
    registry.RegisterKernelObject(
        0x1800, "kernel2",
        rocr::amd::hsa::loader::HotSwapKernelKind::LazySource, nullptr);
    _exit(0);
  }

  EXPECT_NE(child, static_cast<pid_t>(-1));
  int status = 0;
  EXPECT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFSIGNALED(status));
  EXPECT_EQ(WTERMSIG(status), SIGABRT);
}

void TestInvalidRocrBlitDuplicateAddressAborts() {
  const pid_t child = fork();
  if (child == 0) {
    rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
    registry.RegisterKernelObject(
        0x1850, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
        nullptr);
    registry.RegisterRocrBlitTargetKernelObject(
        0x1850, 0x80, "gfx942", "embedded_rocr_target_blit_shader",
        "invalid");
    _exit(0);
  }

  EXPECT_NE(child, static_cast<pid_t>(-1));
  int status = 0;
  EXPECT_EQ(waitpid(child, &status, 0), child);
  EXPECT_TRUE(WIFSIGNALED(status));
  EXPECT_EQ(WTERMSIG(status), SIGABRT);
}

void TestRocrBlitUnregisterDoesNotRemoveOtherKinds() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto source = registry.RegisterKernelObject(
      0x1900, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
      nullptr);

  EXPECT_FALSE(registry.UnregisterRocrBlitTargetKernelObject(0x1900));
  EXPECT_TRUE(registry.Lookup(0x1900).get() == source.get());
}

void TestRocrBlitRangeDoesNotHideExactKernelObject() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto blit = registry.RegisterRocrBlitTargetKernelObject(
      0x2000, 0x100, "gfx942", "embedded_rocr_target_blit_shader",
      "invalid");
  auto source = registry.RegisterKernelObject(
      0x2080, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
      nullptr);

  EXPECT_TRUE(registry.Lookup(0x207f).get() == blit.get());
  EXPECT_TRUE(registry.Lookup(0x2080).get() == source.get());
}

void TestLoadedKernelUnregisterKeepsPinnedTranslatedTarget() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto source = registry.RegisterKernelObject(
      0x2000, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
      nullptr);
  registry.RegisterKernelObject(
      0x3000, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::Translated,
      nullptr);
  source->TargetKernelObject = 0x3000;

  registry.UnregisterLoadedKernelObject(0x2000);

  EXPECT_FALSE(registry.Lookup(0x2000));
  EXPECT_TRUE(registry.Lookup(0x3000) != nullptr);
}

void TestLaunchMetadataUpdate() {
  rocr::amd::hsa::loader::HotSwapKernelRegistry registry;
  auto record = registry.RegisterKernelObject(
      0x4000, "kernel", rocr::amd::hsa::loader::HotSwapKernelKind::LazySource,
      nullptr);

  registry.UpdateLaunchMetadata(0x4000, 32, 16);

  EXPECT_EQ(record->SourceGroupSegmentSize, 32u);
  EXPECT_EQ(record->SourcePrivateSegmentSize, 16u);
}

void TestSegmentSizePatchAddsDynamicDelta() {
  rocr::amd::hsa::loader::HotSwapKernelRecord record;
  record.SourcePrivateSegmentSize = 16;
  record.SourceGroupSegmentSize = 32;
  record.TargetPrivateSegmentSize = 64;
  record.TargetGroupSegmentSize = 128;
  record.TargetGroupSegmentLimit = 256;

  uint32_t patchedPrivate = 0;
  uint32_t patchedGroup = 0;
  std::string failure;
  EXPECT_TRUE(rocr::amd::hsa::loader::ComputeHotSwapPatchedSegmentSizes(
      record, 20, 40, patchedPrivate, patchedGroup, failure));
  EXPECT_EQ(patchedPrivate, 68u);
  EXPECT_EQ(patchedGroup, 136u);
  EXPECT_TRUE(failure.empty());
}

void TestTranslatedDispatchPatchRewritesSourcePacket() {
  rocr::amd::hsa::loader::HotSwapKernelRecord record;
  record.Kind.store(rocr::amd::hsa::loader::HotSwapKernelKind::Translated);
  record.SourcePrivateSegmentSize = 16;
  record.SourceGroupSegmentSize = 32;
  record.TargetKernelObject = 0x5000;
  record.TargetPrivateSegmentSize = 64;
  record.TargetGroupSegmentSize = 128;
  record.TargetGroupSegmentLimit = 256;

  uint64_t kernelObject = 0x1000;
  uint32_t privateSegmentSize = 20;
  uint32_t groupSegmentSize = 40;
  std::string failure;
  EXPECT_TRUE(rocr::amd::hsa::loader::PatchHotSwapTranslatedDispatch(
      record, kernelObject, privateSegmentSize, groupSegmentSize, failure));
  EXPECT_EQ(kernelObject, 0x5000u);
  EXPECT_EQ(privateSegmentSize, 68u);
  EXPECT_EQ(groupSegmentSize, 136u);
  EXPECT_TRUE(failure.empty());
}

void TestSegmentSizePatchRefusesUnderflow() {
  rocr::amd::hsa::loader::HotSwapKernelRecord record;
  record.SourcePrivateSegmentSize = 16;
  record.SourceGroupSegmentSize = 32;
  record.TargetPrivateSegmentSize = 64;
  record.TargetGroupSegmentSize = 128;
  record.TargetGroupSegmentLimit = 256;

  uint32_t patchedPrivate = 0;
  uint32_t patchedGroup = 0;
  std::string failure;
  EXPECT_FALSE(rocr::amd::hsa::loader::ComputeHotSwapPatchedSegmentSizes(
      record, 16, 31, patchedPrivate, patchedGroup, failure));
  EXPECT_TRUE(failure.find("group segment size") != std::string::npos);

  failure.clear();
  EXPECT_FALSE(rocr::amd::hsa::loader::ComputeHotSwapPatchedSegmentSizes(
      record, 15, 32, patchedPrivate, patchedGroup, failure));
  EXPECT_TRUE(failure.find("private segment size") != std::string::npos);
}

void TestSegmentSizePatchRefusesTargetLdsOverflow() {
  rocr::amd::hsa::loader::HotSwapKernelRecord record;
  record.SourcePrivateSegmentSize = 0;
  record.SourceGroupSegmentSize = 32;
  record.TargetPrivateSegmentSize = 0;
  record.TargetGroupSegmentSize = 96;
  record.TargetGroupSegmentLimit = 100;

  uint32_t patchedPrivate = 0;
  uint32_t patchedGroup = 0;
  std::string failure;
  EXPECT_FALSE(rocr::amd::hsa::loader::ComputeHotSwapPatchedSegmentSizes(
      record, 0, 40, patchedPrivate, patchedGroup, failure));
  EXPECT_TRUE(failure.find("target LDS limit") != std::string::npos);
}

void TestPresentationTargetRequiresExplicitTarget() {
  std::string target;
  std::string failure;
  EXPECT_FALSE(rocr::amd::hsa::loader::ResolveHotSwapPresentationTarget(
      true, "", "", "gfx942", target, failure));
  EXPECT_TRUE(target.empty());
  EXPECT_TRUE(failure.find("HSA_HOTSWAP_TARGET") != std::string::npos);

  failure.clear();
  EXPECT_FALSE(rocr::amd::hsa::loader::ResolveHotSwapPresentationTarget(
      true, "", "1", "gfx942", target, failure));
  EXPECT_TRUE(target.empty());
  EXPECT_TRUE(failure.find("HSA_HOTSWAP_TARGET") != std::string::npos);
}

void TestPresentationTargetUsesExplicitTarget() {
  std::string target;
  std::string failure;
  EXPECT_TRUE(rocr::amd::hsa::loader::ResolveHotSwapPresentationTarget(
      true, "gfx942", "", "gfx950", target, failure));
  EXPECT_EQ(target, std::string("gfx942"));
  EXPECT_TRUE(failure.empty());

  EXPECT_TRUE(rocr::amd::hsa::loader::ResolveHotSwapPresentationTarget(
      true, "", "gfx950", "gfx942", target, failure));
  EXPECT_EQ(target, std::string("gfx950"));
  EXPECT_TRUE(failure.empty());
}

void TestNonPresentationTargetCanUseExecutionFallback() {
  std::string target;
  std::string failure;
  EXPECT_TRUE(rocr::amd::hsa::loader::ResolveHotSwapPresentationTarget(
      false, "", "", "gfx942", target, failure));
  EXPECT_EQ(target, std::string("gfx942"));
  EXPECT_TRUE(failure.empty());
}

void TestLazyCacheDirUsesDedicatedOverride() {
  EXPECT_EQ(rocr::amd::hsa::loader::ResolveLazyHotSwapCacheDir(
                "/tmp/lazy", "/tmp/eager"),
            std::string("/tmp/lazy"));
}

void TestLazyCacheDirUsesSharedSubdirectory() {
  EXPECT_EQ(rocr::amd::hsa::loader::ResolveLazyHotSwapCacheDir(
                "", "/tmp/hotswap-cache"),
            std::string("/tmp/hotswap-cache/lazy-v2"));
  EXPECT_EQ(rocr::amd::hsa::loader::ResolveLazyHotSwapCacheDir(
                nullptr, "/tmp/hotswap-cache///"),
            std::string("/tmp/hotswap-cache/lazy-v2"));
}

void TestLazyCacheDirCanRemainDisabled() {
  EXPECT_TRUE(rocr::amd::hsa::loader::ResolveLazyHotSwapCacheDir(
                  nullptr, nullptr).empty());
  EXPECT_TRUE(rocr::amd::hsa::loader::ResolveLazyHotSwapCacheDir(
                  "", "").empty());
}

void TestHotSwapQueueTypeSupport() {
  EXPECT_TRUE(rocr::amd::hsa::loader::IsHotSwapQueueTypeSupported(HSA_QUEUE_TYPE_MULTI));
  EXPECT_FALSE(rocr::amd::hsa::loader::IsHotSwapQueueTypeSupported(HSA_QUEUE_TYPE_SINGLE));
  EXPECT_FALSE(rocr::amd::hsa::loader::IsHotSwapQueueTypeSupported(HSA_QUEUE_TYPE_COOPERATIVE));
}

void TestHotSwapEnvFlagParsing() {
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled(""));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("0"));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("off"));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("FALSE"));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("no"));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("n"));
  EXPECT_FALSE(rocr::hotswap::IsEnvFlagValueEnabled("f"));

  EXPECT_TRUE(rocr::hotswap::IsEnvFlagValueEnabled("1"));
  EXPECT_TRUE(rocr::hotswap::IsEnvFlagValueEnabled("true"));
  EXPECT_TRUE(rocr::hotswap::IsEnvFlagValueEnabled("gfx1250"));
}

}  // namespace

int main() {
  TestPatchPublishedKernelDispatchPacket();
  TestPatchPublishedExtKernelDispatchPacket();
  TestPatchScalesDispatchExtent();
  TestPatchLeavesUnscaledDispatchExtent();
  TestPatchSkipsNonKernelPacket();
  TestDoorbellPatchRangeMonotonic();
  TestDoorbellPatchRangeSkipsReplayedDoorbell();
  TestDoorbellPatchRangeRefusesAliasingRange();
  TestPatchFailureAborts();
  TestRocrBlitRegistryRangeAndLifetime();
  TestInvalidRocrBlitRegistrationIsNotAllowed();
  TestHotSwapKernelRecordNameDerivation();
  TestDuplicateKernelObjectRegistrationAborts();
  TestInvalidRocrBlitDuplicateAddressAborts();
  TestRocrBlitUnregisterDoesNotRemoveOtherKinds();
  TestRocrBlitRangeDoesNotHideExactKernelObject();
  TestLoadedKernelUnregisterKeepsPinnedTranslatedTarget();
  TestLaunchMetadataUpdate();
  TestSegmentSizePatchAddsDynamicDelta();
  TestTranslatedDispatchPatchRewritesSourcePacket();
  TestSegmentSizePatchRefusesUnderflow();
  TestSegmentSizePatchRefusesTargetLdsOverflow();
  TestPresentationTargetRequiresExplicitTarget();
  TestPresentationTargetUsesExplicitTarget();
  TestNonPresentationTargetCanUseExecutionFallback();
  TestLazyCacheDirUsesDedicatedOverride();
  TestLazyCacheDirUsesSharedSubdirectory();
  TestLazyCacheDirCanRemainDisabled();
  TestHotSwapQueueTypeSupport();
  TestHotSwapEnvFlagParsing();

  if (Failures != 0) {
    std::cerr << Failures << " HotSwap runtime unit test checks failed\n";
    return EXIT_FAILURE;
  }
  std::cout << "HotSwap runtime unit tests passed\n";
  return EXIT_SUCCESS;
}
