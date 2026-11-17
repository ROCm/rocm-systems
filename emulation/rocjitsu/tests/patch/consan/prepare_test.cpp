// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/isa/isa_family.h"
#include "target_fixture.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/consan/prepare.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/register_set.h"
#include "rocjitsu/isa/target_registry.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu::consan {
namespace {

#ifdef CONSAN_MULTI_ENTRY_FIXTURE
class ConSanPrepareImage : public ::testing::Test {
protected:
  void SetUp() override {
    std::ifstream file(CONSAN_MULTI_ENTRY_FIXTURE, std::ios::binary);
    original_ = {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    ASSERT_FALSE(original_.empty());
  }
  std::vector<uint8_t> original_;
  InputError error_;
};

TEST_F(ConSanPrepareImage, ExecutionTraceOwnsBoundariesAndADistinctSelectionIdentity) {
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> trace = prepare_image(
      original_, *target_profile("gfx1250"), selected, error_, ScopeKind::ExecutionTrace);
  ASSERT_TRUE(trace.succeeded()) << error_.message;
  const FailureOr<PreparedImageArtifact> lds =
      prepare_image(original_, *target_profile("gfx1250"), selected, error_);
  ASSERT_TRUE(lds.succeeded()) << error_.message;
  EXPECT_NE(trace.value().facts.selection_profile, lds.value().facts.selection_profile);
  EXPECT_EQ(trace.value().facts.entries.size(), lds.value().facts.entries.size());
  for (const ImageEntryFacts &entry : trace.value().facts.entries) {
    if (!entry.selected)
      continue;
    const Manifest &manifest = *entry.selected;
    EXPECT_EQ(manifest.scope, ScopeKind::ExecutionTrace);
    EXPECT_EQ(manifest.options, profile_options_digest(ScopeKind::ExecutionTrace));
    EXPECT_EQ(manifest.entry.workgroup_id_scalar, kWorkgroupIdTtmp9Selector);
    EXPECT_TRUE(manifest.sites.empty());
    EXPECT_TRUE(manifest.barriers.empty());
    ASSERT_EQ(manifest.boundaries.size(), 2u);
    EXPECT_EQ(manifest.boundaries.front().kind, BoundaryKind::Entry);
    EXPECT_EQ(manifest.boundaries.back().kind, BoundaryKind::Exit);
    const FailureOr<LaunchLayout> layout =
        plan_launch(manifest, {{512, 1, 1}, {32, 1, 1}}, manifest.entry.original_group_bytes, 0);
    ASSERT_TRUE(layout.succeeded());
    EXPECT_EQ(layout.value().wave_count, 16u);
    EXPECT_EQ(layout.value().row_stride, kRowHeaderBytes);
    const FailureOr<LaunchLayout> maximum = plan_launch(manifest, {{512 * 32, 1, 1}, {32, 1, 1}},
                                                        manifest.entry.original_group_bytes, 0);
    ASSERT_TRUE(maximum.succeeded());
    EXPECT_EQ(maximum.value().workgroup_count, 512u);
    EXPECT_EQ(maximum.value().wave_count, 512u);
    EXPECT_TRUE(plan_launch(manifest, {{513 * 32, 1, 1}, {32, 1, 1}},
                            manifest.entry.original_group_bytes, 0)
                    .failed());
    const Manifest *lds_manifest = selected_manifest(lds.value().facts, entry.key);
    ASSERT_NE(lds_manifest, nullptr);
    EXPECT_TRUE(plan_launch(*lds_manifest, {{17 * 32, 1, 1}, {32, 1, 1}},
                            lds_manifest->entry.original_group_bytes, 0)
                    .failed());
  }
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(trace.value().facts);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), trace.value().facts);
}

TEST_F(ConSanPrepareImage, ExecutionTraceAdmitsAnEntryWithoutLdsSites) {
  const TargetProfile &target = *target_profile("gfx1250");
  const FailureOr<InputProgram> source =
      build_image_entry_program(original_, target, "mk_a", error_);
  ASSERT_TRUE(source.succeeded()) << error_.message;
  ASSERT_FALSE(source.value().accesses.empty());
  constexpr util::FailureOr<uint32_t> nop = build_s_nop(0, IsaFamily::Cdna5);
  static_assert(nop.succeeded());
  for (const NativeLdsAccess &access : source.value().accesses) {
    const size_t offset = source.value().text_file_offset + access.original_offset;
    ASSERT_LE(offset + 2 * sizeof(uint32_t), original_.size());
    std::memcpy(original_.data() + offset, &nop.value(), sizeof(uint32_t));
    std::memcpy(original_.data() + offset + sizeof(uint32_t), &nop.value(), sizeof(uint32_t));
  }
  EXPECT_TRUE(build_image_entry_program(original_, target, "mk_a", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedAccess);
  const std::array<std::string_view, 1> selected = {"mk_a"};
  const FailureOr<PreparedImageArtifact> trace =
      prepare_image(original_, target, selected, error_, ScopeKind::ExecutionTrace);
  ASSERT_TRUE(trace.succeeded()) << error_.message;
  const Manifest *manifest =
      selected_manifest(trace.value().facts, source.value().kernel.descriptor_file_offset);
  ASSERT_NE(manifest, nullptr);
  EXPECT_TRUE(manifest->sites.empty());
  EXPECT_TRUE(manifest->barriers.empty());
  EXPECT_EQ(manifest->boundaries.size(), 2u);
}

struct TraceBodyHazard {
  uint32_t word;
  bool at_epilogue;
  bool native_allowed;
  const char *mnemonic;
  const char *diagnostic;
};

class ConSanTraceBodyQualification : public ConSanPrepareImage,
                                     public ::testing::WithParamInterface<TraceBodyHazard> {};

TEST_P(ConSanTraceBodyQualification, RejectsSelectedHazardAndAppliesNativeSiblingQualification) {
  const TargetProfile &target = *target_profile("gfx1250");
  const FailureOr<InputProgram> baseline =
      build_image_entry_program(original_, target, "mk_a", error_, {}, ScopeKind::ExecutionTrace);
  ASSERT_TRUE(baseline.succeeded()) << error_.message;
  ASSERT_EQ(baseline.value().exit_offsets.size(), 1u);
  const uint64_t begin = baseline.value().kernel.entry_text_offset;
  const uint64_t end = baseline.value().exit_offsets.front();
  const TraceBodyHazard hazard = GetParam();
  const uint64_t offset = hazard.at_epilogue ? end - 4 : begin + 0x10;
  ASSERT_GT(offset, begin);
  ASSERT_LT(offset, end);
  const size_t file_offset = baseline.value().text_file_offset + offset;
  std::memcpy(original_.data() + file_offset, &hazard.word, sizeof(hazard.word));
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target.public_id);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint32_t, 1> words = {hazard.word};
  const DecodeResult decoded = decoder->decode_window(words);
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value()->mnemonic(), hazard.mnemonic);
  if (hazard.at_epilogue) {
    EXPECT_EQ(decoded.value()->raw_encoding()[0] & 0x7f, 3u);
    // This is the valid release/endpgm epilogue, not an arbitrary body release.
    uint32_t actual_exit = 0;
    std::memcpy(&actual_exit, original_.data() + baseline.value().text_file_offset + end, 4);
    const std::array<uint32_t, 1> actual_words = {actual_exit};
    const DecodeResult exit = decoder->decode_window(actual_words);
    ASSERT_TRUE(exit.succeeded());
    EXPECT_EQ(exit.value()->mnemonic(), "s_endpgm");
  }
  EXPECT_TRUE(
      build_image_entry_program(original_, target, "mk_a", error_, {}, ScopeKind::ExecutionTrace)
          .failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
  EXPECT_EQ(error_.text_offset, offset);
  EXPECT_EQ(error_.message, hazard.diagnostic);
  const std::array<std::string_view, 1> selected = {"mk_a"};
  EXPECT_TRUE(
      prepare_image(original_, target, selected, error_, ScopeKind::ExecutionTrace).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedFlow);
  EXPECT_EQ(error_.text_offset, offset);

  const std::array<std::string_view, 1> sibling = {"mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original_, target, sibling, error_, ScopeKind::ExecutionTrace);
  if (!hazard.native_allowed) {
    ASSERT_TRUE(artifact.failed());
    EXPECT_EQ(error_.code, InputErrorCode::UnsupportedInstruction);
    EXPECT_EQ(error_.text_offset, offset);
    EXPECT_EQ(error_.message,
              "native entry 'mk_a' reads PC; PC-relative data references are not preserved");
    return;
  }
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  AmdGpuCodeObject object(artifact.value().image.data(), artifact.value().image.size());
  bool checked_text = false;
  for (const std::unique_ptr<Section> &section : object.all_sections()) {
    if (section->name() != ".text")
      continue;
    ASSERT_GE(section->size(), end + 4);
    EXPECT_EQ(std::memcmp(section->data() + begin,
                          original_.data() + baseline.value().text_file_offset + begin,
                          end + 4 - begin),
              0);
    checked_text = true;
  }
  EXPECT_TRUE(checked_text);
}

INSTANTIATE_TEST_SUITE_P(
    OriginalBody, ConSanTraceBodyQualification,
    ::testing::Values(
        TraceBodyHazard{0xbe8a4700u, false, false, "s_get_pc_i64",
                        "execution tracing does not admit guest code-address introspection"},
        TraceBodyHazard{0xbfb60003u, true, true, "s_sendmsg",
                        "execution tracing does not admit guest VGPR deallocation"}));

TEST_F(ConSanPrepareImage, WholeImageHasIndependentEntryPlansAndNativeSibling) {
  const std::array<std::string_view, 2> entries = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original_, *target_profile("gfx1250"), entries, error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const ImageFacts &facts = artifact.value().facts;
  ASSERT_EQ(facts.entries.size(), 3u);
  const ImageEntryFacts &first = facts.entries[0];
  const ImageEntryFacts &second = facts.entries[1];
  const ImageEntryFacts &native = facts.entries[2];
  ASSERT_TRUE(first.selected);
  ASSERT_TRUE(second.selected);
  EXPECT_FALSE(native.selected);
  EXPECT_EQ(first.binding.name, "mk_a");
  EXPECT_EQ(second.binding.name, "mk_b");
  EXPECT_EQ(native.binding.name, "mk_c");
  EXPECT_EQ(first.binding.original_kernarg_bytes, 16u);
  EXPECT_EQ(second.binding.original_kernarg_bytes, 32u);
  EXPECT_NE(first.binding.prepared_sgprs, second.binding.prepared_sgprs);
  EXPECT_NE(first.binding.prepared_vgprs, second.binding.prepared_vgprs);
  EXPECT_NE(first.binding.prepared_entry_offset, second.binding.prepared_entry_offset);
  EXPECT_EQ(first.selected->sites.size(), 2u);
  EXPECT_EQ(second.selected->sites.size(), 2u);
  EXPECT_EQ(first.selected->entry.id, 0u);
  EXPECT_EQ(second.selected->entry.id, 0u);
  EXPECT_NE(first.selected->sites[0].original_offset, second.selected->sites[0].original_offset);
  EXPECT_EQ(native.binding.original_kernarg_bytes, 24u);
  EXPECT_EQ(native.binding.original_group_bytes, native.binding.prepared_group_bytes);
  EXPECT_EQ(native.original_sgprs, native.binding.prepared_sgprs);
  EXPECT_EQ(native.original_vgprs, native.binding.prepared_vgprs);
  util::StringDiagnostic contract_error;
  EXPECT_TRUE(
      verify_artifact_images(facts, original_, artifact.value().image, contract_error.emitter())
          .succeeded())
      << contract_error.message();
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(facts);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), facts);
  AmdGpuCodeObject object(artifact.value().image.data(), artifact.value().image.size());
  bool extensions_found = false;
  for (const std::unique_ptr<Section> &section : object.all_sections()) {
    if (section->name() != kKernargExtensionMetadataSectionName) {
      continue;
    }
    const std::optional<std::vector<KernargExtensionMetadata>> records =
        parse_kernarg_extension_metadata(
            {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
    ASSERT_TRUE(records);
    ASSERT_EQ(records->size(), 2u);
    EXPECT_EQ((*records)[0].kernel_name, "mk_a");
    EXPECT_EQ((*records)[1].kernel_name, "mk_b");
    EXPECT_EQ((*records)[0].original_kernarg_size, 16u);
    EXPECT_EQ((*records)[1].original_kernarg_size, 32u);
    extensions_found = true;
  }
  EXPECT_TRUE(extensions_found);
}

TEST_F(ConSanPrepareImage, PreservesNativeVgprsAboveSelectedLimit) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() /
      "multi-entry-native-wide-vgprs.hsaco";
  const std::vector<uint8_t> baseline = test::read_fixture(path);
  ASSERT_FALSE(baseline.empty());
  for (const uint32_t count : {384u, 1024u}) {
    SCOPED_TRACE(count);
    std::vector<uint8_t> original = baseline;
    const AmdGpuCodeObject object(original.data(), original.size());
    ASSERT_TRUE(object.is_valid());
    ASSERT_EQ(object.text_sections().size(), 1u);
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original, text.sectionOffset(), text.size());
    const std::vector<KernelDescriptorInfo>::const_iterator sibling =
        std::ranges::find(kernels, std::string("mk_c"), &KernelDescriptorInfo::kernel_name);
    ASSERT_NE(sibling, kernels.end());
    KernelDescriptorInfo kernel = *sibling;
    RJ_AMDHSA_BITS_SET(kernel.descriptor.compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, count / 16 - 1);
    std::memcpy(original.data() + kernel.descriptor_file_offset, &kernel.descriptor,
                sizeof(kernel.descriptor));
    ASSERT_TRUE(
        amdhsa::update_resource_metadata(original, kernel, kernel.descriptor, std::nullopt, count)
            .succeeded());
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(original, kernel, nullptr, MetadataPolicy::Native);
    ASSERT_TRUE(metadata.succeeded());
    EXPECT_EQ(metadata.value().vgprs, count);
    ASSERT_TRUE(inspect_native_image(original, "gfx1250").succeeded());
    const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, *target_profile("gfx1250"), selected, error_);
    EXPECT_TRUE(artifact.succeeded()) << error_.message;
    if (artifact.failed())
      continue;
    const ImageEntryFacts &native = artifact.value().facts.entries.back();
    ASSERT_EQ(native.binding.name, "mk_c");
    EXPECT_EQ(native.disposition, EntryDisposition::Native);
    EXPECT_EQ(native.original_vgprs, count);
    EXPECT_EQ(native.binding.prepared_vgprs, count);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                       diagnostic.emitter())
                    .succeeded())
        << diagnostic.message();
    const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(artifact.value().facts);
    ASSERT_TRUE(encoded.succeeded());
    const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), artifact.value().facts);

    // Counts can fit the target while exceeding an individual descriptor.
    KernelDescriptorInfo undersized = kernel;
    RJ_AMDHSA_BITS_SET(undersized.descriptor.compute_pgm_rsrc1,
                       amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, count / 16 - 2);
    std::vector<uint8_t> invalid_original = original;
    std::memcpy(invalid_original.data() + kernel.descriptor_file_offset, &undersized.descriptor,
                sizeof(undersized.descriptor));
    EXPECT_TRUE(
        inspect_metadata(invalid_original, undersized, nullptr, MetadataPolicy::Native).failed());
    EXPECT_TRUE(inspect_native_image(invalid_original, "gfx1250").failed());
    EXPECT_TRUE(
        prepare_image(invalid_original, *target_profile("gfx1250"), selected, error_).failed());

    std::vector<uint8_t> changed = artifact.value().image;
    // Preserve the prepared entry address; only shrink its register allocation.
    const size_t rsrc1_offset = native.binding.prepared_descriptor_offset +
                                offsetof(amdhsa::kernel_descriptor_t, compute_pgm_rsrc1);
    std::memcpy(changed.data() + rsrc1_offset, &undersized.descriptor.compute_pgm_rsrc1,
                sizeof(undersized.descriptor.compute_pgm_rsrc1));
    ImageFacts facts = artifact.value().facts;
    facts.prepared.sha256 = digest(changed);
    for (ImageEntryFacts &entry : facts.entries)
      if (entry.selected)
        entry.selected->prepared = facts.prepared;
    ASSERT_TRUE(validate_image_facts(facts).succeeded());
    EXPECT_TRUE(verify_artifact_images(facts, original, changed, diagnostic.emitter()).failed());
    EXPECT_NE(diagnostic.message().find("metadata"), std::string::npos);
  }
}

TEST_F(ConSanPrepareImage, SelectedCodeCanBranchPastNativeEntries) {
  const std::vector<uint8_t> baseline = original_;
  for (const size_t relocated : {0u, 1u}) {
    SCOPED_TRACE(relocated);
    original_ = baseline;
    const AmdGpuCodeObject object(original_.data(), original_.size());
    ASSERT_TRUE(object.is_valid());
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original_, text.sectionOffset(), text.size());
    ASSERT_EQ(kernels.size(), 3u);
    const std::span<const uint8_t> source =
        std::span(original_).subspan(text.sectionOffset(), text.size());
    std::vector<uint8_t> code(source.begin(), source.end());
    const uint64_t entry = kernels[relocated].entry_text_offset;
    const uint64_t body_bytes = kernels[relocated + 1].entry_text_offset - entry;
    code.insert(code.end(), source.begin() + entry, source.begin() + entry + body_bytes);
    const std::optional<int16_t> displacement = compute_sopp_branch_simm16(entry, source.size());
    ASSERT_TRUE(displacement);
    const FailureOr<uint32_t> branch = build_s_branch(*displacement, IsaFamily::Cdna5);
    const FailureOr<uint32_t> nop = build_s_nop(0, IsaFamily::Cdna5);
    ASSERT_TRUE(branch.succeeded() && nop.succeeded());
    std::memcpy(code.data() + entry, &branch.value(), sizeof(uint32_t));
    // The original first SMEM instruction occupies eight bytes; leave a valid
    // full-section decode after replacing its first word with the branch.
    std::memcpy(code.data() + entry + 4, &nop.value(), sizeof(uint32_t));
    CodeObjectPatcher patcher(object);
    ASSERT_TRUE(patcher.replace_text(code));
    original_ = patcher.emit();
    const std::array<std::string_view, 1> selected = {"mk_a"};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original_, *target_profile("gfx1250"), selected, error_);
    ASSERT_TRUE(artifact.succeeded()) << error_.message;
    util::StringDiagnostic contract_error;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original_, artifact.value().image,
                                       contract_error.emitter())
                    .succeeded())
        << contract_error.message();
    if (relocated == 1) {
      ImageFacts facts = artifact.value().facts;
      ASSERT_GE(facts.entries[1].native_code.size(), 2u);
      const NativeCodeRange &island = facts.entries[1].native_code.back();
      EXPECT_GT(island.original_offset, facts.entries[2].binding.original_entry_offset);
      std::vector<uint8_t> changed = artifact.value().image;
      const AmdGpuCodeObject prepared(changed.data(), changed.size());
      changed[prepared.text_sections().front()->sectionOffset() + island.prepared_offset] ^= 1;
      facts.prepared.sha256 = digest(changed);
      for (ImageEntryFacts &row : facts.entries) {
        if (row.selected) {
          row.selected->prepared = facts.prepared;
        }
      }
      EXPECT_TRUE(
          verify_artifact_images(facts, original_, changed, contract_error.emitter()).failed());
      EXPECT_EQ(contract_error.message(), "native entry body changed");
    }
  }
}

TEST_F(ConSanPrepareImage, SelectionOrderIsCanonicalAndSingleSubsetPreservesOtherEntries) {
  const std::array<std::string_view, 2> forward = {"mk_a", "mk_b"};
  const std::array<std::string_view, 2> reverse = {"mk_b", "mk_a"};
  const FailureOr<PreparedImageArtifact> first =
      prepare_image(original_, *target_profile("gfx1250"), forward, error_);
  ASSERT_TRUE(first.succeeded()) << error_.message;
  const FailureOr<PreparedImageArtifact> second =
      prepare_image(original_, *target_profile("gfx1250"), reverse, error_);
  ASSERT_TRUE(second.succeeded()) << error_.message;
  EXPECT_EQ(first.value().facts, second.value().facts);
  EXPECT_EQ(first.value().image, second.value().image);
  for (const std::string_view name : {"mk_a", "mk_b"}) {
    const std::array<std::string_view, 1> selected = {name};
    const FailureOr<PreparedImageArtifact> subset =
        prepare_image(original_, *target_profile("gfx1250"), selected, error_);
    ASSERT_TRUE(subset.succeeded()) << name << ": " << error_.message;
    EXPECT_NE(subset.value().facts.selection_profile, first.value().facts.selection_profile);
    for (const ImageEntryFacts &entry : subset.value().facts.entries) {
      EXPECT_EQ(entry.disposition == EntryDisposition::Selected, entry.binding.name == name);
    }
  }
}

TEST_F(ConSanPrepareImage, InvalidNativeNameCannotPublishFactsThatJsonAnalysisRejects) {
  constexpr std::array<uint8_t, 4> name = {'m', 'k', '_', 'c'};
  size_t replacements = 0;
  for (size_t offset = 0; offset + name.size() <= original_.size(); ++offset) {
    if (std::equal(name.begin(), name.end(), original_.begin() + offset)) {
      original_[offset] = 0xff;
      ++replacements;
    }
  }
  ASSERT_GT(replacements, 0u);
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original_, *target_profile("gfx1250"), selected, error_);
  EXPECT_TRUE(artifact.failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  EXPECT_EQ(error_.message, "entry name is not valid UTF-8 for JSON output");
}

TEST_F(ConSanPrepareImage, PreservesCompilerHiddenArgumentsAndNativePrivateAbi) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() / "multi-entry-abi.hsaco";
  const std::vector<uint8_t> original = test::read_fixture(path);
  ASSERT_FALSE(original.empty());
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original, *target_profile("gfx1250"), selected, error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const ImageEntryFacts &native = artifact.value().facts.entries.back();
  EXPECT_EQ(native.disposition, EntryDisposition::Native);
  EXPECT_EQ(native.binding.kernarg_alignment, 32u);
  EXPECT_EQ(native.binding.original_private_bytes, 68u);
  EXPECT_EQ(native.binding.prepared_private_bytes, 68u);
  EXPECT_EQ(native.binding.required_workgroup_size, (std::array<uint32_t, 3>{32, 2, 1}));
  const Manifest &manifest = *artifact.value().facts.entries.front().selected;
  for (const ImageEntryFacts &entry : artifact.value().facts.entries)
    EXPECT_EQ(entry.binding.workgroup_id_scalar, kWorkgroupIdTtmp9Selector);
  EXPECT_EQ(manifest.entry.required_workgroup_size, (std::array<uint32_t, 3>{64, 1, 1}));
  EXPECT_TRUE(plan_launch(manifest, {{64, 1, 1}, {64, 1, 1}}, 8, 0).succeeded());
  EXPECT_TRUE(plan_launch(manifest, {{32, 1, 1}, {32, 1, 1}}, 8, 0).failed());
  util::StringDiagnostic contract_error;
  EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                     contract_error.emitter())
                  .succeeded())
      << contract_error.message();
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(artifact.value().facts);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), artifact.value().facts);
}

TEST_F(ConSanPrepareImage, PreservesFourByteAlignedNativeScalarSibling) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() /
      "multi-entry-native-scalar.hsaco";
  const std::vector<uint8_t> original = test::read_fixture(path);
  ASSERT_FALSE(original.empty());
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original, *target_profile("gfx1250"), selected, error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const ImageEntryFacts &native = artifact.value().facts.entries.back();
  EXPECT_EQ(native.binding.name, "mk_c");
  EXPECT_EQ(native.disposition, EntryDisposition::Native);
  EXPECT_FALSE(native.selected);
  EXPECT_EQ(native.binding.original_kernarg_bytes, 4u);
  EXPECT_EQ(native.binding.kernarg_alignment, 4u);
  EXPECT_EQ(native.binding.required_workgroup_size, (std::array<uint32_t, 3>{32, 2, 1}));
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                     diagnostic.emitter())
                  .succeeded())
      << diagnostic.message();
  const FailureOr<std::vector<uint8_t>> encoded =
      encode_image_facts(artifact.value().facts, diagnostic.emitter());
  ASSERT_TRUE(encoded.succeeded()) << diagnostic.message();
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), diagnostic.emitter());
  ASSERT_TRUE(decoded.succeeded()) << diagnostic.message();
  EXPECT_EQ(decoded.value(), artifact.value().facts);
}

TEST_F(ConSanPrepareImage, SelectedMultidimensionalRequirementsCannotPublishPreparedFacts) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() / "multi-entry-abi.hsaco";
  const std::vector<uint8_t> original = test::read_fixture(path);
  ASSERT_FALSE(original.empty());
  for (const size_t dimension : {1u, 2u}) {
    SCOPED_TRACE(dimension);
    std::vector<uint8_t> image = original;
    const FailureOr<amdhsa::MetadataDocument> metadata = amdhsa::MetadataDocument::parse(image);
    ASSERT_TRUE(metadata.succeeded());
    const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
    ASSERT_NE(entries, nullptr);
    const amdhsa::MetadataValue *shape = entries->array.front().field(".reqd_workgroup_size");
    ASSERT_NE(shape, nullptr);
    ASSERT_EQ(shape->array.size(), 3u);
    const size_t payload_offset = metadata.value().payload().data() - image.data();
    ASSERT_EQ(shape->array[0].integer_bytes, 0u);
    ASSERT_EQ(shape->array[dimension].integer_bytes, 0u);
    image[payload_offset + shape->array[0].offset] = 32;
    image[payload_offset + shape->array[dimension].offset] = 2;
    const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(image, *target_profile("gfx1250"), selected, error_);
    EXPECT_TRUE(artifact.failed());
    EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
    EXPECT_EQ(error_.message, "selected entry requires a 1D workgroup shape");
  }
}

TEST_F(ConSanPrepareImage, SelectedDynamicStackCannotPublishPreparedFacts) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() / "multi-entry-abi.hsaco";
  std::vector<uint8_t> image = test::read_fixture(path);
  ASSERT_FALSE(image.empty());
  const FailureOr<amdhsa::MetadataDocument> metadata = amdhsa::MetadataDocument::parse(image);
  ASSERT_TRUE(metadata.succeeded());
  const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
  ASSERT_NE(entries, nullptr);
  ASSERT_EQ(entries->array.size(), 3u);
  const amdhsa::MetadataValue *dynamic_stack = entries->array.front().field(".uses_dynamic_stack");
  ASSERT_NE(dynamic_stack, nullptr);
  ASSERT_EQ(dynamic_stack->kind, amdhsa::MetadataValue::Kind::Boolean);
  const size_t payload_offset = metadata.value().payload().data() - image.data();
  const size_t flag_offset = payload_offset + dynamic_stack->offset;
  ASSERT_LT(flag_offset, image.size());
  ASSERT_EQ(image[flag_offset], 0xc2u);
  image[flag_offset] = 0xc3u;
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(image, *target_profile("gfx1250"), selected, error_);
  EXPECT_TRUE(artifact.failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  EXPECT_EQ(error_.message, "selected entry requires a statically bounded private frame");
}

TEST_F(ConSanPrepareImage, CompilerAbiCompatibilityRetainsStructuralRejections) {
  const std::filesystem::path path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_FIXTURE).parent_path() / "multi-entry-abi.hsaco";
  const std::vector<uint8_t> original = test::read_fixture(path);
  ASSERT_FALSE(original.empty());
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  for (const std::pair<std::string_view, std::string_view> mutation :
       {std::pair{"amdgpu12.50", "amdgpu12.51"},
        {"hidden_block_count_x", "hidden_block_count_q"},
        {"constant", "private_"}}) {
    std::vector<uint8_t> image = original;
    const std::vector<uint8_t>::iterator found =
        std::search(image.begin(), image.end(), mutation.first.begin(), mutation.first.end());
    ASSERT_NE(found, image.end());
    ASSERT_EQ(mutation.first.size(), mutation.second.size());
    std::copy(mutation.second.begin(), mutation.second.end(), found);
    EXPECT_TRUE(prepare_image(image, *target_profile("gfx1250"), selected, error_).failed());
    EXPECT_NE(error_.message.find("metadata"), std::string::npos);
  }
  for (const uint32_t variant : {0u, 1u, 2u, 3u}) {
    SCOPED_TRACE(variant);
    std::vector<uint8_t> image = original;
    const FailureOr<amdhsa::MetadataDocument> metadata = amdhsa::MetadataDocument::parse(image);
    ASSERT_TRUE(metadata.succeeded());
    const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
    ASSERT_NE(entries, nullptr);
    ASSERT_EQ(entries->array.size(), 3u);
    const amdhsa::MetadataValue &first = entries->array.front();
    const amdhsa::MetadataValue &native = entries->array.back();
    const amdhsa::MetadataValue *number = nullptr;
    uint8_t replacement = 0;
    if (variant == 0) {
      number = native.field(".kernarg_segment_align");
      replacement = 3;
    } else if (variant == 1) {
      number = &first.field(".reqd_workgroup_size")->array.front();
    } else if (variant == 2) {
      number = &native.field(".reqd_workgroup_size")->array.front();
      replacement = 64; // 64 * 2 exceeds the declared flat bound of 64.
    } else {
      number = first.field(".args")->array[1].field(".size");
      replacement = 2; // hidden_block_count_x is a four-byte ABI scalar.
    }
    ASSERT_NE(number, nullptr);
    ASSERT_EQ(number->kind, amdhsa::MetadataValue::Kind::Integer);
    ASSERT_EQ(number->integer_bytes, 0u);
    const size_t payload_offset = metadata.value().payload().data() - image.data();
    image[payload_offset + number->offset] = replacement;
    EXPECT_TRUE(prepare_image(image, *target_profile("gfx1250"), selected, error_).failed());
    EXPECT_NE(error_.message.find("metadata"), std::string::npos);
  }
}

TEST_F(ConSanPrepareImage, InvalidOrIncompleteSelectionPublishesNoResult) {
  const TargetProfile &target = *target_profile("gfx1250");
  EXPECT_TRUE(prepare_image(original_, target, {}, error_).failed());
  const std::array<std::string_view, 2> duplicate = {"mk_a", "mk_a"};
  EXPECT_TRUE(prepare_image(original_, target, duplicate, error_).failed());
  const std::array<std::string_view, 2> missing = {"mk_a", "missing"};
  EXPECT_TRUE(prepare_image(original_, target, missing, error_).failed());
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  Elf64_Ehdr header = {};
  std::memcpy(&header, original_.data(), sizeof(header));
  for (size_t index = 0; index < header.e_phnum; ++index) {
    const size_t offset = header.e_phoff + index * sizeof(Elf64_Phdr);
    Elf64_Phdr segment = {};
    std::memcpy(&segment, original_.data() + offset, sizeof(segment));
    if ((segment.p_flags & PF_X) != 0) {
      segment.p_flags &= ~PF_X;
      std::memcpy(original_.data() + offset, &segment, sizeof(segment));
    }
  }
  EXPECT_TRUE(prepare_image(original_, target, selected, error_).failed());
}
#endif

#ifdef CONSAN_FIXTURE_DIR
TEST(ConSanPrepareImageCdna, NativeSgprMetadataIncludesReservedRegisters) {
  for (const char *name : {"gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    const TargetProfile &target = *target_profile(name);
    const std::vector<uint8_t> original = test::read_fixture(
        test::target_fixture_path({.target = name, .wave_size = 64}, "cdna-multi-0"));
    ASSERT_FALSE(original.empty());
    InputError error;
    const std::array<std::string_view, 2> both = {"sel", "sib"};
    ASSERT_TRUE(prepare_image(original, target, both, error).succeeded()) << error.message;
    const AmdGpuCodeObject object(original.data(), original.size());
    ASSERT_TRUE(object.is_valid());
    ASSERT_EQ(object.text_sections().size(), 1u);
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original, text.sectionOffset(), text.size());
    const std::vector<KernelDescriptorInfo>::const_iterator sibling =
        std::ranges::find(kernels, std::string("sib"), &KernelDescriptorInfo::kernel_name);
    ASSERT_NE(sibling, kernels.end());
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(original, *sibling, nullptr, MetadataPolicy::Native);
    ASSERT_TRUE(metadata.succeeded());
    const uint32_t allocated =
        (RJ_AMDHSA_BITS_GET(sibling->descriptor.compute_pgm_rsrc1,
                            amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
         1) *
        8;
    EXPECT_EQ(metadata.value().sgprs, 11u);
    EXPECT_LE(metadata.value().sgprs, allocated);
    EXPECT_GT(metadata.value().sgprs, ordinary_sgpr_bound(target.arch, allocated));
    const std::array<std::string_view, 1> selected = {"sel"};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error);
    EXPECT_TRUE(artifact.succeeded()) << error.message;
    if (artifact.failed())
      continue;
    const ImageEntryFacts *native =
        find_image_entry(artifact.value().facts, sibling->descriptor_file_offset);
    ASSERT_NE(native, nullptr);
    EXPECT_EQ(native->disposition, EntryDisposition::Native);
    EXPECT_EQ(native->binding.prepared_sgprs, metadata.value().sgprs);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                       diagnostic.emitter())
                    .succeeded())
        << diagnostic.message();
    const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(artifact.value().facts);
    ASSERT_TRUE(encoded.succeeded());
    const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), artifact.value().facts);
  }
}

TEST(ConSanPrepareImageCdna, NativeVgprMetadataIncludesAccumulatorRegisters) {
  for (const char *name : {"gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    const TargetProfile &target = *target_profile(name);
    const std::vector<uint8_t> original = test::read_fixture(
        test::target_fixture_path({.target = name, .wave_size = 64}, "cdna-multi-1"));
    ASSERT_FALSE(original.empty());
    const AmdGpuCodeObject object(original.data(), original.size());
    ASSERT_TRUE(object.is_valid());
    ASSERT_EQ(object.text_sections().size(), 1u);
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original, text.sectionOffset(), text.size());
    const std::vector<KernelDescriptorInfo>::const_iterator sibling =
        std::ranges::find(kernels, std::string("sib"), &KernelDescriptorInfo::kernel_name);
    ASSERT_NE(sibling, kernels.end());
    const uint32_t accumulator_base =
        (RJ_AMDHSA_BITS_GET(sibling->descriptor.compute_pgm_rsrc3,
                            amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
         1) *
        4;
    EXPECT_EQ(accumulator_base, 4u);
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(original, *sibling, nullptr, MetadataPolicy::Native);
    EXPECT_TRUE(metadata.succeeded());
    EXPECT_TRUE(inspect_metadata(original, *sibling).failed());
    InputError error;
    const std::array<std::string_view, 1> selected = {"sel"};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error);
    EXPECT_TRUE(artifact.succeeded()) << error.message;
    if (artifact.failed() || metadata.failed())
      continue;
    EXPECT_EQ(metadata.value().vgprs, 5u);
    EXPECT_GT(metadata.value().vgprs, accumulator_base);
    const ImageEntryFacts *native =
        find_image_entry(artifact.value().facts, sibling->descriptor_file_offset);
    ASSERT_NE(native, nullptr);
    EXPECT_EQ(native->binding.prepared_vgprs, metadata.value().vgprs);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                       diagnostic.emitter())
                    .succeeded())
        << diagnostic.message();
  }
}

TEST(ConSanPrepareImageCdna, NativeUnifiedVgprMetadataUsesFullDescriptorCapacity) {
  for (const char *name : {"gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    const TargetProfile &target = *target_profile(name);
    const std::vector<uint8_t> original = test::read_fixture(
        test::target_fixture_path({.target = name, .wave_size = 64}, "cdna-multi-2"));
    ASSERT_FALSE(original.empty());
    const AmdGpuCodeObject object(original.data(), original.size());
    ASSERT_TRUE(object.is_valid());
    ASSERT_EQ(object.text_sections().size(), 1u);
    const Section &text = *object.text_sections().front();
    const std::vector<KernelDescriptorInfo> kernels =
        scan_kernel_descriptors(original, text.sectionOffset(), text.size());
    const std::vector<KernelDescriptorInfo>::const_iterator sibling =
        std::ranges::find(kernels, std::string("sib"), &KernelDescriptorInfo::kernel_name);
    ASSERT_NE(sibling, kernels.end());
    const uint32_t accumulator_base =
        (RJ_AMDHSA_BITS_GET(sibling->descriptor.compute_pgm_rsrc3,
                            amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
         1) *
        4;
    const uint32_t allocated =
        (RJ_AMDHSA_BITS_GET(sibling->descriptor.compute_pgm_rsrc1,
                            amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT) +
         1) *
        descriptor_vgpr_granularity_for_wavefront(target.arch, 64);
    EXPECT_EQ(accumulator_base, 192u);
    EXPECT_EQ(allocated, 384u);
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(original, *sibling, nullptr, MetadataPolicy::Native);
    ASSERT_TRUE(metadata.succeeded());
    EXPECT_EQ(metadata.value().vgprs, 384u);
    const std::array<std::string_view, 1> selected = {"sel"};
    InputError error;
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error);
    EXPECT_TRUE(artifact.succeeded()) << error.message;
    if (artifact.failed())
      continue;
    const ImageEntryFacts *native =
        find_image_entry(artifact.value().facts, sibling->descriptor_file_offset);
    ASSERT_NE(native, nullptr);
    EXPECT_EQ(native->disposition, EntryDisposition::Native);
    EXPECT_EQ(native->original_vgprs, 384u);
    EXPECT_EQ(native->binding.prepared_vgprs, 384u);
    util::StringDiagnostic diagnostic;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                       diagnostic.emitter())
                    .succeeded())
        << diagnostic.message();
    const FailureOr<std::vector<uint8_t>> encoded =
        encode_image_facts(artifact.value().facts, diagnostic.emitter());
    ASSERT_TRUE(encoded.succeeded()) << diagnostic.message();
    const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value(), diagnostic.emitter());
    ASSERT_TRUE(decoded.succeeded()) << diagnostic.message();
    EXPECT_EQ(decoded.value(), artifact.value().facts);
  }
}

TEST(ConSanPrepareImageCdna, NativeMetadataRejectsCountsAboveDescriptorCapacity) {
  for (const char *name : {"gfx942", "gfx950"}) {
    SCOPED_TRACE(name);
    for (const bool vector : {false, true}) {
      SCOPED_TRACE(vector);
      std::vector<uint8_t> image = test::read_fixture(test::target_fixture_path(
          {.target = name, .wave_size = 64}, vector ? "cdna-multi-1" : "cdna-multi-0"));
      ASSERT_FALSE(image.empty());
      const FailureOr<amdhsa::MetadataDocument> document = amdhsa::MetadataDocument::parse(image);
      ASSERT_TRUE(document.succeeded());
      const amdhsa::MetadataValue *entries = document.value().root().field("amdhsa.kernels");
      ASSERT_NE(entries, nullptr);
      const amdhsa::MetadataValue *sibling = nullptr;
      for (const amdhsa::MetadataValue &entry : entries->array)
        if (entry.text_field_is(".name", "sib"))
          sibling = &entry;
      ASSERT_NE(sibling, nullptr);
      const amdhsa::MetadataValue *count = sibling->field(vector ? ".vgpr_count" : ".sgpr_count");
      ASSERT_NE(count, nullptr);
      ASSERT_EQ(count->integer_bytes, 0u);
      const size_t payload_offset = document.value().payload().data() - image.data();
      image[payload_offset + count->offset] = vector ? 9 : 17;
      EXPECT_TRUE(inspect_native_image(image, name).failed());
      InputError error;
      const std::array<std::string_view, 1> selected = {"sel"};
      EXPECT_TRUE(prepare_image(image, *target_profile(name), selected, error).failed());
    }
  }
}
#endif

#ifdef CONSAN_MULTI_ENTRY_NATIVE_GLOBAL_FIXTURE
TEST(ConSanPrepareImagePcReads, RejectsNativePcRelativeGlobalStore) {
  std::ifstream file(CONSAN_MULTI_ENTRY_NATIVE_GLOBAL_FIXTURE, std::ios::binary);
  const std::vector<uint8_t> original = {std::istreambuf_iterator<char>(file),
                                         std::istreambuf_iterator<char>()};
  ASSERT_FALSE(original.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  const AmdGpuCodeObject object(original.data(), original.size());
  ASSERT_TRUE(object.is_valid());
  ASSERT_EQ(object.text_sections().size(), 1u);
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(original, text.sectionOffset(), text.size());
  const std::vector<KernelDescriptorInfo>::const_iterator native =
      std::ranges::find(kernels, std::string("mk_c"), &KernelDescriptorInfo::kernel_name);
  ASSERT_NE(native, kernels.end());
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target.public_id);
  ASSERT_NE(decoder, nullptr);
  const std::array<uint64_t, 1> native_entries = {native->entry_text_offset};
  const FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, target.arch, native_entries);
  ASSERT_TRUE(blocks.succeeded());
  ASSERT_GT(blocks.value().size(), 1u);
  std::optional<uint64_t> pc_read_offset;
  for (const std::unique_ptr<BasicBlock> &block : blocks.value()) {
    for (const Instruction &instruction : block->instructions()) {
      if (instruction.reads_pc()) {
        pc_read_offset = instruction.src_loc();
      }
    }
  }
  ASSERT_TRUE(pc_read_offset);
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  InputError error;
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original, target, selected, error);
  ASSERT_TRUE(artifact.failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedInstruction);
  EXPECT_EQ(error.text_offset, pc_read_offset);
  EXPECT_EQ(error.message,
            "native entry 'mk_c' reads PC; PC-relative data references are not preserved");
}
#endif

#ifdef CONSAN_MULTI_ENTRY_UNREACHABLE_PC_FIXTURE
TEST(ConSanPrepareImagePcReads, AllowsUnreachableNativePcRead) {
  std::ifstream file(CONSAN_MULTI_ENTRY_UNREACHABLE_PC_FIXTURE, std::ios::binary);
  const std::vector<uint8_t> original = {std::istreambuf_iterator<char>(file),
                                         std::istreambuf_iterator<char>()};
  ASSERT_FALSE(original.empty());
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  InputError error;
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original, *target_profile("gfx1250"), selected, error);
  ASSERT_TRUE(artifact.succeeded()) << error.message;
  util::StringDiagnostic diagnostic;
  EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                     diagnostic.emitter())
                  .succeeded())
      << diagnostic.message();
}
#endif

#if defined(CONSAN_MULTI_ENTRY_BARRIER_FIXTURE) || defined(CONSAN_MULTI_ENTRY_LOOP_FIXTURE) ||     \
    defined(CONSAN_MULTI_ENTRY_MIXED_FIXTURE)
TEST(ConSanPrepareTrace, BarrierLoopAndMixedBodiesUseOnlyBoundaryFacts) {
  const TargetProfile &target = *target_profile("gfx1250");
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  for (const char *fixture : {
#ifdef CONSAN_MULTI_ENTRY_BARRIER_FIXTURE
           CONSAN_MULTI_ENTRY_BARRIER_FIXTURE,
#endif
#ifdef CONSAN_MULTI_ENTRY_LOOP_FIXTURE
           CONSAN_MULTI_ENTRY_LOOP_FIXTURE,
#endif
#ifdef CONSAN_MULTI_ENTRY_MIXED_FIXTURE
           CONSAN_MULTI_ENTRY_MIXED_FIXTURE,
#endif
       }) {
    SCOPED_TRACE(fixture);
    const std::vector<uint8_t> original = test::read_fixture(fixture);
    ASSERT_FALSE(original.empty());
    InputError error;
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, selected, error, ScopeKind::ExecutionTrace);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    EXPECT_EQ(selected_scope(artifact.value().facts), ScopeKind::ExecutionTrace);
    size_t selected_count = 0;
    for (const ImageEntryFacts &entry : artifact.value().facts.entries) {
      if (!entry.selected)
        continue;
      ++selected_count;
      EXPECT_EQ(entry.selected->scope, ScopeKind::ExecutionTrace);
      EXPECT_EQ(entry.selected->entry.workgroup_id_scalar, kWorkgroupIdTtmp9Selector);
      EXPECT_TRUE(entry.selected->sites.empty());
      EXPECT_TRUE(entry.selected->barriers.empty());
      ASSERT_EQ(entry.selected->boundaries.size(), 2u);
      EXPECT_EQ(entry.selected->boundaries.front().kind, BoundaryKind::Entry);
      EXPECT_EQ(entry.selected->boundaries.back().kind, BoundaryKind::Exit);
    }
    EXPECT_EQ(selected_count, 2u);
    util::StringDiagnostic contract_error;
    EXPECT_TRUE(verify_artifact_images(artifact.value().facts, original, artifact.value().image,
                                       contract_error.emitter())
                    .succeeded())
        << contract_error.message();
  }
}
#endif

#if defined(CONSAN_MULTI_ENTRY_BARRIER_FIXTURE) || defined(CONSAN_MULTI_ENTRY_LOOP_FIXTURE)
void expect_scoped_multi_entry_image(const char *fixture, ScopeKind expected_scope, bool repeated) {
  std::ifstream file(fixture, std::ios::binary);
  const std::vector<uint8_t> original = {std::istreambuf_iterator<char>(file),
                                         std::istreambuf_iterator<char>()};
  ASSERT_FALSE(original.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  InputError error;
  const FailureOr<PreparedImageArtifact> artifact =
      prepare_image(original, target, selected, error);
  ASSERT_TRUE(artifact.succeeded()) << error.message;
  const ImageFacts &facts = artifact.value().facts;
  ASSERT_EQ(facts.entries.size(), 3u);
  EXPECT_EQ(selected_scope(facts), expected_scope);
  const AmdGpuCodeObject prepared(artifact.value().image.data(), artifact.value().image.size());
  ASSERT_TRUE(prepared.is_valid());
  ASSERT_EQ(prepared.text_sections().size(), 1u);
  const Section &text = *prepared.text_sections().front();
  std::vector<EntryKey> keys;
  std::vector<uint64_t> wait_offsets;
  size_t native_count = 0;
  for (const ImageEntryFacts &entry : facts.entries) {
    if (!entry.selected) {
      ++native_count;
      EXPECT_EQ(entry.binding.name, "mk_c");
      EXPECT_EQ(entry.binding.original_group_bytes, entry.binding.prepared_group_bytes);
      EXPECT_EQ(entry.binding.prepared_sgprs, entry.original_sgprs);
      EXPECT_EQ(entry.binding.prepared_vgprs, entry.original_vgprs);
      continue;
    }
    keys.push_back(entry.key);
    const Manifest &manifest = *entry.selected;
    const FailureOr<InputProgram> program =
        build_image_entry_program(original, target, entry.binding.name, error);
    ASSERT_TRUE(program.succeeded()) << error.message;
    EXPECT_EQ(program.value().repeated, repeated);
    ASSERT_EQ(program.value().barriers.size(), 1u);
    ASSERT_EQ(manifest.barriers.size(), 1u);
    EXPECT_EQ(manifest.scope, expected_scope);
    EXPECT_EQ(manifest.entry.workgroup_id_scalar, kWorkgroupIdTtmp9Selector);
    ASSERT_EQ(manifest.sites.size(), program.value().accesses.size());
    for (size_t index = 0; index < manifest.sites.size(); ++index) {
      EXPECT_EQ(manifest.sites[index].width, program.value().accesses[index].width);
      EXPECT_EQ(manifest.sites[index].original_offset,
                program.value().accesses[index].original_offset);
    }
    const BarrierFacts &barrier = manifest.barriers.front();
    const BarrierFacts &source_barrier = program.value().barriers.front();
    EXPECT_EQ(barrier.original_drain, source_barrier.original_drain);
    EXPECT_EQ(barrier.original_signal, source_barrier.original_signal);
    EXPECT_EQ(barrier.original_wait, source_barrier.original_wait);
    EXPECT_EQ(barrier.identity, source_barrier.identity);
    EXPECT_EQ(barrier.prepared_signal, barrier.original_signal);
    ASSERT_LT(barrier.prepared_wait + 4, text.size());
    EXPECT_GE(barrier.completion_probe, barrier.prepared_wait + 4);
    EXPECT_GT(barrier.probe_size, 0u);
    EXPECT_EQ(std::memcmp(
                  original.data() + program.value().text_file_offset + barrier.original_wait,
                  artifact.value().image.data() + text.sectionOffset() + barrier.prepared_wait, 4),
              0);
    wait_offsets.push_back(barrier.original_wait);
    const LaunchDimensions dimensions = {.grid = {128, 1, 1}, .workgroup = {64, 1, 1}};
    const FailureOr<LaunchLayout> layout =
        plan_launch(manifest, dimensions, manifest.entry.original_group_bytes, 0);
    ASSERT_TRUE(layout.succeeded());
    EXPECT_EQ(layout.value().workgroup_count, 2u);
    EXPECT_EQ(layout.value().waves_per_workgroup, 2u);
    EXPECT_EQ(layout.value().wave_count, 4u);
    EXPECT_EQ(layout.value().row_stride, report_row_bytes(manifest.sites.size(), true));
    EXPECT_GT(layout.value().row_stride, report_row_bytes(manifest.sites.size(), false));
  }
  EXPECT_EQ(native_count, 1u);
  ASSERT_EQ(keys.size(), 2u);
  ASSERT_EQ(wait_offsets.size(), 2u);
  EXPECT_NE(wait_offsets[0], wait_offsets[1]);
  EXPECT_EQ(facts.selection_profile, selection_profile_digest(keys, expected_scope));
  util::StringDiagnostic contract_error;
  EXPECT_TRUE(
      verify_artifact_images(facts, original, artifact.value().image, contract_error.emitter())
          .succeeded())
      << contract_error.message();
  const FailureOr<std::vector<uint8_t>> encoded = encode_image_facts(facts);
  ASSERT_TRUE(encoded.succeeded());
  const FailureOr<ImageFacts> decoded = decode_image_facts(encoded.value());
  ASSERT_TRUE(decoded.succeeded());
  EXPECT_EQ(decoded.value(), facts);
}
#endif

#ifdef CONSAN_MULTI_ENTRY_BARRIER_FIXTURE
TEST(ConSanPrepareImageScopes, RequiredBarrierWorkgroupMustPermitTwoWavesBeforePreparation) {
  const std::filesystem::path abi_path =
      std::filesystem::path(CONSAN_MULTI_ENTRY_BARRIER_FIXTURE).parent_path() /
      "multi-entry-abi.hsaco";
  const std::vector<uint8_t> original = test::read_fixture(abi_path);
  const std::vector<uint8_t> barrier_image = test::read_fixture(CONSAN_MULTI_ENTRY_BARRIER_FIXTURE);
  ASSERT_FALSE(original.empty());
  ASSERT_FALSE(barrier_image.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  InputError error;
  const FailureOr<InputProgram> no_sync =
      build_image_entry_program(original, target, "mk_a", error);
  ASSERT_TRUE(no_sync.succeeded()) << error.message;
  ASSERT_EQ(no_sync.value().wave_size, 32u);
  ASSERT_EQ(no_sync.value().accesses.size(), 2u);
  const FailureOr<InputProgram> barrier =
      build_image_entry_program(barrier_image, target, "mk_a", error);
  ASSERT_TRUE(barrier.succeeded()) << error.message;
  ASSERT_EQ(barrier.value().barriers.size(), 1u);
  const BarrierFacts &group = barrier.value().barriers.front();
  ASSERT_EQ(group.original_signal, group.original_drain + 4);
  ASSERT_EQ(group.original_wait, group.original_drain + 8);
  ASSERT_GE(no_sync.value().accesses.back().original_offset, 12u);
  const uint64_t wait_offset =
      no_sync.value().text_file_offset + no_sync.value().accesses.back().original_offset - 12;
  struct Requirement {
    bool barrier;
    uint32_t required;
    uint32_t maximum;
  };
  const std::array<Requirement, 6> requirements = {{
      {.barrier = true, .required = 16, .maximum = 64},
      {.barrier = true, .required = 32, .maximum = 64},
      {.barrier = true, .required = 33, .maximum = 64},
      {.barrier = true, .required = 64, .maximum = 64},
      {.barrier = true, .required = 32, .maximum = 32},
      {.barrier = false, .required = 32, .maximum = 64},
  }};
  for (const Requirement &requirement : requirements) {
    SCOPED_TRACE(requirement.barrier);
    SCOPED_TRACE(requirement.required);
    SCOPED_TRACE(requirement.maximum);
    std::vector<uint8_t> image = original;
    // Replace the fixture's three wait words after the write with a canonical
    // drain/signal/wait group, retaining the argument note's required shape.
    if (requirement.barrier)
      std::copy_n(barrier_image.begin() + barrier.value().text_file_offset + group.original_drain,
                  12, image.begin() + wait_offset);
    const FailureOr<amdhsa::MetadataDocument> metadata = amdhsa::MetadataDocument::parse(image);
    ASSERT_TRUE(metadata.succeeded());
    const amdhsa::MetadataValue *entries = metadata.value().root().field("amdhsa.kernels");
    ASSERT_NE(entries, nullptr);
    const amdhsa::MetadataValue &entry = entries->array.front();
    const amdhsa::MetadataValue *shape = entry.field(".reqd_workgroup_size");
    const amdhsa::MetadataValue *maximum = entry.field(".max_flat_workgroup_size");
    ASSERT_NE(shape, nullptr);
    ASSERT_NE(maximum, nullptr);
    ASSERT_EQ(shape->array.size(), 3u);
    ASSERT_EQ(shape->array.front().integer_bytes, 0u);
    ASSERT_EQ(maximum->integer_bytes, 0u);
    const size_t payload_offset = metadata.value().payload().data() - image.data();
    image[payload_offset + shape->array.front().offset] =
        static_cast<uint8_t>(requirement.required);
    image[payload_offset + maximum->offset] = static_cast<uint8_t>(requirement.maximum);
    const std::array<std::string_view, 1> selected = {"mk_a"};
    const FailureOr<PreparedImageArtifact> artifact = prepare_image(image, target, selected, error);
    if (requirement.barrier && requirement.required <= no_sync.value().wave_size) {
      EXPECT_TRUE(artifact.failed());
      EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry);
      const char *diagnostic =
          requirement.maximum <= no_sync.value().wave_size
              ? "barrier profile requires a maximum workgroup size permitting two waves"
              : "required workgroup size must permit two waves for barrier profile";
      EXPECT_EQ(error.message, diagnostic);
      continue;
    }
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    ASSERT_TRUE(artifact.value().facts.entries.front().selected);
    const Manifest &manifest = *artifact.value().facts.entries.front().selected;
    EXPECT_EQ(manifest.entry.required_workgroup_size,
              (std::array<uint32_t, 3>{requirement.required, 1, 1}));
    EXPECT_EQ(manifest.scope, requirement.barrier ? ScopeKind::NativeLdsWorkgroupBarrier
                                                  : ScopeKind::NativeLdsNoSynchronization);
    EXPECT_EQ(manifest.barriers.size(), requirement.barrier ? 1u : 0u);
    EXPECT_TRUE(plan_launch(manifest, {{requirement.required, 1, 1}, {requirement.required, 1, 1}},
                            manifest.entry.original_group_bytes, 0)
                    .succeeded());
  }
}

TEST(ConSanPrepareImageScopes, WholeImageKeepsPerEntryBarrierCompletionAndWorkgroupFacts) {
  ASSERT_NO_FATAL_FAILURE(expect_scoped_multi_entry_image(
      CONSAN_MULTI_ENTRY_BARRIER_FIXTURE, ScopeKind::NativeLdsWorkgroupBarrier, false));
}
#endif

#ifdef CONSAN_MULTI_ENTRY_LOOP_FIXTURE
TEST(ConSanPrepareImageScopes, WholeImageKeepsPerEntryLoopCompletionAndWorkgroupFacts) {
  ASSERT_NO_FATAL_FAILURE(expect_scoped_multi_entry_image(
      CONSAN_MULTI_ENTRY_LOOP_FIXTURE, ScopeKind::NativeLdsWorkgroupBarrierLoop, true));
}
#endif

#ifdef CONSAN_MULTI_ENTRY_MIXED_FIXTURE
TEST(ConSanPrepareImageScopes, MixedSelectedScopesRejectWhileHomogeneousSubsetsRemainValid) {
  std::ifstream file(CONSAN_MULTI_ENTRY_MIXED_FIXTURE, std::ios::binary);
  const std::vector<uint8_t> original = {std::istreambuf_iterator<char>(file),
                                         std::istreambuf_iterator<char>()};
  ASSERT_FALSE(original.empty());
  const TargetProfile &target = *target_profile("gfx1250");
  const std::array<std::string_view, 2> selected = {"mk_a", "mk_b"};
  InputError error;
  EXPECT_TRUE(prepare_image(original, target, selected, error).failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedEntry);
  EXPECT_EQ(error.message, "selected entries have inconsistent analysis scopes");
  for (const std::string_view name : selected) {
    const std::array<std::string_view, 1> subset = {name};
    const FailureOr<PreparedImageArtifact> artifact =
        prepare_image(original, target, subset, error);
    ASSERT_TRUE(artifact.succeeded()) << error.message;
    const ScopeKind expected = name == "mk_a" ? ScopeKind::NativeLdsWorkgroupBarrier
                                              : ScopeKind::NativeLdsNoSynchronization;
    EXPECT_EQ(selected_scope(artifact.value().facts), expected);
  }
}
#endif

class ConSanPrepare : public ::testing::Test {
protected:
  void SetUp() override {
#ifndef CONSAN_FIXTURE_DIR
    GTEST_SKIP() << "gfx1100 assembler/linker fixtures unavailable";
#else
    original_ = load("race");
    ASSERT_FALSE(original_.empty());
#endif
  }
  static std::vector<uint8_t> load(std::string_view name) {
#ifdef CONSAN_FIXTURE_DIR
    std::ifstream file(std::string(CONSAN_FIXTURE_DIR) + "/" + std::string(name) + ".hsaco",
                       std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
#else
    (void)name;
    return {};
#endif
  }
  std::vector<uint8_t> original_;
  InputError error_;
};

TEST_F(ConSanPrepare, AllAdmittedCompanionsHaveOwningValidatedReplayableArtifacts) {
  for (const std::string_view name :
       {"race", "disjoint", "read_read", "collision", "single_lane", "alias", "immediate",
        "four_ranges", "exhausted", "empty", "branched"}) {
    const std::vector<uint8_t> original = load(name);
    const FailureOr<PreparedArtifact> artifact =
        prepare(original, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(artifact.succeeded()) << name << ": " << error_.message;
    EXPECT_TRUE(validate_prepared(original, artifact.value(), error_).succeeded())
        << error_.message;
    util::StringDiagnostic image_error;
    EXPECT_TRUE(verify_artifact_images(artifact.value().manifest, original, artifact.value().image,
                                       image_error.emitter())
                    .succeeded())
        << name << ": " << image_error.message();
    const FailureOr<std::vector<uint8_t>> encoded = encode_manifest(artifact.value().manifest);
    ASSERT_TRUE(encoded.succeeded());
    const FailureOr<Manifest> decoded = decode_manifest(encoded.value());
    ASSERT_TRUE(decoded.succeeded());
    EXPECT_EQ(decoded.value(), artifact.value().manifest);
    const FailureOr<PreparedArtifact> repeated =
        prepare(original, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(repeated.succeeded());
    EXPECT_EQ(repeated.value().image, artifact.value().image);
    EXPECT_EQ(repeated.value().manifest, artifact.value().manifest);
  }
}

TEST_F(ConSanPrepare, DropBarrierMutantReceivesFreshPreparation) {
  std::vector<uint8_t> source_image = load("barrier");
  ASSERT_FALSE(source_image.empty());
  const std::vector<uint8_t> pristine = source_image;
  const TargetProfile *target = target_profile("gfx1100");
  ASSERT_NE(target, nullptr);
  EXPECT_TRUE(prepare(pristine, *target, "lds_producer_consumer", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedSynchronization);

  std::vector<uint8_t> mutant;
  {
    FaultMutationContext context(source_image);
    FaultMutationError mutation_error;
    const FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, "lds_producer_consumer", mutation_error);
    ASSERT_TRUE(inventory.succeeded()) << mutation_error.message;
    ASSERT_EQ(inventory.value().owner().entry_offset, 0u);
    ASSERT_EQ(inventory.value().barriers().size(), 1u);
    const FailureOr<DropBarrierSelection> selected =
        select_drop_barrier(inventory.value(), {}, mutation_error);
    ASSERT_TRUE(selected.succeeded()) << mutation_error.message;
    ASSERT_EQ(selected.value().barrier().members.size(), 1u);
    EXPECT_EQ(selected.value().barrier().members.front().offset, 0x34u);
    const FailureOr<DropBarrierResult> dropped =
        apply_drop_barrier(context, selected.value(), mutation_error);
    ASSERT_TRUE(dropped.succeeded()) << mutation_error.message;
    EXPECT_EQ(dropped.value().logical_mutation_count, 1u);
    EXPECT_TRUE(std::ranges::equal(context.image(), pristine));
    EXPECT_EQ(source_image, pristine);
    mutant = dropped.value().image;
  }
  ASSERT_FALSE(mutant.empty());
  EXPECT_NE(mutant, pristine);

  // Preparation must own and reanalyze the emitted image after the FI context,
  // inventory, selection and result have expired, including a second DBI pass.
  const FailureOr<PreparedArtifact> artifact =
      prepare(mutant, *target, "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const Manifest &manifest = artifact.value().manifest;
  EXPECT_EQ(manifest.original.sha256, digest(mutant));
  EXPECT_NE(manifest.original.sha256, digest(pristine));
  EXPECT_EQ(manifest.original.bytes, mutant.size());
  ASSERT_EQ(manifest.sites.size(), 2u);
  // native_lds.s case 7 has ds_store_b32 at 0x28 and ds_load_b32 at 0x40.
  // Keep these source expectations independent of the mutant input analysis.
  EXPECT_EQ(manifest.sites[0].id, 0u);
  EXPECT_EQ(manifest.sites[0].original_offset, 0x28u);
  EXPECT_EQ(manifest.sites[0].access, AccessKind::Write);
  EXPECT_EQ(manifest.sites[0].original_size, 8u);
  EXPECT_EQ(manifest.sites[1].id, 1u);
  EXPECT_EQ(manifest.sites[1].original_offset, 0x40u);
  EXPECT_EQ(manifest.sites[1].access, AccessKind::Read);
  EXPECT_EQ(manifest.sites[1].original_size, 8u);
  util::StringDiagnostic image_error;
  EXPECT_TRUE(
      verify_artifact_images(manifest, mutant, artifact.value().image, image_error.emitter())
          .succeeded())
      << image_error.message();
  EXPECT_EQ(source_image, pristine);
}

TEST_F(ConSanPrepare, NonExecutableOriginalCannotProduceAPublishableArtifact) {
  Elf64_Ehdr header{};
  std::memcpy(&header, original_.data(), sizeof(header));
  bool changed = false;
  for (size_t index = 0; index < header.e_phnum; ++index) {
    const size_t offset = header.e_phoff + index * sizeof(Elf64_Phdr);
    Elf64_Phdr segment{};
    std::memcpy(&segment, original_.data() + offset, sizeof(segment));
    if (segment.p_type == PT_LOAD && (segment.p_flags & PF_X) != 0) {
      segment.p_flags &= ~PF_X;
      std::memcpy(original_.data() + offset, &segment, sizeof(segment));
      changed = true;
    }
  }
  ASSERT_TRUE(changed);
  ASSERT_TRUE(
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_)
          .succeeded());
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_NE(error_.message.find("executable .text"), std::string::npos);
}

TEST_F(ConSanPrepare, EntrySelectionIsIndependentOfTheExampleKernelName) {
  const std::vector<uint8_t> bytes = load("renamed");
  const FailureOr<PreparedArtifact> artifact =
      prepare(bytes, *target_profile("gfx1100"), "renamed_input", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  EXPECT_EQ(artifact.value().manifest.entry.name, "renamed_input");
  EXPECT_EQ(artifact.value().manifest.sites.size(), 2u);
  EXPECT_TRUE(prepare(bytes, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
}

TEST_F(ConSanPrepare, EntryPrologueOffsetsMustMatchTheRuntimeWrapper) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded()) << error_.message;
  const std::optional<KernargExtensionLayout> kernarg = make_kernarg_extension_layout(
      input.value().kernel.descriptor.kernarg_size, kContextPointerPayloads);
  ASSERT_TRUE(kernarg.has_value());
  const FailureOr<PreparedArtifact> artifact =
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  AmdGpuCodeObject object(artifact.value().image.data(), artifact.value().image.size());
  bool found = false;
  for (const std::unique_ptr<Section> &section : object.all_sections()) {
    if (section->name() != kKernargExtensionMetadataSectionName) {
      continue;
    }
    const std::optional<std::vector<KernargExtensionMetadata>> records =
        parse_kernarg_extension_metadata(
            {reinterpret_cast<const uint8_t *>(section->data()), section->size()});
    ASSERT_TRUE(records);
    ASSERT_EQ(records->size(), 1u);
    const KernargExtensionMetadata &record = records->front();
    EXPECT_EQ(record.kernel_name, "lds_producer_consumer");
    EXPECT_EQ(record.original_kernarg_size, kernarg->original_kernarg_size);
    ASSERT_EQ(record.payloads.size(), 1u);
    const std::array payloads{KernargExtensionPayloadLayout{record.payloads.front().size,
                                                            record.payloads.front().alignment}};
    const std::optional<KernargExtensionLayout> declared =
        make_kernarg_extension_layout(record.original_kernarg_size, payloads);
    ASSERT_TRUE(declared);
    EXPECT_EQ(declared->wrapper_size, kernarg->wrapper_size);
    EXPECT_EQ(declared->payload_offsets, kernarg->payload_offsets);
    EXPECT_EQ(declared->original_kernarg_pointer_offset, kernarg->original_kernarg_pointer_offset);
    found = true;
  }
  EXPECT_TRUE(found);
}

TEST_F(ConSanPrepare, EntryNamesMustBeUsableByTheShippedJsonAnalyzer) {
  constexpr std::string_view original_name = "lds_producer_consumer";
  for (const bool valid : {false, true}) {
    std::string name(original_name);
    name[0] = valid ? '\xc3' : '\xff';
    if (valid) {
      name[1] = '\xa9';
    }
    std::vector<uint8_t> image = original_;
    size_t replacements = 0;
    for (std::vector<uint8_t>::iterator position = image.begin();;) {
      position = std::search(position, image.end(), original_name.begin(), original_name.end());
      if (position == image.end()) {
        break;
      }
      std::memcpy(&*position, name.data(), name.size());
      position += name.size();
      ++replacements;
    }
    ASSERT_GE(replacements, 3u); // ELF symbols and AMDHSA metadata agree on the name.
    const FailureOr<InputProgram> input =
        build_input_program(image, *target_profile("gfx1100"), name, error_);
    ASSERT_TRUE(input.succeeded()) << error_.message;
    const FailureOr<PreparedArtifact> artifact =
        prepare(image, *target_profile("gfx1100"), name, error_);
    if (valid) {
      ASSERT_TRUE(artifact.succeeded()) << error_.message;
      EXPECT_EQ(artifact.value().manifest.entry.name, name);
    } else {
      EXPECT_TRUE(artifact.failed());
      EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
      EXPECT_EQ(error_.message, "entry name is not valid UTF-8 for JSON output");
    }
  }
}

TEST_F(ConSanPrepare, ModifiedPreparedIsaAndMappingsFailAfterRecomputingTheirImageDigest) {
  FailureOr<PreparedArtifact> artifact =
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  PreparedArtifact broken = artifact.value();
  // Keep the altered envelope disjoint so canonical-plan validation, rather
  // than the structural overlap guard, rejects this mapping.
  ASSERT_GT(broken.manifest.sites[0].probe_size, 4u);
  broken.manifest.sites[0].probe_offset += 4;
  broken.manifest.sites[0].probe_size -= 4;
  EXPECT_TRUE(validate_manifest(broken.manifest).succeeded());
  EXPECT_TRUE(validate_prepared(original_, broken, error_).failed());
  EXPECT_NE(error_.message.find("canonical preservation plan"), std::string::npos);
  broken = artifact.value();
  // Mutate an original guest operand, not a file checksum or framing byte.
  broken
      .image[input.value().text_file_offset + broken.manifest.sites[0].prepared_guest_offset + 4] ^=
      1;
  broken.manifest.prepared.sha256 = digest(broken.image);
  EXPECT_TRUE(match_images(broken.manifest, original_, broken.image).succeeded());
  EXPECT_TRUE(validate_prepared(original_, broken, error_).failed());
}

TEST_F(ConSanPrepare, RegisterPressureReturnsAResourceErrorInsteadOfAnInternalAssertion) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded()) << error_.message;
  std::string metadata_error;
  const FailureOr<MetadataResources> metadata =
      inspect_metadata(original_, input.value().kernel, &metadata_error);
  ASSERT_TRUE(metadata.succeeded()) << metadata_error;
  const uint32_t unavailable = static_cast<uint32_t>(REGISTER_SET_ALLOCATABLE_SGPRS);
  for (const uint32_t declared : {metadata.value().sgprs, unavailable}) {
    SCOPED_TRACE(declared);
    std::vector<uint8_t> image = original_;
    MetadataResources resources = metadata.value();
    resources.sgprs = declared;
    ASSERT_TRUE(amdhsa::update_resource_metadata(image, input.value().kernel,
                                                 input.value().kernel.descriptor, resources.sgprs,
                                                 resources.vgprs, &metadata_error)
                    .succeeded())
        << metadata_error;
    const FailureOr<InputProgram> qualified =
        build_input_program(image, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    ASSERT_TRUE(qualified.succeeded()) << error_.message;
    EXPECT_EQ(qualified.value().original_sgprs, declared);
    const FailureOr<PreparedArtifact> artifact =
        prepare(image, *target_profile("gfx1100"), "lds_producer_consumer", error_);
    if (declared == metadata.value().sgprs) {
      ASSERT_TRUE(artifact.succeeded()) << error_.message;
      const Manifest &manifest = artifact.value().manifest;
      EXPECT_GE(manifest.entry.prepared_sgprs, declared);
      EXPECT_LE(manifest.entry.prepared_sgprs, REGISTER_SET_ALLOCATABLE_SGPRS);
      EXPECT_GE(manifest.entry.prepared_vgprs, metadata.value().vgprs);
      EXPECT_LE(manifest.entry.prepared_vgprs, 256u);
      util::StringDiagnostic image_error;
      EXPECT_TRUE(
          verify_artifact_images(manifest, image, artifact.value().image, image_error.emitter())
              .succeeded())
          << image_error.message();
    } else {
      EXPECT_TRUE(artifact.failed());
      EXPECT_EQ(error_.code, InputErrorCode::ResourceLimit);
      EXPECT_FALSE(error_.message.empty());
    }
  }
}

TEST_F(ConSanPrepare, DescriptorAndMetadataMismatchCannotProduceAnArtifact) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  rocjitsu::amdhsa::kernel_descriptor_t descriptor = input.value().kernel.descriptor;
  descriptor.kernarg_size += 8;
  std::memcpy(original_.data() + input.value().kernel.descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_NE(error_.message.find("metadata"), std::string::npos);
}

TEST_F(ConSanPrepare, MetadataGrowthFailureDoesNotPartiallyMutateTheImage) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  const std::vector<uint8_t> saved = original_;
  std::string message;
  amdhsa::kernel_descriptor_t descriptor = input.value().kernel.descriptor;
  descriptor.kernarg_size = 4096;
  EXPECT_TRUE(amdhsa::update_resource_metadata(original_, input.value().kernel, descriptor, 28, 16,
                                               &message)
                  .failed());
  EXPECT_EQ(original_, saved);
  EXPECT_NE(message.find("wider"), std::string::npos);
}

TEST_F(ConSanPrepare, ConcreteElfTargetMismatchFailsEvenWhenTheRequestedTargetIsGfx1100) {
  Elf64_Ehdr header{};
  std::memcpy(&header, original_.data(), sizeof(header));
  header.e_flags ^= 1;
  std::memcpy(original_.data(), &header, sizeof(header));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedTarget);
}

TEST_F(ConSanPrepare, Gfx1100Wave32RequiresTheInputSgprPaddingWorkaround) {
  const FailureOr<InputProgram> input =
      build_input_program(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(input.succeeded());
  rocjitsu::amdhsa::kernel_descriptor_t descriptor = input.value().kernel.descriptor;
  set_kernel_descriptor_user_sgpr_count(rocjitsu::IsaFamily::Rdna3, descriptor, 2);
  std::memcpy(original_.data() + input.value().kernel.descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  EXPECT_TRUE(
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_).failed());
  EXPECT_EQ(error_.code, InputErrorCode::UnsupportedEntry);
  EXPECT_NE(error_.message.find("SGPR padding"), std::string::npos);
}

TEST_F(ConSanPrepare, TransportStartsUnenteredAndOwnsExactParentAndGeometryBindings) {
  const FailureOr<PreparedArtifact> artifact =
      prepare(original_, *target_profile("gfx1100"), "lds_producer_consumer", error_);
  ASSERT_TRUE(artifact.succeeded()) << error_.message;
  const Manifest &manifest = artifact.value().manifest;
  const LaunchDimensions shape{{64, 1, 1}, {64, 1, 1}};
  const FailureOr<TransportImage> image = initialize_transport(manifest, shape, 8, 0, 101, 0x10000);
  ASSERT_TRUE(image.succeeded());
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextSerial), 101u);
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextReportPointer), 0x10000u);
  EXPECT_EQ(wire::read_u64(image.value().context, wire::kContextReportBytes),
            image.value().report.size());
  for (uint32_t wave = 0; wave < 2; ++wave) {
    const size_t row = kReportHeaderBytes + wave * 576;
    EXPECT_EQ(wire::read_u32(image.value().report, row + wire::kRowStatus), 0u);
    EXPECT_EQ(wire::read_u64(image.value().report, row + wire::kRowSerial), 101u);
    EXPECT_EQ(wire::read_u32(image.value().report, row + wire::kRowWave), wave);
  }
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 0, 0x10000).failed());
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 101, 0).failed());
  EXPECT_TRUE(initialize_transport(manifest, shape, 8, 0, 101, UINT64_MAX - 63).failed());
  EXPECT_FALSE(wire::read_u32({}, 0).has_value());
  EXPECT_FALSE(wire::read_u64({}, SIZE_MAX).has_value());
}

} // namespace
} // namespace rocjitsu::consan
