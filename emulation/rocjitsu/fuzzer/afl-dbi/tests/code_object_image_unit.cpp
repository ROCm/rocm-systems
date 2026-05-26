#include "code_object_image.h"
#include "md5_util.h"

#include <zstd.h>

#include <assert.h>
#include <stdint.h>
#include <string.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace afl_dbi = rocjitsu::fuzzer::afl_dbi;

namespace {

constexpr uint32_t kHipfMagic = 0x48495046;
constexpr uint32_t kHipkMagic = 0x4b504948;

struct HipFatBinaryWrapper {
  uint32_t magic = 0;
  uint32_t version = 0;
  const void *binary = nullptr;
  const void *reserved = nullptr;
};

void append_u32(std::vector<uint8_t> *data, uint32_t value) {
  for (unsigned i = 0; i < 4; ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

void append_u64(std::vector<uint8_t> *data, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i)
    data->push_back(static_cast<uint8_t>((value >> (i * 8)) & 0xffu));
}

uint64_t read_u64(const uint8_t *data) {
  uint64_t value = 0;
  memcpy(&value, data, sizeof(value));
  return value;
}

std::vector<uint8_t>
make_bundle(const std::vector<std::pair<std::string, std::vector<uint8_t>>> &entries) {
  constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  std::vector<uint8_t> bundle;
  bundle.insert(bundle.end(), kBundleMagic, kBundleMagic + sizeof(kBundleMagic) - 1);
  append_u64(&bundle, entries.size());

  uint64_t offset = bundle.size();
  for (const auto &[id, _] : entries)
    offset += 3 * sizeof(uint64_t) + id.size();

  uint64_t data_offset = offset;
  for (const auto &[id, image] : entries) {
    append_u64(&bundle, data_offset);
    append_u64(&bundle, image.size());
    append_u64(&bundle, id.size());
    bundle.insert(bundle.end(), id.begin(), id.end());
    data_offset += image.size();
  }

  for (const auto &[_, image] : entries)
    bundle.insert(bundle.end(), image.begin(), image.end());
  return bundle;
}

std::vector<uint8_t> make_ccob(const std::vector<uint8_t> &uncompressed, uint16_t version) {
  const size_t bound = ZSTD_compressBound(uncompressed.size());
  std::vector<uint8_t> compressed(bound);
  const size_t compressed_size =
      ZSTD_compress(compressed.data(), compressed.size(), uncompressed.data(), uncompressed.size(),
                    /*compressionLevel=*/1);
  assert(!ZSTD_isError(compressed_size));
  compressed.resize(compressed_size);

  const uint64_t header_size = version == 3 ? 32 : 24;
  const uint64_t total_size = header_size + compressed.size();
  std::optional<uint64_t> hash = afl_dbi::truncated_md5_hash64(uncompressed);
  assert(hash.has_value());
  std::vector<uint8_t> ccob;
  ccob.insert(ccob.end(), {'C', 'C', 'O', 'B'});
  append_u32(&ccob, static_cast<uint32_t>(version) | (1u << 16));
  if (version == 3) {
    append_u64(&ccob, total_size);
    append_u64(&ccob, uncompressed.size());
    append_u64(&ccob, *hash);
  } else {
    append_u32(&ccob, static_cast<uint32_t>(total_size));
    append_u32(&ccob, static_cast<uint32_t>(uncompressed.size()));
    append_u64(&ccob, *hash);
  }
  ccob.insert(ccob.end(), compressed.begin(), compressed.end());
  return ccob;
}

uint64_t ccob_hash_field(const std::vector<uint8_t> &ccob) {
  assert(ccob.size() >= 24);
  const uint16_t version = static_cast<uint16_t>(ccob[4] | (static_cast<uint16_t>(ccob[5]) << 8));
  return read_u64(ccob.data() + (version == 3 ? 24 : 16));
}

void check_extracts_device_elf_from_bundle() {
  const std::vector<uint8_t> host = {'h', 'o', 's', 't'};
  const std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'F', 1, 2, 3, 4};
  std::vector<uint8_t> bundle = make_bundle(
      {{"host-x86_64-unknown-linux-gnu", host}, {"hipv4-amdgcn-amd-amdhsa--gfx1100", elf}});

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(bundle);
  assert(images.size() == 1);
  assert(images[0].id == "hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].target_id == "gfx1100");
  assert(images[0].bytes == elf);
}

void check_extracts_device_elf_from_ccob(uint16_t version) {
  const std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'F', 9, 8, 7, 6};
  const std::vector<uint8_t> bundle = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", elf}});
  const std::vector<uint8_t> ccob = make_ccob(bundle, version);
  std::optional<uint64_t> expected_hash = afl_dbi::truncated_md5_hash64(bundle);
  assert(expected_hash.has_value());
  assert(ccob_hash_field(ccob) == *expected_hash);

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(ccob);
  assert(images.size() == 1);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[0].target_id == "gfx1201");
  assert(images[0].bytes == elf);

  const std::vector<uint8_t> copied = afl_dbi::copy_module_data_image(ccob.data());
  assert(copied == ccob);
}

void check_copy_module_data_unwraps_hip_fatbin_wrappers() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> bundle = make_bundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100},
      {"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201},
  });
  const HipFatBinaryWrapper hipf_wrapper{kHipfMagic, 1, bundle.data(), nullptr};
  assert(afl_dbi::copy_module_data_image(&hipf_wrapper) == bundle);

  const std::vector<uint8_t> ccob = make_ccob(bundle, /*version=*/3);
  const HipFatBinaryWrapper hipk_wrapper{kHipkMagic, 1, ccob.data(), nullptr};
  assert(afl_dbi::copy_module_data_image(&hipk_wrapper) == ccob);

  const HipFatBinaryWrapper wrong_version{kHipfMagic, 2, bundle.data(), nullptr};
  assert(afl_dbi::copy_module_data_image(&wrong_version).empty());

  HipFatBinaryWrapper self_reference{kHipfMagic, 1, nullptr, nullptr};
  self_reference.binary = &self_reference;
  assert(afl_dbi::copy_module_data_image(&self_reference).empty());
}

void check_summarizes_raw_elf_and_bundle() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const afl_dbi::CodeObjectImageSummary raw_summary = afl_dbi::summarize_code_object_image(gfx1100);
  assert(raw_summary.top_level_raw_elf);
  assert(!raw_summary.top_level_bundle);
  assert(!raw_summary.top_level_ccob);
  assert(raw_summary.device_image_count == 1);
  assert(!raw_summary.raw_elf_bypass_drops_sibling_payloads);

  const std::vector<uint8_t> bundle = make_bundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100},
      {"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201},
  });
  const afl_dbi::CodeObjectImageSummary bundle_summary =
      afl_dbi::summarize_code_object_image(bundle);
  assert(!bundle_summary.top_level_raw_elf);
  assert(bundle_summary.top_level_bundle);
  assert(!bundle_summary.top_level_ccob);
  assert(bundle_summary.device_image_count == 2);
  assert(bundle_summary.top_level_ccob_device_image_count == 0);
  assert(bundle_summary.raw_elf_bypass_drops_sibling_payloads);
}

void check_summarizes_single_payload_ccob() {
  const std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> bundle = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", elf}});
  const std::vector<uint8_t> ccob = make_ccob(bundle, /*version=*/3);

  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(ccob);
  assert(!summary.top_level_raw_elf);
  assert(!summary.top_level_bundle);
  assert(summary.top_level_ccob);
  assert(summary.device_image_count == 1);
  assert(summary.top_level_ccob_device_image_count == 1);
  assert(!summary.raw_elf_bypass_drops_sibling_payloads);
}

void check_summarizes_multi_payload_ccob() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> bundle = make_bundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100},
      {"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201},
  });
  const std::vector<uint8_t> ccob = make_ccob(bundle, /*version=*/2);

  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(ccob);
  assert(summary.top_level_ccob);
  assert(summary.device_image_count == 2);
  assert(summary.top_level_ccob_device_image_count == 2);
  assert(summary.raw_elf_bypass_drops_sibling_payloads);
}

void check_rebuilds_multi_payload_ccob(uint16_t version) {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> replacement = {0x7f, 'E', 'L', 'F', 9, 8, 7, 6, 5, 4};
  const std::vector<uint8_t> host = {'h', 'o', 's', 't'};
  const std::vector<uint8_t> bundle = make_bundle({
      {"host-x86_64-unknown-linux-gnu", host},
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100},
      {"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201},
  });
  const std::vector<uint8_t> ccob = make_ccob(bundle, version);

  std::optional<std::vector<uint8_t>> rebuilt = afl_dbi::rebuild_ccob_with_replaced_device_image(
      ccob, "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201", replacement);
  assert(rebuilt.has_value());
  assert(ccob_hash_field(*rebuilt) != 0);
  assert(ccob_hash_field(*rebuilt) != ccob_hash_field(ccob));

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(*rebuilt);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].target_id == "gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].target_id == "gfx1201");
  assert(images[1].bytes == replacement);

  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(*rebuilt);
  assert(summary.top_level_ccob);
  assert(summary.device_image_count == 2);
  assert(summary.raw_elf_bypass_drops_sibling_payloads);
}

void check_rebuild_ccob_rejects_missing_payload() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> replacement = {0x7f, 'E', 'L', 'F', 9};
  const std::vector<uint8_t> bundle = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> ccob = make_ccob(bundle, /*version=*/3);

  std::optional<std::vector<uint8_t>> rebuilt = afl_dbi::rebuild_ccob_with_replaced_device_image(
      ccob, "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201", replacement);
  assert(!rebuilt.has_value());
}

void check_rebuilds_concatenated_bundles_preserving_siblings() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> replacement = {0x7f, 'E', 'L', 'F', 9, 8, 7};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});
  std::vector<uint8_t> fatbin = {'p', 'a', 'd'};
  fatbin.insert(fatbin.end(), first.begin(), first.end());
  fatbin.insert(fatbin.end(), {0, 0, 0, 0});
  fatbin.insert(fatbin.end(), second.begin(), second.end());

  std::optional<std::vector<uint8_t>> rebuilt =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          fatbin, "hipv4-amdgcn-amd-amdhsa--gfx1201", replacement);
  assert(rebuilt.has_value());
  assert((*rebuilt)[0] == 'p' && (*rebuilt)[1] == 'a' && (*rebuilt)[2] == 'd');

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(*rebuilt);
  assert(images.size() == 2);
  assert(images[0].id == "hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].bytes == replacement);
}

void check_rebuilds_mixed_concatenated_containers_preserving_siblings() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> gfx1100_replacement = {0x7f, 'E', 'L', 'F', 9};
  const std::vector<uint8_t> gfx1201_replacement = {0x7f, 'E', 'L', 'F', 8, 7};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});
  std::vector<uint8_t> fatbin = make_ccob(first, /*version=*/3);
  fatbin.insert(fatbin.end(), {0xaa, 0xbb});
  fatbin.insert(fatbin.end(), second.begin(), second.end());

  std::optional<std::vector<uint8_t>> rebuilt_ccob_payload =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          fatbin, "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100_replacement);
  assert(rebuilt_ccob_payload.has_value());
  std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(*rebuilt_ccob_payload);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].bytes == gfx1100_replacement);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].bytes == gfx1201);

  std::optional<std::vector<uint8_t>> rebuilt_bundle_payload =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          fatbin, "hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201_replacement);
  assert(rebuilt_bundle_payload.has_value());
  images = afl_dbi::extract_device_images(*rebuilt_bundle_payload);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].bytes == gfx1201_replacement);
}

void check_rebuilds_ccob_concatenated_bundles_preserving_siblings() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> replacement = {0x7f, 'E', 'L', 'F', 6, 5, 4};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});
  std::vector<uint8_t> concatenated = first;
  concatenated.insert(concatenated.end(), second.begin(), second.end());
  const std::vector<uint8_t> ccob = make_ccob(concatenated, /*version=*/2);

  std::optional<std::vector<uint8_t>> rebuilt =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          ccob, "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201", replacement);
  assert(rebuilt.has_value());
  assert(ccob_hash_field(*rebuilt) != 0);
  assert(ccob_hash_field(*rebuilt) != ccob_hash_field(ccob));

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(*rebuilt);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].bytes == replacement);
}

void check_rebuilds_duplicate_device_ids_by_extracted_image() {
  const std::vector<uint8_t> first_elf = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> second_elf = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> first_replacement = {0x7f, 'E', 'L', 'F', 3};
  const std::vector<uint8_t> second_replacement = {0x7f, 'E', 'L', 'F', 4, 5};
  const std::vector<uint8_t> first =
      make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", first_elf}});
  const std::vector<uint8_t> second =
      make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", second_elf}});
  std::vector<uint8_t> fatbin = first;
  fatbin.insert(fatbin.end(), second.begin(), second.end());

  std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(fatbin);
  assert(images.size() == 2);
  assert(images[0].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[0].index == 0);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].index == 1);

  std::optional<std::vector<uint8_t>> rebuilt_by_id =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          fatbin, images[1].id, first_replacement);
  assert(rebuilt_by_id.has_value());
  images = afl_dbi::extract_device_images(*rebuilt_by_id);
  assert(images[0].bytes == first_replacement);
  assert(images[1].bytes == second_elf);

  std::optional<std::vector<uint8_t>> rebuilt_by_image =
      afl_dbi::rebuild_code_object_image_with_replaced_device_image(
          fatbin, images[1], second_replacement);
  assert(rebuilt_by_image.has_value());
  images = afl_dbi::extract_device_images(*rebuilt_by_image);
  assert(images[0].bytes == first_elf);
  assert(images[1].bytes == second_replacement);
}

void check_extracts_device_elf_from_nested_bundle_ccob() {
  const std::vector<uint8_t> elf = {0x7f, 'E', 'L', 'F', 5, 4, 3, 2};
  const std::vector<uint8_t> inner = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", elf}});
  const std::vector<uint8_t> outer = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", inner}});
  const std::vector<uint8_t> ccob = make_ccob(outer, /*version=*/2);

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(ccob);
  assert(images.size() == 1);
  assert(images[0].target_id == "gfx1201");
  assert(images[0].bytes == elf);
}

void check_extracts_device_elf_from_concatenated_bundles() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});
  std::vector<uint8_t> fatbin = {'p', 'a', 'd'};
  fatbin.insert(fatbin.end(), first.begin(), first.end());
  fatbin.insert(fatbin.end(), {0, 0, 0, 0});
  fatbin.insert(fatbin.end(), second.begin(), second.end());

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(fatbin);
  assert(images.size() == 2);
  assert(images[0].id == "hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].target_id == "gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].target_id == "gfx1201");
  assert(images[1].bytes == gfx1201);
}

void check_extracts_device_elf_from_mixed_concatenated_containers() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});

  std::vector<uint8_t> fatbin = make_ccob(first, /*version=*/3);
  fatbin.insert(fatbin.end(), second.begin(), second.end());

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(fatbin);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].target_id == "gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].target_id == "gfx1201");
  assert(images[1].bytes == gfx1201);

  const afl_dbi::CodeObjectImageSummary summary = afl_dbi::summarize_code_object_image(fatbin);
  assert(summary.top_level_ccob);
  assert(summary.device_image_count == 2);
  assert(summary.top_level_ccob_device_image_count == 1);
  assert(summary.raw_elf_bypass_drops_sibling_payloads);
}

void check_extracts_device_elf_from_ccob_concatenated_bundles() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> first = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100}});
  const std::vector<uint8_t> second = make_bundle({{"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201}});
  std::vector<uint8_t> concatenated = first;
  concatenated.insert(concatenated.end(), second.begin(), second.end());

  const std::vector<uint8_t> ccob = make_ccob(concatenated, /*version=*/2);
  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(ccob);
  assert(images.size() == 2);
  assert(images[0].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1100");
  assert(images[0].target_id == "gfx1100");
  assert(images[0].bytes == gfx1100);
  assert(images[1].id == "ccob:hipv4-amdgcn-amd-amdhsa--gfx1201");
  assert(images[1].target_id == "gfx1201");
  assert(images[1].bytes == gfx1201);
}

void check_orders_device_images_by_current_target() {
  const std::vector<uint8_t> gfx1100 = {0x7f, 'E', 'L', 'F', 1};
  const std::vector<uint8_t> gfx1201 = {0x7f, 'E', 'L', 'F', 2};
  const std::vector<uint8_t> gfx942 = {0x7f, 'E', 'L', 'F', 3};
  const std::vector<uint8_t> bundle = make_bundle({
      {"hipv4-amdgcn-amd-amdhsa--gfx1100", gfx1100},
      {"hipv4-amdgcn-amd-amdhsa--gfx1201", gfx1201},
      {"hipv4-amdgcn-amd-amdhsa--gfx942:xnack-", gfx942},
  });

  const std::vector<afl_dbi::DeviceImage> images = afl_dbi::extract_device_images(bundle);
  assert(images.size() == 3);

  std::vector<afl_dbi::DeviceImage> ordered =
      afl_dbi::order_device_images_for_target(images, "gfx1201");
  assert(ordered.size() == 3);
  assert(ordered[0].target_id == "gfx1201");
  assert(ordered[0].bytes == gfx1201);

  ordered = afl_dbi::order_device_images_for_target(images, "gfx942:sramecc+:xnack-");
  assert(ordered.size() == 3);
  assert(ordered[0].target_id == "gfx942:xnack-");
  assert(ordered[0].bytes == gfx942);

  ordered = afl_dbi::order_device_images_for_target(images, "gfx90a");
  assert(ordered.size() == 3);
  assert(ordered[0].target_id == "gfx1100");
}

void check_normalizes_target_ids() {
  assert(afl_dbi::normalize_amdgpu_target_id("amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-") ==
         "gfx942:sramecc+:xnack-");
  assert(afl_dbi::normalize_amdgpu_target_id("gfx90a-xnack+") == "gfx90a:xnack+");
  assert(afl_dbi::normalize_amdgpu_target_id("gfx942-sramecc+-xnack-") == "gfx942:sramecc+:xnack-");
}

} // namespace

int main() {
  check_extracts_device_elf_from_bundle();
  check_extracts_device_elf_from_ccob(/*version=*/2);
  check_extracts_device_elf_from_ccob(/*version=*/3);
  check_copy_module_data_unwraps_hip_fatbin_wrappers();
  check_summarizes_raw_elf_and_bundle();
  check_summarizes_single_payload_ccob();
  check_summarizes_multi_payload_ccob();
  check_rebuilds_multi_payload_ccob(/*version=*/2);
  check_rebuilds_multi_payload_ccob(/*version=*/3);
  check_rebuild_ccob_rejects_missing_payload();
  check_rebuilds_concatenated_bundles_preserving_siblings();
  check_rebuilds_mixed_concatenated_containers_preserving_siblings();
  check_rebuilds_ccob_concatenated_bundles_preserving_siblings();
  check_rebuilds_duplicate_device_ids_by_extracted_image();
  check_extracts_device_elf_from_nested_bundle_ccob();
  check_extracts_device_elf_from_concatenated_bundles();
  check_extracts_device_elf_from_mixed_concatenated_containers();
  check_extracts_device_elf_from_ccob_concatenated_bundles();
  check_orders_device_images_by_current_target();
  check_normalizes_target_ids();
  return 0;
}
