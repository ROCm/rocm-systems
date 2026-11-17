// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/facts.h"

#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/target.h"
#include "util/bit.h"
#include "util/sha256.h"

#include <algorithm>
#include <bit>
#include <set>

namespace rocjitsu::consan {
namespace {
constexpr std::array<uint8_t, 8> kImageMagic = {'R', 'J', 'C', 'S', 'I', 'M', 'G', 'F'};
constexpr uint64_t kEnvelopeBytes = 64;
constexpr uint64_t kImageFixedBytes = 144;
constexpr uint64_t kEntryFixedBytes = 108;
constexpr uint64_t kManifestFixedBytes = 64 + 232;
constexpr uint64_t kSiteBytes = 68;
constexpr uint64_t kBarrierBytes = 56;

Result reject(ContractError *error, ContractErrorCode code, std::string_view message) {
  if (error != nullptr)
    *error = {.code = code, .message = std::string(message)};
  return Result::failure();
}

class Encoder {
public:
  void u32(uint32_t value) {
    for (unsigned shift = 0; shift < 32; shift += 8)
      data_.push_back(static_cast<uint8_t>(value >> shift));
  }
  void u64(uint64_t value) {
    u32(static_cast<uint32_t>(value));
    u32(static_cast<uint32_t>(value >> 32));
  }
  void bytes(std::span<const uint8_t> bytes) {
    data_.insert(data_.end(), bytes.begin(), bytes.end());
  }
  void string(std::string_view value) {
    u32(static_cast<uint32_t>(value.size()));
    data_.insert(data_.end(), value.begin(), value.end());
  }
  std::vector<uint8_t> take() { return std::move(data_); }

private:
  std::vector<uint8_t> data_;
};

class Decoder {
public:
  explicit Decoder(std::span<const uint8_t> bytes) : bytes_(bytes) {}
  bool u32(uint32_t &value) {
    const std::optional<uint32_t> read = wire::read_u32(bytes_, offset_);
    if (!read)
      return false;
    value = *read;
    offset_ += 4;
    return true;
  }
  bool u64(uint64_t &value) {
    const std::optional<uint64_t> read = wire::read_u64(bytes_, offset_);
    if (!read)
      return false;
    value = *read;
    offset_ += 8;
    return true;
  }
  bool bytes(std::span<uint8_t> value) {
    if (value.size() > remaining())
      return false;
    std::copy_n(bytes_.begin() + offset_, value.size(), value.begin());
    offset_ += value.size();
    return true;
  }
  bool string(std::string &value, uint32_t limit) {
    uint32_t size = 0;
    if (!u32(size) || size > limit || size > remaining())
      return false;
    value.assign(reinterpret_cast<const char *>(bytes_.data() + offset_), size);
    offset_ += size;
    return true;
  }
  std::span<const uint8_t> remaining_bytes() const { return bytes_.subspan(offset_); }
  bool skip(uint64_t size) {
    if (size > remaining())
      return false;
    offset_ += size;
    return true;
  }
  size_t remaining() const { return bytes_.size() - offset_; }

private:
  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
};

void encode_identity(Encoder &encoder, const ImageIdentity &identity) {
  encoder.bytes(identity.sha256);
  encoder.u64(identity.bytes);
}
bool decode_identity(Decoder &decoder, ImageIdentity &identity) {
  return decoder.bytes(identity.sha256) && decoder.u64(identity.bytes);
}
void encode_binding(Encoder &encoder, const EntryFacts &entry) {
  encoder.u32(entry.id);
  encoder.string(entry.name);
  encoder.u64(entry.original_descriptor_offset);
  encoder.u64(entry.prepared_descriptor_offset);
  encoder.u64(entry.original_entry_offset);
  encoder.u64(entry.prepared_entry_offset);
  encoder.u32(entry.original_kernarg_bytes);
  encoder.u32(entry.kernarg_alignment);
  encoder.u32(entry.original_group_bytes);
  encoder.u32(entry.prepared_group_bytes);
  encoder.u32(entry.original_private_bytes);
  encoder.u32(entry.prepared_private_bytes);
  encoder.u32(entry.prepared_sgprs);
  encoder.u32(entry.prepared_vgprs);
  encoder.u32(entry.maximum_workgroup_size);
  encoder.u32(entry.workgroup_id_scalar);
}
bool decode_binding(Decoder &decoder, EntryFacts &entry, uint32_t string_limit) {
  return decoder.u32(entry.id) && decoder.string(entry.name, string_limit) &&
         decoder.u64(entry.original_descriptor_offset) &&
         decoder.u64(entry.prepared_descriptor_offset) &&
         decoder.u64(entry.original_entry_offset) && decoder.u64(entry.prepared_entry_offset) &&
         decoder.u32(entry.original_kernarg_bytes) && decoder.u32(entry.kernarg_alignment) &&
         decoder.u32(entry.original_group_bytes) && decoder.u32(entry.prepared_group_bytes) &&
         decoder.u32(entry.original_private_bytes) && decoder.u32(entry.prepared_private_bytes) &&
         decoder.u32(entry.prepared_sgprs) && decoder.u32(entry.prepared_vgprs) &&
         decoder.u32(entry.maximum_workgroup_size) && decoder.u32(entry.workgroup_id_scalar);
}

bool add_extent(uint64_t &total, uint64_t amount, uint64_t limit) {
  if (total > limit || amount > limit - total)
    return false;
  total += amount;
  return true;
}
} // namespace

Digest selection_profile_digest(std::span<const EntryKey> selected_keys, ScopeKind scope) {
  util::Sha256 hash;
  hash.update("consan-image-selection-v2");
  std::array<uint8_t, 4> scope_bytes{};
  wire::write_u32(scope_bytes, 0, static_cast<uint32_t>(scope));
  hash.update(scope_bytes.data(), scope_bytes.size());
  const Digest options = mvp_options_digest();
  hash.update(options.data(), options.size());
  std::array<uint8_t, 4> count{};
  wire::write_u32(count, 0, static_cast<uint32_t>(selected_keys.size()));
  hash.update(count.data(), count.size());
  for (const EntryKey key : selected_keys) {
    std::array<uint8_t, 8> bytes{};
    wire::write_u64(bytes, 0, key);
    hash.update(bytes.data(), bytes.size());
  }
  return hash.finish();
}

std::optional<ScopeKind> selected_scope(const ImageFacts &image) {
  std::optional<ScopeKind> scope;
  for (const ImageEntryFacts &entry : image.entries) {
    if (!entry.selected)
      continue;
    if (scope && *scope != entry.selected->scope)
      return std::nullopt;
    scope = entry.selected->scope;
  }
  return scope;
}

const ImageEntryFacts *find_image_entry(const ImageFacts &image, EntryKey key) {
  const std::vector<ImageEntryFacts>::const_iterator found =
      std::ranges::lower_bound(image.entries, key, {}, &ImageEntryFacts::key);
  return found != image.entries.end() && found->key == key ? &*found : nullptr;
}

const Manifest *selected_manifest(const ImageFacts &image, EntryKey key) {
  const ImageEntryFacts *entry = find_image_entry(image, key);
  return entry != nullptr && entry->disposition == EntryDisposition::Selected && entry->selected
             ? &*entry->selected
             : nullptr;
}

ImageFacts single_entry_image_facts(const Manifest &manifest, uint32_t original_sgprs,
                                    uint32_t original_vgprs) {
  const std::array<EntryKey, 1> keys = {manifest.entry.original_descriptor_offset};
  return ImageFacts{.original = manifest.original,
                    .prepared = manifest.prepared,
                    .selection_profile = selection_profile_digest(keys, manifest.scope),
                    .target = manifest.target,
                    .wave_size = manifest.wave_size,
                    .code_object_abi = manifest.code_object_abi,
                    .original_text_bytes = manifest.original_text_bytes,
                    .prepared_text_bytes = manifest.prepared_text_bytes,
                    .entries = {{.key = keys.front(),
                                 .disposition = EntryDisposition::Selected,
                                 .binding = manifest.entry,
                                 .original_sgprs = original_sgprs,
                                 .original_vgprs = original_vgprs,
                                 .selected = manifest}}};
}

Result validate_image_facts(const ImageFacts &image, ContractError *error,
                            const ContractLimits &limits) {
  if (error != nullptr)
    *error = {};
  const TargetProfile *target = target_profile(image.target);
  if (image.entries.empty() || image.entries.size() > kImageEntryLimit ||
      image.original.bytes > limits.image_bytes || image.prepared.bytes > limits.image_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "image entry or byte budget exceeded");
  if (target == nullptr || !supports_wave(*target, image.wave_size) || image.code_object_abi != 5 ||
      image.original.bytes == 0 || image.prepared.bytes == 0 || image.original_text_bytes == 0 ||
      image.original_text_bytes > image.original.bytes || image.prepared_text_bytes == 0 ||
      image.prepared_text_bytes > image.prepared.bytes)
    return reject(error, ContractErrorCode::InvalidFacts, "invalid common image facts");

  uint64_t encoded_bytes = kEnvelopeBytes + kImageFixedBytes;
  uint64_t total_sites = 0;
  std::vector<EntryKey> selected_keys;
  std::set<std::string> names;
  std::set<uint64_t> original_entries;
  std::set<uint64_t> prepared_entries;
  std::vector<std::pair<uint64_t, uint64_t>> prepared_descriptors;
  std::vector<std::pair<uint64_t, uint64_t>> original_sites;
  std::vector<std::pair<uint64_t, uint64_t>> prepared_observations;
  uint64_t previous_descriptor_end = 0;
  for (const ImageEntryFacts &entry : image.entries) {
    const EntryFacts &binding = entry.binding;
    if (binding.name.size() > limits.string_bytes)
      return reject(error, ContractErrorCode::ResourceLimit, "entry name exceeds budget");
    if (entry.key != binding.original_descriptor_offset || entry.key < previous_descriptor_end ||
        !util::is_range_in_bounds(entry.key, 64, image.original.bytes) ||
        !util::is_range_in_bounds(binding.prepared_descriptor_offset, 64, image.prepared.bytes) ||
        !original_entries.insert(binding.original_entry_offset).second ||
        !prepared_entries.insert(binding.prepared_entry_offset).second ||
        !names.insert(binding.name).second || binding.name.empty() ||
        binding.name.find('\0') != std::string::npos || binding.id != 0 ||
        !util::is_range_in_bounds(binding.original_entry_offset, 4, image.original_text_bytes) ||
        !util::is_range_in_bounds(binding.prepared_entry_offset, 4, image.prepared_text_bytes) ||
        binding.original_entry_offset % 4 != 0 || binding.prepared_entry_offset % 4 != 0 ||
        !std::has_single_bit(binding.kernarg_alignment) || binding.kernarg_alignment < 8 ||
        binding.original_kernarg_bytes == 0 || binding.maximum_workgroup_size == 0 ||
        binding.maximum_workgroup_size > 1024 || binding.original_group_bytes > 65536 ||
        binding.prepared_group_bytes > 65536 || binding.original_private_bytes != 0 ||
        binding.prepared_private_bytes != 0 || entry.original_sgprs == 0 ||
        entry.original_sgprs > 106 || entry.original_vgprs == 0 || entry.original_vgprs > 256 ||
        binding.prepared_sgprs < entry.original_sgprs || binding.prepared_sgprs > 106 ||
        binding.prepared_vgprs < entry.original_vgprs || binding.prepared_vgprs > 256)
      return reject(error, ContractErrorCode::InvalidFacts,
                    "invalid or duplicate image entry binding");
    previous_descriptor_end = entry.key + 64;
    prepared_descriptors.emplace_back(binding.prepared_descriptor_offset,
                                      binding.prepared_descriptor_offset + 64);
    if (!add_extent(encoded_bytes, kEntryFixedBytes, limits.metadata_bytes) ||
        !add_extent(encoded_bytes, binding.name.size(), limits.metadata_bytes))
      return reject(error, ContractErrorCode::ResourceLimit, "image facts exceed metadata budget");
    if (entry.disposition == EntryDisposition::Native) {
      if (entry.selected || binding.original_group_bytes != binding.prepared_group_bytes ||
          entry.original_sgprs != binding.prepared_sgprs ||
          entry.original_vgprs != binding.prepared_vgprs)
        return reject(error, ContractErrorCode::InvalidFacts,
                      "native entry changes ABI or resources");
    } else if (entry.disposition == EntryDisposition::Selected) {
      if (!entry.selected)
        return reject(error, ContractErrorCode::InvalidFacts, "selected entry has no manifest");
      const Manifest &manifest = *entry.selected;
      if (validate_manifest(manifest, error, limits).failed())
        return Result::failure();
      if (manifest.entry != binding || manifest.original != image.original ||
          manifest.prepared != image.prepared || manifest.target != image.target ||
          manifest.wave_size != image.wave_size ||
          manifest.code_object_abi != image.code_object_abi ||
          manifest.original_text_bytes != image.original_text_bytes ||
          manifest.prepared_text_bytes != image.prepared_text_bytes)
        return reject(error, ContractErrorCode::IdentityMismatch,
                      "entry manifest differs from image");
      if (!add_extent(total_sites, manifest.sites.size(), limits.sites) ||
          !add_extent(encoded_bytes,
                      kManifestFixedBytes + manifest.entry.name.size() +
                          uint64_t{manifest.sites.size()} * kSiteBytes +
                          uint64_t{manifest.barriers.size()} * kBarrierBytes,
                      limits.metadata_bytes))
        return reject(error, ContractErrorCode::ResourceLimit,
                      "aggregate image facts exceed budget");
      for (const SiteFacts &site : manifest.sites) {
        original_sites.emplace_back(site.original_offset,
                                    site.original_offset + site.original_size);
        prepared_observations.emplace_back(site.prepared_guest_offset,
                                           site.prepared_guest_offset + site.prepared_guest_size);
        prepared_observations.emplace_back(site.probe_offset, site.probe_offset + site.probe_size);
      }
      for (const BarrierFacts &barrier : manifest.barriers) {
        original_sites.emplace_back(barrier.original_drain, barrier.original_wait + 4);
        if (barrier.prepared_signal != barrier.prepared_wait)
          prepared_observations.emplace_back(barrier.prepared_signal, barrier.prepared_signal + 4);
        prepared_observations.emplace_back(barrier.prepared_wait,
                                           barrier.completion_probe + barrier.probe_size);
      }
      selected_keys.push_back(entry.key);
    } else {
      return reject(error, ContractErrorCode::InvalidFacts, "unknown entry disposition");
    }
  }
  for (std::vector<std::pair<uint64_t, uint64_t>> *ranges :
       {&prepared_descriptors, &original_sites, &prepared_observations}) {
    std::ranges::sort(*ranges);
    for (size_t index = 1; index < ranges->size(); ++index)
      if ((*ranges)[index].first < (*ranges)[index - 1].second)
        return reject(error, ContractErrorCode::InvalidFacts, "overlapping image entry ownership");
  }
  const std::optional<ScopeKind> scope = selected_scope(image);
  if (!scope)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "selected entries must share one qualified scope");
  if (selected_keys.empty() ||
      image.selection_profile != selection_profile_digest(selected_keys, *scope))
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "image selection profile differs from entries");
  return Result::success();
}

FailureOr<std::vector<uint8_t>> encode_image_facts(const ImageFacts &image, ContractError *error,
                                                   const ContractLimits &limits) {
  if (validate_image_facts(image, error, limits).failed())
    return Result::failure();
  Encoder payload;
  encode_identity(payload, image.original);
  encode_identity(payload, image.prepared);
  payload.bytes(image.selection_profile);
  payload.u32(image.target);
  payload.u32(image.wave_size);
  payload.u32(image.code_object_abi);
  payload.u64(image.original_text_bytes);
  payload.u64(image.prepared_text_bytes);
  payload.u32(static_cast<uint32_t>(image.entries.size()));
  for (const ImageEntryFacts &entry : image.entries) {
    payload.u64(entry.key);
    payload.u32(static_cast<uint32_t>(entry.disposition));
    payload.u32(entry.original_sgprs);
    payload.u32(entry.original_vgprs);
    encode_binding(payload, entry.binding);
    if (entry.selected) {
      const FailureOr<std::vector<uint8_t>> manifest =
          encode_manifest(*entry.selected, error, limits);
      if (manifest.failed())
        return Result::failure();
      payload.u64(manifest.value().size());
      payload.bytes(manifest.value());
    } else {
      payload.u64(0);
    }
  }
  const std::vector<uint8_t> contents = payload.take();
  Encoder envelope;
  envelope.bytes(kImageMagic);
  envelope.bytes(kProducerContract);
  envelope.u64(contents.size());
  envelope.bytes(digest(contents));
  envelope.bytes(contents);
  return envelope.take();
}

FailureOr<ImageFacts> decode_image_facts(std::span<const uint8_t> bytes, ContractError *error,
                                         const ContractLimits &limits) {
  if (error != nullptr)
    *error = {};
  if (bytes.size() > limits.metadata_bytes)
    return reject(error, ContractErrorCode::ResourceLimit, "image facts exceed metadata budget");
  Decoder envelope(bytes);
  std::array<uint8_t, 8> magic{};
  std::array<uint8_t, 16> contract{};
  uint64_t size = 0;
  Digest checksum{};
  if (!envelope.bytes(magic) || !envelope.bytes(contract) || !envelope.u64(size) ||
      !envelope.bytes(checksum) || magic != kImageMagic || size != envelope.remaining())
    return reject(error, ContractErrorCode::InvalidEncoding, "invalid image-facts framing");
  if (contract != kProducerContract)
    return reject(error, ContractErrorCode::IncompatibleContract, "producer contract mismatch");
  if (digest(envelope.remaining_bytes()) != checksum)
    return reject(error, ContractErrorCode::IdentityMismatch, "image-facts digest mismatch");
  Decoder payload(envelope.remaining_bytes());
  ImageFacts image;
  uint32_t count = 0;
  if (!decode_identity(payload, image.original) || !decode_identity(payload, image.prepared) ||
      !payload.bytes(image.selection_profile) || !payload.u32(image.target) ||
      !payload.u32(image.wave_size) || !payload.u32(image.code_object_abi) ||
      !payload.u64(image.original_text_bytes) || !payload.u64(image.prepared_text_bytes) ||
      !payload.u32(count))
    return reject(error, ContractErrorCode::InvalidEncoding, "truncated common image facts");
  if (count == 0 || count > kImageEntryLimit)
    return reject(error, ContractErrorCode::ResourceLimit, "image entry count exceeds budget");
  if (uint64_t{count} * kEntryFixedBytes > payload.remaining())
    return reject(error, ContractErrorCode::InvalidEncoding, "entry count disagrees with extent");
  image.entries.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    ImageEntryFacts entry;
    uint32_t disposition = 0;
    uint64_t manifest_bytes = 0;
    if (!payload.u64(entry.key) || !payload.u32(disposition) ||
        !payload.u32(entry.original_sgprs) || !payload.u32(entry.original_vgprs) ||
        !decode_binding(payload, entry.binding, limits.string_bytes) ||
        !payload.u64(manifest_bytes) || manifest_bytes > payload.remaining())
      return reject(error, ContractErrorCode::InvalidEncoding, "truncated image entry facts");
    entry.disposition = static_cast<EntryDisposition>(disposition);
    if (manifest_bytes != 0) {
      FailureOr<Manifest> manifest =
          decode_manifest(payload.remaining_bytes().first(manifest_bytes), error, limits);
      if (manifest.failed())
        return Result::failure();
      entry.selected = std::move(manifest.value());
    }
    if (!payload.skip(manifest_bytes))
      return reject(error, ContractErrorCode::InvalidEncoding, "truncated entry manifest");
    image.entries.push_back(std::move(entry));
  }
  if (payload.remaining() != 0)
    return reject(error, ContractErrorCode::InvalidEncoding, "trailing image facts");
  if (validate_image_facts(image, error, limits).failed())
    return Result::failure();
  return image;
}

Result match_image_facts(const ImageFacts &image, std::span<const uint8_t> original,
                         std::span<const uint8_t> prepared,
                         const Digest &expected_selection_profile, ContractError *error,
                         const ContractLimits &limits) {
  if (validate_image_facts(image, error, limits).failed())
    return Result::failure();
  if (image.selection_profile != expected_selection_profile ||
      image.original.bytes != original.size() || image.prepared.bytes != prepared.size() ||
      image.original.sha256 != digest(original) || image.prepared.sha256 != digest(prepared))
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "image or configured selection identity mismatch");
  return Result::success();
}

} // namespace rocjitsu::consan
