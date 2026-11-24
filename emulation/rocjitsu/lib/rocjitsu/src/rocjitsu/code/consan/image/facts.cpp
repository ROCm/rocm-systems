// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/facts.h"

#include "rocjitsu/code/consan/contract_internal.h"
#include "rocjitsu/code/consan/target.h"
#include "util/bit.h"
#include "util/sha256.h"

#include <algorithm>
#include <bit>
#include <set>

namespace rocjitsu::consan {
namespace {
constexpr std::array<uint8_t, 8> kImageFactsMagic = {'R', 'J', 'C', 'S', 'I', 'M', 'G', 'F'};
constexpr uint64_t kEnvelopeBytes = 64;
constexpr uint64_t kImageFixedBytes = 144;
constexpr uint64_t kEntryFixedBytes = 124;
constexpr uint64_t kNativeCodeRangeBytes = 24;
constexpr uint64_t kManifestFixedBytes = 64 + 244;
constexpr uint64_t kSiteBytes = 68;
constexpr uint64_t kBarrierBytes = 56;

bool add_extent(uint64_t &total, uint64_t amount, uint64_t limit) {
  if (total > limit || amount > limit - total) {
    return false;
  }
  total += amount;
  return true;
}
} // namespace

Digest selection_profile_digest(std::span<const EntryKey> selected_keys, ScopeKind scope,
                                const Digest &options) {
  util::Sha256 hash;
  hash.update("consan-image-selection-v2");
  std::array<uint8_t, 4> scope_bytes{};
  wire::write_u32(scope_bytes, 0, static_cast<uint32_t>(scope));
  hash.update(scope_bytes.data(), scope_bytes.size());
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
    if (!entry.selected) {
      continue;
    }
    if (scope && *scope != entry.selected->scope) {
      return std::nullopt;
    }
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
                    .selection_profile =
                        selection_profile_digest(keys, manifest.scope, manifest.options),
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

Result validate_image_facts(const ImageFacts &image, util::DiagnosticEmitter diagnostic,
                            const ContractLimits &limits) {
  const TargetProfile *target = target_profile(image.target);
  if (image.entries.empty() || image.entries.size() > kImageEntryLimit ||
      image.original.bytes > limits.image_bytes || image.prepared.bytes > limits.image_bytes) {
    return diagnostic.emit() << "image entry or byte budget exceeded";
  }
  if (target == nullptr || !supports_wave(*target, image.wave_size) || image.code_object_abi != 5 ||
      image.original.bytes == 0 || image.prepared.bytes == 0 || image.original_text_bytes == 0 ||
      image.original_text_bytes > image.original.bytes || image.prepared_text_bytes == 0 ||
      image.prepared_text_bytes > image.prepared.bytes) {
    return diagnostic.emit() << "invalid common image facts";
  }

  uint64_t encoded_bytes = kEnvelopeBytes + kImageFixedBytes;
  uint64_t total_sites = 0;
  std::vector<EntryKey> selected_keys;
  const Digest *selected_options = nullptr;
  std::set<std::string> names;
  std::set<uint64_t> original_entries;
  std::set<uint64_t> prepared_entries;
  std::vector<std::pair<uint64_t, uint64_t>> prepared_descriptors;
  std::vector<std::pair<uint64_t, uint64_t>> original_sites;
  std::vector<std::pair<uint64_t, uint64_t>> prepared_observations;
  uint64_t previous_descriptor_end = 0;
  for (const ImageEntryFacts &entry : image.entries) {
    const EntryFacts &binding = entry.binding;
    if (binding.name.size() > limits.string_bytes) {
      return diagnostic.emit() << "entry name exceeds budget";
    }
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
        !std::has_single_bit(binding.kernarg_alignment) || binding.original_kernarg_bytes == 0 ||
        binding.maximum_workgroup_size == 0 || binding.maximum_workgroup_size > 1024 ||
        binding.original_group_bytes > 65536 || binding.prepared_group_bytes > 65536 ||
        !valid_workgroup_requirement(binding.required_workgroup_size,
                                     binding.maximum_workgroup_size) ||
        entry.original_sgprs == 0 || entry.original_sgprs > 106 || entry.original_vgprs == 0 ||
        entry.original_vgprs > 256 || binding.prepared_sgprs < entry.original_sgprs ||
        binding.prepared_sgprs > 106 || binding.prepared_vgprs < entry.original_vgprs ||
        binding.prepared_vgprs > 256) {
      return diagnostic.emit() << "invalid or duplicate image entry binding";
    }
    previous_descriptor_end = entry.key + 64;
    prepared_descriptors.emplace_back(binding.prepared_descriptor_offset,
                                      binding.prepared_descriptor_offset + 64);
    if (!add_extent(encoded_bytes, kEntryFixedBytes, limits.metadata_bytes) ||
        !add_extent(encoded_bytes, binding.name.size(), limits.metadata_bytes)) {
      return diagnostic.emit() << "image facts exceed metadata budget";
    }
    if (entry.disposition == EntryDisposition::Native) {
      if (entry.selected || entry.native_code.empty() ||
          binding.original_group_bytes != binding.prepared_group_bytes ||
          binding.original_private_bytes != binding.prepared_private_bytes ||
          entry.original_sgprs != binding.prepared_sgprs ||
          entry.original_vgprs != binding.prepared_vgprs) {
        return diagnostic.emit() << "native entry changes ABI or resources";
      }
      if (entry.native_code.size() > UINT32_MAX ||
          entry.native_code.size() > limits.metadata_bytes / kNativeCodeRangeBytes ||
          !add_extent(encoded_bytes, entry.native_code.size() * kNativeCodeRangeBytes,
                      limits.metadata_bytes)) {
        return diagnostic.emit() << "native code ranges exceed metadata budget";
      }
      uint64_t previous_end = 0;
      bool entry_covered = false;
      for (const NativeCodeRange &range : entry.native_code) {
        if (range.bytes == 0 || range.bytes % 4 != 0 || range.original_offset % 4 != 0 ||
            range.prepared_offset % 4 != 0 || range.original_offset < previous_end ||
            !util::is_range_in_bounds(range.original_offset, range.bytes,
                                      image.original_text_bytes) ||
            !util::is_range_in_bounds(range.prepared_offset, range.bytes,
                                      image.prepared_text_bytes)) {
          return diagnostic.emit() << "invalid native code range";
        }
        previous_end = range.original_offset + range.bytes;
        if (binding.original_entry_offset >= range.original_offset &&
            binding.original_entry_offset - range.original_offset < range.bytes &&
            binding.prepared_entry_offset ==
                range.prepared_offset + binding.original_entry_offset - range.original_offset) {
          entry_covered = true;
        }
        original_sites.emplace_back(range.original_offset, previous_end);
        prepared_observations.emplace_back(range.prepared_offset,
                                           range.prepared_offset + range.bytes);
      }
      if (!entry_covered) {
        return diagnostic.emit() << "native code ranges omit the entry binding";
      }
    } else if (entry.disposition == EntryDisposition::Selected) {
      if (!entry.selected) {
        return diagnostic.emit() << "selected entry has no manifest";
      }
      if (!entry.native_code.empty()) {
        return diagnostic.emit() << "selected entry declares native code ranges";
      }
      const Manifest &manifest = *entry.selected;
      if (validate_manifest(manifest, diagnostic, limits).failed()) {
        return Result::failure();
      }
      if (manifest.entry != binding || manifest.original != image.original ||
          manifest.prepared != image.prepared || manifest.target != image.target ||
          manifest.wave_size != image.wave_size ||
          manifest.code_object_abi != image.code_object_abi ||
          manifest.original_text_bytes != image.original_text_bytes ||
          manifest.prepared_text_bytes != image.prepared_text_bytes) {
        return diagnostic.emit() << "entry manifest differs from image";
      }
      if (!add_extent(total_sites, manifest.sites.size(), limits.sites) ||
          !add_extent(encoded_bytes,
                      kManifestFixedBytes + manifest.entry.name.size() +
                          uint64_t{manifest.sites.size()} * kSiteBytes +
                          uint64_t{manifest.barriers.size()} * kBarrierBytes,
                      limits.metadata_bytes)) {
        return diagnostic.emit() << "aggregate image facts exceed budget";
      }
      for (const SiteFacts &site : manifest.sites) {
        original_sites.emplace_back(site.original_offset,
                                    site.original_offset + site.original_size);
        prepared_observations.emplace_back(site.prepared_guest_offset,
                                           site.prepared_guest_offset + site.prepared_guest_size);
        prepared_observations.emplace_back(site.probe_offset, site.probe_offset + site.probe_size);
      }
      for (const BarrierFacts &barrier : manifest.barriers) {
        original_sites.emplace_back(barrier.original_drain, barrier.original_wait + 4);
        if (barrier.prepared_signal != barrier.prepared_wait) {
          prepared_observations.emplace_back(barrier.prepared_signal, barrier.prepared_signal + 4);
        }
        prepared_observations.emplace_back(barrier.prepared_wait,
                                           barrier.completion_probe + barrier.probe_size);
      }
      if (selected_options && *selected_options != manifest.options) {
        return diagnostic.emit() << "selected entries must share one options profile";
      }
      selected_options = &manifest.options;
      selected_keys.push_back(entry.key);
    } else {
      return diagnostic.emit() << "unknown entry disposition";
    }
  }
  for (std::vector<std::pair<uint64_t, uint64_t>> *ranges :
       {&prepared_descriptors, &original_sites, &prepared_observations}) {
    std::ranges::sort(*ranges);
    for (size_t index = 1; index < ranges->size(); ++index) {
      if ((*ranges)[index].first < (*ranges)[index - 1].second) {
        return diagnostic.emit() << "overlapping image entry ownership";
      }
    }
  }
  const std::optional<ScopeKind> scope = selected_scope(image);
  if (!scope) {
    return diagnostic.emit() << "selected entries must share one qualified scope";
  }
  if (!selected_options || image.selection_profile !=
                               selection_profile_digest(selected_keys, *scope, *selected_options)) {
    return diagnostic.emit() << "image selection profile differs from entries";
  }
  return Result::success();
}

FailureOr<std::vector<uint8_t>> encode_image_facts(const ImageFacts &image,
                                                   util::DiagnosticEmitter diagnostic,
                                                   const ContractLimits &limits) {
  if (validate_image_facts(image, diagnostic, limits).failed()) {
    return Result::failure();
  }
  wire::Encoder payload;
  wire::encode_identity(payload, image.original);
  wire::encode_identity(payload, image.prepared);
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
    wire::encode_entry(payload, entry.binding);
    payload.u32(static_cast<uint32_t>(entry.native_code.size()));
    for (const NativeCodeRange &range : entry.native_code) {
      payload.u64(range.original_offset);
      payload.u64(range.prepared_offset);
      payload.u64(range.bytes);
    }
    if (entry.selected) {
      const FailureOr<std::vector<uint8_t>> manifest =
          encode_manifest(*entry.selected, diagnostic, limits);
      if (manifest.failed()) {
        return Result::failure();
      }
      payload.u64(manifest.value().size());
      payload.bytes(manifest.value());
    } else {
      payload.u64(0);
    }
  }
  const std::vector<uint8_t> contents = payload.take();
  wire::Encoder envelope;
  envelope.bytes(kImageFactsMagic);
  envelope.bytes(kProducerContract);
  envelope.u64(contents.size());
  envelope.bytes(digest(contents));
  envelope.bytes(contents);
  return envelope.take();
}

FailureOr<ImageFacts> decode_image_facts(std::span<const uint8_t> bytes,
                                         util::DiagnosticEmitter diagnostic,
                                         const ContractLimits &limits) {
  if (bytes.size() > limits.metadata_bytes) {
    return diagnostic.emit() << "image facts exceed metadata budget";
  }
  wire::Decoder envelope(bytes);
  std::array<uint8_t, 8> magic{};
  std::array<uint8_t, 16> contract{};
  uint64_t size = 0;
  Digest checksum{};
  if (!envelope.bytes(magic) || !envelope.bytes(contract) || !envelope.u64(size) ||
      !envelope.bytes(checksum) || magic != kImageFactsMagic || size != envelope.remaining()) {
    return diagnostic.emit() << "invalid image-facts framing";
  }
  if (contract != kProducerContract) {
    return diagnostic.emit() << "producer contract mismatch";
  }
  if (digest(envelope.remaining_bytes()) != checksum) {
    return diagnostic.emit() << "image-facts digest mismatch";
  }
  wire::Decoder payload(envelope.remaining_bytes());
  ImageFacts image;
  uint32_t count = 0;
  if (!wire::decode_identity(payload, image.original) ||
      !wire::decode_identity(payload, image.prepared) || !payload.bytes(image.selection_profile) ||
      !payload.u32(image.target) || !payload.u32(image.wave_size) ||
      !payload.u32(image.code_object_abi) || !payload.u64(image.original_text_bytes) ||
      !payload.u64(image.prepared_text_bytes) || !payload.u32(count)) {
    return diagnostic.emit() << "truncated common image facts";
  }
  if (count == 0 || count > kImageEntryLimit) {
    return diagnostic.emit() << "image entry count exceeds budget";
  }
  if (uint64_t{count} * kEntryFixedBytes > payload.remaining()) {
    return diagnostic.emit() << "entry count disagrees with extent";
  }
  image.entries.reserve(count);
  for (uint32_t index = 0; index < count; ++index) {
    ImageEntryFacts entry;
    uint32_t disposition = 0;
    uint32_t native_ranges = 0;
    uint64_t manifest_bytes = 0;
    if (!payload.u64(entry.key) || !payload.u32(disposition) ||
        !payload.u32(entry.original_sgprs) || !payload.u32(entry.original_vgprs) ||
        !wire::decode_entry(payload, entry.binding, limits.string_bytes) ||
        !payload.u32(native_ranges)) {
      return diagnostic.emit() << "truncated image entry facts";
    }
    if (uint64_t{native_ranges} * kNativeCodeRangeBytes > payload.remaining()) {
      return diagnostic.emit() << "truncated native code ranges";
    }
    entry.native_code.reserve(native_ranges);
    for (uint32_t index = 0; index < native_ranges; ++index) {
      NativeCodeRange range;
      if (!payload.u64(range.original_offset) || !payload.u64(range.prepared_offset) ||
          !payload.u64(range.bytes)) {
        return diagnostic.emit() << "truncated native code range";
      }
      entry.native_code.push_back(range);
    }
    if (!payload.u64(manifest_bytes) || manifest_bytes > payload.remaining()) {
      return diagnostic.emit() << "truncated image entry manifest";
    }
    entry.disposition = static_cast<EntryDisposition>(disposition);
    if (manifest_bytes != 0) {
      FailureOr<Manifest> manifest =
          decode_manifest(payload.remaining_bytes().first(manifest_bytes), diagnostic, limits);
      if (manifest.failed()) {
        return Result::failure();
      }
      entry.selected = std::move(manifest.value());
    }
    if (!payload.skip(manifest_bytes)) {
      return diagnostic.emit() << "truncated entry manifest";
    }
    image.entries.push_back(std::move(entry));
  }
  if (payload.remaining() != 0) {
    return diagnostic.emit() << "trailing image facts";
  }
  if (validate_image_facts(image, diagnostic, limits).failed()) {
    return Result::failure();
  }
  return image;
}

Result match_image_facts(const ImageFacts &image, std::span<const uint8_t> original,
                         std::span<const uint8_t> prepared,
                         const Digest &expected_selection_profile,
                         util::DiagnosticEmitter diagnostic, const ContractLimits &limits) {
  if (validate_image_facts(image, diagnostic, limits).failed()) {
    return Result::failure();
  }
  if (image.selection_profile != expected_selection_profile ||
      image.original.bytes != original.size() || image.prepared.bytes != prepared.size() ||
      image.original.sha256 != digest(original) || image.prepared.sha256 != digest(prepared)) {
    return diagnostic.emit() << "image or configured selection identity mismatch";
  }
  return Result::success();
}

} // namespace rocjitsu::consan
