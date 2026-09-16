// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/sidecar_registry.h"

#include "rocjitsu/code/amdgpu_elf.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu::hooks {
namespace {

[[nodiscard]] bool elf_range_in_bounds(size_t image_size, uint64_t offset, uint64_t size) {
  const uint64_t limit = static_cast<uint64_t>(image_size);
  return offset <= limit && size <= limit - offset;
}

[[nodiscard]] std::string normalize_kernel_symbol_name(std::string_view symbol_name) {
  constexpr std::string_view kDescriptorSuffix = ".kd";
  if (symbol_name.ends_with(kDescriptorSuffix))
    symbol_name.remove_suffix(kDescriptorSuffix.size());
  return std::string(symbol_name);
}

} // namespace

std::optional<std::vector<uint8_t>> read_metadata_section(std::span<const uint8_t> image,
                                                          std::string_view section_name) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return std::vector<uint8_t>{};

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_machine != EM_AMDGPU) {
    return std::vector<uint8_t>{};
  }
  if (header.e_shoff == 0 || header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shnum == 0)
    return std::vector<uint8_t>{};
  if (header.e_shstrndx == SHN_UNDEF || header.e_shstrndx >= header.e_shnum)
    return std::nullopt;
  if (!elf_range_in_bounds(image.size(), header.e_shoff,
                           static_cast<uint64_t>(header.e_shnum) * sizeof(Elf64_Shdr))) {
    return std::nullopt;
  }

  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));

  const Elf64_Shdr &shstrtab = sections[header.e_shstrndx];
  if (shstrtab.sh_type != SHT_STRTAB ||
      !elf_range_in_bounds(image.size(), shstrtab.sh_offset, shstrtab.sh_size)) {
    return std::nullopt;
  }
  const char *section_names = reinterpret_cast<const char *>(image.data() + shstrtab.sh_offset);

  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type == SHT_NULL || section.sh_name >= shstrtab.sh_size)
      continue;
    const size_t max_name = static_cast<size_t>(shstrtab.sh_size - section.sh_name);
    const std::string_view name(section_names + section.sh_name,
                                strnlen(section_names + section.sh_name, max_name));
    if (name != section_name)
      continue;
    if (!elf_range_in_bounds(image.size(), section.sh_offset, section.sh_size))
      return std::nullopt;
    const auto payload =
        image.subspan(static_cast<size_t>(section.sh_offset), static_cast<size_t>(section.sh_size));
    return std::vector<uint8_t>(payload.begin(), payload.end());
  }
  return std::vector<uint8_t>{};
}

std::optional<std::vector<SidecarVariantMetadata>>
parse_sidecar_metadata_section(std::span<const uint8_t> image) {
  auto bytes = read_metadata_section(image, kSidecarMetadataSectionName);
  if (!bytes)
    return std::nullopt;
  if (bytes->empty())
    return std::vector<SidecarVariantMetadata>{};
  return parse_sidecar_metadata(*bytes);
}

std::optional<uint64_t> ResolvedSidecarKernel::variant_object(std::string_view variant_name) const {
  const auto it = std::ranges::find(variants, variant_name, &ResolvedSidecarVariant::variant_name);
  if (it == variants.end() || it->kernel_object == 0)
    return std::nullopt;
  return it->kernel_object;
}

SidecarRegistry &SidecarRegistry::instance() {
  static SidecarRegistry registry;
  return registry;
}

void SidecarRegistry::record_load(uint64_t executable, uint64_t load_id,
                                  std::vector<SidecarVariantMetadata> metadata) {
  // HSA makes the loaded-code-object output optional. A zero load_id therefore
  // means "not reported", not "the load did not happen".
  if (executable == 0 || metadata.empty())
    return;

  std::erase_if(metadata, [](const SidecarVariantMetadata &variant) {
    return variant.kernel_name.empty() || variant.variant_name.empty() ||
           variant.normal_descriptor_vaddr == 0 || variant.variant_descriptor_vaddr == 0;
  });
  if (metadata.empty())
    return;

  std::lock_guard lock(mutex_);
  // Keep the wire records flat. Grouping them into nested owning vectors made
  // pending metadata outlive the simple load/symbol hand-off and was the source
  // of the teardown invalid free this registry is intended to prevent.
  loads_.push_back(
      LoadEntry{.executable = executable, .load_id = load_id, .variants = std::move(metadata)});
}

void SidecarRegistry::record_symbol(uint64_t executable, std::string_view symbol_name,
                                    uint64_t symbol) {
  if (executable == 0 || symbol == 0 || symbol_name.empty())
    return;
  const std::string kernel_name = normalize_kernel_symbol_name(symbol_name);

  std::lock_guard lock(mutex_);
  SymbolRecord &known = known_symbols_[symbol];
  if (known.executable != 0 && known.executable != executable) {
    // HSA handles may be reused after executable destruction. Be defensive if
    // a client observes reuse before the old executable callback reaches us.
    if (known.kernel_object != 0) {
      const auto old_object = kernel_object_symbols_.find(known.kernel_object);
      if (old_object != kernel_object_symbols_.end() && old_object->second == symbol)
        kernel_object_symbols_.erase(old_object);
    }
    const auto old_sidecar = sidecar_symbols_.find(symbol);
    if (old_sidecar != sidecar_symbols_.end()) {
      const uint64_t old_object = old_sidecar->second.resolved.normal_kernel_object;
      const auto reverse = sidecar_object_symbols_.find(old_object);
      if (reverse != sidecar_object_symbols_.end() && reverse->second == symbol)
        sidecar_object_symbols_.erase(reverse);
      sidecar_symbols_.erase(old_sidecar);
    }
    known = {};
  }
  // Do not replace the record here. Symbol lookup is repeatable and often
  // occurs after KERNEL_OBJECT was already queried; replacing it would erase
  // the object and private-size facts needed by packet rewriting.
  known.executable = executable;
  known.kernel_name = kernel_name;

  const auto associated = sidecar_symbols_.find(symbol);
  if (associated != sidecar_symbols_.end() &&
      associated->second.resolved.executable == executable &&
      associated->second.resolved.kernel_name == kernel_name) {
    return;
  }
  for (size_t load_index = loads_.size(); load_index > 0; --load_index) {
    LoadEntry &load = loads_[load_index - 1];
    if (load.executable != executable)
      continue;
    const auto first =
        std::ranges::find(load.variants, kernel_name, &SidecarVariantMetadata::kernel_name);
    if (first == load.variants.end())
      continue;

    PendingResolvedKernel pending{};
    pending.resolved.executable = executable;
    pending.resolved.load_id = load.load_id;
    pending.resolved.symbol = symbol;
    pending.resolved.kernel_name = kernel_name;
    pending.normal_descriptor_vaddr = first->normal_descriptor_vaddr;
    bool valid = true;
    for (const SidecarVariantMetadata &variant : load.variants) {
      if (variant.kernel_name != kernel_name)
        continue;
      if (variant.normal_descriptor_vaddr != pending.normal_descriptor_vaddr ||
          std::ranges::find(pending.resolved.variants, variant.variant_name,
                            &ResolvedSidecarVariant::variant_name) !=
              pending.resolved.variants.end()) {
        valid = false;
        continue;
      }
      pending.resolved.variants.push_back(
          ResolvedSidecarVariant{.variant_name = variant.variant_name});
      pending.variant_descriptor_vaddrs.push_back(variant.variant_descriptor_vaddr);
    }
    std::erase_if(load.variants, [&](const SidecarVariantMetadata &variant) {
      return variant.kernel_name == kernel_name;
    });
    if (load.variants.empty())
      loads_.erase(loads_.begin() + static_cast<std::ptrdiff_t>(load_index - 1));
    if (valid && !pending.resolved.variants.empty())
      sidecar_symbols_[symbol] = std::move(pending);
    return;
  }
}

void SidecarRegistry::note_kernel_object(uint64_t symbol, uint64_t kernel_object,
                                         uint32_t private_segment_size) {
  if (symbol == 0 || kernel_object == 0)
    return;

  std::lock_guard lock(mutex_);
  auto known = known_symbols_.find(symbol);
  if (known != known_symbols_.end()) {
    if (known->second.kernel_object != 0 && known->second.kernel_object != kernel_object) {
      const auto old = kernel_object_symbols_.find(known->second.kernel_object);
      if (old != kernel_object_symbols_.end() && old->second == symbol)
        kernel_object_symbols_.erase(old);
    }
    known->second.kernel_object = kernel_object;
    known->second.private_segment_size = private_segment_size;
    kernel_object_symbols_[kernel_object] = symbol;
  }

  auto sidecars = sidecar_symbols_.find(symbol);
  if (sidecars == sidecar_symbols_.end() || sidecars->second.normal_descriptor_vaddr == 0 ||
      kernel_object < sidecars->second.normal_descriptor_vaddr) {
    return;
  }
  PendingResolvedKernel &pending = sidecars->second;
  const uint64_t load_base = kernel_object - pending.normal_descriptor_vaddr;
  for (const uint64_t variant_vaddr : pending.variant_descriptor_vaddrs) {
    if (variant_vaddr > std::numeric_limits<uint64_t>::max() - load_base)
      return;
  }

  if (pending.resolved.normal_kernel_object != 0 &&
      pending.resolved.normal_kernel_object != kernel_object) {
    const auto old = sidecar_object_symbols_.find(pending.resolved.normal_kernel_object);
    if (old != sidecar_object_symbols_.end() && old->second == symbol)
      sidecar_object_symbols_.erase(old);
  }
  pending.resolved.normal_kernel_object = kernel_object;
  for (size_t i = 0; i < pending.resolved.variants.size(); ++i)
    pending.resolved.variants[i].kernel_object = load_base + pending.variant_descriptor_vaddrs[i];
  sidecar_object_symbols_[kernel_object] = symbol;
}

std::optional<ResolvedSidecarKernel>
SidecarRegistry::find_by_kernel_object(uint64_t kernel_object) {
  if (kernel_object == 0)
    return std::nullopt;
  std::lock_guard lock(mutex_);
  const auto object = sidecar_object_symbols_.find(kernel_object);
  if (object == sidecar_object_symbols_.end())
    return std::nullopt;
  const auto pending = sidecar_symbols_.find(object->second);
  if (pending == sidecar_symbols_.end() ||
      pending->second.resolved.normal_kernel_object != kernel_object)
    return std::nullopt;
  return pending->second.resolved;
}

std::optional<std::string> SidecarRegistry::kernel_name_for_object(uint64_t kernel_object) {
  if (kernel_object == 0)
    return std::nullopt;
  std::lock_guard lock(mutex_);
  const auto object = kernel_object_symbols_.find(kernel_object);
  if (object == kernel_object_symbols_.end())
    return std::nullopt;
  const auto symbol = known_symbols_.find(object->second);
  if (symbol == known_symbols_.end())
    return std::nullopt;
  return symbol->second.kernel_name;
}

uint32_t SidecarRegistry::private_segment_size_for_object(uint64_t kernel_object) {
  if (kernel_object == 0)
    return 0;
  std::lock_guard lock(mutex_);
  const auto object = kernel_object_symbols_.find(kernel_object);
  if (object == kernel_object_symbols_.end())
    return 0;
  const auto symbol = known_symbols_.find(object->second);
  return symbol == known_symbols_.end() ? 0 : symbol->second.private_segment_size;
}

void SidecarRegistry::erase_executable(uint64_t executable) {
  std::lock_guard lock(mutex_);
  std::erase_if(loads_, [=](const LoadEntry &load) { return load.executable == executable; });
  for (auto it = sidecar_symbols_.begin(); it != sidecar_symbols_.end();) {
    if (it->second.resolved.executable != executable) {
      ++it;
      continue;
    }
    const uint64_t object = it->second.resolved.normal_kernel_object;
    const auto reverse = sidecar_object_symbols_.find(object);
    if (reverse != sidecar_object_symbols_.end() && reverse->second == it->first)
      sidecar_object_symbols_.erase(reverse);
    it = sidecar_symbols_.erase(it);
  }
  for (auto it = known_symbols_.begin(); it != known_symbols_.end();) {
    if (it->second.executable != executable) {
      ++it;
      continue;
    }
    if (it->second.kernel_object != 0) {
      const auto reverse = kernel_object_symbols_.find(it->second.kernel_object);
      if (reverse != kernel_object_symbols_.end() && reverse->second == it->first)
        kernel_object_symbols_.erase(reverse);
    }
    it = known_symbols_.erase(it);
  }
}

void SidecarRegistry::clear() {
  std::lock_guard lock(mutex_);
  loads_.clear();
  sidecar_symbols_.clear();
  known_symbols_.clear();
  kernel_object_symbols_.clear();
  sidecar_object_symbols_.clear();
}

} // namespace rocjitsu::hooks
