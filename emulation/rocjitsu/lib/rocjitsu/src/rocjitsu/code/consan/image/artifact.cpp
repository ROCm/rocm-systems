// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/consan/target.h"
#include "util/bit.h"

#include <cstring>
#include <fstream>

namespace rocjitsu::consan {
namespace {

Result reject(ContractError *error, ContractErrorCode code, std::string message) {
  if (error != nullptr)
    *error = {code, std::move(message)};
  return Result::failure();
}

Result inspect_image(const Manifest &manifest, std::span<const uint8_t> image, bool prepared,
                     ContractError *error) {
  const auto *target = target_profile(manifest.target);
  if (target == nullptr)
    return reject(error, ContractErrorCode::InvalidFacts, "unsupported artifact target");
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return reject(error, ContractErrorCode::InvalidEncoding, "artifact has truncated ELF header");
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA ||
      header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU || (header.e_flags & EF_AMDGPU_MACH) != target->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0 || header.e_shentsize != sizeof(Elf64_Shdr) ||
      header.e_phentsize != sizeof(Elf64_Phdr) ||
      !util::is_range_in_bounds(header.e_shoff, uint64_t{header.e_shnum} * sizeof(Elf64_Shdr),
                                image.size()) ||
      !util::is_range_in_bounds(header.e_phoff, uint64_t{header.e_phnum} * sizeof(Elf64_Phdr),
                                image.size()) ||
      header.e_shstrndx >= header.e_shnum)
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "artifact ELF target, ABI or tables are inconsistent");
  Elf64_Shdr strings{};
  std::memcpy(&strings, image.data() + header.e_shoff + header.e_shstrndx * sizeof(strings),
              sizeof(strings));
  if (!util::is_range_in_bounds(strings.sh_offset, strings.sh_size, image.size()))
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "artifact section names are truncated");
  std::optional<Elf64_Shdr> text;
  size_t text_index = 0;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr section{};
    std::memcpy(&section, image.data() + header.e_shoff + index * sizeof(section), sizeof(section));
    if ((section.sh_flags & (SHF_EXECINSTR | SHF_ALLOC)) != (SHF_EXECINSTR | SHF_ALLOC))
      continue;
    if (text || section.sh_name >= strings.sh_size || section.sh_type != SHT_PROGBITS ||
        !util::is_range_in_bounds(section.sh_offset, section.sh_size, image.size()) ||
        section.sh_size > UINT64_MAX - section.sh_addr)
      return reject(error, ContractErrorCode::InvalidEncoding,
                    "artifact executable section is ambiguous");
    const auto name =
        image.subspan(strings.sh_offset + section.sh_name, strings.sh_size - section.sh_name);
    if (name.size() < 6 || std::memcmp(name.data(), ".text", 6) != 0)
      return reject(error, ContractErrorCode::InvalidEncoding,
                    "artifact executable section is not .text");
    text = section;
    text_index = index;
  }
  const uint64_t expected_text =
      prepared ? manifest.prepared_text_bytes : manifest.original_text_bytes;
  if (!text || text->sh_size != expected_text)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "artifact text extent differs from facts");
  const auto kernels = scan_kernel_descriptors(image, text->sh_offset, text->sh_size, text_index);
  const auto &entry = manifest.entry;
  if (kernels.size() != 1 || kernels.front().kernel_name != entry.name)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "artifact entry/symbol does not match facts");
  const auto &kernel = kernels.front();
  const uint64_t descriptor_offset =
      prepared ? entry.prepared_descriptor_offset : entry.original_descriptor_offset;
  const uint64_t entry_offset =
      prepared ? entry.prepared_entry_offset : entry.original_entry_offset;
  const std::optional<KernargExtensionLayout> wrapper =
      make_kernarg_extension_layout(entry.original_kernarg_bytes, kContextPointerPayloads);
  if (!wrapper)
    return reject(error, ContractErrorCode::ResourceLimit, "artifact kernarg wrapper overflows");
  const uint32_t kernarg = prepared ? wrapper->wrapper_size : entry.original_kernarg_bytes;
  if (kernel.descriptor_file_offset != descriptor_offset ||
      kernel.entry_text_offset != entry_offset || entry_offset > UINT64_MAX - text->sh_addr ||
      ((text->sh_addr + entry_offset) % 256) != 0 ||
      kernel_wavefront_size(target->arch, kernel.descriptor) != manifest.wave_size ||
      kernel.descriptor.kernarg_size != kernarg ||
      kernel.descriptor.private_segment_fixed_size != 0 ||
      kernel.descriptor.group_segment_fixed_size !=
          (prepared ? entry.prepared_group_bytes : entry.original_group_bytes) ||
      (target->id == 1100 && manifest.wave_size == 32 &&
       kernel_descriptor_initial_sgpr_count(target->arch, kernel.descriptor) < 16))
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "artifact descriptor differs from its binding facts");
  bool loaded_descriptor = false;
  bool loaded_text = false;
  for (size_t index = 0; index < header.e_phnum; ++index) {
    Elf64_Phdr segment{};
    std::memcpy(&segment, image.data() + header.e_phoff + index * sizeof(segment), sizeof(segment));
    if (segment.p_type != PT_LOAD)
      continue;
    if (!util::is_range_in_bounds(segment.p_offset, segment.p_filesz, image.size()) ||
        segment.p_filesz > segment.p_memsz || segment.p_memsz > UINT64_MAX - segment.p_vaddr)
      return reject(error, ContractErrorCode::InvalidEncoding,
                    "artifact load segment has invalid extent");
    if (util::is_range_in_bounds(descriptor_offset, 64, segment.p_offset + segment.p_filesz) &&
        descriptor_offset >= segment.p_offset) {
      const uint64_t descriptor_vaddr = segment.p_vaddr + (descriptor_offset - segment.p_offset);
      const int64_t relative = kernel.descriptor.kernel_code_entry_byte_offset;
      const uint64_t expected = text->sh_addr + entry_offset;
      if (relative >= 0)
        loaded_descriptor |= uint64_t(relative) <= UINT64_MAX - descriptor_vaddr &&
                             descriptor_vaddr + uint64_t(relative) == expected;
      else {
        const uint64_t magnitude = uint64_t(-(relative + 1)) + 1;
        loaded_descriptor |=
            magnitude <= descriptor_vaddr && descriptor_vaddr - magnitude == expected;
      }
    }
    // This MVP has one executable section. Require its complete instruction
    // image, not just entry bytes or zero-filled memory beyond p_filesz.
    loaded_text |= text->sh_offset >= segment.p_offset &&
                   util::is_range_in_bounds(text->sh_offset, text->sh_size,
                                            segment.p_offset + segment.p_filesz) &&
                   (segment.p_flags & PF_X) != 0 &&
                   segment.p_vaddr + (text->sh_offset - segment.p_offset) == text->sh_addr;
  }
  if (!loaded_text)
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "artifact executable .text is not fully file-backed at its declared address");
  if (!loaded_descriptor)
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "artifact entry/descriptor is not loaded");
  std::string metadata_error;
  const auto metadata = inspect_metadata(image, kernel, &metadata_error);
  if (metadata.failed())
    return reject(error, ContractErrorCode::InvalidFacts, metadata_error);
  if (metadata.value().maximum_workgroup_size != entry.maximum_workgroup_size)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "kernel launch bound differs from facts");
  if (prepared && (metadata.value().sgprs != entry.prepared_sgprs ||
                   metadata.value().vgprs != entry.prepared_vgprs))
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "prepared metadata resources differ from facts");
  if (prepared &&
      (target->arch == ROCJITSU_CODE_ARCH_CDNA3 || target->arch == ROCJITSU_CODE_ARCH_CDNA4)) {
    const uint32_t allocated_sgprs =
        (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc1,
                            amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
         1) *
        8;
    if (ordinary_sgpr_bound(target->arch, allocated_sgprs) < entry.prepared_sgprs)
      return reject(error, ContractErrorCode::IdentityMismatch,
                    "prepared descriptor does not allocate its declared ordinary SGPRs");
  }
  return Result::success();
}

FailureOr<std::vector<uint8_t>> read(const std::filesystem::path &path, uint64_t maximum,
                                     ContractError *error) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) > maximum)
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "missing or oversized artifact file: " + path.string());
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file)
    return reject(error, ContractErrorCode::InvalidEncoding,
                  "short artifact file read: " + path.string());
  return bytes;
}

} // namespace

Result verify_artifact_images(const Manifest &manifest, std::span<const uint8_t> original,
                              std::span<const uint8_t> prepared, ContractError *error) {
  if (match_images(manifest, original, prepared, error).failed() ||
      inspect_image(manifest, original, false, error).failed() ||
      inspect_image(manifest, prepared, true, error).failed())
    return Result::failure();
  return Result::success();
}

FailureOr<LoadedArtifact> load_artifact(const std::filesystem::path &directory,
                                        std::span<const uint8_t> original,
                                        std::string_view concrete_target, ContractError *error) {
  const auto *target = target_profile(concrete_target);
  if (target == nullptr)
    return reject(error, ContractErrorCode::InvalidFacts,
                  "ConSan artifact requires a supported concrete GPU agent");
  const auto ready = read(directory / "ready", kProducerContract.size(), error);
  if (ready.failed())
    return Result::failure();
  if (!std::ranges::equal(ready.value(), kProducerContract))
    return reject(error, ContractErrorCode::IncompatibleContract,
                  "artifact ready marker has wrong contract");
  const ContractLimits limits;
  const auto bytes = read(directory / "facts.bin", limits.metadata_bytes, error);
  if (bytes.failed())
    return Result::failure();
  auto manifest = decode_manifest(bytes.value(), error);
  if (manifest.failed())
    return Result::failure();
  if (manifest.value().target != target->id)
    return reject(error, ContractErrorCode::IdentityMismatch,
                  "concrete agent differs from artifact target");
  auto image = read(directory / "prepared.hsaco", limits.image_bytes, error);
  if (image.failed() ||
      verify_artifact_images(manifest.value(), original, image.value(), error).failed())
    return Result::failure();
  return LoadedArtifact{std::move(manifest.value()), std::move(image.value())};
}

} // namespace rocjitsu::consan
