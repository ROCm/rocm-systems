// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file code_object.h
/// @brief Base classes for ELF headers, sections, and code objects.

#ifndef ROCJITSU_CODE_CODE_OBJECT_H_
#define ROCJITSU_CODE_CODE_OBJECT_H_

#include "rocjitsu/code/amdgpu_elf.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief Abstract base class for ELF headers.
class Header {
public:
  virtual ~Header() = default;
  /// @brief Offset of the program header table in the ELF file.
  /// @returns Byte offset from the start of the file.
  virtual uint64_t programHeaderOff() const = 0;

  /// @brief Number of entries in the program header table.
  /// @returns Program header count.
  virtual int numProgramHeaders() const = 0;

  /// @brief Offset of the section header table in the ELF file.
  /// @returns Byte offset from the start of the file.
  virtual uint64_t sectionHeaderOff() const = 0;

  /// @brief Number of entries in the section header table.
  /// @returns Section header count.
  virtual int numSectionHeaders() const = 0;

  /// @brief Byte stride between section-header table entries.
  /// @returns Byte stride declared by e_shentsize.
  virtual uint64_t sectionHeaderEntrySize() const = 0;

  /// @brief Index of the section header string table section.
  /// @returns Section index.
  virtual int sectionHeaderStrIdx() const = 0;

  /// @brief Processor-specific ELF header flags.
  /// @returns Flags value from the ELF header.
  virtual uint32_t flags() const = 0;
};

/// @brief Abstract base class for ELF sections.
class Section {
public:
  /// @brief Construct a section with the given name and raw data.
  /// @param[in] name Section name (e.g. ".text", ".rodata").
  /// @param[in] data Owned buffer holding the section contents.
  Section(std::string name, std::unique_ptr<char[]> data)
      : name_(std::move(name)), data_(std::move(data)) {}
  virtual ~Section() = default;

  /// @brief Section name (e.g. ".text", ".rodata").
  /// @returns Reference to the section name string.
  const std::string &name() const { return name_; }

  /// @brief Size of the section data in bytes.
  /// @returns Section size.
  virtual std::size_t size() const = 0;

  /// @brief Virtual address of the section (from ELF sh_addr).
  /// @returns Section virtual address, or 0 if not set.
  virtual uint64_t vaddr() const { return 0; }

  /// @brief Raw ELF section flags.
  /// @returns Section flags from sh_flags, or 0 when not backed by ELF metadata.
  virtual uint64_t flags() const { return 0; }

  /// @brief Raw section data.
  /// @returns Pointer to the section contents, or nullptr if empty.
  const char *data() const { return data_.get(); }

  /// @brief Index into the section header string table for this section's name.
  /// @returns String table index.
  virtual uint32_t sectionHeaderNameIdx() const = 0;

  /// @brief Index of this section in its owning ELF section-header table.
  /// @returns Section index when the section is backed by ELF metadata.
  virtual std::optional<size_t> sectionHeaderIndex() const { return std::nullopt; }

  /// @brief File offset of this section in the ELF image.
  /// @returns Byte offset from the start of the file.
  virtual uint64_t sectionOffset() const = 0;

protected:
  std::string name_;
  std::unique_ptr<char[]> data_;
};

/// @brief Base class for code objects containing machine code and metadata.
class CodeObject {
public:
  virtual ~CodeObject() = default;

  /// @brief Whether the code object was loaded and parsed successfully.
  /// @retval true The code object was loaded and parsed successfully.
  /// @retval false Loading or parsing failed.
  bool is_valid() const { return is_valid_; }

  /// @brief Size of the code object image in bytes.
  /// @returns Image size in bytes.
  std::size_t size() const { return image_.size(); }

  /// @brief .text sections containing executable machine code.
  /// @returns Vector of pointers to .text sections.
  const std::vector<const Section *> &text_sections() const { return text_sections_; }

  /// @brief Runtime-loaded ranges declared executable by ELF metadata.
  /// @returns Sections carrying both SHF_ALLOC and SHF_EXECINSTR, including
  ///          unmaterialized SHT_NOBITS ranges used for layout validation.
  const std::vector<const Section *> &allocated_executable_sections() const {
    return allocated_executable_sections_;
  }

  /// @brief .rodata sections containing read-only data.
  /// @returns Vector of pointers to .rodata sections.
  const std::vector<const Section *> &rodata_sections() const { return rodata_sections_; }

  /// @brief All sections in the code object.
  const std::vector<std::unique_ptr<Section>> &all_sections() const { return sections_; }

  /// @brief Raw ELF image data.
  const char *image_data() const { return image_.data(); }

  /// @brief Raw ELF image size in bytes.
  std::size_t image_size() const { return image_.size(); }

  /// @brief Memory size required to load the ELF at its virtual addresses.
  /// @returns The highest VA + size across all LOAD segments.
  std::size_t mem_size() const {
    uint64_t max_va = 0;
    for (const Elf64_Phdr &phdr : program_headers_) {
      if (phdr.p_type != PT_LOAD)
        continue;
      max_va = std::max(max_va, phdr.p_vaddr + phdr.p_memsz);
    }
    if (max_va > std::numeric_limits<std::size_t>::max())
      return image_.size();
    return max_va > 0 ? static_cast<std::size_t>(max_va) : image_.size();
  }

  /// @brief Load the ELF image into memory respecting LOAD segment VA mapping.
  /// @details Each PT_LOAD segment is copied from its file offset to its virtual
  /// address offset relative to base_addr. This handles GPU ELFs where .text has
  /// a different VA than file offset (common with AMDGPU code objects).
  /// @param memory Target memory to load into.
  /// @param base_addr Base address for loading.
  /// @returns False if adding the base address would overflow any load range;
  ///          no memory writes occur in that case.
  template <typename Memory> bool load_to_memory(Memory *memory, uint64_t base_addr) const {
    bool loaded_any = false;
    for (const Elf64_Phdr &phdr : program_headers_) {
      if (phdr.p_type != PT_LOAD)
        continue;
      if (phdr.p_vaddr > std::numeric_limits<uint64_t>::max() - base_addr ||
          phdr.p_memsz > std::numeric_limits<uint64_t>::max() - base_addr - phdr.p_vaddr)
        return false;
      loaded_any = true;
    }
    if (!loaded_any) {
      if (image_.size() > std::numeric_limits<uint64_t>::max() - base_addr)
        return false;
      memory->load_image(reinterpret_cast<const uint8_t *>(image_.data()), image_.size(),
                         base_addr);
      return true;
    }

    const auto *image = reinterpret_cast<const uint8_t *>(image_.data());
    for (const Elf64_Phdr &phdr : program_headers_) {
      if (phdr.p_type == PT_LOAD && phdr.p_filesz != 0)
        memory->load_image(image + phdr.p_offset, static_cast<std::size_t>(phdr.p_filesz),
                           base_addr + phdr.p_vaddr);
    }
    return true;
  }

  /// @brief Find the kernel descriptor symbol offset for a named kernel.
  /// @param kernel_name The kernel function name (e.g., "vector_add").
  /// @returns Offset of the ".kd" symbol within the loaded image, or 0 if not found.
  virtual uint64_t kernel_descriptor_offset(const std::string & /*kernel_name*/) const { return 0; }

protected:
  /// @brief Validate and retain program headers for device-image loading.
  /// @returns False for unsupported strides, invalid table ranges, or LOAD
  ///          segments whose file range or virtual-address arithmetic overflows.
  bool load_program_headers(const Elf64_Ehdr &ehdr) {
    // The patcher rewrites contiguous records and cannot preserve extensions.
    if (ehdr.e_phnum != 0 && ehdr.e_phentsize != sizeof(Elf64_Phdr))
      return false;
    const std::span image(reinterpret_cast<const uint8_t *>(image_.data()), image_.size());
    std::optional<std::vector<Elf64_Phdr>> phdrs =
        read_elf_table<Elf64_Phdr>(image, ehdr.e_phoff, ehdr.e_phnum, ehdr.e_phentsize);
    if (!phdrs)
      return false;
    for (const Elf64_Phdr &phdr : *phdrs) {
      if (phdr.p_type != PT_LOAD)
        continue;
      if ((phdr.p_filesz != 0 &&
           (phdr.p_offset > image_.size() || phdr.p_filesz > image_.size() - phdr.p_offset)) ||
          phdr.p_filesz > phdr.p_memsz ||
          phdr.p_memsz > std::numeric_limits<uint64_t>::max() - phdr.p_vaddr) {
        return false;
      }
    }
    program_headers_ = std::move(*phdrs);
    return true;
  }

  bool is_valid_ = true;
  std::vector<char> image_;
  std::unique_ptr<Header> header_;
  std::vector<std::unique_ptr<Section>> sections_;
  std::vector<const Section *> text_sections_;
  std::vector<const Section *> allocated_executable_sections_;
  std::vector<const Section *> rodata_sections_;
  std::vector<Elf64_Phdr> program_headers_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_CODE_OBJECT_H_
