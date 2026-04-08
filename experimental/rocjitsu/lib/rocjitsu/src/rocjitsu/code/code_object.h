// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file code_object.h
/// @brief Base classes for ELF headers, sections, and code objects.

#ifndef ROCJITSU_CODE_CODE_OBJECT_H_
#define ROCJITSU_CODE_CODE_OBJECT_H_

#include <cstddef>
#include <cstdint>
#include <memory>
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

  /// @brief Raw section data.
  /// @returns Pointer to the section contents, or nullptr if empty.
  const char *data() const { return data_.get(); }

  /// @brief Index into the section header string table for this section's name.
  /// @returns String table index.
  virtual uint32_t sectionHeaderNameIdx() const = 0;

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

  /// @brief .rodata sections containing read-only data.
  /// @returns Vector of pointers to .rodata sections.
  const std::vector<const Section *> &rodata_sections() const { return rodata_sections_; }

protected:
  bool is_valid_ = true;
  std::vector<char> image_;
  std::unique_ptr<Header> header_;
  std::vector<std::unique_ptr<Section>> sections_;
  std::vector<const Section *> text_sections_;
  std::vector<const Section *> rodata_sections_;
};

} // namespace rocjitsu

#endif // ROCJITSU_CODE_CODE_OBJECT_H_
