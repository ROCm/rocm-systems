// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/checked_text_edits.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/text_padding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

Result reject(TextEditError &error, TextEditErrorCode code, std::string message,
              uint64_t offset = 0) {
  error = {.code = code, .offset = offset, .message = std::move(message)};
  return Result::failure();
}

std::span<const uint8_t> image_bytes(const AmdGpuCodeObject &object) {
  return {reinterpret_cast<const uint8_t *>(object.image_data()), object.image_size()};
}

bool reference_enters_edit(uint64_t offset, std::span<const CheckedTextEdit> edits) {
  const std::span<const CheckedTextEdit>::iterator next = std::upper_bound(
      edits.begin(), edits.end(), offset,
      [](uint64_t value, const CheckedTextEdit &edit) { return value < edit.offset; });
  if (next == edits.begin())
    return false;
  const CheckedTextEdit &edit = *std::prev(next);
  return offset > edit.offset && offset - edit.offset < edit.expected.size();
}

Result validate_load_file_aliases(const AmdGpuCodeObject &source,
                                  std::span<const CheckedTextEdit> edits, TextEditError &error) {
  const std::span<const uint8_t> image = image_bytes(source);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return reject(error, TextEditErrorCode::InvalidImage, "invalid PT_LOAD program header table",
                  edits.front().offset);
  for (const Elf64_Phdr &segment : *segments) {
    if (segment.p_type != PT_LOAD || segment.p_filesz == 0)
      continue;
    if (segment.p_offset > image.size() || segment.p_filesz > image.size() - segment.p_offset ||
        segment.p_filesz > segment.p_memsz ||
        segment.p_memsz > std::numeric_limits<uint64_t>::max() - segment.p_vaddr)
      return reject(error, TextEditErrorCode::InvalidImage, "invalid PT_LOAD extent",
                    edits.front().offset);
  }
  const Section &text = *source.text_sections().front();
  for (size_t left_index = 0; left_index < segments->size(); ++left_index) {
    const Elf64_Phdr &left = (*segments)[left_index];
    if (left.p_type != PT_LOAD || left.p_filesz == 0)
      continue;
    for (size_t right_index = left_index + 1; right_index < segments->size(); ++right_index) {
      const Elf64_Phdr &right = (*segments)[right_index];
      if (right.p_type != PT_LOAD || right.p_filesz == 0)
        continue;
      const uint64_t shared_begin = std::max(left.p_offset, right.p_offset);
      const uint64_t shared_end =
          std::min(left.p_offset + left.p_filesz, right.p_offset + right.p_filesz);
      if (shared_begin >= shared_end)
        continue;
      // Both file ranges and virtual extents were checked above. Their affine
      // mappings agree throughout the shared interval if they agree at its start.
      const uint64_t left_address = left.p_vaddr + (shared_begin - left.p_offset);
      const uint64_t right_address = right.p_vaddr + (shared_begin - right.p_offset);
      if (left_address == right_address)
        continue;
      for (const CheckedTextEdit &edit : edits) {
        // Source extent and per-edit bounds have already been validated, so
        // these file coordinates stay inside the immutable source image.
        const uint64_t edit_begin = text.sectionOffset() + edit.offset;
        const uint64_t edit_end = edit_begin + edit.expected.size();
        if (shared_begin < edit_end && edit_begin < shared_end)
          return reject(
              error, TextEditErrorCode::InvalidImage,
              "edited source bytes have PT_LOAD file aliases at different virtual addresses",
              edit.offset);
      }
    }
  }
  for (const CheckedTextEdit &edit : edits) {
    if (!elf_load_file_mappings_match_range(*segments, text.sectionOffset() + edit.offset,
                                            text.vaddr() + edit.offset, edit.expected.size()))
      return reject(error, TextEditErrorCode::InvalidImage,
                    "edited virtual addresses have conflicting PT_LOAD file backing", edit.offset);
  }
  return Result::success();
}

Result decode_replacement(Decoder &decoder, std::span<const uint8_t> bytes, uint64_t offset,
                          TextEditError &error) {
  std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  size_t cursor = 0;
  while (cursor < words.size()) {
    DecodeResult decoded =
        decoder.decode_window(std::span(words).subspan(cursor), offset + cursor * sizeof(uint32_t));
    if (decoded.failed() || decoded.value()->size() == 0)
      return reject(error, TextEditErrorCode::InvalidInstruction,
                    "replacement contains an invalid or truncated instruction", offset);
    if (decoded.value()->mnemonic() == "s_clause")
      return reject(error, TextEditErrorCode::Clause, "replacement may not introduce a clause",
                    offset);
    cursor += decoded.value()->size() / sizeof(uint32_t);
  }
  return Result::success();
}

} // namespace

CheckedTextEdits::CheckedTextEdits(const AmdGpuCodeObject &source, Decoder &decoder,
                                   const InstructionBoundaries &boundaries,
                                   const TextEditReferences &references)
    : source_(source), decoder_(decoder), boundaries_(boundaries), references_(references) {}

Result CheckedTextEdits::validate_source_extent(const AmdGpuCodeObject &source,
                                                TextEditError &error) {
  error = {};
  if (!source.is_valid() || source.text_sections().size() != 1)
    return reject(error, TextEditErrorCode::InvalidImage,
                  "expected one allocated executable .text section");
  const Section &text = *source.text_sections().front();
  if ((text.flags() & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR) ||
      !text.sectionHeaderIndex() || text.size() == 0 || text.size() % 4 != 0 ||
      text.vaddr() > std::numeric_limits<uint64_t>::max() - text.size() ||
      text.sectionOffset() > source.image_size() ||
      text.size() > source.image_size() - text.sectionOffset())
    return reject(error, TextEditErrorCode::InvalidImage, "invalid .text extent");
  return Result::success();
}

FailureOr<TextEditReferences> CheckedTextEdits::collect_references(
    const AmdGpuCodeObject &source, rj_code_arch_t architecture,
    const InstructionBoundaries &boundaries, const CodeObjectPatcher &patcher,
    std::span<const uint64_t> kernel_entries, TextEditError &error) {
  if (validate_source_extent(source, error).failed())
    return Result::failure();
  const Section &text = *source.text_sections().front();
  const std::vector<Elf64_Shdr> sections = patcher.section_headers();
  const auto aliases_text = [&](uint64_t offset, uint64_t size) {
    return size != 0 && offset < text.sectionOffset() + text.size() &&
           (offset >= text.sectionOffset() || size > text.sectionOffset() - offset);
  };
  Elf64_Ehdr header = {};
  std::memcpy(&header, source.image_data(), sizeof(header));
  if (aliases_text(0, sizeof(header)) ||
      aliases_text(header.e_shoff, uint64_t{header.e_shnum} * header.e_shentsize) ||
      aliases_text(header.e_phoff, uint64_t{header.e_phnum} * header.e_phentsize))
    return reject(error, TextEditErrorCode::InvalidImage, ".text aliases ELF headers");
  for (size_t index = 0; index < sections.size(); ++index) {
    const Elf64_Shdr &section = sections[index];
    if (index != *text.sectionHeaderIndex() && section.sh_type != SHT_NULL &&
        section.sh_type != SHT_NOBITS && aliases_text(section.sh_offset, section.sh_size))
      return reject(error, TextEditErrorCode::InvalidImage, ".text aliases another section");
    // Relocatable sections have section-relative coordinates, commonly all
    // zero; only linked images share the virtual address space checked here.
    if (header.e_type != ET_REL && index != *text.sectionHeaderIndex() &&
        (section.sh_flags & SHF_ALLOC) != 0 && section.sh_size != 0 &&
        section.sh_addr < text.vaddr() + text.size() &&
        (section.sh_addr >= text.vaddr() || section.sh_size > text.vaddr() - section.sh_addr))
      return reject(error, TextEditErrorCode::InvalidImage,
                    ".text virtual addresses alias another allocated section");
  }
  if (patcher.has_relocations_within_text() || patcher.has_malformed_rocr_relocation_section() ||
      patcher.has_rocr_rejected_none_relocation() || patcher.has_unsupported_relocation_to_text())
    return reject(error, TextEditErrorCode::UnsupportedReferences,
                  "unsupported relocation place or reference to .text");
  for (const Elf64_Shdr &section : sections) {
    if ((section.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) == (SHF_ALLOC | SHF_EXECINSTR) &&
        section.sh_size != 0 &&
        elf_has_rocr_dynamic_write_in_range(image_bytes(source), header, sections, section.sh_addr,
                                            section.sh_size))
      return reject(error, TextEditErrorCode::UnsupportedReferences,
                    "dynamic relocation overwrites executable source instructions");
  }
  TextEditReferences references;
  references.branch_addresses.assign(boundaries.reference_addresses().begin(),
                                     boundaries.reference_addresses().end());
  references.relative_text_references = patcher.relative_text_addend_offsets();
  references.kernel_entries.assign(kernel_entries.begin(), kernel_entries.end());
  const std::span<const uint32_t> words(reinterpret_cast<const uint32_t *>(text.data()),
                                        text.size() / sizeof(uint32_t));
  size_t cursor = 0;
  size_t boundary_index = 0;
  while (cursor < words.size()) {
    if (is_text_padding_word(architecture, words[cursor])) {
      ++cursor;
      continue;
    }
    // Both callers supply a full-section index. Verify non-padding coverage
    // before relying on the collected instruction and reference facts.
    const std::span<const InstructionBoundaries::Entry> entries = boundaries.entries();
    const InstructionBoundaries::Entry *entry =
        boundary_index < entries.size() ? &entries[boundary_index++] : nullptr;
    if (!entry || entry->offset != cursor * 4 || entry->size == 0 || entry->size % 4 != 0 ||
        entry->size / 4 > words.size() - cursor)
      return reject(error, TextEditErrorCode::InvalidInstruction,
                    "decoded index does not cover the complete source text", cursor * 4);
    cursor += entry->size / 4;
  }
  if (boundary_index != boundaries.entries().size())
    return reject(error, TextEditErrorCode::InvalidInstruction,
                  "decoded index extends outside source text");
  if (boundaries.reference_overflow())
    return reject(error, TextEditErrorCode::UnsupportedReferences, "branch reference overflows");
  // Fixed addresses preserve text references only at retained boundaries. A
  // symbol start/end in a replaced literal would become a new instruction entry.
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM)
      continue;
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0)
      return reject(error, TextEditErrorCode::UnsupportedReferences, "malformed symbol table");
    const std::optional<std::vector<Elf64_Sym>> symbols =
        read_elf_table<Elf64_Sym>(image_bytes(source), section.sh_offset,
                                  section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    if (!symbols)
      return reject(error, TextEditErrorCode::UnsupportedReferences, "malformed symbol table");
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_shndx != *text.sectionHeaderIndex())
        continue;
      // ET_REL symbol values already use section-relative coordinates. Linked
      // images use virtual addresses, as in CodeObjectPatcher's symbol updates.
      uint64_t start = symbol.st_value;
      if (header.e_type != ET_REL) {
        if (start < text.vaddr())
          return reject(error, TextEditErrorCode::UnsupportedReferences, "symbol leaves .text");
        start -= text.vaddr();
      }
      if (start > text.size() || symbol.st_size > text.size() - start)
        return reject(error, TextEditErrorCode::UnsupportedReferences, "symbol leaves .text");
      references.symbol_endpoints.push_back(start);
      references.symbol_endpoints.push_back(start + symbol.st_size);
    }
  }
  return references;
}

Result CheckedTextEdits::validate_source_edits(std::span<const CheckedTextEdit> edits,
                                               TextEditError &error) const {
  error = {};
  const Section &text = *source_.text_sections().front();
  std::vector<CheckedTextEdit> sorted(edits.begin(), edits.end());
  std::ranges::sort(sorted, {}, &CheckedTextEdit::offset);
  edits = sorted;
  if (edits.empty())
    return reject(error, TextEditErrorCode::InvalidRange, "at least one edit is required");

  // Validate arithmetic before constructing subspans or adding offsets. All
  // consumers use the same ordering for duplicate/overlap detection and commit.
  uint64_t previous_end = 0;
  for (const CheckedTextEdit &edit : edits) {
    const size_t size = edit.expected.size();
    if (edit.offset % 4 != 0 || (size != 4 && size != 8 && size != 12) ||
        edit.offset > text.size() || size > text.size() - edit.offset)
      return reject(error, TextEditErrorCode::InvalidRange,
                    "edit requires an in-bounds, aligned 4/8/12-byte source extent", edit.offset);
    if (edit.offset < previous_end)
      return reject(error, TextEditErrorCode::Overlap, "duplicate or overlapping edits",
                    edit.offset);
    previous_end = edit.offset + size;
    if (std::memcmp(text.data() + edit.offset, edit.expected.data(), size) != 0)
      return reject(error, TextEditErrorCode::StaleBytes, "expected bytes do not match source",
                    edit.offset);
  }
  if (validate_load_file_aliases(source_, edits, error).failed())
    return Result::failure();

  for (const CheckedTextEdit &edit : edits) {
    const InstructionBoundaries::Entry *instruction = boundaries_.find(edit.offset);
    if (!instruction || instruction->size != edit.expected.size())
      return reject(error, TextEditErrorCode::InvalidInstruction,
                    "edit does not cover one complete decoded source instruction", edit.offset);
    if (instruction->clause_member || instruction->clause)
      return reject(error, TextEditErrorCode::Clause, "cannot edit a clause or its members",
                    edit.offset);
  }
  for (const uint64_t address : references_.branch_addresses) {
    // A valid target before .text cannot enter an original-text anchor.
    if (address < text.vaddr())
      continue;
    const uint64_t reference = address - text.vaddr();
    if (reference_enters_edit(reference, edits))
      return reject(error, TextEditErrorCode::UnsupportedReferences,
                    "branch reference enters an edited instruction", reference);
  }
  for (const uint64_t endpoint : references_.symbol_endpoints) {
    if (reference_enters_edit(endpoint, edits))
      return reject(error, TextEditErrorCode::UnsupportedReferences,
                    "symbol endpoint lies inside an edited instruction", endpoint);
  }
  for (const uint64_t reference : references_.relative_text_references) {
    if (reference_enters_edit(reference, edits))
      return reject(error, TextEditErrorCode::UnsupportedReferences,
                    "relocation reference enters an edited instruction", reference);
  }
  for (const uint64_t entry : references_.kernel_entries) {
    if (reference_enters_edit(entry, edits))
      return reject(error, TextEditErrorCode::UnsupportedReferences,
                    "kernel entry lies inside an edited instruction", entry);
  }
  return Result::success();
}

Result CheckedTextEdits::validate(std::span<const CheckedTextEdit> edits,
                                  TextEditError &error) const {
  if (validate_source_edits(edits, error).failed())
    return Result::failure();
  for (const CheckedTextEdit &edit : edits) {
    if (edit.replacement.size() != edit.expected.size())
      return reject(error, TextEditErrorCode::InvalidRange,
                    "anchor edit must preserve its source extent", edit.offset);
    if (decode_replacement(decoder_, edit.replacement, edit.offset, error).failed())
      return Result::failure();
  }
  return Result::success();
}

Result CheckedTextEdits::apply(std::span<const CheckedTextEdit> edits,
                               std::span<uint8_t> destination, TextEditError &error) const {
  if (destination.size() != source_.text_sections().front()->size())
    return reject(error, TextEditErrorCode::InvalidRange,
                  "edit destination must cover the complete original text");
  if (validate(edits, error).failed())
    return Result::failure();
  // No write occurs until the entire batch is valid against the immutable source.
  for (const CheckedTextEdit &edit : edits)
    std::memcpy(destination.data() + edit.offset, edit.replacement.data(), edit.replacement.size());
  return Result::success();
}
} // namespace rocjitsu
