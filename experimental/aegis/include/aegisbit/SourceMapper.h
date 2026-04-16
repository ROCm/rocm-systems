//===-- aegisbit/SourceMapper.h - DWARF Source Mapping -----------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps instruction PCs in AMDGPU code objects to source file/line using
/// DWARF .debug_line tables. Falls back gracefully when debug info is absent.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_SOURCE_MAPPER_H
#define AEGISBIT_SOURCE_MAPPER_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace aegisbit {

/// Source location for a single instruction address.
struct SourceLocation {
  std::string File;       ///< Source file path (may be relative)
  uint32_t Line = 0;      ///< 1-based line number (0 = unknown)
  uint32_t Column = 0;    ///< 1-based column (0 = unknown)

  bool isValid() const { return Line != 0; }

  /// Short filename (basename only).
  std::string shortFile() const;
};

/// Parses DWARF debug info from AMDGPU ELF code objects and maps
/// instruction addresses to source locations.
///
/// Usage:
/// \code
///   auto Mapper = SourceMapper::create(ELFBytes);
///   if (Mapper) {
///     auto Loc = Mapper->lookup(0x0148);
///     if (Loc.isValid())
///       printf("%s:%u\n", Loc.File.c_str(), Loc.Line);
///   }
/// \endcode
class SourceMapper {
public:
  /// Create a source mapper from raw ELF bytes.
  /// Returns nullptr (not an error) if the ELF has no debug info.
  static std::unique_ptr<SourceMapper> create(llvm::ArrayRef<uint8_t> ELFBytes);

  /// Look up the source location for a given PC (offset within .text).
  SourceLocation lookup(uint64_t PC) const;

  /// Batch-lookup: populate source locations for a list of PCs.
  std::vector<SourceLocation> lookup(const std::vector<uint64_t> &PCs) const;

  /// Returns true if this mapper has any debug line info.
  bool hasLineInfo() const { return !Entries.empty(); }

  /// Number of line-table entries parsed.
  size_t entryCount() const { return Entries.size(); }

  ~SourceMapper();

private:
  SourceMapper() = default;

  struct LineEntry {
    uint64_t Address;
    std::string File;
    uint32_t Line;
    uint32_t Column;
  };

  /// Sorted by address for binary search lookup.
  std::vector<LineEntry> Entries;

  /// .text section virtual address (for address normalization).
  uint64_t TextSectionAddr = 0;
};

} // namespace aegisbit

#endif // AEGISBIT_SOURCE_MAPPER_H
