//===-- SourceMapper.cpp - DWARF Source Mapping ------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps AMDGPU instruction PCs to source file/line via DWARF .debug_line.
///
/// AMDGPU code objects produced by Triton (or any LLVM-based compiler with -g)
/// embed standard DWARF debug info. We use LLVM's DWARFContext to parse the
/// .debug_line section and build a sorted address→source table for O(log n)
/// lookup at report time.
///
/// When debug info is absent, create() returns nullptr and callers degrade
/// gracefully (no source annotations in the report).
///
//===----------------------------------------------------------------------===//

#include "aegisbit/SourceMapper.h"

#include "llvm/DebugInfo/DWARF/DWARFContext.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/MemoryBuffer.h"

#include <algorithm>

using namespace llvm;
using namespace llvm::object;

namespace aegisbit {

std::string SourceLocation::shortFile() const {
  if (File.empty())
    return {};
  auto Pos = File.find_last_of("/\\");
  return (Pos == std::string::npos) ? File : File.substr(Pos + 1);
}

SourceMapper::~SourceMapper() = default;

std::unique_ptr<SourceMapper> SourceMapper::create(ArrayRef<uint8_t> ELFBytes) {
  if (ELFBytes.size() < 64)
    return nullptr;

  auto MemBuf = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char *>(ELFBytes.data()),
                ELFBytes.size()),
      /*BufferName=*/"", /*RequiresNullTerminator=*/false);

  auto ObjOrErr = ObjectFile::createELFObjectFile(MemBuf->getMemBufferRef());
  if (!ObjOrErr) {
    consumeError(ObjOrErr.takeError());
    return nullptr;
  }

  auto *ELF = dyn_cast<ELFObjectFile<ELF64LE>>(ObjOrErr->get());
  if (!ELF)
    return nullptr;

  // Find .text section address for offset normalization.
  uint64_t TextAddr = 0;
  for (const auto &Sec : ELF->sections()) {
    auto NameOrErr = Sec.getName();
    if (NameOrErr && *NameOrErr == ".text") {
      TextAddr = Sec.getAddress();
      break;
    } else if (!NameOrErr) {
      consumeError(NameOrErr.takeError());
    }
  }

  // Create DWARFContext — this parses all .debug_* sections.
  auto DWARFCtx = DWARFContext::create(*ELF);
  if (!DWARFCtx)
    return nullptr;

  // Check if there are any compile units with line info.
  bool HasAnyLineInfo = false;
  for (const auto &CU : DWARFCtx->compile_units()) {
    if (CU) {
      const auto *LT = DWARFCtx->getLineTableForUnit(CU.get());
      if (LT && !LT->Rows.empty()) {
        HasAnyLineInfo = true;
        break;
      }
    }
  }

  if (!HasAnyLineInfo)
    return nullptr;

  // Build sorted entry table from all compile units' line tables.
  auto Mapper = std::unique_ptr<SourceMapper>(new SourceMapper());
  Mapper->TextSectionAddr = TextAddr;

  for (const auto &CU : DWARFCtx->compile_units()) {
    if (!CU)
      continue;

    const auto *LT = DWARFCtx->getLineTableForUnit(CU.get());
    if (!LT)
      continue;

    const char *RawCompDir = CU->getCompilationDir();
    StringRef CompDir = RawCompDir ? RawCompDir : "";

    for (const auto &Row : LT->Rows) {
      if (Row.EndSequence)
        continue;

      std::string FileName;
      if (Row.File > 0)
        LT->getFileNameByIndex(Row.File, CompDir,
                               DILineInfoSpecifier::FileLineInfoKind::AbsoluteFilePath,
                               FileName);

      // Store addresses as offsets from .text base so they match the
      // PC values used in InstrumentationSite (which are .text offsets).
      LineEntry E;
      E.Address = Row.Address.Address - TextAddr;
      E.File = std::move(FileName);
      E.Line = Row.Line;
      E.Column = Row.Column;
      Mapper->Entries.push_back(std::move(E));
    }
  }

  // Sort by address for binary search.
  std::sort(Mapper->Entries.begin(), Mapper->Entries.end(),
            [](const LineEntry &A, const LineEntry &B) {
              return A.Address < B.Address;
            });

  // Deduplicate: keep only the first entry for each address.
  auto Last = std::unique(Mapper->Entries.begin(), Mapper->Entries.end(),
                          [](const LineEntry &A, const LineEntry &B) {
                            return A.Address == B.Address;
                          });
  Mapper->Entries.erase(Last, Mapper->Entries.end());

  return Mapper;
}

SourceLocation SourceMapper::lookup(uint64_t PC) const {
  if (Entries.empty())
    return {};

  // Binary search: find the last entry with Address <= PC.
  LineEntry Key;
  Key.Address = PC;
  auto It = std::upper_bound(
      Entries.begin(), Entries.end(), Key,
      [](const LineEntry &A, const LineEntry &B) {
        return A.Address < B.Address;
      });

  if (It == Entries.begin())
    return {};

  --It;

  SourceLocation Loc;
  Loc.File = It->File;
  Loc.Line = It->Line;
  Loc.Column = It->Column;
  return Loc;
}

std::vector<SourceLocation>
SourceMapper::lookup(const std::vector<uint64_t> &PCs) const {
  std::vector<SourceLocation> Results;
  Results.reserve(PCs.size());
  for (uint64_t PC : PCs)
    Results.push_back(lookup(PC));
  return Results;
}

} // namespace aegisbit
