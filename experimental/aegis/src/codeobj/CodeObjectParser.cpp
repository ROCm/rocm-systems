//===-- CodeObjectParser.cpp - ELF Parsing ----------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of AMD GPU ELF code object parsing.
///
/// Uses LLVM's Object library to parse ELF files and extract sections,
/// symbols, and kernel information.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectParser.h"
#include "aegisbit/DescriptorUpdater.h"
#include "aegisbit/Endian.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include <set>

#define DEBUG_TYPE "aegisbit-codeobj"

using namespace llvm;
using namespace llvm::object;

namespace aegisbit {

bool CodeObjectParser::isAMDGPUCodeObject(ArrayRef<uint8_t> Bytes) {
  // Check minimum size for ELF header
  if (Bytes.size() < ELF_HEADER_SIZE)
    return false;

  // Check ELF magic
  if (Bytes[0] != 0x7F || Bytes[1] != 'E' || Bytes[2] != 'L' || Bytes[3] != 'F')
    return false;

  // Check class (ELF64)
  if (Bytes[4] != 2)
    return false;

  // Check endianness (little-endian)
  if (Bytes[5] != 1)
    return false;

  // Check e_machine at offset 18 (ELF64)
  uint16_t EMachine = readLE16(&Bytes[18]);
  return EMachine == EM_AMDGPU;
}

std::string CodeObjectParser::getGPUArch(uint32_t EFlags) {
  uint32_t Mach = EFlags & CodeObjectParser::EF_AMDGPU_MACH_MASK;

  // Machine codes from CodeObjectParser.h (which mirrors LLVM's ELF.h)
  switch (Mach) {
  // GFX9 family (CDNA/Vega)
  case CodeObjectParser::EF_AMDGPU_MACH_GFX900:
    return "gfx900";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX906:
    return "gfx906";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX908:
    return "gfx908";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX90A:
    return "gfx90a";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX940:
    return "gfx940";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX941:
    return "gfx941";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX942:
    return "gfx942";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX950:
    return "gfx950";

  // GFX10 family (RDNA)
  case CodeObjectParser::EF_AMDGPU_MACH_GFX1010:
    return "gfx1010";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX1030:
    return "gfx1030";

  // GFX11 family (RDNA3)
  case CodeObjectParser::EF_AMDGPU_MACH_GFX1100:
    return "gfx1100";

  // GFX12 family (RDNA4 / future)
  case CodeObjectParser::EF_AMDGPU_MACH_GFX1200:
    return "gfx1200";
  case CodeObjectParser::EF_AMDGPU_MACH_GFX1250:
    return "gfx1250";

  default:
    // Return with hex suffix for unknown architectures to aid debugging
    char Buf[32];
    snprintf(Buf, sizeof(Buf), "gfx_unknown_0x%03x", Mach);
    return Buf;
  }
}

Expected<ParsedCodeObject> CodeObjectParser::parse(ArrayRef<uint8_t> Bytes) {
  if (!isAMDGPUCodeObject(Bytes)) {
    return createStringError(inconvertibleErrorCode(),
                             "Not a valid AMDGPU ELF code object");
  }

  // Create memory buffer for LLVM Object library
  auto MemBuf = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char*>(Bytes.data()), Bytes.size()),
      "", false);

  // Parse as ELF
  auto ObjOrErr = ObjectFile::createELFObjectFile(MemBuf->getMemBufferRef());
  if (!ObjOrErr) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to parse ELF: " +
                                 toString(ObjOrErr.takeError()));
  }

  auto* ELF = dyn_cast<ELFObjectFile<ELF64LE>>(ObjOrErr->get());
  if (!ELF) {
    return createStringError(inconvertibleErrorCode(),
                             "Not an ELF64 little-endian file");
  }

  ParsedCodeObject Result;

  // Store complete original ELF bytes (for byte surgery approach)
  Result.OriginalBytes.assign(Bytes.begin(), Bytes.end());

  // Extract header info
  const auto& Header = ELF->getELFFile().getHeader();
  Result.EType = Header.e_type;
  Result.EMachine = Header.e_machine;
  Result.EFlags = Header.e_flags;
  Result.GPUArch = getGPUArch(Header.e_flags);

  // Extract program headers if present (needed for ET_DYN)
  if (Header.e_phnum > 0) {
    Result.ProgramHeaderCount = Header.e_phnum;
    size_t PhdrSize = Header.e_phentsize * Header.e_phnum;
    const uint8_t* PhdrStart = Bytes.data() + Header.e_phoff;
    Result.ProgramHeaders.assign(PhdrStart, PhdrStart + PhdrSize);
  }

  // Extract sections
  uint16_t SectionIdx = 0;
  for (const auto& Sec : ELF->sections()) {
    auto NameOrErr = Sec.getName();
    if (!NameOrErr) {
      LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get section name at index "
                              << SectionIdx << "\n");
      consumeError(NameOrErr.takeError());
      SectionIdx++;
      continue;
    }

    auto ContentsOrErr = Sec.getContents();
    if (!ContentsOrErr) {
      LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get contents of section "
                              << *NameOrErr << "\n");
      consumeError(ContentsOrErr.takeError());
      SectionIdx++;
      continue;
    }

    StringRef Name = *NameOrErr;
    ArrayRef<uint8_t> Contents(
        reinterpret_cast<const uint8_t*>(ContentsOrErr->data()),
        ContentsOrErr->size());

    if (Name == ".text") {
      Result.TextSection.assign(Contents.begin(), Contents.end());
      Result.TextSectionIndex = SectionIdx;
      Result.TextSectionAddress = Sec.getAddress();
      // Get file offset from raw section header
      auto RawSection = ELF->getELFFile().getSection(SectionIdx);
      if (RawSection) {
        Result.TextSectionFileOffset = (*RawSection)->sh_offset;
      }
    } else if (Name == ".rodata") {
      Result.RodataSection.assign(Contents.begin(), Contents.end());
      Result.RodataSectionIndex = SectionIdx;
      Result.RodataSectionAddress = Sec.getAddress();
      auto RawSection = ELF->getELFFile().getSection(SectionIdx);
      if (RawSection) {
        Result.RodataSectionFileOffset = (*RawSection)->sh_offset;
      }
    } else if (Name.starts_with(".note")) {
      // May be .note.AMDGPU or similar
      Result.NoteSection.assign(Contents.begin(), Contents.end());
      Result.NoteSectionIndex = SectionIdx;
      auto RawSection = ELF->getELFFile().getSection(SectionIdx);
      if (RawSection) {
        Result.NoteSectionFileOffset = (*RawSection)->sh_offset;
      }
    }
    SectionIdx++;
  }

  // Parse symbols and identify kernels
  if (auto Err = parseSymbols(Bytes, Result)) {
    return Err;
  }

  return Result;
}

Error CodeObjectParser::parseSymbols(ArrayRef<uint8_t> ELFBytes,
                                      ParsedCodeObject& Result) {
  auto MemBuf = MemoryBuffer::getMemBuffer(
      StringRef(reinterpret_cast<const char*>(ELFBytes.data()), ELFBytes.size()),
      "", false);

  auto ObjOrErr = ObjectFile::createELFObjectFile(MemBuf->getMemBufferRef());
  if (!ObjOrErr) {
    return ObjOrErr.takeError();
  }

  auto* ELF = cast<ELFObjectFile<ELF64LE>>(ObjOrErr->get());

  // Collect all symbols, tracking which are in .text section
  std::unordered_map<std::string, SymbolEntry> SymbolMap;
  std::set<std::string> TextSymbols;  // Names of symbols in .text

  for (const auto& Sym : ELF->symbols()) {
    auto NameOrErr = Sym.getName();
    if (!NameOrErr) {
      LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get symbol name\n");
      consumeError(NameOrErr.takeError());
      continue;
    }

    SymbolEntry Entry;
    Entry.Name = NameOrErr->str();

    auto ValueOrErr = Sym.getValue();
    if (!ValueOrErr) {
      LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get value for symbol "
                              << Entry.Name << "\n");
      consumeError(ValueOrErr.takeError());
      continue;
    }
    Entry.Value = *ValueOrErr;

    // Get raw ELF symbol type using ELFSymbolRef
    auto ELFSym = cast<ELFSymbolRef>(Sym);

    // Use ELFSymbolRef::getSize() instead of getCommonSize() — the latter
    // asserts SF_Common which not all symbols have.
    Entry.Size = ELFSym.getSize();
    Entry.Type = ELFSym.getELFType();
    Entry.Binding = ELFSym.getBinding();

    // Check if this symbol is in .text section by looking at section name
    auto SecOrErr = Sym.getSection();
    if (SecOrErr) {
      auto SI = SecOrErr.get();
      if (SI != ELF->section_end()) {
        auto SecNameOrErr = SI->getName();
        if (SecNameOrErr && *SecNameOrErr == ".text") {
          TextSymbols.insert(Entry.Name);
          Entry.SectionIndex = Result.TextSectionIndex;
        } else if (SecNameOrErr && *SecNameOrErr == ".rodata") {
          Entry.SectionIndex = Result.RodataSectionIndex;
        }
        if (SecNameOrErr) {
          // Find section index by iterating
          uint16_t Idx = 0;
          for (auto It = ELF->section_begin(); It != ELF->section_end(); ++It, ++Idx) {
            if (It == SI) {
              Entry.SectionIndex = Idx;
              break;
            }
          }
        } else {
          LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get section name for symbol "
                                  << Entry.Name << "\n");
          consumeError(SecNameOrErr.takeError());
        }
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to get section for symbol "
                              << Entry.Name << "\n");
      consumeError(SecOrErr.takeError());
    }

    Result.Symbols.push_back(Entry);
    SymbolMap[Entry.Name] = Entry;
  }

  // Collect .text function symbols sorted by address for size inference
  std::vector<const SymbolEntry*> TextFuncs;
  for (const auto& Sym : Result.Symbols) {
    if (Sym.Type == 2 && TextSymbols.count(Sym.Name))
      TextFuncs.push_back(&Sym);
  }
  std::sort(TextFuncs.begin(), TextFuncs.end(),
            [](const SymbolEntry* A, const SymbolEntry* B) {
              return A->Value < B->Value;
            });

  // Find kernels: look for STT_FUNC symbols in .text and matching .kd symbols
  for (size_t FI = 0; FI < TextFuncs.size(); ++FI) {
    const auto& Sym = *TextFuncs[FI];
    {
      // This is a kernel function
      std::string KDName = Sym.Name + ".kd";
      auto KDIt = SymbolMap.find(KDName);

      KernelInfo KI;
      KI.Name = Sym.Name;
      // Symbol value is virtual address; convert to offset within .text section
      KI.CodeOffset = Sym.Value - Result.TextSectionAddress;
      KI.CodeSize = Sym.Size;

      // Tensile/rocBLAS code objects from CCOB bundles often have Size=0.
      // Infer size from the gap to the next function or end of .text.
      if (KI.CodeSize == 0) {
        uint64_t NextAddr = Result.TextSectionAddress + Result.TextSection.size();
        if (FI + 1 < TextFuncs.size())
          NextAddr = TextFuncs[FI + 1]->Value;
        KI.CodeSize = NextAddr - Sym.Value;
      }

      if (KDIt != SymbolMap.end()) {
        // Descriptor symbol value is also virtual address; convert to offset in .rodata
        KI.DescriptorOffset = KDIt->second.Value - Result.RodataSectionAddress;

        // Parse kernel descriptor from .rodata
        if (KI.DescriptorOffset + DescriptorUpdater::DESCRIPTOR_SIZE <=
            Result.RodataSection.size()) {
          // Pass GPU architecture for correct VGPR granularity (8 for gfx90a+, 4 for older)
          auto KDOrErr = DescriptorUpdater::parse(
              ArrayRef<uint8_t>(Result.RodataSection.data() + KI.DescriptorOffset,
                                DescriptorUpdater::DESCRIPTOR_SIZE),
              Result.GPUArch);
          if (KDOrErr) {
            KI.Descriptor = std::move(*KDOrErr);
          } else {
            LLVM_DEBUG(llvm::dbgs() << "CodeObjectParser: Failed to parse descriptor for kernel "
                                    << KI.Name << "\n");
            consumeError(KDOrErr.takeError());
          }
        }
      }

      Result.Kernels.push_back(std::move(KI));
    }
  }

  return Error::success();
}

Expected<ParsedCodeObject> CodeObjectParser::parseFile(StringRef Path) {
  auto BufOrErr = MemoryBuffer::getFile(Path);
  if (!BufOrErr) {
    return createStringError(BufOrErr.getError(),
                             "Failed to read file: " + Path.str());
  }

  auto& Buf = *BufOrErr;
  ArrayRef<uint8_t> Bytes(
      reinterpret_cast<const uint8_t*>(Buf->getBufferStart()),
      Buf->getBufferSize());

  return parse(Bytes);
}

} // namespace aegisbit
