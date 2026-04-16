//===-- CodeObjectHandler.cpp - Unified Interface ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of unified AMD GPU code object handling.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DescriptorUpdater.h"
#include "aegisbit/RuntimeConfig.h"

#include <cstring>
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"

#define DEBUG_TYPE "aegisbit-handler"

using namespace llvm;

namespace aegisbit {

Expected<CodeObjectHandler> CodeObjectHandler::loadFromBytes(
    ArrayRef<uint8_t> Bytes) {
  auto ParsedOrErr = CodeObjectParser::parse(Bytes);
  if (!ParsedOrErr) {
    return ParsedOrErr.takeError();
  }

  CodeObjectHandler Handler;
  Handler.Parsed = std::move(*ParsedOrErr);
  Handler.ModifiedText = Handler.Parsed.TextSection;
  Handler.ModifiedRodata = Handler.Parsed.RodataSection;

  // Parse note section if present. A missing or unparseable note section is a
  // normal condition on trimmed or partial code objects, so only surface these
  // warnings when logging is explicitly enabled — otherwise they generate
  // thousands of stderr lines when handling multi-kernel libraries like
  // rocBLAS/Tensile.
  if (!Handler.Parsed.NoteSection.empty()) {
    auto NoteOrErr = NoteMetadataHandler::parse(Handler.Parsed.NoteSection);
    if (NoteOrErr) {
      Handler.Note = std::move(*NoteOrErr);
    } else {
      if (RuntimeConfig::getInstance().LogEnabled) {
        llvm::errs() << "[WARN] Note metadata parsing failed: "
                     << llvm::toString(NoteOrErr.takeError()) << "\n";
      } else {
        llvm::consumeError(NoteOrErr.takeError());
      }
    }
  } else if (RuntimeConfig::getInstance().LogEnabled) {
    llvm::errs() << "[WARN] No note section found in code object\n";
  }

  // Create builder from parsed code object
  // No builder needed — build() does in-place ELF patching.

  return Handler;
}

Expected<CodeObjectHandler> CodeObjectHandler::loadFromFile(StringRef Path) {
  auto BufOrErr = MemoryBuffer::getFile(Path);
  if (!BufOrErr) {
    return createStringError(BufOrErr.getError(),
                             "Failed to read file: " + Path.str());
  }

  auto& Buf = *BufOrErr;
  ArrayRef<uint8_t> Bytes(
      reinterpret_cast<const uint8_t*>(Buf->getBufferStart()),
      Buf->getBufferSize());

  return loadFromBytes(Bytes);
}

std::string CodeObjectHandler::getGPUArch() const {
  return Parsed.GPUArch;
}

size_t CodeObjectHandler::getKernelCount() const {
  return Parsed.Kernels.size();
}

std::vector<std::string> CodeObjectHandler::getKernelNames() const {
  std::vector<std::string> Names;
  Names.reserve(Parsed.Kernels.size());
  for (const auto& K : Parsed.Kernels) {
    Names.push_back(K.Name);
  }
  return Names;
}

const KernelInfo* CodeObjectHandler::getKernel(StringRef Name) const {
  // rocprofiler-sdk provides kernel names with .kd suffix, but CodeObjectParser
  // stores kernel names without it (the function symbol name, not the descriptor)
  StringRef LookupName = Name;
  if (LookupName.ends_with(".kd")) {
    LookupName = LookupName.drop_back(3);  // Remove ".kd" suffix
  }

  for (const auto& K : Parsed.Kernels) {
    if (K.Name == LookupName) {
      return &K;
    }
  }
  return nullptr;
}

std::vector<std::pair<uint64_t, uint64_t>>
CodeObjectHandler::getTextFunctionRanges() const {
  std::vector<std::pair<uint64_t, uint64_t>> Ranges;
  for (const auto &Sym : Parsed.Symbols) {
    if (Sym.Type == 2 /* STT_FUNC */ &&
        Sym.SectionIndex == Parsed.TextSectionIndex && Sym.Size > 0) {
      uint64_t Off = Sym.Value - Parsed.TextSectionAddress;
      Ranges.emplace_back(Off, Off + Sym.Size);
    }
  }
  return Ranges;
}

ArrayRef<uint8_t> CodeObjectHandler::getTextSection() const {
  if (TextModified) {
    return ModifiedText;
  }
  return Parsed.TextSection;
}

ArrayRef<uint8_t> CodeObjectHandler::getRodataSection() const {
  if (RodataModified) {
    return ModifiedRodata;
  }
  return Parsed.RodataSection;
}

ArrayRef<uint8_t> CodeObjectHandler::getNoteSection() const {
  return Parsed.NoteSection;
}

void CodeObjectHandler::setTextSection(ArrayRef<uint8_t> Text) {
  ModifiedText.assign(Text.begin(), Text.end());
  TextModified = true;
}

Error CodeObjectHandler::applyPatch(ArrayRef<uint8_t> PatchedCode,
                                     StringRef KernelName,
                                     uint32_t AdditionalVGPRs,
                                     uint32_t AdditionalSGPRs,
                                     uint32_t AdditionalKernargSize,
                                     uint32_t AdditionalScratchSize,
                                     int64_t CodeEntryOffsetAdjust) {
  // rocprofiler-sdk provides kernel names with .kd suffix, but CodeObjectParser
  // stores kernel names without it (the function symbol name, not the descriptor)
  StringRef LookupName = KernelName;
  if (LookupName.ends_with(".kd")) {
    LookupName = LookupName.drop_back(3);  // Remove ".kd" suffix
  }

  // Find the kernel
  const KernelInfo* KI = nullptr;
  size_t KernelIdx = 0;
  for (size_t i = 0; i < Parsed.Kernels.size(); ++i) {
    if (Parsed.Kernels[i].Name == LookupName) {
      KI = &Parsed.Kernels[i];
      KernelIdx = i;
      break;
    }
  }

  if (!KI) {
    return createStringError(inconvertibleErrorCode(),
                             "Kernel not found: " + KernelName.str());
  }

  // Update text section
  ModifiedText.assign(PatchedCode.begin(), PatchedCode.end());
  TextModified = true;

  // Update kernel descriptor
  KernelDescriptor NewDesc = KI->Descriptor;

  if (CodeEntryOffsetAdjust != 0)
    NewDesc.KernelCodeEntryByteOffset += CodeEntryOffsetAdjust;

  // Increase register counts
  uint32_t NewVGPRCount = NewDesc.VGPRCount + AdditionalVGPRs;
  uint32_t NewSGPRCount = NewDesc.SGPRCount + AdditionalSGPRs;

  if (auto Err = DescriptorUpdater::updateVGPRCount(NewDesc, NewVGPRCount)) {
    return Err;
  }
  if (AdditionalSGPRs > 0) {
    if (auto Err = DescriptorUpdater::updateSGPRCount(NewDesc, NewSGPRCount)) {
      return Err;
    }
  }

  // AccumOffset is left unchanged — scratch VGPRs go above the total
  // allocation (above both regular and AccVGPRs). The hardware allocates
  // the full VGPRCount; our scratch sits in the newly-extended region.

  // Increase kernarg size
  DescriptorUpdater::updateKernargSize(NewDesc,
      NewDesc.KernargSize + AdditionalKernargSize);

  // Increase scratch size (for VGPR spill to scratch memory)
  if (AdditionalScratchSize > 0) {
    // If kernel didn't originally use scratch, we need to enable it
    if (KI->Descriptor.PrivateSegmentFixedSize == 0) {
      DescriptorUpdater::enablePrivateSegment(NewDesc);
    }
    DescriptorUpdater::updateScratchSize(NewDesc,
        NewDesc.PrivateSegmentFixedSize + AdditionalScratchSize);
  }

  // Serialize descriptor to rodata
  auto DescBytes = DescriptorUpdater::serialize(NewDesc);

  // Copy rodata if not already modified
  if (!RodataModified) {
    ModifiedRodata = Parsed.RodataSection;
    RodataModified = true;
  }

  // Update descriptor in rodata
  if (KI->DescriptorOffset + DescriptorUpdater::DESCRIPTOR_SIZE <=
      ModifiedRodata.size()) {
    std::copy(DescBytes.begin(), DescBytes.end(),
              ModifiedRodata.begin() + KI->DescriptorOffset);
  }

  // Update note metadata if present.
  // The .note stores logical (pre-granularity) register counts, while the
  // descriptor has hardware-aligned counts. Apply deltas to preserve the
  // original logical values.
  if (Note.has_value()) {
    KernelDescriptor NoteDesc = NewDesc;

    StringRef NoteLookup = LookupName;
    for (const auto &KM : Note->Kernels) {
      if (KM.Name == NoteLookup) {
        NoteDesc.VGPRCount = KM.VGPRCount + AdditionalVGPRs;
        NoteDesc.SGPRCount = KM.SGPRCount + AdditionalSGPRs;
        break;
      }
    }

    auto UpdateErr = NoteMetadataHandler::updateKernelMetadata(
        *Note, KernelName, NoteDesc);
    if (UpdateErr) {
      return UpdateErr;
    }
  }

  return Error::success();
}

Expected<std::vector<uint8_t>> CodeObjectHandler::build() {
  // In-place ELF patching: start with original bytes, patch sections,
  // append island at file end, extend LOAD segment. Preserves all
  // dynamic linking metadata (.dynsym, .gnu.hash, .dynamic, etc.)
  // that yaml2elf was dropping.

  if (Parsed.OriginalBytes.empty()) {
    return createStringError(inconvertibleErrorCode(),
                             "No original ELF bytes available");
  }

  std::vector<uint8_t> ELF = Parsed.OriginalBytes;

  // ---- Step 1: Overwrite .rodata in-place (descriptor updates) ----
  if (RodataModified && Parsed.RodataSectionFileOffset > 0) {
    // .rodata size should not change (same number of descriptors)
    if (ModifiedRodata.size() == Parsed.RodataSection.size()) {
      std::memcpy(ELF.data() + Parsed.RodataSectionFileOffset,
                  ModifiedRodata.data(), ModifiedRodata.size());
    }
  }

  // ---- Step 2: Overwrite .note in-place (metadata updates) ----
  if (Note.has_value() && Parsed.NoteSectionFileOffset > 0) {
    auto NoteBytes = NoteMetadataHandler::serialize(*Note);
    // .note size should not change (same metadata, just updated values)
    if (NoteBytes.size() <= Parsed.NoteSection.size()) {
      std::memcpy(ELF.data() + Parsed.NoteSectionFileOffset,
                  NoteBytes.data(), NoteBytes.size());
    }
  }

  // ---- Step 3: Handle .text changes ----
  if (TextModified) {
    uint64_t origTextSize = Parsed.TextSection.size();
    uint64_t newTextSize = ModifiedText.size();

    if (newTextSize <= origTextSize) {
      std::memcpy(ELF.data() + Parsed.TextSectionFileOffset,
                  ModifiedText.data(), newTextSize);
    } else {
      // .text grew (trampoline island). Shift all file content after
      // the original .text end forward to make room, preserving
      // .dynamic, .symtab, .strtab, section headers, etc.
      uint64_t origTextEnd = Parsed.TextSectionFileOffset + origTextSize;
      uint64_t delta = newTextSize - origTextSize;
      uint64_t tailSize = ELF.size() - origTextEnd;

      ELF.resize(ELF.size() + delta, 0x00);

      // Shift the tail (everything after original .text) forward by delta.
      // Use memmove since regions may overlap.
      std::memmove(ELF.data() + origTextEnd + delta,
                   ELF.data() + origTextEnd,
                   tailSize);

      // Write the full modified .text (original patches + padding + island)
      std::memcpy(ELF.data() + Parsed.TextSectionFileOffset,
                  ModifiedText.data(), newTextSize);

      // --- Fix all ELF offsets and virtual addresses shifted by delta ---
      //
      // When .text grows, the file bytes after it shift forward.  We must
      // also shift virtual addresses of sections/segments that sat after
      // .text in VA space, otherwise the expanded .text LOAD segment will
      // overlap the next LOAD segment, corrupting the GPU memory mapping.
      uint64_t origTextVAEnd =
          Parsed.TextSectionAddress + origTextSize;

      // Round delta up to the .text LOAD segment's alignment (usually 0x1000)
      // so that the next segment's VA stays properly aligned.
      uint64_t alignedDelta = delta;

      // a) Section headers: update sh_offset (file) and sh_addr (VA)
      uint64_t e_shoff = *(uint64_t*)(ELF.data() + 40);
      uint16_t e_shentsize = *(uint16_t*)(ELF.data() + 58);
      uint16_t e_shnum = *(uint16_t*)(ELF.data() + 60);

      // Section header table itself moved
      if (e_shoff >= origTextEnd) {
        e_shoff += delta;
        std::memcpy(ELF.data() + 40, &e_shoff, 8);
      }

      for (int i = 0; i < e_shnum; i++) {
        uint8_t *sh = ELF.data() + e_shoff + i * e_shentsize;
        uint32_t sh_type = *(uint32_t*)(sh + 4);
        uint64_t sh_offset = *(uint64_t*)(sh + 24);
        uint64_t sh_addr = *(uint64_t*)(sh + 16);
        if (sh_type == 0 /* SHT_NULL */)
          continue;

        if (sh_offset == Parsed.TextSectionFileOffset) {
          // This is .text — update its size
          std::memcpy(sh + 32, &newTextSize, 8);
        } else if (sh_offset >= origTextEnd) {
          // Section after .text — shift its file offset
          uint64_t new_offset = sh_offset + delta;
          std::memcpy(sh + 24, &new_offset, 8);
        }

        // Shift virtual addresses of sections that were after .text in VA space
        if (sh_addr >= origTextVAEnd && sh_addr != 0) {
          uint64_t new_addr = sh_addr + alignedDelta;
          std::memcpy(sh + 16, &new_addr, 8);
        }
      }

      // b) Program headers: fix p_offset, p_vaddr, p_paddr for segments
      //    after .text, and extend the .text LOAD segment.
      uint64_t e_phoff = *(uint64_t*)(ELF.data() + 32);
      uint16_t e_phentsize = *(uint16_t*)(ELF.data() + 54);
      uint16_t e_phnum = *(uint16_t*)(ELF.data() + 56);

      for (int i = 0; i < e_phnum; i++) {
        uint8_t *ph = ELF.data() + e_phoff + i * e_phentsize;
        uint32_t p_type = *(uint32_t*)(ph);
        uint64_t p_offset = *(uint64_t*)(ph + 8);
        uint64_t p_vaddr = *(uint64_t*)(ph + 16);

        if (p_type == 1 /* PT_LOAD */ && p_offset == Parsed.TextSectionFileOffset) {
          // .text LOAD segment — expand filesz/memsz
          uint64_t orig_filesz;
          std::memcpy(&orig_filesz, ph + 32, 8);
          uint64_t new_filesz = orig_filesz + delta;
          std::memcpy(ph + 32, &new_filesz, 8); // p_filesz
          std::memcpy(ph + 40, &new_filesz, 8); // p_memsz
        } else if (p_vaddr >= origTextVAEnd && p_vaddr != 0) {
          // Segment after .text in VA space — shift file offset + VAs
          uint64_t new_offset = p_offset + delta;
          uint64_t new_vaddr = p_vaddr + alignedDelta;
          std::memcpy(ph + 8, &new_offset, 8);  // p_offset
          std::memcpy(ph + 16, &new_vaddr, 8);  // p_vaddr
          std::memcpy(ph + 24, &new_vaddr, 8);  // p_paddr
        } else if (p_offset >= origTextEnd) {
          // Segment after .text in file but VA is 0 (e.g. GNU_STACK)
          uint64_t new_offset = p_offset + delta;
          std::memcpy(ph + 8, &new_offset, 8);
        }
      }

      // c) .dynamic DT_* entries reference VAs in the first LOAD segment
      //    (before .text), which didn't move. No fixup needed.
    }
  }

  return ELF;
}

} // namespace aegisbit
