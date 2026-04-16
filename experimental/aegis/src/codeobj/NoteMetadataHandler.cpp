//===-- NoteMetadataHandler.cpp - Note Section Handling --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of AMD GPU .note section metadata handling.
///
/// Parses NT_AMDGPU_METADATA notes containing msgpack-formatted kernel metadata.
/// Uses LLVM's MsgPackDocument for parsing the binary format.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/NoteMetadataHandler.h"
#include "aegisbit/Endian.h"
#include "llvm/BinaryFormat/MsgPackDocument.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <regex>
#include <sstream>

using namespace llvm;
using namespace llvm::msgpack;

namespace aegisbit {

Expected<std::string> NoteMetadataHandler::extractMsgPack(
    ArrayRef<uint8_t> NoteSection) {
  if (NoteSection.size() < 12) {
    return createStringError(inconvertibleErrorCode(),
                             "Note section too small");
  }

  // Note header: namesz (4), descsz (4), type (4), name (aligned), desc (aligned)
  size_t Offset = 0;

  while (Offset + 12 <= NoteSection.size()) {
    uint32_t NameSz = readLE32(&NoteSection[Offset]);
    uint32_t DescSz = readLE32(&NoteSection[Offset + 4]);
    uint32_t Type = readLE32(&NoteSection[Offset + 8]);

    size_t NameOffset = Offset + 12;
    size_t AlignedNameSz = align4(NameSz);
    size_t DescOffset = NameOffset + AlignedNameSz;
    size_t AlignedDescSz = align4(DescSz);

    if (DescOffset + DescSz > NoteSection.size()) {
      break;
    }

    // Check for AMDGPU metadata note
    if (Type == NT_AMDGPU_METADATA) {
      StringRef Name(reinterpret_cast<const char*>(&NoteSection[NameOffset]),
                     NameSz > 0 ? NameSz - 1 : 0);  // Exclude null terminator

      if (Name == AMDGPU_NOTE_NAME || Name.starts_with("AMD")) {
        return std::string(
            reinterpret_cast<const char*>(&NoteSection[DescOffset]), DescSz);
      }
    }

    Offset = DescOffset + AlignedDescSz;
  }

  return createStringError(inconvertibleErrorCode(),
                           "No AMDGPU metadata note found");
}

Error NoteMetadataHandler::parseMsgPack(const std::string& MsgPackData,
                                         NoteContent& Content) {
  // Parse msgpack using LLVM's MsgPackDocument
  Document Doc;
  if (!Doc.readFromBlob(StringRef(MsgPackData), /*Multi=*/false)) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to parse msgpack data");
  }

  DocNode Root = Doc.getRoot();
  if (!Root.isMap()) {
    return createStringError(inconvertibleErrorCode(),
                             "Root node is not a map");
  }

  MapDocNode &RootMap = Root.getMap();

  // Extract GPU architecture from "amdhsa.target"
  DocNode TargetKey = Doc.getNode(".target", /*Copy=*/false);
  if (TargetKey.isEmpty()) {
    TargetKey = Doc.getNode("amdhsa.target", /*Copy=*/false);
  }

  if (!TargetKey.isEmpty()) {
    auto It = RootMap.find(TargetKey);
    if (It != RootMap.end() && It->second.isString()) {
      StringRef TargetStr = It->second.getString();
      // Format: "amdgcn-amd-amdhsa--gfx950"
      size_t Pos = TargetStr.rfind("--");
      if (Pos != StringRef::npos && Pos + 2 < TargetStr.size()) {
        Content.GPUArch = TargetStr.substr(Pos + 2).str();
      }
    }
  }

  // Extract kernels from "amdhsa.kernels" array
  // Try different key formats
  DocNode KernelsKey = Doc.getNode("amdhsa.kernels", /*Copy=*/false);
  if (KernelsKey.isEmpty()) {
    KernelsKey = Doc.getNode(".kernels", /*Copy=*/false);
  }

  if (KernelsKey.isEmpty()) {
    // No kernels found - this might be valid for some code objects
    return Error::success();
  }

  auto KernelsIt = RootMap.find(KernelsKey);
  if (KernelsIt == RootMap.end()) {
    return Error::success();
  }

  if (!KernelsIt->second.isArray()) {
    return createStringError(inconvertibleErrorCode(),
                             "amdhsa.kernels is not an array");
  }

  ArrayDocNode &KernelsArray = KernelsIt->second.getArray();

  // Parse each kernel
  for (size_t i = 0; i < KernelsArray.size(); ++i) {
    DocNode &KernelNode = KernelsArray[i];
    if (!KernelNode.isMap()) {
      continue;
    }

    MapDocNode &KernelMap = KernelNode.getMap();
    KernelMetadata KM;

    // Helper to get string field
    auto getStringField = [&](const char* FieldName) -> std::string {
      DocNode Key = Doc.getNode(FieldName, /*Copy=*/false);
      if (!Key.isEmpty()) {
        auto It = KernelMap.find(Key);
        if (It != KernelMap.end() && It->second.isString()) {
          return It->second.getString().str();
        }
      }
      return "";
    };

    // Helper to get uint field
    auto getUIntField = [&](const char* FieldName) -> uint32_t {
      DocNode Key = Doc.getNode(FieldName, /*Copy=*/false);
      if (!Key.isEmpty()) {
        auto It = KernelMap.find(Key);
        if (It != KernelMap.end()) {
          if (It->second.getKind() == Type::UInt) {
            return static_cast<uint32_t>(It->second.getUInt());
          } else if (It->second.getKind() == Type::Int) {
            return static_cast<uint32_t>(It->second.getInt());
          }
        }
      }
      return 0;
    };

    // Extract fields
    KM.Name = getStringField(".name");
    KM.SGPRCount = getUIntField(".sgpr_count");
    KM.VGPRCount = getUIntField(".vgpr_count");
    KM.KernargSize = getUIntField(".kernarg_segment_size");
    KM.LDSSize = getUIntField(".group_segment_fixed_size");
    KM.ScratchSize = getUIntField(".private_segment_fixed_size");

    if (!KM.Name.empty()) {
      Content.Kernels.push_back(KM);
    }
  }

  Content.RawYAML = MsgPackData;
  Content.IsMsgPack = true;

  return Error::success();
}

Error NoteMetadataHandler::parseYAML(const std::string& YAML,
                                      NoteContent& Content) {
  // Deprecated - modern code objects use msgpack
  // Keep this for backwards compatibility with old YAML-based metadata
  Content.RawYAML = YAML;
  // Minimal parsing - just return success
  return Error::success();
}

Expected<NoteContent> NoteMetadataHandler::parse(ArrayRef<uint8_t> NoteSection) {
  // Modern ROCm uses msgpack format
  auto MsgPackOrErr = extractMsgPack(NoteSection);
  if (!MsgPackOrErr) {
    return MsgPackOrErr.takeError();
  }

  NoteContent Content;
  if (auto Err = parseMsgPack(*MsgPackOrErr, Content)) {
    return std::move(Err);
  }

  return Content;
}

Error NoteMetadataHandler::updateMsgPack(std::string& MsgPackData,
                                          StringRef KernelName,
                                          const KernelDescriptor& Descriptor) {
  Document Doc;
  if (!Doc.readFromBlob(StringRef(MsgPackData), /*Multi=*/false)) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to re-parse msgpack for update");
  }

  DocNode Root = Doc.getRoot();
  if (!Root.isMap())
    return createStringError(inconvertibleErrorCode(),
                             "msgpack root is not a map");

  MapDocNode &RootMap = Root.getMap();

  DocNode KernelsKey = Doc.getNode("amdhsa.kernels", /*Copy=*/false);
  if (KernelsKey.isEmpty())
    return createStringError(inconvertibleErrorCode(),
                             "No amdhsa.kernels key in msgpack");

  auto KernelsIt = RootMap.find(KernelsKey);
  if (KernelsIt == RootMap.end() || !KernelsIt->second.isArray())
    return createStringError(inconvertibleErrorCode(),
                             "amdhsa.kernels not found or not an array");

  ArrayDocNode &KernelsArray = KernelsIt->second.getArray();

  bool Found = false;
  for (size_t i = 0; i < KernelsArray.size(); ++i) {
    DocNode &KernelNode = KernelsArray[i];
    if (!KernelNode.isMap())
      continue;

    MapDocNode &KernelMap = KernelNode.getMap();

    DocNode NameKey = Doc.getNode(".name", /*Copy=*/false);
    if (NameKey.isEmpty())
      continue;

    auto NameIt = KernelMap.find(NameKey);
    if (NameIt == KernelMap.end() || !NameIt->second.isString())
      continue;

    if (NameIt->second.getString() != KernelName)
      continue;

    Found = true;

    auto setUInt = [&](const char *Field, uint64_t Value) {
      DocNode Key = Doc.getNode(Field, /*Copy=*/false);
      if (Key.isEmpty())
        return;
      auto It = KernelMap.find(Key);
      if (It != KernelMap.end()) {
        It->second = Doc.getNode(Value);
      }
    };

    setUInt(".vgpr_count", Descriptor.VGPRCount);
    setUInt(".sgpr_count", Descriptor.SGPRCount);
    setUInt(".kernarg_segment_size", Descriptor.KernargSize);
    setUInt(".group_segment_fixed_size", Descriptor.GroupSegmentFixedSize);
    setUInt(".private_segment_fixed_size", Descriptor.PrivateSegmentFixedSize);
    break;
  }

  if (!Found)
    return createStringError(inconvertibleErrorCode(),
                             "Kernel '" + KernelName.str() +
                                 "' not found in msgpack metadata");

  std::string Output;
  Doc.writeToBlob(Output);
  MsgPackData = Output;
  return Error::success();
}

std::string NoteMetadataHandler::updateYAML(const std::string& YAML,
                                             StringRef KernelName,
                                             const KernelDescriptor& Descriptor) {
  std::string Result = YAML;

  // Find the kernel section in YAML and update its values
  // This is a simple find-and-replace approach

  // Build regex to find the kernel's section
  std::string KernelPattern = R"(-\s*\.name:\s*)" + KernelName.str();
  std::regex kernelRegex(KernelPattern);
  std::smatch match;

  if (!std::regex_search(Result, match, kernelRegex)) {
    return Result;  // Kernel not found, return unchanged
  }

  size_t KernelStart = match.position();
  size_t KernelEnd = Result.size();

  // Find next kernel or end of kernels section
  std::regex nextKernelRegex(R"(\n\s*-\s*\.name:)");
  std::string afterMatch = Result.substr(KernelStart + match.length());
  std::smatch nextMatch;
  if (std::regex_search(afterMatch, nextMatch, nextKernelRegex)) {
    KernelEnd = KernelStart + match.length() + nextMatch.position();
  }

  std::string KernelSection = Result.substr(KernelStart, KernelEnd - KernelStart);

  // Update VGPR count
  std::regex vgprRegex(R"(\.vgpr_count:\s*\d+)");
  KernelSection = std::regex_replace(KernelSection, vgprRegex,
      ".vgpr_count: " + std::to_string(Descriptor.VGPRCount));

  // Update SGPR count
  std::regex sgprRegex(R"(\.sgpr_count:\s*\d+)");
  KernelSection = std::regex_replace(KernelSection, sgprRegex,
      ".sgpr_count: " + std::to_string(Descriptor.SGPRCount));

  // Update kernarg size
  std::regex kernargRegex(R"(\.kernarg_segment_size:\s*\d+)");
  KernelSection = std::regex_replace(KernelSection, kernargRegex,
      ".kernarg_segment_size: " + std::to_string(Descriptor.KernargSize));

  // Update LDS size
  std::regex ldsRegex(R"(\.group_segment_fixed_size:\s*\d+)");
  KernelSection = std::regex_replace(KernelSection, ldsRegex,
      ".group_segment_fixed_size: " + std::to_string(Descriptor.GroupSegmentFixedSize));

  // Update scratch size
  std::regex scratchRegex(R"(\.private_segment_fixed_size:\s*\d+)");
  KernelSection = std::regex_replace(KernelSection, scratchRegex,
      ".private_segment_fixed_size: " + std::to_string(Descriptor.PrivateSegmentFixedSize));

  // Reconstruct result
  Result = Result.substr(0, KernelStart) + KernelSection +
           Result.substr(KernelEnd);

  return Result;
}

Error NoteMetadataHandler::updateKernelMetadata(NoteContent& Content,
                                                 StringRef KernelName,
                                                 const KernelDescriptor& Descriptor) {
  // rocprofiler-sdk provides kernel names with .kd suffix, but metadata
  // stores kernel names without it (the function symbol name, not the descriptor)
  StringRef LookupName = KernelName;
  if (LookupName.ends_with(".kd")) {
    LookupName = LookupName.drop_back(3);  // Remove ".kd" suffix
  }

  // Update the kernel in our parsed list
  bool Found = false;
  for (auto& KM : Content.Kernels) {
    if (KM.Name == LookupName) {
      KM.SGPRCount = Descriptor.SGPRCount;
      KM.VGPRCount = Descriptor.VGPRCount;
      KM.KernargSize = Descriptor.KernargSize;
      KM.LDSSize = Descriptor.GroupSegmentFixedSize;
      KM.ScratchSize = Descriptor.PrivateSegmentFixedSize;
      Found = true;
      break;
    }
  }

  if (!Found) {
    llvm::errs() << "[ERROR] Available kernels in metadata: ";
    for (const auto& KM : Content.Kernels) {
      llvm::errs() << "'" << KM.Name << "' ";
    }
    llvm::errs() << "\n";
    return createStringError(inconvertibleErrorCode(),
                             "Kernel not found in metadata: " + LookupName.str());
  }

  if (Content.IsMsgPack) {
    if (auto Err = updateMsgPack(Content.RawYAML, LookupName, Descriptor))
      return Err;
  } else {
    Content.RawYAML = updateYAML(Content.RawYAML, LookupName, Descriptor);
  }

  return Error::success();
}

std::vector<uint8_t> NoteMetadataHandler::buildNoteSection(
    const std::string& Data) {
  const char* Name = AMDGPU_NOTE_NAME;
  size_t NameLen = strlen(Name) + 1;
  size_t DescLen = Data.size();

  size_t AlignedNameLen = align4(NameLen);
  size_t AlignedDescLen = align4(DescLen);
  size_t TotalSize = 12 + AlignedNameLen + AlignedDescLen;

  std::vector<uint8_t> Result(TotalSize, 0);

  writeLE32(&Result[0], NameLen);
  writeLE32(&Result[4], DescLen);
  writeLE32(&Result[8], NT_AMDGPU_METADATA);
  std::memcpy(&Result[12], Name, NameLen);
  std::memcpy(&Result[12 + AlignedNameLen], Data.data(), Data.size());

  return Result;
}

std::vector<uint8_t> NoteMetadataHandler::serialize(const NoteContent& Content) {
  return buildNoteSection(Content.RawYAML);
}

} // namespace aegisbit
