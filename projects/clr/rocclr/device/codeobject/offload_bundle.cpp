/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/offload_bundle.hpp"

#include "device/codeobject/elf_header.hpp"

#include "device/device.hpp"
#include "elf/elfio/elf_types.hpp"

#include <cstring>

namespace amd::codeobject {

using amd::ELFIO::Elf64_Ehdr;

namespace {

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t kBundleMagicSize = sizeof(kBundleMagic) - 1;

//! Bounds-checked little-endian reader over a byte range
class Reader {
 public:
  Reader(const char* data, size_t size) : cur_(data), end_(data + size) {}

  bool readU64(uint64_t* value) {
    if (remaining() < sizeof(uint64_t)) return false;
    uint64_t v = 0;
    for (size_t i = 0; i < sizeof(uint64_t); ++i) {
      v |= static_cast<uint64_t>(static_cast<uint8_t>(cur_[i])) << (8 * i);
    }
    cur_ += sizeof(uint64_t);
    *value = v;
    return true;
  }

  bool readString(uint64_t size, std::string* value) {
    if (size > remaining()) return false;
    value->assign(cur_, static_cast<size_t>(size));
    cur_ += size;
    return true;
  }

  bool skip(size_t size) {
    if (size > remaining()) return false;
    cur_ += size;
    return true;
  }

  const char* position() const { return cur_; }

 private:
  size_t remaining() const { return static_cast<size_t>(end_ - cur_); }

  const char* cur_;  //!< Next byte to read
  const char* end_;  //!< One past the last readable byte
};

//! Target identifier: arch-vendor-os-environment-processor[:feature±]...
struct TargetId {
  std::string processor;  //!< e.g. gfx90a, gfx9-4-generic
  char xnack = ' ';       //!< '+', '-', or ' ' when unset
  char sramecc = ' ';     //!< '+', '-', or ' ' when unset
};

/**
 * Parses @p name the way comgr's parseTargetIdentifier does: split at the first four dashes, the
 * last part is the processor followed by ':'-separated features. The processor must be a known
 * ISA and each feature must be xnack or sramecc with +/-, supported by that ISA. The SPIR-V names
 * take no features.
 */
bool ParseTargetId(const std::string& name, TargetId* id) {
  size_t dash[4];
  size_t pos = 0;
  for (size_t i = 0; i < 4; ++i) {
    dash[i] = name.find('-', pos);
    if (dash[i] == std::string::npos) return false;
    pos = dash[i] + 1;
  }
  const std::string last = name.substr(pos);
  const size_t colon = last.find(':');
  id->processor = last.substr(0, colon);
  std::vector<std::string> features;
  for (size_t start = colon; start != std::string::npos;) {
    const size_t next = last.find(':', start + 1);
    features.push_back(last.substr(start + 1, next == std::string::npos ? next : next - start - 1));
    start = next;
  }

  if (name == "spirv64-amd-amdhsa--amdgcnspirv" ||
      name == "spirv64-amd-amdhsa-unknown-amdgcnspirv") {
    return features.empty();
  }

  const std::string arch = name.substr(0, dash[0]);
  const std::string vendor = name.substr(dash[0] + 1, dash[1] - dash[0] - 1);
  const std::string os = name.substr(dash[1] + 1, dash[2] - dash[1] - 1);
  if ((arch != "amdgcn" && arch != "amdgpu") || vendor != "amd" || os != "amdhsa") {
    return false;
  }
  const amd::Isa* isa = amd::Isa::findIsa(("amdgcn-amd-amdhsa--" + id->processor).c_str());
  if (isa == nullptr) return false;

  for (const std::string& feature : features) {
    if (feature.size() < 2) return false;
    const char sign = feature.back();
    if (sign != '+' && sign != '-') return false;
    const std::string featureName = feature.substr(0, feature.size() - 1);
    if (featureName == "xnack" && isa->isXnackSupported()) {
      id->xnack = sign;
    } else if (featureName == "sramecc" && isa->isSrameccSupported()) {
      id->sramecc = sign;
    } else {
      return false;
    }
  }
  return true;
}

/**
 * Names the code object of a bundle entry by its ELF header, the way the loader names it, e.g.
 * "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-".
 *
 * @returns False for an ELF whose ISA isn't known; the entry is skipped. True otherwise, with
 *          @p target left empty when the entry isn't an ELF or lies outside @p imageBound (the
 *          caller rejects such offsets).
 */
bool ElfTarget(const char* bundle, size_t imageBound, uint64_t offset, uint64_t size,
               std::string* target) {
  target->clear();
  if (size < sizeof(Elf64_Ehdr) ||
      (imageBound != kUnknownSize &&
       (offset > imageBound || imageBound - offset < sizeof(Elf64_Ehdr)))) {
    return true;
  }
  const char* image = bundle + offset;
  if (!IsElf(image)) {
    return true;
  }
  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image, sizeof(ehdr));
  return GetIsaFromHeader(ehdr.e_ident[EI_CLASS], ehdr.e_machine, ehdr.e_ident[EI_OSABI],
                          ehdr.e_ident[EI_ABIVERSION], ehdr.e_flags, *target);
}

//! The generic target of @p input, a full target name without features
//! ("amdgcn-amd-amdhsa--gfx1100"), from its amd::Isa row; empty if it has none
std::string TargetGenericMap(const std::string& input) {
  const amd::Isa* isa = amd::Isa::findIsa(input.c_str());
  if (isa == nullptr || isa->genericTarget() == nullptr) {
    return {};
  }
  return std::string("amdgcn-amd-amdhsa--") + isa->genericTarget();
}

// For sramecc and xnack
std::string TargetFeatureCheck(const std::string& input, const std::string &feature) {
  if (input.find(feature) != std::string::npos) {
    auto feature_p = feature + "+";  // feature present eg: xnack+
    auto feature_m = feature + "-";  // feature absent eg: xnack-
    if (input.find(feature_p) != std::string::npos) {
      return feature_p;
    } else if (input.find(feature_m) != std::string::npos) {
      return feature_m;
    }
  }
  return "";
}

}  // namespace

// ================================================================================================
std::string TargetToGeneric(const std::string &input) {
  auto sramecc = TargetFeatureCheck(input, "sramecc");
  auto xnack = TargetFeatureCheck(input, "xnack");

  // Remove all features
  size_t index = input.find_first_of(":");
  std::string name_without_feature = input.substr(0, index);

  // Look up generic name
  auto generic_name = TargetGenericMap(name_without_feature);
  if (generic_name.empty()) {
    return generic_name;  // No generic exists
  }

  // reappend feature
  if (!sramecc.empty()) {
    generic_name += ":";
    generic_name += sramecc;
  }
  if (!xnack.empty()) {
    generic_name += ":";
    generic_name += xnack;
  }
  return generic_name;
}

// ================================================================================================
bool IsOffloadBundle(const void* image, size_t imageBound) {
  return imageBound >= kBundleMagicSize && std::memcmp(image, kBundleMagic, kBundleMagicSize) == 0;
}

// ================================================================================================
uint64_t OffloadBundleSize(const void* bundle, size_t headerBound) {
  // Same limits as the lookup: a 4 KiB header window, at most 4096 entries and ids
  constexpr size_t kHeaderWindow = 4096;
  constexpr uint64_t kMaxEntries = 4096;
  constexpr uint64_t kMaxIdSize = 4096;
  if (!IsOffloadBundle(bundle, headerBound)) {
    return 0;
  }
  const size_t window = (headerBound < kHeaderWindow) ? headerBound : kHeaderWindow;
  const char* start = static_cast<const char*>(bundle);
  Reader reader(start, window);
  uint64_t count = 0;
  if (!reader.skip(kBundleMagicSize) || !reader.readU64(&count) || count == 0 ||
      count > kMaxEntries) {
    return 0;
  }
  uint64_t end = 0;
  for (uint64_t i = 0; i < count; ++i) {
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t idSize = 0;
    if (!reader.readU64(&offset) || !reader.readU64(&size) || !reader.readU64(&idSize) ||
        idSize > kMaxIdSize || !reader.skip(static_cast<size_t>(idSize)) ||
        offset > ~uint64_t(0) - size) {
      return 0;
    }
    if (offset + size > end) end = offset + size;
  }
  // The header itself ends where the reader stopped
  const uint64_t headerEnd = static_cast<uint64_t>(reader.position() - start);
  return (headerEnd > end) ? headerEnd : end;
}

// ================================================================================================
bool IsCompatibleTarget(const std::string& target, const std::string& codeObjectTarget) {
  if (target == codeObjectTarget) {
    return true;
  }
  TargetId codeObjectId;
  TargetId targetId;
  if (!ParseTargetId(codeObjectTarget, &codeObjectId) || !ParseTargetId(target, &targetId)) {
    return false;
  }
  if (codeObjectId.processor != targetId.processor) {
    return false;
  }
  if (codeObjectId.xnack != ' ' && codeObjectId.xnack != targetId.xnack) {
    return false;
  }
  if (codeObjectId.sramecc != ' ' && codeObjectId.sramecc != targetId.sramecc) {
    return false;
  }
  return true;
}

// ================================================================================================
bool LookupCodeObjects(const void* bundle, size_t headerBound, size_t imageBound,
                       const std::vector<std::string>& targets,
                       std::vector<CodeObjectRef>& found) {
  found.assign(targets.size(), CodeObjectRef{});
  if (!IsOffloadBundle(bundle, headerBound)) {
    return false;
  }
  Reader reader(static_cast<const char*>(bundle), headerBound);
  uint64_t numEntries = 0;
  if (!reader.skip(kBundleMagicSize) || !reader.readU64(&numEntries)) {
    return false;
  }

  const char* start = static_cast<const char*>(bundle);
  size_t filled = 0;
  for (uint64_t i = 0; i < numEntries && filled < targets.size(); ++i) {
    uint64_t offset = 0;
    uint64_t size = 0;
    uint64_t idSize = 0;
    std::string id;
    if (!reader.readU64(&offset) || !reader.readU64(&size) || !reader.readU64(&idSize) ||
        !reader.readString(idSize, &id)) {
      return false;
    }
    // The encoded size may include a null terminator
    id.erase(id.find_last_not_of('\0') + 1);

    // Entry id: <offload kind>-<target>
    const size_t dash = id.find('-');
    const std::string kind = id.substr(0, dash);
    if (kind != "hip" && kind != "hipv4" && kind != "hcc") {
      continue;
    }
    // An AMDGPU ELF entry is named by its own header, the way the loader names it, so the
    // spelling of the target in the id (amdgcn-amd-amdhsa--gfxN or LLVM's amdgpuN.M triples)
    // doesn't matter. Other entries (SPIR-V) are named by the id.
    std::string target;
    if (!ElfTarget(start, imageBound, offset, size, &target)) {
      continue;
    }
    if (target.empty() && dash != std::string::npos) {
      target = id.substr(dash + 1);
    }

    for (size_t q = 0; q < targets.size(); ++q) {
      if (found[q].size == 0 && IsCompatibleTarget(targets[q], target)) {
        found[q] = CodeObjectRef{offset, size};
        ++filled;
        break;
      }
    }
  }
  return true;
}

}  // namespace amd::codeobject
