/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/metadata_notes.hpp"

#include <algorithm>
#include <string_view>

namespace amd::codeobject {

// ================================================================================================
bool MetadataNotes::parse(const char* note, uint32_t noteSize) {
  // Top-level map: amdhsa.version [major, minor], amdhsa.kernels [maps], amdhsa.printf [strings]
  error_.clear();
  MsgPackReader reader(note, noteSize);
  uint32_t numEntries = 0;
  if (!reader.readMapSize(&numEntries)) {
    error_ = "Malformed code object metadata";
    return false;
  }
  for (uint32_t i = 0; i < numEntries; ++i) {
    std::string_view key;
    bool ok = reader.readString(&key);
    if (ok && key == "amdhsa.version") {
      ok = parseVersion(reader);
    } else if (ok && key == "amdhsa.kernels") {
      uint32_t count = 0;
      ok = reader.readArraySize(&count);
      hasKernels_ = hasKernels_ || ok;
      for (uint32_t k = 0; ok && k < count; ++k) {
        ok = kernelFn_(reader);
      }
    } else if (ok && key == "amdhsa.printf") {
      ok = parsePrintf(reader);
    } else if (ok) {
      ok = reader.skip();
    }
    if (!error_.empty()) {
      return false;
    }
    if (!ok) {
      error_ = "Malformed code object metadata";
      return false;
    }
  }
  return true;
}

// ================================================================================================
bool MetadataNotes::parseVersion(MsgPackReader& reader) {
  uint32_t count = 0;
  uint64_t major = 0;
  uint64_t minor = 0;
  bool ok = reader.readArraySize(&count) && count >= 2 && reader.readUint(&major) &&
            reader.readUint(&minor);
  for (uint32_t v = 2; ok && v < count; ++v) {
    ok = reader.skip();
  }
  if (!ok) {
    return false;
  }
  if (hasVersion_ && version_ != std::make_pair(major, minor)) {
    error_ = "Code object metadata notes have different versions";
    return false;
  }
  hasVersion_ = true;
  version_ = {major, minor};
  return true;
}

// ================================================================================================
bool MetadataNotes::parsePrintf(MsgPackReader& reader) {
  // The first list is kept; a later one is compared in place
  uint32_t count = 0;
  if (!reader.readArraySize(&count)) {
    return false;
  }
  const bool first = !hasPrintf_;
  bool same = (first || count == printfStrings_.size());
  if (first) {
    // The count comes from the note: never reserve more than the note can hold
    printfStrings_.reserve(std::min<size_t>(count, reader.remaining()));
  }
  for (uint32_t p = 0; p < count; ++p) {
    std::string_view format;
    if (!reader.readString(&format)) {
      return false;
    }
    if (first) {
      printfStrings_.emplace_back(format);
    } else if (same) {
      same = (format == printfStrings_[p]);
    }
  }
  hasPrintf_ = true;
  if (!same) {
    error_ = "Code object metadata notes have different printf strings";
    return false;
  }
  return true;
}

}  // namespace amd::codeobject
