/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// libFuzzer harness for the parsers that read application-supplied code object bytes: bundle
// lookup, compressed bundles, ELF image size, .dynsym names and metadata notes, and MessagePack.
// Run with a corpus of real fat binaries and code objects, e.g.
// ./codeobject_fuzz corpus/ -max_len=65536

#include "device/codeobject/compressed_bundle.hpp"
#include "device/codeobject/elf_header.hpp"
#include "device/codeobject/metadata_notes.hpp"
#include "device/codeobject/msgpack.hpp"
#include "device/codeobject/offload_bundle.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace amd::codeobject;

namespace {


//! Reads a metadata note the way Program does, with a kernel callback that walks each kernel map
bool ParseNote(const char* note, uint32_t size) {
  std::vector<std::string> printfStrings;
  MetadataNotes notes(printfStrings, [](MsgPackReader& reader) {
    uint32_t fields = 0;
    if (!reader.readMapSize(&fields)) return false;
    for (uint32_t i = 0; i < fields; ++i) {
      std::string_view key;
      if (!reader.readString(&key) || !reader.skip()) return false;
    }
    return true;
  });
  return notes.parse(note, size);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  static const std::vector<std::string> kTargets = {
      "amdgcn-amd-amdhsa--gfx1100", "amdgcn-amd-amdhsa--gfx11-generic",
      "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-", "spirv64-amd-amdhsa--amdgcnspirv"};
  const char* bytes = reinterpret_cast<const char*>(data);

  std::vector<CodeObjectRef> found;
  if (LookupCodeObjects(bytes, size, size, kTargets, found)) {
    for (const CodeObjectRef& ref : found) {
      // What the runtime does next with a selected entry
      if (ref.size != 0 && ref.offset <= size && ref.size <= size - ref.offset) {
        ElfImageSize(bytes + ref.offset, static_cast<size_t>(ref.size));
      }
    }
  }
  OffloadBundleSize(bytes, size);

  CompressedBundleSize(bytes, size);
  // No size pre-filter: DecompressBundle must reject a header size the data can't produce
  if (IsKnownCompressedBundle(bytes, size)) {
    std::vector<char> out;
    std::string error;
    if (DecompressBundle(bytes, size, out, &error)) {
      LookupCodeObjects(out.data(), out.size(), out.size(), kTargets, found);
    }
  }

  if (size >= 4 && IsElf(bytes)) {
    // What hipModuleLoadData does with a raw ELF: measure it, then read its symbols and notes
    ElfImageSize(bytes, size);
    std::vector<std::string> funcs;
    std::vector<std::string> vars;
    DynamicSymbolNames(bytes, size, funcs, vars);
    ForEachNote(bytes, size, kAmdgpuNoteName, kAmdgpuMetadataNoteType,
                [](const char* note, uint32_t noteSize) { return ParseNote(note, noteSize); });
  }

  // The input as one metadata note
  ParseNote(bytes, static_cast<uint32_t>(size));
  MsgPackReader reader(bytes, size);
  while (reader.skip()) {
  }
  return 0;
}
