/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Bundle selection: entries holding an AMDGPU ELF are matched by the ISA in their ELF header,
// other entries (SPIR-V) by their id, with comgr's first-compatible rule.

// amd_hsa_elf.h first: elf_types.hpp defines R_AMDGPU_* and EM_AMDGPU as macros, which clash
// with its enums. It needs <cstdint> first.
#include <cstdint>
#include "hsa/amd_hsa_elf.h"

#include "device/codeobject/elf_header.hpp"
#include "device/codeobject/offload_bundle.hpp"
#include "elf/elfio/elf_types.hpp"
#undef EM_AMDGPU

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <utility>
#include <vector>

using namespace amd::codeobject;
using amd::ELFIO::Elf64_Ehdr;

namespace {

const std::string kGfx1100 = "amdgcn-amd-amdhsa--gfx1100";
const std::string kGfx11Generic = "amdgcn-amd-amdhsa--gfx11-generic";
const std::string kGfx90aXnackOn = "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack+";
const std::string kGfx90aXnackOff = "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-";
const std::string kSpirv = "spirv64-amd-amdhsa--amdgcnspirv";

//! An ELF header of a code object v5 for @p mach with @p features (EF_AMDGPU_FEATURE_*_V4)
std::vector<char> MakeElf(uint32_t mach, uint32_t features = 0,
                          uint8_t abiVersion = ELF::ELFABIVERSION_AMDGPU_HSA_V5) {
  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, "\177ELF", 4);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_DATA] = ELFDATA2LSB;
  ehdr.e_ident[EI_VERSION] = EV_CURRENT;
  ehdr.e_ident[EI_OSABI] = ELF::ELFOSABI_AMDGPU_HSA;
  ehdr.e_ident[EI_ABIVERSION] = abiVersion;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = ELF::EM_AMDGPU;
  ehdr.e_version = EV_CURRENT;
  ehdr.e_flags = mach | features;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  std::vector<char> elf(sizeof(ehdr));
  std::memcpy(elf.data(), &ehdr, sizeof(ehdr));
  return elf;
}

void AppendU64(std::vector<char>& out, uint64_t value) {
  for (unsigned i = 0; i < 8; ++i) out.push_back(static_cast<char>(value >> (8u * i)));
}

//! An uncompressed clang offload bundle with @p entries (id, contents) in that order
std::vector<char> MakeBundle(const std::vector<std::pair<std::string, std::vector<char>>>& entries) {
  constexpr char kMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
  size_t headerSize = sizeof(kMagic) - 1 + 8;
  // Per entry: offset, size and id size (3 x 8 bytes), then the id
  for (const auto& entry : entries) headerSize += 24 + entry.first.size();

  std::vector<char> bundle(kMagic, kMagic + sizeof(kMagic) - 1);
  AppendU64(bundle, entries.size());
  uint64_t offset = headerSize;
  for (const auto& entry : entries) {
    AppendU64(bundle, offset);
    AppendU64(bundle, entry.second.size());
    AppendU64(bundle, entry.first.size());
    bundle.insert(bundle.end(), entry.first.begin(), entry.first.end());
    offset += entry.second.size();
  }
  for (const auto& entry : entries) {
    bundle.insert(bundle.end(), entry.second.begin(), entry.second.end());
  }
  return bundle;
}

//! Runs the lookup over the whole bundle; @returns the found entry per query
std::vector<CodeObjectRef> Lookup(const std::vector<char>& bundle,
                                  const std::vector<std::string>& targets) {
  std::vector<CodeObjectRef> found;
  REQUIRE(LookupCodeObjects(bundle.data(), bundle.size(), bundle.size(), targets, found));
  return found;
}

//! Offset of entry @p index in a bundle built by MakeBundle
uint64_t EntryOffset(const std::vector<char>& bundle, size_t index) {
  uint64_t offset = 0;
  size_t pos = 24 + 8;
  for (size_t i = 0; i <= index; ++i) {
    std::memcpy(&offset, bundle.data() + pos, 8);
    uint64_t idSize = 0;
    std::memcpy(&idSize, bundle.data() + pos + 16, 8);
    pos += 24 + idSize;
  }
  return offset;
}

}  // namespace

TEST_CASE("Bundle entry is selected by its ELF header", "[bundle]") {
  const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  const auto found = Lookup(bundle, {kGfx1100});
  CHECK(found[0].offset == EntryOffset(bundle, 0));
  CHECK(found[0].size == sizeof(Elf64_Ehdr));
}

TEST_CASE("LLVM amdgpuN.M bundle ids are matched through the ELF header", "[bundle]") {
  // Triples from "AMDGPU: Introduce amdgpu triple arch"; the id's spelling isn't parsed
  const auto bundle = MakeBundle(
      {{"hipv4-amdgpu11.00-amd-amdhsa-unknown-gfx1100", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)},
       {"hipv4-amdgpu11-amd-amdhsa-unknown", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC)}});
  const auto found = Lookup(bundle, {kGfx1100, kGfx11Generic});
  CHECK(found[0].offset == EntryOffset(bundle, 0));
  CHECK(found[1].offset == EntryOffset(bundle, 1));
}

TEST_CASE("The ELF header wins over the id", "[bundle]") {
  // The id claims gfx1100 but the code object is gfx90a: never load it on gfx1100
  const auto bundle = MakeBundle({{"hipv4-" + kGfx1100,
                                   MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A,
                                           ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4 |
                                               ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4)}});
  const auto found = Lookup(bundle, {kGfx1100, kGfx90aXnackOn});
  CHECK(found[0].size == 0);
  CHECK(found[1].offset == EntryOffset(bundle, 0));
}

TEST_CASE("A generic code object matches the generic query only", "[bundle]") {
  const auto bundle = MakeBundle(
      {{"hipv4-" + kGfx11Generic, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX11_GENERIC)}});
  const auto found = Lookup(bundle, {kGfx1100, kGfx11Generic});
  CHECK(found[0].size == 0);
  CHECK(found[1].offset == EntryOffset(bundle, 0));
}

TEST_CASE("xnack and sramecc settings from e_flags", "[bundle]") {
  const uint32_t sramecc = ELF::EF_AMDGPU_FEATURE_SRAMECC_ON_V4;
  SECTION("xnack+ code object runs on xnack+ only") {
    const auto bundle = MakeBundle({{"hipv4-x", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A,
                                                         sramecc | ELF::EF_AMDGPU_FEATURE_XNACK_ON_V4)}});
    const auto found = Lookup(bundle, {kGfx90aXnackOff, kGfx90aXnackOn});
    CHECK(found[0].size == 0);
    CHECK(found[1].size != 0);
  }
  SECTION("xnack any matches either setting, first query first") {
    const auto bundle = MakeBundle({{"hipv4-x", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A,
                                                         sramecc | ELF::EF_AMDGPU_FEATURE_XNACK_ANY_V4)}});
    const auto found = Lookup(bundle, {kGfx90aXnackOff, kGfx90aXnackOn});
    CHECK(found[0].size != 0);
    CHECK(found[1].size == 0);  // each entry fills one query, as in comgr
  }
  SECTION("sramecc- code object doesn't run on sramecc+") {
    const auto bundle = MakeBundle({{"hipv4-x", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A,
                                                         ELF::EF_AMDGPU_FEATURE_SRAMECC_OFF_V4)}});
    const auto found = Lookup(bundle, {kGfx90aXnackOn});
    CHECK(found[0].size == 0);
  }
}

TEST_CASE("Code object v3 feature bits", "[bundle]") {
  // v3 has one bit per feature: set is on, clear is off
  const auto bundle = MakeBundle({{"hipv4-x", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A,
                                                       ELF::EF_AMDGPU_FEATURE_SRAMECC_V3,
                                                       ELF::ELFABIVERSION_AMDGPU_HSA_V3)}});
  const auto found = Lookup(bundle, {kGfx90aXnackOn, kGfx90aXnackOff});
  CHECK(found[0].size == 0);
  CHECK(found[1].size != 0);
}

TEST_CASE("The first compatible entry wins", "[bundle]") {
  const auto bundle = MakeBundle({{"hipv4-a", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)},
                                  {"hipv4-b", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  const auto found = Lookup(bundle, {kGfx1100});
  CHECK(found[0].offset == EntryOffset(bundle, 0));
}

TEST_CASE("Only hip, hipv4 and hcc entries are used", "[bundle]") {
  const auto bundle = MakeBundle({{"host-x86_64-unknown-linux-gnu", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)},
                                  {"openmp-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)},
                                  {"hip-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  const auto found = Lookup(bundle, {kGfx1100});
  CHECK(found[0].offset == EntryOffset(bundle, 2));
}

TEST_CASE("SPIR-V entries are matched by id", "[bundle]") {
  const std::vector<char> spirv = {0x03, 0x02, 0x23, 0x07, 0, 0, 0, 0};
  const auto bundle = MakeBundle({{"hipv4-" + kSpirv, spirv},
                                  {"hipv4-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  const auto found = Lookup(bundle, {kGfx1100, kSpirv});
  CHECK(found[0].offset == EntryOffset(bundle, 1));
  CHECK(found[1].offset == EntryOffset(bundle, 0));
}

TEST_CASE("Entries the loader can't name are skipped", "[bundle]") {
  SECTION("unknown machine") {
    const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, MakeElf(0xfe)}});
    CHECK(Lookup(bundle, {kGfx1100})[0].size == 0);
  }
  SECTION("code object v2") {
    const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100, 0,
                                                                   /*abiVersion=*/0)}});
    CHECK(Lookup(bundle, {kGfx1100})[0].size == 0);
  }
  SECTION("32-bit ELF") {
    auto elf = MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100);
    elf[EI_CLASS] = ELFCLASS32;
    const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, elf}});
    CHECK(Lookup(bundle, {kGfx1100})[0].size == 0);
  }
}

TEST_CASE("ELF headers outside the image aren't read", "[bundle]") {
  // The entry's code object lies past imageBound: it is named by its id, and the caller
  // rejects the offset
  const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  std::vector<CodeObjectRef> found;
  const size_t headerOnly = static_cast<size_t>(EntryOffset(bundle, 0));
  REQUIRE(LookupCodeObjects(bundle.data(), headerOnly, headerOnly, {kGfx1100}, found));
  CHECK(found[0].size == sizeof(Elf64_Ehdr));
}

TEST_CASE("Malformed bundle headers fail the lookup", "[bundle]") {
  const auto bundle = MakeBundle({{"hipv4-" + kGfx1100, MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)}});
  std::vector<CodeObjectRef> found;
  SECTION("truncated inside the entry table") {
    CHECK_FALSE(LookupCodeObjects(bundle.data(), 40, bundle.size(), {kGfx1100}, found));
  }
  SECTION("not a bundle") {
    std::vector<char> other = bundle;
    other[0] = 'X';
    CHECK_FALSE(LookupCodeObjects(other.data(), other.size(), other.size(), {kGfx1100}, found));
  }
  SECTION("id longer than the header") {
    std::vector<char> other = bundle;
    const uint64_t hugeId = ~uint64_t(0);
    std::memcpy(other.data() + 24 + 8 + 16, &hugeId, 8);
    CHECK_FALSE(LookupCodeObjects(other.data(), other.size(), other.size(), {kGfx1100}, found));
  }
}

TEST_CASE("Target compatibility follows comgr's isCompatibleIsaName", "[bundle]") {
  // Identical names always match
  CHECK(IsCompatibleTarget(kGfx1100, kGfx1100));
  CHECK(IsCompatibleTarget(kSpirv, kSpirv));
  // A code object that leaves a feature unset runs with either setting
  CHECK(IsCompatibleTarget(kGfx90aXnackOn, "amdgcn-amd-amdhsa--gfx90a"));
  CHECK(IsCompatibleTarget(kGfx90aXnackOff, "amdgcn-amd-amdhsa--gfx90a:sramecc+"));
  // A feature the code object sets must match the device
  CHECK_FALSE(IsCompatibleTarget(kGfx90aXnackOff, kGfx90aXnackOn));
  CHECK_FALSE(IsCompatibleTarget("amdgcn-amd-amdhsa--gfx90a", kGfx90aXnackOn));
  // Different processors, including a generic target by name
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, "amdgcn-amd-amdhsa--gfx1101"));
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, kGfx11Generic));
  // A feature the ISA doesn't support makes the name invalid
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, "amdgcn-amd-amdhsa--gfx1100:xnack+"));
  CHECK_FALSE(IsCompatibleTarget(kGfx90aXnackOn, "amdgcn-amd-amdhsa--gfx90a:wavefrontsize64+"));
  // SPIR-V takes no features
  CHECK_FALSE(IsCompatibleTarget(kSpirv, kSpirv + ":xnack+"));
  // The amdgpu arch spelling is accepted like amdgcn
  CHECK(IsCompatibleTarget(kGfx1100, "amdgpu-amd-amdhsa--gfx1100"));
  // Malformed and unknown names
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, "amdgcn-amd-amdhsa-gfx1100"));
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, "amdgcn-amd-amdhsa--gfx9999"));
  CHECK_FALSE(IsCompatibleTarget(kGfx1100, "x86_64-amd-amdhsa--gfx1100"));
}

TEST_CASE("Bundle size covers the header and every entry", "[bundle]") {
  const auto bundle = MakeBundle({{"hipv4-a", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX1100)},
                                  {"hipv4-b", MakeElf(ELF::EF_AMDGPU_MACH_AMDGCN_GFX90A)}});
  CHECK(OffloadBundleSize(bundle.data(), bundle.size()) == bundle.size());
  CHECK(OffloadBundleSize(bundle.data(), 20) == 0);
}
