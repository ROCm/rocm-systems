/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// The amd::Isa register limits and image support (read at device init) against comgr's ISA
// metadata, where roc::Device read them before. comgr is loaded here only; the test is skipped
// if it isn't installed.

#include "device/device.hpp"
#include "os/os.hpp"

#include <catch2/catch_test_macros.hpp>

#include "amd_comgr/amd_comgr.h"

#include <cstdlib>
#include <string>

namespace {

//! The comgr entry points the comparison needs, loaded from the installed library
struct ComgrIsaMetadata {
  decltype(&amd_comgr_get_isa_metadata) getIsaMetadata = nullptr;
  decltype(&amd_comgr_metadata_lookup) lookup = nullptr;
  decltype(&amd_comgr_get_metadata_string) getString = nullptr;
  decltype(&amd_comgr_destroy_metadata) destroy = nullptr;

  bool load() {
    void* handle = amd::Os::loadLibrary("libamd_comgr.so.3");
    if (handle == nullptr) return false;
    getIsaMetadata = reinterpret_cast<decltype(getIsaMetadata)>(
        amd::Os::getSymbol(handle, "amd_comgr_get_isa_metadata"));
    lookup = reinterpret_cast<decltype(lookup)>(
        amd::Os::getSymbol(handle, "amd_comgr_metadata_lookup"));
    getString = reinterpret_cast<decltype(getString)>(
        amd::Os::getSymbol(handle, "amd_comgr_get_metadata_string"));
    destroy = reinterpret_cast<decltype(destroy)>(
        amd::Os::getSymbol(handle, "amd_comgr_destroy_metadata"));
    return getIsaMetadata && lookup && getString && destroy;
  }

  //! @returns The integer value of @p key in @p isa's metadata, or -1 if it can't be read
  long value(amd_comgr_metadata_node_t isa, const char* key) const {
    amd_comgr_metadata_node_t node;
    if (lookup(isa, key, &node) != AMD_COMGR_STATUS_SUCCESS) return -1;
    size_t size = 0;
    getString(node, &size, nullptr);
    std::string text(size, '\0');
    getString(node, &size, text.data());
    destroy(node);
    return std::strtol(text.c_str(), nullptr, 10);
  }
};

}  // namespace

TEST_CASE("ISA register limits and image support match comgr's ISA metadata", "[isa]") {
  ComgrIsaMetadata comgr;
  if (!comgr.load()) {
    SKIP("libamd_comgr.so.3 not found");
  }
  for (const amd::Isa* isa = amd::Isa::begin(); isa != amd::Isa::end(); ++isa) {
    INFO(isa->isaName());
    amd_comgr_metadata_node_t meta;
    REQUIRE(comgr.getIsaMetadata(isa->isaName().c_str(), &meta) == AMD_COMGR_STATUS_SUCCESS);
    CHECK(comgr.value(meta, "AddressableNumVGPRs") == isa->addressableVgprs());
    CHECK(comgr.value(meta, "TotalNumVGPRs") == isa->vgprsPerSimd());
    CHECK(comgr.value(meta, "VGPRAllocGranule") == isa->vgprAllocGranule());
    CHECK(comgr.value(meta, "AddressableNumSGPRs") == isa->addressableSgprs());
    CHECK(comgr.value(meta, "SGPRAllocGranule") == isa->sgprAllocGranule());
    CHECK(comgr.value(meta, "TrapHandlerEnabled") == (isa->trapHandlerEnabled() ? 1 : 0));
    CHECK(comgr.value(meta, "ImageSupport") == (isa->hasImageSupport() ? 1 : 0));
    comgr.destroy(meta);
  }
}
