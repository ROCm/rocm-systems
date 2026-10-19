/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Reader for the full-ELF kernel binaries aiecc emits (`aiecc --get-full-elf`).
//
// A full ELF carries the PDIs, the control code, and the relocations that say where addresses go
// in the control code. The loader parses the full ELF found in a unified hsaco's AIE section,
// places the PDIs in the agent's device memory and keeps the control code in host memory with the
// PDI addresses written in. No address is patched here. At dispatch time the driver copies the
// control code into a per-dispatch device buffer and patches the argument addresses into that copy,
// so concurrent dispatches of the same kernel never share a patch site.

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_

#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "inc/hsa.h"

namespace rocr {
namespace AMD {
namespace aie_elf {

/// @brief Size of an aie4 control-code page. Page N of a kernel's control code starts at
/// N * kAie4PageSize, each page's ctrltext and ctrldata padded out to it, as aiebu's
/// elf_reader_gen2plus lays it out; the firmware addresses pages by that position.
constexpr uint32_t kAie4PageSize = 8192;

/// @brief One place in the control code that takes an address.
struct PatchSite {
  /// @brief Index into Kernel::pages naming which page's buffer @c offset is relative to.
  uint32_t page = 0;
  /// @brief True if @c offset is into that page's ctrldata buffer; false for its ctrltext one.
  bool in_ctrldata = false;
  /// @brief Byte offset into the named buffer.
  uint32_t offset = 0;
  /// @brief Added to the address before it is written.
  uint32_t addend = 0;
};

/// @brief One place a PDI's device address is stored. PDI sites are aie2p-only today and always
/// land in a page's ctrltext buffer (aie2p has no ctrldata).
struct PdiPatchSite {
  /// @brief Index into Kernel::pages naming which page's ctrltext buffer @c offset is relative to.
  uint32_t page = 0;
  /// @brief Byte offset into that page's ctrltext buffer.
  uint32_t offset = 0;
};

/// @brief A PDI the control code loads, and where its device address goes.
struct Pdi {
  /// @brief PDI bytes.
  std::vector<uint8_t> bytes;
  /// @brief Sites taking this PDI's device address; never empty. Parse() rejects a site at
  /// offset 0.
  std::vector<PdiPatchSite> patch_sites;
};

/// @brief One page of a kernel's control code: a chunk of instructions (ctrltext) and, for an
/// aie4 full ELF, the buffer descriptors it addresses (ctrldata). aie2p has exactly one page and
/// no ctrldata: its argument and PDI patch sites all live in ctrltext. aie4 bundles several pages
/// per kernel, each potentially with its own argument patch sites in ctrldata. A page is
/// identified by the suffix aiecc gave its ctrltext/ctrldata section names, "<col>.<page>[.<id>]"
/// (e.g. ".ctrltext.0.2.0" and ".ctrldata.0.2.0" share the page).
struct Page {
  /// @brief Page number: the <page> of the section names' suffix on aie4, where it places the page
  /// at number * kAie4PageSize; always 0 on aie2p.
  uint32_t number = 0;
  /// @brief Control-code bytes.
  std::vector<uint8_t> ctrltext;
  /// @brief Buffer-descriptor bytes the page's ctrltext addresses; empty if the page has none
  /// (aie2p, always).
  std::vector<uint8_t> ctrldata;
};

/// @brief A parsed full-ELF kernel: the bytes to load and where addresses go.
struct Kernel {
  /// @brief "<kernel>:<instance>", e.g. "main:sequence".
  std::string name;
  /// @brief The PDIs the control code loads; always empty on aie4, whose model has no separate
  /// PDI construct. A control code may load several PDIs, or one more than once (MLIR-AIR's loads
  /// an empty PDI before and after the design runs). Each distinct PDI appears once: sites naming
  /// byte-identical PDIs share an entry, so it is placed in device memory once.
  std::vector<Pdi> pdis;
  /// @brief The kernel's control-code pages, in increasing page number. Never empty.
  std::vector<Page> pages;
  /// @brief Patch sites per argument index. Entries may be empty for unused arguments.
  std::vector<std::vector<PatchSite>> arg_sites;

  /// @brief Number of arguments the control code references.
  uint32_t num_args() const { return static_cast<uint32_t>(arg_sites.size()); }
};

/// @brief Parses `image` and returns every dispatchable kernel in it, keyed by
/// "<kernel>:<instance>".
///
/// @param [in] image Pointer to the ELF image bytes.
/// @param [in] size Size of `image` in bytes.
/// @param [in] arch Arch name of the hsaco section the ELF was found in, e.g. "aie2p". The ELF's
/// OS/ABI must be the one for this arch.
/// @param [out] out Kernels found in the image, keyed by name. Replaced on success; left untouched
/// on failure.
/// @param [out] error Human-readable message describing the failure; only touched on error.
/// @retval HSA_STATUS_SUCCESS `image` is a well-formed full ELF for `arch` and at least one
/// dispatchable kernel was found.
/// @retval HSA_STATUS_ERROR_INVALID_CODE_OBJECT `image` is not a well-formed full ELF for `arch`,
/// `arch` has no full-ELF format, or `image` uses a feature this reader does not implement.
hsa_status_t Parse(const void* image, size_t size, std::string_view arch,
                   std::map<std::string, Kernel>* out, std::string* error);

}  // namespace aie_elf
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_
