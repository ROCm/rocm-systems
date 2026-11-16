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
// PDI and external-symbol addresses written in. No address is patched here. At dispatch time the
// driver copies the control code into a per-dispatch device buffer and patches the argument
// addresses into that copy, so concurrent dispatches of the same kernel never share a patch site.

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
  /// @brief Size of an aie4 page. Each page's ctrltext and ctrldata are padded out to it, as
  /// aiebu's elf_reader_gen2plus lays them out; the firmware addresses pages by that position.
  static constexpr uint32_t kAie4Size = 8192;

  /// @brief Page number: the <page> of the section names' suffix on aie4, where it places the page
  /// at number * kAie4Size in the kernel's control code; always 0 on aie2p.
  uint32_t number = 0;
  /// @brief Control-code bytes.
  std::vector<uint8_t> ctrltext;
  /// @brief Buffer-descriptor bytes the page's ctrltext addresses; empty if the page has none
  /// (aie2p, always).
  std::vector<uint8_t> ctrldata;
};

/// @brief A symbol the control code takes the address of but the ELF does not carry, such as the
/// control scratchpad (`scratch-pad-ctrl`). The application allocates it and defines it with
/// hsa_executable_agent_global_variable_define before loading.
struct ExternalSymbol {
  /// @brief Bytes the control code expects at the address: the symbol's `st_size`.
  uint32_t size = 0;
  /// @brief Places in the control code taking the symbol's device address; never empty.
  std::vector<PatchSite> sites;
};

/// @brief A parsed full-ELF kernel: the bytes to load and where addresses go.
struct Kernel {
  /// @brief "<kernel>:<instance>", e.g. "main:sequence".
  std::string name;
  /// @brief The PDIs the control code loads; always empty on aie4, whose model has no separate
  /// PDI construct. A control code may load several PDIs, or one more than once (MLIR-AIR's loads
  /// an empty PDI before and after the design runs). Entries are keyed by the ELF *section* a patch
  /// site names, not by byte content: two sites naming the same section share one entry and one
  /// device placement, but two different sections are placed separately even when their bytes are
  /// identical.
  ///
  /// This was not always true: an earlier version keyed on byte content instead, so two sections
  /// with identical bytes (e.g. an empty PDI loaded both before and after the design) collapsed to
  /// one placement at one device address. A kernel built that way dispatched correctly once and
  /// incorrectly on every dispatch after. Nothing in this reader explains why -- it does not touch
  /// the firmware or driver layer that actually loads a PDI at an address -- so this is recorded as
  /// an observed fix, not a diagnosed root cause. Keying by section instead of content does not
  /// conflict with anything the format document requires: it never asked for content-based merging
  /// across distinct sections, only that sites naming the *same* section (by name, resolved once by
  /// `find_pdi_section`) share a copy.
  std::vector<Pdi> pdis;
  /// @brief The kernel's control-code pages, in increasing page number. Never empty.
  std::vector<Page> pages;
  /// @brief Patch sites per argument index. Entries may be empty for unused arguments.
  std::vector<std::vector<PatchSite>> arg_sites;
  /// @brief External symbols the control code references, by name.
  std::map<std::string, ExternalSymbol> external_symbols;

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
