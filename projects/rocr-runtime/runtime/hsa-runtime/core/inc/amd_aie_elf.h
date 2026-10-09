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

/// @brief One place in the control code that takes an address.
struct PatchSite {
  /// @brief Byte offset into the control code.
  uint32_t offset = 0;
  /// @brief Added to the address before it is written.
  uint32_t addend = 0;
};

/// @brief A PDI the control code loads, and where its device address goes.
struct Pdi {
  /// @brief PDI bytes.
  std::vector<uint8_t> bytes;
  /// @brief Byte offsets in the control code taking this PDI's device address; never empty.
  /// Parse() rejects a site at offset 0.
  std::vector<uint32_t> patch_offsets;
};

/// @brief A parsed full-ELF kernel: the bytes to load and where addresses go.
struct Kernel {
  /// @brief "<kernel>:<instance>", e.g. "main:sequence".
  std::string name;
  /// @brief The PDIs the control code loads; empty if it loads none. A control code may load
  /// several PDIs, or one more than once (MLIR-AIR's loads an empty PDI before and after the
  /// design runs). Each distinct PDI appears once: sites naming byte-identical PDIs share an
  /// entry, so it is placed in device memory once.
  std::vector<Pdi> pdis;
  /// @brief Control-code bytes.
  std::vector<uint8_t> ctrl_code;
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

/// @brief Folds a buffer address into a shim DMA buffer descriptor, the scheme the NPU firmware
/// defines. This *adds* to the descriptor already in place, so it must only ever be applied to a
/// pristine copy of the control code.
///
/// @param [in,out] site Pointer to the three-dword patch site inside the control code.
/// @param [in] addr Device address to fold in.
void PatchShimDma48(uint32_t* site, uint64_t addr);

}  // namespace aie_elf
}  // namespace AMD
}  // namespace rocr

#endif  // HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_
