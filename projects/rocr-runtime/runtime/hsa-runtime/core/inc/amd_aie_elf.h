/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Reader for the full-ELF kernel binaries aiecc emits (`aiecc --get-full-elf`).
//
// A full ELF carries the PDI and the control code in one file, along with the relocations that
// say where addresses have to be written into the control code. The runtime deliberately knows
// nothing about any of that: an application (or, for the unified hsaco flow, the runtime itself)
// extracts the pieces, allocates them from the agent's device memory pool, patches its own
// argument addresses in, and names the buffers in an ordinary dispatch packet. The one address
// it cannot know is the PDI's device address, so it passes the offset of that patch site in
// hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset and the runtime fills it in; that
// non-zero offset is also what selects the full-ELF dispatch shape.
//
// Only what the vector_scalar_add design needs is supported: one PDI, one control-code section
// per kernel, and buffer arguments. Control packets, preemption save/restore sections and
// scalar arguments are not handled, and an ELF using them is rejected rather than silently
// mispatched.

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_ELF_H_

#include <cstdint>
#include <map>
#include <string>
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

/// @brief A parsed full-ELF kernel: the bytes to load and where addresses go.
struct Kernel {
  /// @brief "<kernel>:<instance>", e.g. "main:sequence".
  std::string name;
  /// @brief PDI bytes; empty if the kernel has no PDI.
  std::vector<uint8_t> pdi;
  /// @brief Control-code bytes.
  std::vector<uint8_t> ctrl_code;
  /// @brief Where the PDI's device address goes in the control code. Passed to the runtime as
  /// hsa_amd_aie_kernel_dispatch_packet_t::pdi_patch_offset.
  uint64_t pdi_patch_offset = 0;
  /// @brief Whether @ref pdi_patch_offset is valid.
  bool has_pdi_patch = false;
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
/// @param [out] out Kernels found in the image, keyed by name. Cleared before use.
/// @param [out] error Human-readable message describing the failure; only touched on error.
/// @retval HSA_STATUS_SUCCESS `image` is a well-formed aie2p full ELF and at least one
/// dispatchable kernel was found.
/// @retval HSA_STATUS_ERROR_INVALID_CODE_OBJECT `image` is not a well-formed aie2p full ELF or
/// uses a feature this reader does not implement.
hsa_status_t Parse(const void* image, size_t size, std::map<std::string, Kernel>* out,
                   std::string* error);

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
