/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Reader for the full-ELF kernel binaries aiecc emits (`aiecc --get-full-elf`).
//
// A full ELF carries the PDI and the control code in one file, along with the relocations that
// say where addresses have to be written into the control code. This reader is used at load time,
// nested inside a unified hsaco's AIE section: the loader parses it, keeps the control code
// pristine in host memory, places the PDI in the agent's device memory, and patches the PDI's
// device address into the pristine copy right away (that address is not known until placement).
// Application argument addresses are not patched here -- at dispatch time, the driver copies the
// pristine control code into a per-dispatch device buffer and patches the argument addresses into
// that copy, so concurrent dispatches of the same kernel never share a patch site.
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
  /// @brief Byte offset in @ref ctrl_code where the PDI's device address is patched in at load
  /// time, once the PDI has been placed in device memory.
  /// Zero when the kernel has no patch site; Parse() rejects a real site at offset 0.
  uint64_t pdi_patch_offset = 0;
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
