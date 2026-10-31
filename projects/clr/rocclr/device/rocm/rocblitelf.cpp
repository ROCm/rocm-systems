/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rocblitelf.hpp"
#include "device/rocm/rocdevice.hpp"

#if ROCCLR_OFFLINE_BLIT_KERNELS
#include "device/codeobject/elf_header.hpp"
#include "device/codeobject/offload_bundle.hpp"
#include "elf/elfio/elf_types.hpp"
#include "blit_kernels_elf.hpp"

#include <cstring>
#include <string>
#endif

namespace amd::roc {

// ================================================================================================
bool FindOfflineBlitProgram(const Device& device, const void** image, size_t* size) {
#if ROCCLR_OFFLINE_BLIT_KERNELS
#if DEVICE_ADDRESS_SANITIZER
  // ASan instruments the blit kernels at runtime
  return false;
#endif
  if (GPU_DUMP_BLIT_KERNELS) {
    return false;
  }
  const device::Settings& settings = device.settings();
  const std::string isa = device.isa().isaName();
  for (const std::string& target : {isa, codeobject::TargetToGeneric(isa)}) {
    if (target.empty()) {
      continue;
    }
    for (const BlitKernelElf& elf : kBlitKernelElfs) {
      amd::ELFIO::Elf64_Ehdr ehdr;
      std::memcpy(&ehdr, elf.image, sizeof(ehdr));
      std::string elfIsa;
      if (!codeobject::GetIsaFromHeader(ehdr.e_ident[EI_CLASS], ehdr.e_machine,
                                        ehdr.e_ident[EI_OSABI], ehdr.e_ident[EI_ABIVERSION],
                                        ehdr.e_flags, elfIsa) ||
          !codeobject::IsCompatibleTarget(target, elfIsa)) {
        continue;
      }
      // Built for other settings than the device has now (e.g. GPU_ENABLE_WAVE32_MODE)
      if (elf.wave64 != settings.lcWavefrontSize64_ || elf.wgpMode != settings.enableWgpMode_ ||
          elf.kernargPreload != settings.kernel_arg_opt_ ||
          (amd::IS_HIP && elf.gwsInit != settings.gwsInitSupported_) ||
          (!elf.images && device.info().imageSupport_)) {
        ClPrint(amd::LOG_INFO, amd::LOG_KERN,
                "Blit kernels for %s were built for other device settings", elfIsa.c_str());
        return false;
      }
      ClPrint(amd::LOG_INFO, amd::LOG_KERN, "Using the blit kernels built for %s",
              elfIsa.c_str());
      *image = elf.image;
      *size = elf.size;
      return true;
    }
  }
#endif
  return false;
}

}  // namespace amd::roc
