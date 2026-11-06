/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace amd::roc {

class Device;

//! A blit program code object built offline (device/blit_kernels), with the device settings
//! it was compiled for
struct BlitKernelElf {
  const uint8_t* image;  //!< Code object
  size_t size;           //!< Code object size
  bool wave64;           //!< Built for wave64 (Settings::lcWavefrontSize64_)
  bool wgpMode;          //!< Built for WGP mode (Settings::enableWgpMode_)
  bool kernargPreload;   //!< Built with kernarg preload (Settings::kernel_arg_opt_)
  bool gwsInit;          //!< Has the GWS init kernel (Settings::gwsInitSupported_)
  bool images;           //!< Has the image blit kernels (Info::imageSupport_)
};

/**
 * Finds the blit program built offline for @p device: the code object for its processor, else
 * for its generic target, built for the device's current settings.
 *
 * @returns False if there is none (no offline build, another processor, settings changed by
 *          the environment, ASan, GPU_DUMP_BLIT_KERNELS); the blit kernels are then compiled
 *          at runtime.
 */
bool FindOfflineBlitProgram(const Device& device, const void** image, size_t* size);

}  // namespace amd::roc
