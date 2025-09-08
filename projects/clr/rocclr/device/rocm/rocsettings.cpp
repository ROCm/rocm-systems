/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "rocsettings.hpp"
#include "device/rocm/rocglinterop.hpp"

namespace amd::roc {

// ================================================================================================
Settings::Settings() {
  // Initialize the HSA device default settings

  // Set this to true when we drop the flag
  doublePrecision_ = ::CL_KHR_FP64;

  enableLocalMemory_ = HSA_LOCAL_MEMORY_ENABLE;

  maxWorkGroupSize_ = 1024;
  preferredWorkGroupSize_ = 256;

  kernargPoolSize_ = HSA_KERNARG_POOL_SIZE;

  // Determine if user is requesting Non-Coherent mode
  // for system memory. By default system memory is
  // operates or is programmed to be in Coherent mode.
  // Users can turn it off for hardware that does not
  // support this feature naturally
  char* nonCoherentMode = getenv("OPENCL_USE_NC_MEMORY_POLICY");
  enableNCMode_ = (nonCoherentMode) ? true : false;

  // Disable image DMA by default (ROCM runtime doesn't support it)
  imageDMA_ = false;

  stagedXferSize_ = flagIsDefault(GPU_STAGING_BUFFER_SIZE) ? 1 * Mi : GPU_STAGING_BUFFER_SIZE * Mi;

  // Initialize transfer buffer size to 1MB by default
  xferBufSize_ = 1024 * Ki;

  pinnedXferSize_ = GPU_PINNED_XFER_SIZE * Mi;
  pinnedMinXferSize_ =
      flagIsDefault(GPU_PINNED_MIN_XFER_SIZE) ? 1 * Mi : GPU_PINNED_MIN_XFER_SIZE * Mi;

  sdmaCopyThreshold_ = GPU_FORCE_BLIT_COPY_SIZE * Ki;

  // Don't support Denormals for single precision by default
  singleFpDenorm_ = false;

  apuSystem_ = false;

  // Device enqueuing settings
  numDeviceEvents_ = 1024;
  numWaitEvents_ = 8;

  useLightning_ = (!flagIsDefault(GPU_ENABLE_LC)) ? GPU_ENABLE_LC : true;

  lcWavefrontSize64_ = true;
  imageBufferWar_ = false;

  sdma_p2p_threshold_ = ROC_P2P_SDMA_SIZE * Ki;
  hmmFlags_ = (!flagIsDefault(ROC_HMM_FLAGS)) ? ROC_HMM_FLAGS : 0;

  rocr_backend_ = true;

  cpu_wait_for_signal_ = !AMD_DIRECT_DISPATCH;
  cpu_wait_for_signal_ =
      (!flagIsDefault(ROC_CPU_WAIT_FOR_SIGNAL)) ? ROC_CPU_WAIT_FOR_SIGNAL : cpu_wait_for_signal_;
  system_scope_signal_ = ROC_SYSTEM_SCOPE_SIGNAL;

  // Use coarse grain system memory for kernel arguments by default (to keep GPU cache)
  fgs_kernel_arg_ = false;
  barrier_value_packet_ = false;
  kernel_arg_impl_ = KernelArgImpl::HostKernelArgs;
  gwsInitSupported_ = true;
  limit_blit_wg_ = 16;

  dynamic_queues_ = amd::IS_HIP ? DEBUG_HIP_DYNAMIC_QUEUES : false;
  // note: OCL user events don't allow CPU blocking calls in DD mode
  blocking_blit_ = amd::IS_HIP || !AMD_DIRECT_DISPATCH;
}

// ================================================================================================
bool Settings::create(bool fullProfile, const amd::Isa& isa, bool enableXNACK, bool coop_groups,
                      bool isXgmi, bool hasValidHDPFlush) {
  uint32_t gfxipMajor = isa.versionMajor();
  uint32_t gfxipMinor = isa.versionMinor();
  uint32_t gfxStepping = isa.versionStepping();

  customHostAllocator_ = false;

  if (fullProfile) {
    pinnedXferSize_ = 0;
    stagedXferSize_ = 0;
    xferBufSize_ = 0;
    apuSystem_ = true;
  } else {
    pinnedXferSize_ = std::max(pinnedXferSize_, pinnedMinXferSize_);
  }
  enableXNACK_ = enableXNACK;
  hsailExplicitXnack_ = enableXNACK;

  // Enable extensions
  enableExtension(ClKhrByteAddressableStore);
  enableExtension(ClKhrGlobalInt32BaseAtomics);
  enableExtension(ClKhrGlobalInt32ExtendedAtomics);
  enableExtension(ClKhrLocalInt32BaseAtomics);
  enableExtension(ClKhrLocalInt32ExtendedAtomics);
  enableExtension(ClKhrInt64BaseAtomics);
  enableExtension(ClKhrInt64ExtendedAtomics);
  enableExtension(ClKhr3DImageWrites);
  enableExtension(ClAmdMediaOps);
  enableExtension(ClAmdMediaOps2);
  enableExtension(ClKhrImage2dFromBuffer);

  if (MesaInterop::Supported()) {
    enableExtension(ClKhrGlSharing);
  }

  // Enable platform extension
  enableExtension(ClAmdDeviceAttributeQuery);

  // Enable KHR double precision extension
  enableExtension(ClKhrFp64);
  enableExtension(ClKhrSubGroups);
  enableExtension(ClKhrDepthImages);
  enableExtension(ClAmdCopyBufferP2P);
  enableExtension(ClKhrFp16);
  supportDepthsRGB_ = true;

  if (useLightning_) {
    enableExtension(ClAmdAssemblyProgram);
    // enable subnormals for gfx900 and later
    if (gfxipMajor >= 9) {
      singleFpDenorm_ = true;
      enableCoopGroups_ = GPU_ENABLE_COOP_GROUPS & coop_groups;
      enableCoopMultiDeviceGroups_ = GPU_ENABLE_COOP_GROUPS & coop_groups;
    }
  } else {
    // Also enable AMD double precision extension?
    enableExtension(ClAmdFp64);
  }

  if ((gfxipMajor == 9 && gfxipMinor == 0 && gfxStepping == 10) ||
      ((gfxipMajor == 9 && gfxipMinor >= 4 &&
        (gfxStepping == 0 || gfxStepping == 1 || gfxStepping == 2)))) {
    // Enable Barrier Value packet is only for MI2XX/300
    barrier_value_packet_ = true;
  }

  setKernelArgImpl(isa, isXgmi, hasValidHDPFlush);

  if (gfxipMajor >= 10) {
    enableWave32Mode_ = true;
    enableWgpMode_ = GPU_ENABLE_WGP_MODE;
    if (gfxipMinor == 1) {
      // GFX10.1 HW doesn't support custom pitch. Enable double copy workaround
      // TODO: This should be updated when ROCr support custom pitch
      imageBufferWar_ = GPU_IMAGE_BUFFER_WAR;
    }
  }

  if (!flagIsDefault(GPU_ENABLE_WAVE32_MODE)) {
    enableWave32Mode_ = GPU_ENABLE_WAVE32_MODE;
  }

  lcWavefrontSize64_ = !enableWave32Mode_;

  if (gfxipMajor > 10 || (gfxipMajor == 9 && gfxipMinor >= 4)) {
    gwsInitSupported_ = false;
  }

  // Override current device settings
  override();

  return true;
}

// ================================================================================================
void Settings::override() {
  // Limit reported workgroup size
  if (GPU_MAX_WORKGROUP_SIZE != 0) {
    preferredWorkGroupSize_ = GPU_MAX_WORKGROUP_SIZE;
  }

  if (!flagIsDefault(GPU_XFER_BUFFER_SIZE)) {
    xferBufSize_ = GPU_XFER_BUFFER_SIZE * Ki;
  }

  if (!flagIsDefault(AMD_GPU_FORCE_SINGLE_FP_DENORM)) {
    switch (AMD_GPU_FORCE_SINGLE_FP_DENORM) {
      case 0:
        singleFpDenorm_ = false;
        break;
      case 1:
        singleFpDenorm_ = true;
        break;
      default:
        break;
    }
  }
  if (!flagIsDefault(GPU_ENABLE_COOP_GROUPS)) {
    enableCoopGroups_ = GPU_ENABLE_COOP_GROUPS;
    enableCoopMultiDeviceGroups_ = GPU_ENABLE_COOP_GROUPS;
  }

  if (!flagIsDefault(ROC_USE_FGS_KERNARG)) {
    fgs_kernel_arg_ = ROC_USE_FGS_KERNARG;
  }

  if (!flagIsDefault(DEBUG_CLR_BLIT_KERNARG_OPT)) {
    kernel_arg_opt_ = DEBUG_CLR_BLIT_KERNARG_OPT;
  }
}

// ================================================================================================
void Settings::setKernelArgImpl(const amd::Isa& isa, bool isXgmi, bool hasValidHDPFlush) {
  const uint32_t gfxipMajor = isa.versionMajor();
  const uint32_t gfxipMinor = isa.versionMinor();
  const uint32_t gfxStepping = isa.versionStepping();

  const bool isGfx94x = gfxipMajor == 9 && gfxipMinor >= 4 &&
                        (gfxStepping == 0 || gfxStepping == 1 || gfxStepping == 2);
  const bool isGfx90a = (gfxipMajor == 9 && gfxipMinor == 0 && gfxStepping == 10);
  const bool isPreGfx908 =
      (gfxipMajor < 9) || ((gfxipMajor == 9) && (gfxipMinor == 0) && (gfxStepping < 8));
  const bool isGfx101x = (gfxipMajor == 10) && ((gfxipMinor == 0) || (gfxipMinor == 1));

  auto kernelArgImpl = KernelArgImpl::HostKernelArgs;

  hasValidHDPFlush &= DEBUG_CLR_KERNARG_HDP_FLUSH_WA;

  if (isXgmi) {
    // The XGMI-connected path does not require the manual memory ordering
    // workarounds that the PCIe connected path requires
    kernelArgImpl = KernelArgImpl::DeviceKernelArgs;
  } else if (hasValidHDPFlush) {
    // If the HDP flush register is valid implement the HDP flush to MMIO
    // workaround.
    if (!(isPreGfx908 || isGfx101x)) {
      kernelArgImpl = KernelArgImpl::DeviceKernelArgsHDP;
    }
  } else if (isGfx94x || isGfx90a) {
    // Implement the kernel argument readback workaround
    // (write all args -> sfence -> write last byte -> mfence -> read last byte)
    kernelArgImpl = KernelArgImpl::DeviceKernelArgsReadback;
  }

  // Enable device kernel args for gfx94x for now
  if (isGfx94x) {
    kernel_arg_impl_ = kernelArgImpl;
    kernel_arg_opt_ = true;
  }

  if (!flagIsDefault(HIP_FORCE_DEV_KERNARG)) {
    kernel_arg_impl_ = kernelArgImpl & (HIP_FORCE_DEV_KERNARG ? 0xF : 0x0);
  }

  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Using dev kernel arg wa = %d", kernel_arg_impl_);
}
}  // namespace amd::roc
