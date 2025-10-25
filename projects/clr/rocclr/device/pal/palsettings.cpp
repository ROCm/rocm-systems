/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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
#include "device/pal/paldefs.hpp"
#include "device/pal/palsettings.hpp"

#include <algorithm>

#if defined(_WIN32)
#include "Windows.h"
#include "VersionHelpers.h"
#endif

namespace amd::pal {

Settings::Settings() {
  // Initialize the GPU device default settings
  oclVersion_ = OpenCL12;
  debugFlags_ = 0;
  remoteAlloc_ = REMOTE_ALLOC;

  stagedXferRead_ = true;
  stagedXferWrite_ = true;
  stagedXferSize_ = GPU_STAGING_BUFFER_SIZE * Mi;

  // We will enable staged read/write if we use local memory
  disablePersistent_ = false;

  imageSupport_ = false;
  hwLDSSize_ = 0;

  // Set this to true when we drop the flag
  doublePrecision_ = ::CL_KHR_FP64;

  // Fill workgroup info size
  maxWorkGroupSize_ = 1024;
  preferredWorkGroupSize_ = 256;

  hostMemDirectAccess_ = HostMemDisable;

  libSelector_ = amd::LibraryUndefined;

  // By default use host blit
  blitEngine_ = BlitEngineHost;
  pinnedXferSize_ = GPU_PINNED_XFER_SIZE * Mi;
  size_t defaultMinXferSize = amd::IS_HIP ? 128 : 4;
  pinnedMinXferSize_ = flagIsDefault(GPU_PINNED_MIN_XFER_SIZE) ? defaultMinXferSize * Mi
                                                               : GPU_PINNED_MIN_XFER_SIZE * Mi;

  // GPU device by default
  apuSystem_ = false;

  // Disable 64 bit pointers support by default
  use64BitPtr_ = false;

  // Max alloc size is 16GB
  maxAllocSize_ = 16 * static_cast<uint64_t>(Gi);

  // Disable memory dependency tracking by default
  numMemDependencies_ = 0;

  // By default cache isn't present
  cacheLineSize_ = 0;
  cacheSize_ = 0;

  // Initialize transfer buffer size to 1MB by default
  xferBufSize_ = 1024 * Ki;

  // Use image DMA if requested
  imageDMA_ = GPU_IMAGE_DMA;

  // Number of compute rings.
  numComputeRings_ = 0;

  // Device enqueuing settings
  numDeviceEvents_ = 1024;
  numWaitEvents_ = 8;

  numScratchWavesPerCu_ = 32;

  // Don't support platform atomics by default.
  svmAtomics_ = false;

  // Use host queue for device enqueuing by default
  useDeviceQueue_ = GPU_USE_DEVICE_QUEUE;

  // Don't support Denormals for single precision by default
  singleFpDenorm_ = false;

  // SQTT buffer size in bytes
  rgpSqttDispCount_ = PAL_RGP_DISP_COUNT;
  rgpSqttWaitIdle_ = true;
  rgpSqttForceDisable_ = false;

  // Sub allocation parameters
  subAllocationMinSize_ = 4 * Ki;
  subAllocationChunkSize_ = 64 * Mi;
  subAllocationMaxSize_ =
      std::min(static_cast<uint64_t>(GPU_MAX_SUBALLOC_SIZE) * Ki, subAllocationChunkSize_);

  maxCmdBuffers_ = 12;
  useLightning_ = amd::IS_HIP ? true : ((!flagIsDefault(GPU_ENABLE_LC)) ? GPU_ENABLE_LC : false);
  enableWgpMode_ = false;
  enableWave32Mode_ = false;
  hsailExplicitXnack_ = false;
  lcWavefrontSize64_ = true;
  enableHwP2P_ = false;
  imageBufferWar_ = false;
  disableSdma_ = PAL_DISABLE_SDMA;
  mallPolicy_ = 0;
  alwaysResident_ = amd::IS_HIP ? true : false;
  prepinnedMinSize_ = 0;
  cpDmaCopySizeMax_ = GPU_CP_DMA_COPY_SIZE * Ki;
  kernel_arg_impl_ = flagIsDefault(HIP_FORCE_DEV_KERNARG) ? KernelArgImpl::DeviceKernelArgs
                                                          : HIP_FORCE_DEV_KERNARG;

  limit_blit_wg_ = 16;
  DEBUG_CLR_GRAPH_PACKET_CAPTURE = false;  // disable graph performance optimizations for PAL
}

bool Settings::create(const Pal::DeviceProperties& palProp,
                      const Pal::GpuMemoryHeapProperties* heaps, const Pal::WorkStationCaps& wscaps,
                      const amd::Isa& isa, bool reportAsOCL12Device) {
  uint32_t osVer = 0x0;

  // Disable thread trace by default for all devices
  threadTraceEnable_ = false;

  // APU systems
  if (palProp.gpuType == Pal::GpuType::Integrated) {
    apuSystem_ = true;
  }

  enableXNACK_ = (isa.xnack() == amd::Isa::Feature::Enabled);
  hsailExplicitXnack_ = enableXNACK_;
  bool useWavefront64 = false;

  std::string appName = {};
  std::string appPathAndName = {};
  amd::Os::getAppPathAndFileName(appName, appPathAndName);

  switch (palProp.revision) {
    // Fall through for Navi4x ...
    case Pal::AsicRevision::Navi44:
    case Pal::AsicRevision::Navi48:
    // Fall through for Navi3x ...
    case Pal::AsicRevision::Navi33:
    case Pal::AsicRevision::Navi32:
    case Pal::AsicRevision::Navi31:
      gwsInitSupported_ = false;
    // Fall through for Navi2x ...
    case Pal::AsicRevision::StrixHalo:
    case Pal::AsicRevision::Strix1:
    case Pal::AsicRevision::Phoenix1:
    case Pal::AsicRevision::Phoenix2:
    case Pal::AsicRevision::HawkPoint1:
    case Pal::AsicRevision::HawkPoint2:
    case Pal::AsicRevision::Raphael:
    case Pal::AsicRevision::Rembrandt:
    case Pal::AsicRevision::Navi24:
    case Pal::AsicRevision::Navi23:
    case Pal::AsicRevision::Navi22:
    case Pal::AsicRevision::Navi21:
      // set wavefront 64 for Geekbench 5
      {
        if (appName == "Geekbench 5.exe" || appName == "geekbench_x86_64.exe" ||
            appName == "geekbench5.exe") {
          useWavefront64 = true;
        }
      }
    // Fall through for Navi1x ...
    case Pal::AsicRevision::Navi14:
    case Pal::AsicRevision::Navi12:
    case Pal::AsicRevision::Navi10:
      useLightning_ = GPU_ENABLE_LC;
      enableWgpMode_ = GPU_ENABLE_WGP_MODE;
      if (useLightning_) {
        enableWave32Mode_ = true;
      }
      if (!flagIsDefault(GPU_ENABLE_WAVE32_MODE)) {
        enableWave32Mode_ = GPU_ENABLE_WAVE32_MODE;
      }
      if (useWavefront64) {
        enableWave32Mode_ = 0;
      }
      lcWavefrontSize64_ = !enableWave32Mode_;
      if (palProp.gfxLevel == Pal::GfxIpLevel::GfxIp10_1) {
        // GFX10.1 HW doesn't support custom pitch. Enable double copy workaround
        imageBufferWar_ = GPU_IMAGE_BUFFER_WAR;
      }
      enableHwP2P_ = true;
      enableCoopGroups_ = IS_LINUX;
      enableCoopMultiDeviceGroups_ = IS_LINUX;
      if (useLightning_) {
        singleFpDenorm_ = true;
      }
      enableExtension(ClKhrFp16);
      threadTraceEnable_ = AMD_THREAD_TRACE_ENABLE;
      // Cache line size is 64 bytes
      cacheLineSize_ = 64;
      // L1 cache size is 16KB
      cacheSize_ = 16 * Ki;

      libSelector_ = amd::GPU_Library_CI;
      if (LP64_SWITCH(false, true)) {
        oclVersion_ =
            !reportAsOCL12Device ? XCONCAT(OpenCL, XCONCAT(OPENCL_MAJOR, OPENCL_MINOR)) : OpenCL12;
      }
      if (OPENCL_VERSION < 200) {
        oclVersion_ = OpenCL12;
      }
      numComputeRings_ = 8;

      // Cap at OpenCL20 for now
      if (oclVersion_ > OpenCL20) oclVersion_ = OpenCL20;

      use64BitPtr_ = LP64_SWITCH(false, true);

      if (oclVersion_ >= OpenCL20) {
        supportDepthsRGB_ = true;
      }
      if (use64BitPtr_) {
        maxAllocSize_ = 64ULL * Gi;
      } else {
        maxAllocSize_ = 3ULL * Gi;
      }
      supportRA_ = false;
      numMemDependencies_ = GPU_NUM_MEM_DEPENDENCY;
      break;
    default:
      assert(0 && "Unknown ASIC type!");
      return false;
  }

  if (0 == palProp.engineProperties[Pal::EngineTypeDma].engineCount) {
    disableSdma_ = true;
  }

  // Image DMA must be disabled if SDMA is disabled
  imageDMA_ &= !disableSdma_;

  // Enable atomics support
  enableExtension(ClKhrInt64BaseAtomics);
  enableExtension(ClKhrInt64ExtendedAtomics);
  enableExtension(ClKhrGlobalInt32BaseAtomics);
  enableExtension(ClKhrGlobalInt32ExtendedAtomics);
  enableExtension(ClKhrLocalInt32BaseAtomics);
  enableExtension(ClKhrLocalInt32ExtendedAtomics);
  enableExtension(ClKhrByteAddressableStore);
  enableExtension(ClKhr3DImageWrites);
  enableExtension(ClKhrImage2dFromBuffer);
  enableExtension(ClAmdMediaOps);
  enableExtension(ClAmdMediaOps2);

  {
    // Not supported by Unknown device
    enableExtension(ClKhrGlSharing);
    enableExtension(ClKhrGlEvent);
    enableExtension(ClAmdCopyBufferP2P);
  }

  if (!useLightning_) {
    enableExtension(ClAmdPopcnt);
    enableExtension(ClAmdVec3);
    enableExtension(ClAmdPrintf);
  }
  // Enable some platform extensions
  enableExtension(ClAmdDeviceAttributeQuery);

  if (hwLDSSize_ == 0) {
    // Use hardcoded values for now, since PAL properties aren't available with offline devices
    hwLDSSize_ = (IS_LINUX || amd::IS_HIP) ? 64 * Ki : 32 * Ki;
  }

  imageSupport_ = true;

  // Use kernels for blit if appropriate
  blitEngine_ = BlitEngineKernel;

  hostMemDirectAccess_ |= HostMemBuffer;
  // HW doesn't support untiled image writes
  // hostMemDirectAccess_ |= HostMemImage;

  if (doublePrecision_) {
    // Enable KHR double precision extension
    enableExtension(ClKhrFp64);
  }

  if (!useLightning_) {
    // Enable AMD double precision extension
    doublePrecision_ = true;
    enableExtension(ClAmdFp64);
  }

  if (palProp.gpuMemoryProperties.busAddressableMemSize > 0) {
    // Enable bus addressable memory extension
    enableExtension(ClAMDBusAddressableMemory);
  }

  svmFineGrainSystem_ = palProp.gpuMemoryProperties.flags.iommuv2Support;
  svmAtomics_ = svmFineGrainSystem_;

// SVM is not currently supported for DX Interop
#if defined(_WIN32)
  enableExtension(ClKhrD3d9Sharing);
  enableExtension(ClKhrD3d10Sharing);
  enableExtension(ClKhrD3d11Sharing);
#endif  // _WIN32

  // Enable some OpenCL 2.0 extensions
  if (oclVersion_ >= OpenCL20) {
    enableExtension(ClKhrSubGroups);
    enableExtension(ClKhrDepthImages);

    if (GPU_MIPMAP && imageSupport_) {
      enableExtension(ClKhrMipMapImage);
      enableExtension(ClKhrMipMapImageWrites);
    }

#if defined(_WIN32)
    enableExtension(ClAmdPlanarYuv);
#endif
  }

  if (apuSystem_ && ((heaps[Pal::GpuHeapLocal].logicalSize +
                      heaps[Pal::GpuHeapInvisible].logicalSize) < (150 * Mi))) {
    remoteAlloc_ = true;
  }

  // Update resource cache size
  if (remoteAlloc_) {
    resourceCacheSize_ = std::max((heaps[Pal::GpuHeapGartUswc].logicalSize / 8),
                                  (uint64_t)GPU_RESOURCE_CACHE_SIZE * Mi);
  } else {
    if (apuSystem_) {
      resourceCacheSize_ = std::max(
          ((heaps[Pal::GpuHeapLocal].logicalSize + heaps[Pal::GpuHeapInvisible].logicalSize +
            heaps[Pal::GpuHeapGartUswc].logicalSize) /
           8),
          (uint64_t)GPU_RESOURCE_CACHE_SIZE * Mi);
    } else {
      resourceCacheSize_ = std::max(
          ((heaps[Pal::GpuHeapLocal].logicalSize + heaps[Pal::GpuHeapInvisible].logicalSize) / 8),
          (uint64_t)GPU_RESOURCE_CACHE_SIZE * Mi);
    }
#if !defined(_LP64)
    resourceCacheSize_ = std::min(resourceCacheSize_, 1 * Gi);
#endif
  }

  // If is Rebar, override prepinned memory size.
  if ((heaps[Pal::GpuHeapInvisible].logicalSize == 0) &&
      (heaps[Pal::GpuHeapLocal].logicalSize > 256 * Mi)) {
    prepinnedMinSize_ = PAL_PREPINNED_MEMORY_SIZE * Ki;
  }

  limit_blit_wg_ = enableWgpMode_ ? palProp.gfxipProperties.shaderCore.numAvailableCus / 2
                                  : palProp.gfxipProperties.shaderCore.numAvailableCus;

  // Override current device settings
  override();

  return true;
}

void Settings::override() {
  // Limit reported workgroup size
  if (GPU_MAX_WORKGROUP_SIZE != 0) {
    preferredWorkGroupSize_ = GPU_MAX_WORKGROUP_SIZE;
  }

  // Override blit engine type
  if (GPU_BLIT_ENGINE_TYPE != BlitEngineDefault) {
    blitEngine_ = GPU_BLIT_ENGINE_TYPE;
  }

  if (!flagIsDefault(DEBUG_GPU_FLAGS)) {
    debugFlags_ = DEBUG_GPU_FLAGS;
  }

  if (!flagIsDefault(GPU_XFER_BUFFER_SIZE)) {
    xferBufSize_ = GPU_XFER_BUFFER_SIZE * Ki;
  }

  if (!flagIsDefault(GPU_NUM_COMPUTE_RINGS)) {
    numComputeRings_ = GPU_NUM_COMPUTE_RINGS;
  }

  if (!flagIsDefault(GPU_RESOURCE_CACHE_SIZE)) {
    resourceCacheSize_ = GPU_RESOURCE_CACHE_SIZE * Mi;
  }

  if (!flagIsDefault(GPU_ENABLE_HW_P2P)) {
    enableHwP2P_ = GPU_ENABLE_HW_P2P;
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

  if (!flagIsDefault(GPU_MAX_COMMAND_BUFFERS)) {
    maxCmdBuffers_ = GPU_MAX_COMMAND_BUFFERS;
  }

  if (!flagIsDefault(GPU_ENABLE_COOP_GROUPS)) {
    enableCoopGroups_ = GPU_ENABLE_COOP_GROUPS;
    enableCoopMultiDeviceGroups_ = GPU_ENABLE_COOP_GROUPS;
  }

  if (!flagIsDefault(PAL_MALL_POLICY)) {
    mallPolicy_ = PAL_MALL_POLICY;
  }

  if (!flagIsDefault(PAL_ALWAYS_RESIDENT)) {
    alwaysResident_ = PAL_ALWAYS_RESIDENT;
  }

  if (!flagIsDefault(DEBUG_CLR_LIMIT_BLIT_WG)) {
    limit_blit_wg_ = std::max(DEBUG_CLR_LIMIT_BLIT_WG, 0x1U);
  }
}

}  // namespace amd::pal
