/* Copyright (c) 2008 - 2023 Advanced Micro Devices, Inc.

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

#include "platform/program.hpp"
#include "platform/kernel.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/paldevice.hpp"
#include "device/pal/palgpuopen.hpp"
#include "device/pal/palubercapturemgr.hpp"
#include "utils/flags.hpp"
#include "utils/versions.hpp"
#include "thread/monitor.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palsettings.hpp"
#include "device/pal/palblit.hpp"
#include "palLib.h"
#include "palPlatform.h"
#include "palDevice.h"
#include "palQueueSemaphore.h"
#include "hsailctx.hpp"

#include "vdi_common.hpp"

#ifdef _WIN32
#include <d3d9.h>
#include <d3d10_1.h>
#include "CL/cl_d3d10.h"
#include "CL/cl_d3d11.h"
#include "CL/cl_dx9_media_sharing.h"
#endif  // _WIN32

#include <algorithm>
#include <array>
#include <cstring>
#include <ctype.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>

#ifdef PAL_GPUOPEN_OCL
// gpuutil headers
#include "gpuUtil/palGpaSession.h"
#include "palTraceSession.h"
#include "devDriverServer.h"
#include "protocols/rgpServer.h"
#include "protocols/driverControlServer.h"
#endif  // PAL_GPUOPEN_OCL

extern struct r_debug* _amdgpu_r_debug_ptr;

namespace {

//! Define the mapping from PAL asic revision enumeration values to the
//! compiler gfx major/minor/stepping version.
struct PalDevice {
  uint32_t gfxipMajor_;             //!< The core engine GFXIP Major version
  uint32_t gfxipMinor_;             //!< The core engine GFXIP Minor version
  uint32_t gfxipStepping_;          //!< The core engine GFXIP Stepping version
  Pal::GfxIpLevel gfxIpLevel_;      //!< PAL gfx IP level
  const char* palName_;             //!< PAL device name
  Pal::AsicRevision asicRevision_;  //!< PAL AsicRevision
};

static constexpr PalDevice supportedPalDevices[] = {
    // GFX Version PAL GFX IP Level            PAL Name         PAL ASIC Revision
    {10, 1, 0, Pal::GfxIpLevel::GfxIp10_1, "gfx1010", Pal::AsicRevision::Navi10},
    {10, 1, 1, Pal::GfxIpLevel::GfxIp10_1, "gfx1011", Pal::AsicRevision::Navi12},
    {10, 1, 2, Pal::GfxIpLevel::GfxIp10_1, "gfx1012", Pal::AsicRevision::Navi14},
    {10, 3, 0, Pal::GfxIpLevel::GfxIp10_3, "gfx1030", Pal::AsicRevision::Navi21},
    {10, 3, 1, Pal::GfxIpLevel::GfxIp10_3, "gfx1031", Pal::AsicRevision::Navi22},
    {10, 3, 2, Pal::GfxIpLevel::GfxIp10_3, "gfx1032", Pal::AsicRevision::Navi23},
    {10, 3, 4, Pal::GfxIpLevel::GfxIp10_3, "gfx1034", Pal::AsicRevision::Navi24},
    {10, 3, 5, Pal::GfxIpLevel::GfxIp10_3, "gfx1035", Pal::AsicRevision::Rembrandt},
    {10, 3, 6, Pal::GfxIpLevel::GfxIp10_3, "gfx1036", Pal::AsicRevision::Raphael},
    {11, 0, 0, Pal::GfxIpLevel::GfxIp11_0, "gfx1100", Pal::AsicRevision::Navi31},
    {11, 0, 1, Pal::GfxIpLevel::GfxIp11_0, "gfx1101", Pal::AsicRevision::Navi32},
    {11, 0, 2, Pal::GfxIpLevel::GfxIp11_0, "gfx1102", Pal::AsicRevision::Navi33},
    {11, 0, 3, Pal::GfxIpLevel::GfxIp11_0, "gfx1103", Pal::AsicRevision::Phoenix1},
    {11, 0, 3, Pal::GfxIpLevel::GfxIp11_0, "gfx1103", Pal::AsicRevision::Phoenix2},
    {11, 0, 3, Pal::GfxIpLevel::GfxIp11_0, "gfx1103", Pal::AsicRevision::HawkPoint1},
    {11, 0, 3, Pal::GfxIpLevel::GfxIp11_0, "gfx1103", Pal::AsicRevision::HawkPoint2},
    {11, 5, 0, Pal::GfxIpLevel::GfxIp11_5, "gfx1150", Pal::AsicRevision::Strix1},
    {11, 5, 1, Pal::GfxIpLevel::GfxIp11_5, "gfx1151", Pal::AsicRevision::StrixHalo},
    {12, 0, 0, Pal::GfxIpLevel::GfxIp12, "gfx1200", Pal::AsicRevision::Navi44},
    {12, 0, 1, Pal::GfxIpLevel::GfxIp12, "gfx1201", Pal::AsicRevision::Navi48},
};

static std::tuple<const amd::Isa*, const char*> findIsa(uint32_t gfxipMajor, uint32_t gfxipMinor,
                                                        uint32_t gfxipStepping, bool sramecc,
                                                        bool xnack) {
  auto palDeviceIter = std::find_if(std::begin(supportedPalDevices), std::end(supportedPalDevices),
                                    [&](const PalDevice& palDevice) {
                                      return palDevice.gfxipMajor_ == gfxipMajor &&
                                             palDevice.gfxipMinor_ == gfxipMinor &&
                                             palDevice.gfxipStepping_ == (gfxipStepping & 0xF);
                                    });
  if (palDeviceIter == std::end(supportedPalDevices)) {
    return std::make_tuple(nullptr, nullptr);
  }
  const amd::Isa* isa = amd::Isa::findIsa(
      palDeviceIter->gfxipMajor_, palDeviceIter->gfxipMinor_, palDeviceIter->gfxipStepping_,
      sramecc ? amd::Isa::Feature::Enabled : amd::Isa::Feature::Disabled,
      xnack ? amd::Isa::Feature::Enabled : amd::Isa::Feature::Disabled);
  return std::make_tuple(
      isa, (palDeviceIter->gfxipMajor_ > 8) ? isa->hsailName() : palDeviceIter->palName_);
}

static std::tuple<Pal::GfxIpLevel, Pal::AsicRevision, const char*> findPal(uint32_t gfxipMajor,
                                                                           uint32_t gfxipMinor,
                                                                           uint32_t gfxipStepping) {
  auto palDeviceIter = std::find_if(std::begin(supportedPalDevices), std::end(supportedPalDevices),
                                    [&](const PalDevice& palDevice) {
                                      return palDevice.gfxipMajor_ == gfxipMajor &&
                                             palDevice.gfxipMinor_ == gfxipMinor &&
                                             palDevice.gfxipStepping_ == (gfxipStepping & 0xF);
                                    });
  if (palDeviceIter == std::end(supportedPalDevices)) {
    return std::make_tuple(Pal::GfxIpLevel::None, Pal::AsicRevision::Unknown, nullptr);
  }
  return std::make_tuple(palDeviceIter->gfxIpLevel_, palDeviceIter->asicRevision_,
                         palDeviceIter->palName_);
}

}  // namespace

namespace amd::device {
extern const char* HipExtraSourceCode;
extern const char* HipExtraSourceCodeNoGWS;
}  // namespace amd::device

bool PalDeviceLoad() {
  bool ret = false;

  // Create online devices
  ret |= amd::pal::Device::init();
  // Create offline GPU devices
  ret |= amd::pal::NullDevice::init();

  return ret;
}

void PalDeviceUnload() { amd::pal::Device::tearDown(); }

namespace amd::pal {

Util::GenericAllocator NullDevice::allocator_;
char* Device::platformObj_;
Pal::IPlatform* Device::platform_;

#if defined(WITH_COMPILER_LIB)
NullDevice::Compiler* NullDevice::compiler_;
#endif
AppProfile Device::appProfile_;

Pal::IDevice* gDeviceList[Pal::MaxDevices] = {};
uint32_t gStartDevice = 0;
uint32_t gNumDevices = 0;

NullDevice::NullDevice() : amd::Device(), ipLevel_(Pal::GfxIpLevel::None), palName_(nullptr) {}

bool NullDevice::init() {
  // Create offline devices for all ISAs not already associated with an online
  // device. This allows code objects to be compiled for all supported ISAs.
  std::vector<Device*> devices = getDevices(CL_DEVICE_TYPE_GPU, false);
  for (const amd::Isa* isa = amd::Isa::begin(); isa != amd::Isa::end(); isa++) {
    if (!isa->runtimePalSupported() || (isa->sramecc() == amd::Isa::Feature::Any) ||
        (isa->xnack() == amd::Isa::Feature::Any)) {
      continue;
    }
    bool isOnline = false;
    // Check if the particular device is online
    for (size_t i = 0; i < devices.size(); i++) {
      if (&(devices[i]->isa()) == isa) {
        isOnline = true;
        break;
      }
    }
    if (isOnline) {
      continue;
    }

    Pal::GfxIpLevel gfxIpLevel;
    Pal::AsicRevision asicRevision;
    const char* palName;
    std::tie(gfxIpLevel, asicRevision, palName) =
        findPal(isa->versionMajor(), isa->versionMinor(), isa->versionStepping());
    if (asicRevision == Pal::AsicRevision::Unknown) {
      // PAL does not support this asic.
      continue;
    }

    std::unique_ptr<NullDevice> nullDevice(new NullDevice());
    if (!nullDevice) {
      LogPrintfError("Error allocating new instance of offline PAL Device %s", isa->targetId());
      return false;
    }
    if (!nullDevice->create(palName, *isa, gfxIpLevel, asicRevision)) {
      // Skip over unsupported devices
      LogPrintfError("Skipping creating new instance of offline PAL Device %s", isa->targetId());
      continue;
    }
    nullDevice.release()->registerDevice();
  }
  return true;
}

bool NullDevice::create(const char* palName, const amd::Isa& isa, Pal::GfxIpLevel ipLevel,
                        Pal::AsicRevision asicRevision) {
  if (!isa.runtimePalSupported()) {
    LogPrintfError("Offline PAL device %s is not supported", isa.targetId());
    return false;
  }

  online_ = false;
  palName_ = palName;
  Pal::DeviceProperties properties = {};

  // Use fake GFX IP for the device init
  asicRevision_ = asicRevision;
  ipLevel_ = ipLevel;
  properties.revision = asicRevision;
  properties.gfxLevel = ipLevel;
  properties.gfxTriple.major = isa.versionMajor();
  properties.gfxTriple.major = isa.versionMinor();
  properties.gfxTriple.stepping = isa.versionStepping();
  uint subtarget = 0;

  pal::Settings* palSettings = new pal::Settings();
  settings_ = palSettings;

  // Report 512MB for all offline devices
  Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount];
  heaps[Pal::GpuHeapLocal].logicalSize = heaps[Pal::GpuHeapLocal].physicalSize = 512 * Mi;

  Pal::WorkStationCaps wscaps = {};

  // Create setting for the offline target
  if ((palSettings == nullptr) ||
      !palSettings->create(properties, heaps, wscaps, isa)) {
    LogPrintfError("Unable to create PAL setting for offline PAL device %s", isa.targetId());
    return false;
  }
  if (!settings().useLightning_) {
    if ((isa.hsailName() != nullptr)) {
      palName_ = isa.hsailName();
    } else {
      return false;
    }
  }

  if (!ValidateComgr()) {
    LogPrintfError("Code object manager initialization failed for offline PAL device %s",
                   isa.targetId());
    return false;
  }

  if (!ValidateHsail()) {
    LogPrintfError("HSAIL initialization failed for offline PAL device %s", isa.targetId());
    return false;
  }

  if (!amd::Device::create(isa)) {
    LogPrintfError("Unable to setup device for PAL offline device %s", isa.targetId());
    return false;
  }

  // Fill the device info structure
  fillDeviceInfo(properties, heaps, 4096, 1, 0, nullptr);

  // Runtime doesn't know what local size could be on the real board
  info_.maxGlobalVariableSize_ = static_cast<size_t>(512 * Mi);

  info_.wavefrontWidth_ = settings().enableWave32Mode_ ? 32 : 64;

  if (!settings().useLightning_) {
#if defined(WITH_COMPILER_LIB)
    const char* library = getenv("HSA_COMPILER_LIBRARY");
    aclCompilerOptions opts = {sizeof(aclCompilerOptions_0_8),
                               library,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
    // Initialize the compiler handle
    acl_error error;
    compiler_ = amd::Hsail::CompilerInit(&opts, &error);
    if (error != ACL_SUCCESS) {
      LogPrintfError("Error initializing the compiler for offline PAL device %s", isa.targetId());
      return false;
    }
#endif  // defined(WITH_COMPILER_LIB)
  }

  return true;
}

device::Program* NullDevice::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }

  if (program == nullptr) {
    LogError("Memory allocation has failed!");
  }

  return program;
}

void setUUID(Pal::DeviceProperties* devProps, char* uuid) {
  snprintf(uuid, 5, "%04d", devProps->pciProperties.domainNumber);
  snprintf(uuid + 4, 5, "%04d", devProps->pciProperties.busNumber);
  snprintf(uuid + 8, 5, "%04d", devProps->pciProperties.deviceNumber);
  snprintf(uuid + 12, 5, "%04d", devProps->pciProperties.functionNumber);
}

void NullDevice::fillDeviceInfo(const Pal::DeviceProperties& palProp,
                                const Pal::GpuMemoryHeapProperties heaps[Pal::GpuHeapCount],
                                size_t maxTextureSize, uint numComputeRings,
                                uint numExclusiveComputeRings, Pal::IDevice* pal_device) {
  info_.type_ = CL_DEVICE_TYPE_GPU;
  info_.vendorId_ = palProp.vendorId;
  // Set uuid
  Pal::DeviceProperties palPropTmp = palProp;
  setUUID(&palPropTmp, &info_.uuid_[0]);

  info_.maxWorkItemDimensions_ = 3;

  info_.maxComputeUnits_ = settings().enableWgpMode_
                               ? palProp.gfxipProperties.shaderCore.numAvailableCus / 2
                               : palProp.gfxipProperties.shaderCore.numAvailableCus;
  info_.maxPhysicalComputeUnits_ = info_.maxComputeUnits_;
  info_.numberOfShaderEngines = palProp.gfxipProperties.shaderCore.numShaderEngines;

  // SI parts are scalar.  Also, reads don't need to be 128-bits to get peak rates.
  // For example, float4 is not faster than float as long as all threads fetch the same
  // amount of data and the reads are coalesced.  This is from the H/W team and confirmed
  // through experimentation.  May also be true on EG/NI, but no point in confusing
  // developers now.
  info_.nativeVectorWidthChar_ = info_.preferredVectorWidthChar_ = 4;
  info_.nativeVectorWidthShort_ = info_.preferredVectorWidthShort_ = 2;
  info_.nativeVectorWidthInt_ = info_.preferredVectorWidthInt_ = 1;
  info_.nativeVectorWidthLong_ = info_.preferredVectorWidthLong_ = 1;
  info_.nativeVectorWidthFloat_ = info_.preferredVectorWidthFloat_ = 1;
  info_.nativeVectorWidthDouble_ = info_.preferredVectorWidthDouble_ =
      (settings().checkExtension(ClKhrFp64)) ? 1 : 0;
  info_.nativeVectorWidthHalf_ = info_.preferredVectorWidthHalf_ = 0;  // no half support

  info_.maxEngineClockFrequency_ = (palProp.gfxipProperties.performance.maxGpuClock != 0)
                                       ? palProp.gfxipProperties.performance.maxGpuClock
                                       : 555;
  info_.maxMemoryClockFrequency_ = (palProp.gpuMemoryProperties.performance.maxMemClock != 0)
                                       ? palProp.gpuMemoryProperties.performance.maxMemClock
                                       : 555;
  info_.wallClockFrequency_ = palProp.timestampFrequency / 1000;  // in KHz
  info_.vramBusBitWidth_ = palProp.gpuMemoryProperties.performance.vramBusBitWidth;
  info_.l2CacheSize_ = palProp.gfxipProperties.shaderCore.tccSizeInBytes;
  info_.maxParameterSize_ = 1024;
  info_.minDataTypeAlignSize_ = sizeof(int64_t[16]);
  info_.singleFPConfig_ =
      CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_FMA;

  if (settings().singleFpDenorm_) {
    info_.singleFPConfig_ |= CL_FP_DENORM;
  }

  if (settings().checkExtension(ClKhrFp64)) {
    info_.doubleFPConfig_ = info_.singleFPConfig_ | CL_FP_DENORM;
  }
  info_.singleFPConfig_ |= CL_FP_CORRECTLY_ROUNDED_DIVIDE_SQRT;

  if (settings().checkExtension(ClKhrFp16)) {
    info_.halfFPConfig_ = info_.singleFPConfig_;
  }

  info_.globalMemCacheLineSize_ = settings().cacheLineSize_;
  info_.globalMemCacheSize_ = settings().cacheSize_;
  if ((settings().cacheLineSize_ != 0) || (settings().cacheSize_ != 0)) {
    info_.globalMemCacheType_ = CL_READ_WRITE_CACHE;
  } else {
    info_.globalMemCacheType_ = CL_NONE;
  }

  uint64_t localRAM;
  if (GPU_ADD_HBCC_SIZE) {
    localRAM = heaps[Pal::GpuHeapLocal].logicalSize + heaps[Pal::GpuHeapInvisible].logicalSize;
  } else {
    localRAM = heaps[Pal::GpuHeapLocal].physicalSize + heaps[Pal::GpuHeapInvisible].physicalSize;
  }

  info_.globalMemSize_ = (static_cast<uint64_t>(std::min(GPU_MAX_HEAP_SIZE, 100u)) *
                          static_cast<uint64_t>(localRAM) / 100u);

  uint uswcPercentAvailable =
      ((static_cast<uint64_t>(heaps[Pal::GpuHeapGartUswc].logicalSize) / Mi) > 1536 && IS_WINDOWS)
          ? 75
          : 50;
  if (settings().apuSystem_) {
    info_.globalMemSize_ +=
        (static_cast<uint64_t>(heaps[Pal::GpuHeapGartUswc].logicalSize) * uswcPercentAvailable) /
        100;
  }

  // Find the largest heap form FB memory
  if (GPU_ADD_HBCC_SIZE) {
    info_.maxMemAllocSize_ = std::max(uint64_t(heaps[Pal::GpuHeapLocal].logicalSize),
                                      uint64_t(heaps[Pal::GpuHeapInvisible].logicalSize));
  } else {
    info_.maxMemAllocSize_ = std::max(uint64_t(heaps[Pal::GpuHeapLocal].physicalSize),
                                      uint64_t(heaps[Pal::GpuHeapInvisible].physicalSize));
  }

#if IS_WINDOWS
  if (settings().apuSystem_) {
    info_.maxMemAllocSize_ = std::max(
        (static_cast<uint64_t>(heaps[Pal::GpuHeapGartUswc].logicalSize) * uswcPercentAvailable) /
            100,
        info_.maxMemAllocSize_);
  }
#endif
  info_.maxMemAllocSize_ =
      uint64_t(info_.maxMemAllocSize_ * std::min(GPU_SINGLE_ALLOC_PERCENT, 100u) / 100u);

  //! \note Force max single allocation size.
  //! 4GB limit for the blit kernels and 64 bit optimizations.
  info_.maxMemAllocSize_ =
      std::min(info_.maxMemAllocSize_, static_cast<uint64_t>(settings().maxAllocSize_));

  if (info_.maxMemAllocSize_ < uint64_t(128 * Mi)) {
    LogError(
        "We are unable to get a heap large enough to support the OpenCL minimum "
        "requirement for FULL_PROFILE");
  }

  info_.maxMemAllocSize_ = std::max(uint64_t(128 * Mi), info_.maxMemAllocSize_);

  // Clamp max single alloc size to the globalMemSize since it's
  // reduced by default
  info_.maxMemAllocSize_ = std::min(info_.maxMemAllocSize_, info_.globalMemSize_);

  // Maximum system memory allocation size allowed
  info_.maxPhysicalMemAllocSize_ = amd::Os::getPhysicalMemSize();

  // We need to verify that we are not reporting more global memory
  // that 4x single alloc
  info_.globalMemSize_ = std::min(4 * info_.maxMemAllocSize_, info_.globalMemSize_);

  // Use 64 bit pointers
  if (settings().use64BitPtr_) {
    info_.addressBits_ = 64;
  } else {
    info_.addressBits_ = (settings().useLightning_) ? 64 : 32;
    // Limit total size with 3GB for 32 bit
    info_.globalMemSize_ = std::min(info_.globalMemSize_, uint64_t(3 * Gi));
  }

  // Alignment in BITS of the base address of any allocated memory object
  static const size_t MemBaseAlignment = 256;
  //! @note Force 256 bytes alignment, since currently
  //! calAttr.surface_alignment returns 4KB. For pinned memory runtime
  //! should be able to create a view with 256 bytes alignement
  info_.memBaseAddrAlign_ = 8 * MemBaseAlignment;

  info_.preferredConstantBufferSize_ = 16 * Ki;
  info_.maxConstantBufferSize_ = info_.maxMemAllocSize_;
  info_.maxConstantArgs_ = MaxConstArguments;

  // Image support fields
  if (settings().imageSupport_) {
    info_.imageSupport_ = true;
    info_.maxSamplers_ = MaxSamplers;
    info_.maxReadImageArgs_ = MaxReadImage;
    info_.maxWriteImageArgs_ = MaxWriteImage;
    info_.image2DMaxWidth_ = maxTextureSize;
    info_.image2DMaxHeight_ = maxTextureSize;
    info_.image3DMaxWidth_ = std::min(2 * Ki, maxTextureSize);
    info_.image3DMaxHeight_ = std::min(2 * Ki, maxTextureSize);
    info_.image3DMaxDepth_ = std::min(2 * Ki, maxTextureSize);

    info_.imagePitchAlignment_ = 256;        // PAL uses LINEAR_ALIGNED
    info_.imageBaseAddressAlignment_ = 256;  // XXX: 256 byte base address alignment for now

    info_.bufferFromImageSupport_ = true;
  }

  info_.errorCorrectionSupport_ = false;

  if (settings().apuSystem_) {
    info_.hostUnifiedMemory_ = 1;
  }

  info_.iommuv2_ = palProp.gpuMemoryProperties.flags.iommuv2Support;

  info_.profilingTimerResolution_ = 1;
  info_.profilingTimerOffset_ = amd::Os::offsetToEpochNanos();
  info_.littleEndian_ = true;
  info_.available_ = true;
  info_.compilerAvailable_ = true;
  info_.linkerAvailable_ = true;

  info_.executionCapabilities_ = CL_EXEC_KERNEL;
  info_.preferredPlatformAtomicAlignment_ = 0;
  info_.preferredGlobalAtomicAlignment_ = 0;
  info_.preferredLocalAtomicAlignment_ = 0;
  info_.queueProperties_ = CL_QUEUE_PROFILING_ENABLE;

  info_.platform_ = AMD_PLATFORM;

  ::strncpy(info_.name_, settings().useLightning_ ? isa().targetId() : palName_,
            sizeof(info_.name_));
  ::strncpy(info_.vendor_, "Advanced Micro Devices, Inc.", sizeof(info_.vendor_) - 1);
  ::snprintf(info_.driverVersion_, sizeof(info_.driverVersion_) - 1, AMD_BUILD_STRING " (PAL%s)%s",
             settings().useLightning_ ? ",LC" : ",HSAIL", isOnline() ? "" : " [Offline]");

  info_.profile_ = "FULL_PROFILE";
  info_.spirVersions_ = "";
  if (settings().oclVersion_ >= OpenCL20) {
    info_.version_ = "OpenCL 2.0 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 2.0 ";
  } else if (settings().oclVersion_ == OpenCL12) {
    info_.version_ = "OpenCL 1.2 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 1.2 ";
  } else {
    info_.version_ = "OpenCL 1.0 " AMD_PLATFORM_INFO;
    info_.oclcVersion_ = "OpenCL C 1.0 ";
    LogError("Unknown version for support");
  }

  // Fill workgroup info size
  info_.maxWorkGroupSize_ = settings().maxWorkGroupSize_;
  info_.maxWorkItemSizes_[0] = info_.maxWorkGroupSize_;
  info_.maxWorkItemSizes_[1] = info_.maxWorkGroupSize_;
  info_.maxWorkItemSizes_[2] = info_.maxWorkGroupSize_;
  info_.preferredWorkGroupSize_ = settings().preferredWorkGroupSize_;

  info_.localMemType_ = CL_LOCAL;
  info_.localMemSize_ = settings().hwLDSSize_;
  info_.extensions_ = getExtensionString();

  // OpenCL1.2 device info fields
  info_.builtInKernels_ = "";
  // Clamp max image buffer size to the maximum buffer size we can create.
  // Image format has max 4 channels per pixel, 1 DWORD per channel.
  constexpr size_t kPixelRgbaSize = 4 * sizeof(int);
  info_.imageMaxBufferSize_ =
      std::min<size_t>(MaxImageBufferSize, info_.maxMemAllocSize_ / kPixelRgbaSize);
  info_.image1DMaxWidth_ = maxTextureSize;
  info_.imageMaxArraySize_ = MaxImageArraySize;
  info_.image2DAMaxWidth_[0] = MaxImageArraySize;
  info_.image2DAMaxWidth_[1] = MaxImageArraySize;
  info_.image1DAMaxWidth_ = maxTextureSize;
  info_.preferredInteropUserSync_ = true;
  info_.printfBufferSize_ = PrintfDbg::WorkitemDebugSize * info().maxWorkGroupSize_;

  if (settings().oclVersion_ >= OpenCL20) {
    info_.svmCapabilities_ = (CL_DEVICE_SVM_COARSE_GRAIN_BUFFER | CL_DEVICE_SVM_FINE_GRAIN_BUFFER);
    if (settings().svmAtomics_) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
    }
    if (settings().svmFineGrainSystem_) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_FINE_GRAIN_SYSTEM;
    }
    if (amd::IS_HIP) {
      info_.svmCapabilities_ |= CL_DEVICE_SVM_ATOMICS;
    }

    info_.pcie_atomics_ = palProp.pciProperties.flags.atomicOpsSupported ? true : false;

    // OpenCL2.0 device info fields
    info_.maxWriteImageArgs_ = MaxReadWriteImage;  //!< For compatibility
    info_.maxReadWriteImageArgs_ = MaxReadWriteImage;

    info_.maxPipePacketSize_ = info_.maxMemAllocSize_;
    info_.maxPipeActiveReservations_ = 16;
    info_.maxPipeArgs_ = 16;

    info_.queueOnDeviceProperties_ =
        CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE;
    info_.queueOnDevicePreferredSize_ = 256 * Ki;
    info_.queueOnDeviceMaxSize_ = 8 * Mi;
    info_.maxOnDeviceQueues_ = 1;
    info_.maxOnDeviceEvents_ = settings().numDeviceEvents_;
    info_.globalVariablePreferredTotalSize_ = static_cast<size_t>(info_.globalMemSize_);
    //! \todo Remove % calculation.
    //! Use 90% of max single alloc size.
    //! Boards with max single alloc size around 4GB will fail allocations
    info_.maxGlobalVariableSize_ =
        static_cast<size_t>(amd::alignDown(info_.maxMemAllocSize_ * 9 / 10, 256));
  }

  if (settings().checkExtension(ClAmdDeviceAttributeQuery)) {
    ::strncpy(info_.boardName_, palProp.gpuName,
              ::strnlen(palProp.gpuName, sizeof(info_.boardName_) - 1));

    info_.deviceTopology_.pcie.type = CL_DEVICE_TOPOLOGY_TYPE_PCIE_AMD;
    info_.deviceTopology_.pcie.bus = palProp.pciProperties.busNumber;
    info_.deviceTopology_.pcie.device = palProp.pciProperties.deviceNumber;
    info_.deviceTopology_.pcie.function = palProp.pciProperties.functionNumber;

    info_.simdPerCU_ = settings().enableWgpMode_
                           ? (2 * palProp.gfxipProperties.shaderCore.numSimdsPerCu)
                           : palProp.gfxipProperties.shaderCore.numSimdsPerCu;
    info_.cuPerShaderArray_ = palProp.gfxipProperties.shaderCore.numCusPerShaderArray;
    info_.simdWidth_ = isa().simdWidth();
    info_.simdInstructionWidth_ = 1;
    info_.wavefrontWidth_ =
        settings().enableWave32Mode_ ? 32 : palProp.gfxipProperties.shaderCore.nativeWavefrontSize;
    info_.availableSGPRs_ = palProp.gfxipProperties.shaderCore.numAvailableSgprs;

    info_.globalMemChannelBanks_ = 4;
    info_.globalMemChannelBankWidth_ = isa().memChannelBankWidth();
    info_.localMemSizePerCU_ = palProp.gfxipProperties.shaderCore.ldsSizePerCu;
    info_.localMemBanks_ = isa().localMemBanks();

    info_.timeStampFrequency_ = 1000000;
    info_.numAsyncQueues_ = numComputeRings;

    info_.numRTQueues_ = numExclusiveComputeRings;

    const auto& engineProp = palProp.engineProperties[Pal::EngineTypeCompute];

    if (settings().enableWgpMode_) {
      info_.numRTCUs_ = engineProp.maxNumDedicatedCu / 2;
      info_.granularityRTCUs_ = engineProp.dedicatedCuGranularity / 2;
    } else {
      info_.numRTCUs_ = engineProp.maxNumDedicatedCu;
      info_.granularityRTCUs_ = engineProp.dedicatedCuGranularity;
    }

    info_.threadTraceEnable_ = settings().threadTraceEnable_;

    info_.pcieDeviceId_ = palProp.deviceId;
    info_.pcieRevisionId_ = palProp.revisionId;
    info_.maxThreadsPerCU_ = info_.wavefrontWidth_ * info_.simdPerCU_ *
                             palProp.gfxipProperties.shaderCore.numWavefrontsPerSimd;

    info_.cooperativeGroups_ = settings().enableCoopGroups_;
    info_.cooperativeMultiDeviceGroups_ = settings().enableCoopMultiDeviceGroups_;
    // Enable StreamWrite and StreamWait for all devices
    info_.aqlBarrierValue_ = true;

#if defined(_WIN64)
    if (amd::IS_HIP) {
      info_.largeBar_ = false;
    } else if (heaps[Pal::GpuHeapInvisible].logicalSize == 0) {
      info_.largeBar_ = true;
      ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Resizable bar enabled");
    }
#else   // !_WIN64
    info_.largeBar_ = false;
#endif  // _WIN64
  }
  info_.virtualMemoryManagement_ = true;
  info_.virtualMemAllocGranularity_ =
      static_cast<size_t>(palProp.gpuMemoryProperties.virtualMemAllocGranularity);
  info_.vgprAllocGranularity_ = palProp.gfxipProperties.shaderCore.vgprAllocGranularity;
  info_.vgprsPerSimd_ = palProp.gfxipProperties.shaderCore.vgprsPerSimd;
  info_.availableVGPRs_ = palProp.gfxipProperties.shaderCore.numAvailableVgprs;
  info_.sgprsPerSimd_ = palProp.gfxipProperties.shaderCore.sgprsPerSimd;
  info_.availableRegistersPerCU_ = info_.vgprsPerSimd_ * info_.simdPerCU_ * info_.wavefrontWidth_;
#if IS_WINDOWS
  info_.luidLowPart_ = palProp.osProperties.luidLowPart;
  info_.luidHighPart_ = palProp.osProperties.luidHighPart;
#endif
  // Setup the node mask for MGPU only case from the original PAL list of all devices
  if ((gNumDevices > 1) && (pal_device != nullptr)) {
    for (uint32_t i = 0; i < gNumDevices; ++i) {
      if (gDeviceList[i] == pal_device) {
        info_.luidDeviceNodeMask_ = 1 << i;
      }
    }
  }
}

Device::XferBuffers::~XferBuffers() {
  // Destroy temporary buffer for reads
  for (const auto& buf : freeBuffers_) {
    // CPU optimization: unmap staging buffer just once
    if (!buf->desc().cardMemory_) {
      buf->unmap(nullptr);
    }
    delete buf;
  }
  freeBuffers_.clear();
}

bool Device::XferBuffers::create() {
  bool result = false;
  // Create a buffer object
  Memory* xferBuf = new Memory(dev(), bufSize_);

  // Try to allocate memory for the transfer buffer
  if ((nullptr == xferBuf) || !xferBuf->create(type_)) {
    delete xferBuf;
    xferBuf = nullptr;
    LogError("Couldn't allocate a transfer buffer!");
  } else {
    result = true;
    freeBuffers_.push_back(xferBuf);
    // CPU optimization: map staging buffer just once
    if (!xferBuf->desc().cardMemory_) {
      xferBuf->map(nullptr);
    }
  }

  return result;
}

Memory& Device::XferBuffers::acquire() {
  Memory* xferBuf = nullptr;
  size_t listSize;

  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  listSize = freeBuffers_.size();

  // If the list is empty, then attempt to allocate a staged buffer
  if (listSize == 0) {
    // Allocate memory
    xferBuf = new Memory(dev(), bufSize_);

    // Allocate memory for the transfer buffer
    if ((nullptr == xferBuf) || !xferBuf->create(type_)) {
      delete xferBuf;
      xferBuf = nullptr;
      LogError("Couldn't allocate a transfer buffer!");
    } else {
      ++acquiredCnt_;
      // CPU optimization: map staging buffer just once
      if (!xferBuf->desc().cardMemory_) {
        xferBuf->map(nullptr);
      }
    }
  }

  if (xferBuf == nullptr) {
    xferBuf = *(freeBuffers_.begin());
    freeBuffers_.erase(freeBuffers_.begin());
    ++acquiredCnt_;
  }

  return *xferBuf;
}

void Device::XferBuffers::release(VirtualGPU& gpu, Memory& buffer) {
  // Make sure buffer isn't busy on the current VirtualGPU, because
  // the next aquire can come from different queue
  buffer.wait(gpu);
  // Lock the operations with the staged buffer list
  amd::ScopedLock l(lock_);
  freeBuffers_.push_back(&buffer);
  --acquiredCnt_;
}


Device::ScopedLockVgpus::ScopedLockVgpus(const Device& dev) : dev_(dev) {
  // Lock the virtual GPU list
  dev_.vgpusAccess().lock();

  // Find all available virtual GPUs and lock them
  // from the execution of commands
  for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
    dev_.vgpus()[idx]->execution().lock();
  }
}

Device::ScopedLockVgpus::~ScopedLockVgpus() {
  // Find all available virtual GPUs and unlock them
  // for the execution of commands
  for (uint idx = 0; idx < dev_.vgpus().size(); ++idx) {
    dev_.vgpus()[idx]->execution().unlock();
  }

  // Unock the virtual GPU list
  dev_.vgpusAccess().unlock();
}

Device::Device()
    : NullDevice(),
      numOfVgpus_(0),
      lockAsyncOps_(true),    /* Device Async Ops Lock */
      lockForInitHeap_(true), /* Initialization of Heap Resource */
      lockPAL_(true),         /* PAL Ops Lock */
      vgpusAccess_(true),     /* Virtual GPU List Ops Lock */
      scratchAlloc_(true),    /* Scratch Allocation Lock */
      mapCacheOps_(true),     /* Map Cache Lock */
      lockResourceOps_(true), /* Resource List Ops Lock */
      xferRead_(nullptr),
      mapCache_(nullptr),
      resourceCache_(nullptr),
      numDmaEngines_(0),
      heapInitComplete_(false),
      xferQueue_(nullptr),
      globalScratchBuf_(nullptr),
      srdManager_(nullptr),
      resourceList_(nullptr),
      captureMgr_(nullptr) {}

Device::~Device() {
  if (p2p_stage_ != nullptr) {
    p2p_stage_->release();
    p2p_stage_ = nullptr;
  }

  if (glb_ctx_ != nullptr) {
    glb_ctx_->release();
    glb_ctx_ = nullptr;
  }

  delete srdManager_;

  for (uint s = 0; s < scratch_.size(); ++s) {
    delete scratch_[s];
    scratch_[s] = nullptr;
  }

  delete globalScratchBuf_;
  globalScratchBuf_ = nullptr;

  // Release all queues if the app didn't release them
  while (vgpus().size() > 1) {
    delete vgpus()[1];
  }

  // Destroy transfer queue
  delete xferQueue_;

  if (trap_handler_ != nullptr) {
    trap_handler_->release();
  }

  // Destroy blit program
  delete blitProgram_;

  // Release cached map targets
  for (uint i = 0; mapCache_ != nullptr && i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] != nullptr) {
      (*mapCache_)[i]->release();
    }
  }
  delete mapCache_;

  // Destroy temporary buffers for read/write
  delete xferRead_;

  // Destroy resource cache
  delete resourceCache_;

  delete resourceList_;

  if (context_ != nullptr) {
    context_->release();
  }

  device_ = nullptr;

  // Delete developer driver manager
  delete captureMgr_;
}

extern const char* SchedulerSourceCode;
extern const char* SchedulerSourceCode20;

constexpr int TrapHandlerABIVersion = 10;
extern const char* TrapHandlerCode;

// ================================================================================================
bool Device::create(Pal::IDevice* device) {
  resourceList_ = new std::unordered_set<Resource*>();
  if (nullptr == resourceList_) {
    return false;
  }
  appProfile_.init();
  device_ = device;

  // Retrive device properties
  Pal::Result result = iDev()->GetProperties(&properties_);
  if (result != Pal::Result::Success) {
    return false;
  }

  // Save the IP level for the offline detection
  ipLevel_ = properties().gfxLevel;
  asicRevision_ = flagIsDefault(PAL_FORCE_ASIC_REVISION)
                      ? properties().revision
                      : static_cast<Pal::AsicRevision>(PAL_FORCE_ASIC_REVISION);

  // XNACK flag should be set for PageMigration or IOMMUv2 support.
  bool isXNACKEnabled =
      (static_cast<uint>(properties().gpuMemoryProperties.flags.pageMigrationEnabled ||
                         properties().gpuMemoryProperties.flags.iommuv2Support));

  // Temporarily disable reporting sramecc support.
  // PAL currently only reports if the device CAN support it,
  // not if it is ENABLED. This will cause us to enable the feature on
  // the HSAIL path, which is not supported.
  bool isSRAMECCEnabled = false;
  const amd::Isa* isa;
  std::tie(isa, palName_) =
      findIsa(properties().gfxTriple.major, properties().gfxTriple.minor,
              properties().gfxTriple.stepping, isSRAMECCEnabled, isXNACKEnabled);
  if (!isa) {
    LogPrintfError("Unsupported PAL device with ASIC revision #%d", asicRevision_);
    return false;
  }
  if (!isa->runtimePalSupported()) {
    LogPrintfError("Unsupported PAL device with ISA %s", isa->targetId());
    return false;
  }

  if (!amd::Device::create(*isa)) {
    LogPrintfError("Unable to setup device for PAL device %s", isa->targetId());
    return false;
  }

  const auto& computeProp = properties().engineProperties[Pal::EngineTypeCompute];
  // Find the number of available engines
  for (uint i = 0; i < computeProp.engineCount; ++i) {
    const auto& computeCaps = computeProp.capabilities[i];
    if ((computeCaps.queuePrioritySupport & Pal::SupportQueuePriorityRealtime) &&
        (computeProp.maxNumDedicatedCu > 0)) {
      if (exclusiveComputeEnginesId_.find(ExclusiveQueueType::RealTime0) !=
          exclusiveComputeEnginesId_.end()) {
        exclusiveComputeEnginesId_.insert({ExclusiveQueueType::RealTime1, i});
      } else {
        exclusiveComputeEnginesId_.insert({ExclusiveQueueType::RealTime0, i});
      }
    }
    if (computeCaps.queuePrioritySupport & Pal::SupportQueuePriorityMedium) {
      exclusiveComputeEnginesId_.insert({ExclusiveQueueType::Medium, i});
    }

    if ((computeCaps.queuePrioritySupport & Pal::SupportQueuePriorityNormal) ||
        // In Linux all queues have 0 for priority
        (computeCaps.queuePrioritySupport == 0)) {
      computeEnginesId_.push_back(i);
    }
  }
  numDmaEngines_ = properties().engineProperties[Pal::EngineTypeDma].engineCount;

  // Creates device settings
  settings_ = new pal::Settings();
  Pal::PalPublicSettings* const palSettings = iDev()->GetPublicSettings();
  // Modify settings here
  // palSettings ...
  palSettings->forceHighClocks = appProfile_.enableHighPerformanceState();
  palSettings->longRunningSubmissions = true;
  palSettings->cmdBufBatchedSubmitChainLimit = 0;
  palSettings->disableResourceProcessingManager = true;
  // Make sure CP DMA can be used for all possible transfers
  palSettings->cpDmaCmdCopyMemoryMaxBytes = 0xFFFFFFFF;

  // Create RGP / UberTrace capture manager
  // Note: RGP initialization in PAL must be performed before CommitSettingsAndInit()
#if PAL_BUILD_RDF
  if ((platform_->GetTraceSession() != nullptr) &&
      (platform_->GetTraceSession()->IsTracingEnabled())) {
    captureMgr_ = UberTraceCaptureMgr::Create(platform_, *this);
  } else
#endif
  {
    captureMgr_ = RgpCaptureMgr::Create(platform_, *this);
  }

  if (nullptr != captureMgr_) {
    // KMD forced DWORD alignment for debug VMID, request it back to Unaligned
    palSettings->hardwareBufferAlignmentMode = Pal::BufferAlignmentMode::Unaligned;
    Pal::IPlatform::InstallDeveloperCb(iPlat(), &Device::PalDeveloperCallback, this);
  }

  // Commit the new settings for the device
  result = iDev()->CommitSettingsAndInit();
  if (result != Pal::Result::Success) {
    return false;
  }

  iDev()->GetGpuMemoryHeapProperties(heaps_);

  Pal::WorkStationCaps wscaps = {};
  iDev()->QueryWorkStationCaps(&wscaps);

  pal::Settings* gpuSettings = reinterpret_cast<pal::Settings*>(settings_);
  if (!gpuSettings ||
      !gpuSettings->create(properties(), heaps_, wscaps, *isa,
                           appProfile_.reportAsOCL12Device())) {
    return false;
  }

  // Fill the device info structure
  fillDeviceInfo(properties(), heaps_, 16 * Ki, numComputeEngines(), numExclusiveComputeEngines(),
                 iDev());

  // Reserve percentage memory for large frame buffer.
  // Reserve a threshold size for small frame buffer, used by page table for remote memory mapping
  Pal::gpusize invisibleSize = heaps_[Pal::GpuHeapInvisible].logicalSize;
  Pal::gpusize visibleSize = heaps_[Pal::GpuHeapLocal].logicalSize;

  Pal::gpusize maxInvisibleAllocation =
      std::min((invisibleSize / 100) * 98, invisibleSize < 128 * Mi ? 0 : invisibleSize - 128 * Mi);
  Pal::gpusize maxVisibleAllocation =
      std::min((visibleSize / 100) * 98, visibleSize < 128 * Mi ? 0 : visibleSize - 128 * Mi);

  if (invisibleSize < visibleSize && invisibleSize > 0) {
    // Page table is in invisible and its size is smaller. Invisible is the only deciding factor
    maxFrameBufferAllocation_ = maxInvisibleAllocation;
  } else {
    maxFrameBufferAllocation_ = std::max(maxInvisibleAllocation, maxVisibleAllocation);
  }

  if (!ValidateComgr()) {
    LogError("Code object manager initialization failed!");
    return false;
  }

  if (!ValidateHsail()) {
    LogError("Hsail initialization failed!");
    return false;
  }

  computeEnginesId_.resize(std::min(numComputeEngines(), settings().numComputeRings_));

  amd::Context::Info info = {0};
  std::vector<amd::Device*> devices;
  devices.push_back(this);

  // Create a dummy context
  context_ = new amd::Context(devices, info);
  if (context_ == nullptr) {
    return false;
  }

  mapCache_ = new std::vector<amd::Memory*>();
  if (mapCache_ == nullptr) {
    return false;
  }
  // Use just 1 entry by default for the map cache
  mapCache_->push_back(nullptr);

  size_t resourceCacheSize = settings().resourceCacheSize_;
  // Create resource cache.
  // \note Cache must be created before any resource creation to avoid nullptr check
  resourceCache_ = new ResourceCache(this, resourceCacheSize);
  if (nullptr == resourceCache_) {
    return false;
  }

#ifdef DEBUG
  std::stringstream message;
  message << info_.name_;
  if (settings().remoteAlloc_) {
    message << ": Using *Remote* memory";
  } else {
    message << ": Using *Local* memory";
  }

  message << std::endl;
  ClPrint(amd::LOG_INFO, amd::LOG_INIT, message.str().c_str());
#endif  // DEBUG

  for (uint i = 0; i < Pal::GpuHeap::GpuHeapCount; ++i) {
    allocedMem[i] = 0;
  }

  if (!settings().useLightning_) {
#if defined(WITH_COMPILER_LIB)
    const char* library = getenv("HSA_COMPILER_LIBRARY");
    aclCompilerOptions opts = {sizeof(aclCompilerOptions_0_8),
                               library,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr,
                               nullptr};
    // Initialize the compiler handle
    acl_error error;
    compiler_ = amd::Hsail::CompilerInit(&opts, &error);
    if (error != ACL_SUCCESS) {
      LogError("Error initializing the compiler");
      return false;
    }
#endif  // defined(WITH_COMPILER_LIB)
  }

  // Allocate SRD manager
  srdManager_ = new SrdManager(*this, std::max(HsaImageObjectSize, HsaSamplerObjectSize), 64 * Ki);
  if (srdManager_ == nullptr) {
    return false;
  }

  return true;
}

// ================================================================================================
// Master function that handles developer callbacks from PAL.
void PAL_STDCALL Device::PalDeveloperCallback(void* pPrivateData, const Pal::uint32 deviceIndex,
                                              Pal::Developer::CallbackType type, void* pCbData) {
#ifdef PAL_GPUOPEN_OCL
  VirtualGPU* gpu = nullptr;
  Device* device = static_cast<Device*>(pPrivateData);
  const auto& barrier = *static_cast<const Pal::Developer::BarrierData*>(pCbData);

  if ((type == Pal::Developer::CallbackType::BarrierBegin) ||
      (type == Pal::Developer::CallbackType::BarrierEnd)) {
    const auto* pBarrierData = reinterpret_cast<const Pal::Developer::BarrierData*>(pCbData);

    if (pBarrierData->pCmdBuffer != nullptr) {
      // Find which queue the current command buffer belongs
      for (const auto& it : device->vgpus()) {
        if (it->isActiveCmd(pBarrierData->pCmdBuffer)) {
          gpu = it;
          break;
        }
      }
    }
  }

  if (gpu == nullptr) {
    return;
  }

  switch (type) {
    case Pal::Developer::CallbackType::BarrierBegin:
      device->captureMgr()->WriteBarrierStartMarker(gpu, barrier);
      break;
    case Pal::Developer::CallbackType::BarrierEnd:
      device->captureMgr()->WriteBarrierEndMarker(gpu, barrier);
      break;
    case Pal::Developer::CallbackType::ImageBarrier:
      assert(false);
      break;
    case Pal::Developer::CallbackType::DrawDispatch:
      break;
    default:
      break;
  }
#endif  // PAL_GPUOPEN_OCL
}

// ================================================================================================
bool Device::initializeHeapResources() {
  amd::ScopedLock k(lockForInitHeap_);
  if (!heapInitComplete_) {
    Pal::DeviceFinalizeInfo finalizeInfo = {};

    // Request all compute engines
    for (const auto& it : computeEnginesId_) {
      // Request real time compute engines
      finalizeInfo.requestedEngineCounts[Pal::EngineTypeCompute].engines |= (1 << it);
    }

    for (const auto& it : exclusiveComputeEnginesId_) {
      // Request real time compute engines
      finalizeInfo.requestedEngineCounts[Pal::EngineTypeCompute].engines |= (1 << it.second);
    }
    // Request all SDMA engines
    finalizeInfo.requestedEngineCounts[Pal::EngineTypeDma].engines = (1 << numDmaEngines_) - 1;

    if (iDev()->Finalize(finalizeInfo) != Pal::Result::Success) {
      return false;
    }

    // Override the _amdgpu_r_debug.r_version field with the trap handler
    // version.  We can't import the definition of r_debug here as it would
    // conflict with ELF related definitions pulled-in by platform/program.hpp.
    // The layout of r_debug is a stable ABI, so we are guaranteed the
    // r_version field will always be at offset 0.
    *reinterpret_cast<int*>(_amdgpu_r_debug_ptr) = TrapHandlerABIVersion;

    Pal::HipRuntimeSetup setup{.pRdebug = _amdgpu_r_debug_ptr,
                               .runtimeState = 1,  // Always valid debug state
                               .ttmpSetupHint = GPU_DEBUG_ENABLE};
    setup.pRdebug = _amdgpu_r_debug_ptr;
    if (iDev()->RegisterHipRuntimeState(setup) != Pal::Result::Success) {
      LogError("Couldn't register debug state from the loader!");
      // Note: ignore debug state error, since it's not a critical
      // error for the execution
    }

    heapInitComplete_ = true;

    scratch_.resize(GPU_MAX_HW_QUEUES + numExclusiveComputeEngines());

    // Initialize the number of mem object for the scratch buffer
    for (uint s = 0; s < scratch_.size(); ++s) {
      scratch_[s] = new ScratchBuffer();
      if (nullptr == scratch_[s]) {
        return false;
      }
    }

    if (settings().stagedXferSize_ != 0) {
      // Initialize staged read buffers
      if (settings().stagedXferRead_) {
        xferRead_ = new XferBuffers(*this, Resource::Remote,
                                    amd::alignUp(settings().stagedXferSize_, 4 * Ki));
        if ((xferRead_ == nullptr) || !xferRead_->create()) {
          LogError("Couldn't allocate transfer buffer objects for write");
          return false;
        }
      }
    }

    // Update RGP capture manager
    if (captureMgr_ != nullptr) {
      if (!captureMgr_->Update(platform_)) {
        delete captureMgr_;
        captureMgr_ = nullptr;
      }
    }

    // Create a synchronized transfer queue
    xferQueue_ = new VirtualGPU(*this);
    if (!(xferQueue_ && xferQueue_->create(false))) {
      delete xferQueue_;
      xferQueue_ = nullptr;
    }
    if (nullptr == xferQueue_) {
      LogError("Couldn't create the device transfer manager!");
      return false;
    }
    xferQueue_->enableSyncedBlit();
    // Setup trap handler if available
    if (trap_handler_ != nullptr) {
      auto program =
          reinterpret_cast<pal::LightningProgram*>(trap_handler_->getDeviceProgram(*this));
      if (program != nullptr) {
        Pal::Result result{Pal::Result::Success};
        Pal::GpuMemoryRef memRef = {};
        memRef.pGpuMemory = program->codeSegGpu().iMem();
        if (!settings().alwaysResident_) {
          // Make sure trap handler is always resident in memory
          // note: this code path is for OpenCL only, since HIP has alwaysResident_ enabled
          result = iDev()->AddGpuMemoryReferences(1, &memRef, nullptr, Pal::GpuMemoryRefCantTrim);
        }
        if (result == Pal::Result::Success) {
          // Find an offset in memory for the trap handler.
          // Loader returns an absolute address, but PAL accepts base + offset, hense find offset
          auto offset = program->GetTrapHandlerAddress() - memRef.pGpuMemory->Desc().gpuVirtAddr;
          // Bind the trap handler's executable to the kernel mode driver
          result = iDev()->SetHipTrapHandler(memRef.pGpuMemory, offset, nullptr, 0);
          if (result != Pal::Result::Success) {
            LogError("KMD failed to setup the trap handler");
          }
        } else {
          LogError("Failed to make trap handler resident in memory");
        }
      }
    }
  }
  return true;
}

// ================================================================================================
device::VirtualDevice* Device::createVirtualDevice(amd::CommandQueue* queue) {
  bool profiling = false;
  uint rtCUs = amd::CommandQueue::RealTimeDisabled;
  uint deviceQueueSize = 0;

  if (queue != nullptr) {
    profiling = queue->properties().test(CL_QUEUE_PROFILING_ENABLE);
    if (queue->asHostQueue() != nullptr) {
      bool interopQueue = (0 != (queue->context().info().flags_ &
                                 (amd::Context::GLDeviceKhr | amd::Context::D3D10DeviceKhr |
                                  amd::Context::D3D11DeviceKhr)));
      rtCUs = queue->rtCUs();
    } else if (queue->asDeviceQueue() != nullptr) {
      deviceQueueSize = queue->asDeviceQueue()->size();
    }
  }

  // Not safe to add a queue. So lock the device
  amd::ScopedLock k(lockAsyncOps());
  amd::ScopedLock lock(vgpusAccess());

  // Initialization of heap and other resources occur during the command queue creation time.
  if (!initializeHeapResources()) {
    LogError("Heap initializaiton fails!");
    return nullptr;
  }

  VirtualGPU* vgpu = new VirtualGPU(*this);
  if (vgpu && vgpu->create(profiling, deviceQueueSize, rtCUs, queue->priority())) {
    return vgpu;
  } else {
    delete vgpu;
    return nullptr;
  }
}

device::Program* Device::createProgram(amd::Program& owner, amd::option::Options* options) {
  device::Program* program;
  if (settings().useLightning_) {
    program = new LightningProgram(*this, owner);
  } else {
    program = new HSAILProgram(*this, owner);
  }
  if (program == nullptr) {
    LogError("We failed memory allocation for program!");
  }

  return program;
}

//! Requested devices list as configured by the GPU_DEVICE_ORDINAL
typedef std::unordered_map<int, bool> requestedDevices_t;

//! Parses the requested list of devices to be exposed to the user.
static void parseRequestedDeviceList(const char* requestedDeviceList,
                                     requestedDevices_t& requestedDevices, uint32_t numDevices,
                                     Pal::IDevice* deviceList[Pal::MaxDevices]) {
  char* pch = strtok(const_cast<char*>(requestedDeviceList), ",");
  while (pch != nullptr) {
    bool deviceIdValid = true;
    // UUID needs to be specified in the format GPU-<body>, <body> encodes UUID as a 16 chars
    char* deviceUuid = strstr(pch, "GPU-");
    // If Uuid is specified, then convert it to index
    if (deviceUuid != nullptr) {
      for (uint32_t i = 0; i < numDevices; i++) {
        Pal::DeviceProperties properties;
        // Retrieve device properties
        Pal::Result result = deviceList[i]->GetProperties(&properties);
        if (result != Pal::Result::Success) {
          continue;
        }

        // Retrieve uuid
        char uuid[17] = {0};
        setUUID(&properties, &uuid[0]);

        // Convert it to index
        if (strcmp(pch + 4, uuid) == 0) {
          snprintf(pch, strlen(pch), "%d", i);
          break;
        }
      }
    }

    int currentDeviceIndex = atoi(pch);
    // Validate device index.
    for (size_t i = 0; i < strlen(pch); i++) {
      if (!isdigit(reinterpret_cast<unsigned char*>(pch)[i])) {
        deviceIdValid = false;
        break;
      }
    }
    if (currentDeviceIndex < 0 || static_cast<uint32_t>(currentDeviceIndex) >= numDevices) {
      deviceIdValid = false;
    }
    // Get next token.
    pch = strtok(nullptr, ",");

    if (!deviceIdValid) {
      // Exit the loop as anything to the right of invalid deviceId
      // has to be discarded
      break;
    }

    // Requested device is valid.
    requestedDevices[currentDeviceIndex] = deviceIdValid;
  }
}

// ================================================================================================
bool Device::init() {
  gStartDevice = amd::Device::numDevices(CL_DEVICE_TYPE_GPU, true);
  bool useDeviceList = false;
  requestedDevices_t requestedDevices;

  size_t size = Pal::GetPlatformSize();
  platformObj_ = new char[size];
  Pal::PlatformCreateInfo info = {};
  info.flags.disableGpuTimeout = true;
#if !defined(PAL_BUILD_DTIF)
#ifdef ATI_BITS_32
  info.flags.force32BitVaSpace = true;
  info.flags.enableSvmMode = false;
#else
  info.flags.enableSvmMode = true;
#endif
#endif
  info.flags.supportRgpTraces = true;
  info.pSettingsPath = "OCL";
  info.maxSvmSize = static_cast<Pal::gpusize>(OCL_SET_SVM_SIZE * Mi);
  info.clientApiId = (amd::IS_HIP) ? Pal::ClientApi::Hip : Pal::ClientApi::OpenCl;
  info.instrApiVer = RgpSqttInstrumentationApiVersion;

  if (IS_LINUX) {
    //! @note: Linux may have a deadlock if runtime will attempt to reserve
    //! VA range, which is much bigger than sysmem size
    size_t maxVirtualReserve = amd::Os::getPhysicalMemSize() << 1;
    if (info.maxSvmSize > maxVirtualReserve) {
      info.maxSvmSize = maxVirtualReserve;
    }
  }
  info.maxSvmSize = amd::nextPowerOfTwo(info.maxSvmSize - 1);

  // PAL init
  if (Pal::Result::Success != Pal::CreatePlatform(info, platformObj_, &platform_)) {
    LogError("Pal::CreatePlatform failed!");
    return false;
  }

#ifdef PAL_GPUOPEN_OCL
  if ((platform_->GetDevDriverServer() != nullptr) &&
      (platform_->GetDevDriverServer()->GetDriverControlServer() != nullptr)) {
    // Make sure the devdriver initialization is done after Pal platform creation
    // to avoid a timeout in RGP server
    platform_->GetDevDriverServer()->GetDriverControlServer()->StartLateDeviceInit();
  }
#endif  // PAL_GPUOPEN_OCL

  // Get the total number of active devices
  // Count up all the devices in the system.
  platform_->EnumerateDevices(&gNumDevices, &gDeviceList[0]);

  const char* requestedDeviceList =
      amd::IS_HIP ? ((HIP_VISIBLE_DEVICES[0] != '\0') ? HIP_VISIBLE_DEVICES : CUDA_VISIBLE_DEVICES)
                  : GPU_DEVICE_ORDINAL;

  if (requestedDeviceList[0] != '\0') {
    useDeviceList = true;
    parseRequestedDeviceList(requestedDeviceList, requestedDevices, gNumDevices, &gDeviceList[0]);
  }

  bool foundDevice = false;

  // Loop through all active devices and initialize the device info structure
  for (uint ordinal = 0; ordinal < gNumDevices; ++ordinal) {
    bool result = true;
    if (useDeviceList) {
      result = (requestedDevices.find(ordinal) != requestedDevices.end());
    }
    // Create the GPU device object
    Device* d = new Device();
    result = result && (nullptr != d) && d->create(gDeviceList[ordinal]);

#ifdef PAL_GPUOPEN_OCL
    if ((platform_->GetDevDriverServer() != nullptr) &&
        (platform_->GetDevDriverServer()->GetDriverControlServer() != nullptr)) {
      // Finalize DevDriver initialization after device creation
      platform_->GetDevDriverServer()->GetDriverControlServer()->FinishDeviceInit();
    }
#endif  // PAL_GPUOPEN_OCL

    if (result) {
      foundDevice = true;
      d->registerDevice();
    } else {
      delete d;
    }
  }
  if (!foundDevice) {
    Device::tearDown();
  } else {
    // Loop through all available devices
    uint32_t all_devices = devices().size();
    for (uint32_t device0 = gStartDevice; device0 < all_devices; ++device0) {
      // Find all device that can have access to the current device
      for (uint32_t device1 = gStartDevice; device1 < all_devices; ++device1) {
        // If it's not the same device, then validate P2P settings
        if ((devices()[device0] != devices()[device1]) &&
            static_cast<Device*>(devices()[device1])->settings().enableHwP2P_) {
          Pal::GpuCompatibilityInfo comp_info = {};
          // Can device 0 have access to device1?
          static_cast<Device*>(devices()[device0])
              ->iDev()
              ->GetMultiGpuCompatibility(*static_cast<Device*>(devices()[device1])->iDev(),
                                         &comp_info);
          // Check P2P capability
          if (comp_info.flags.peerTransferRead && comp_info.flags.peerTransferWrite) {
            devices()[device0]->p2pDevices_.push_back(as_cl(devices()[device1]));
            devices()[device1]->p2p_access_devices_.push_back(devices()[device0]);
          }
        }
      }
    }

    // Query active devices only
    constexpr bool kNoOfflineDevices = false;
    std::vector<amd::Device*> devices = getDevices(CL_DEVICE_TYPE_GPU, kNoOfflineDevices);
    if (devices.size() > 0) {
      // Create a dummy context for internal memory allocations on all reported devices
      glb_ctx_ = new amd::Context(devices, amd::Context::Info());
      if (glb_ctx_ == nullptr) {
        return false;
      }
      // Allocate a staging buffer for P2P emulation path
      if (devices.size() > 1) {
        amd::Buffer* buf =
            new (*glb_ctx_) amd::Buffer(*glb_ctx_, CL_MEM_ALLOC_HOST_PTR, kP2PStagingSize);
        if ((buf != nullptr) && buf->create()) {
          p2p_stage_ = buf;
        } else {
          delete buf;
          return false;
        }
      }
    }
  }

  return true;
}

// ================================================================================================
void Device::tearDown() {
  if (platform_ != nullptr) {
    platform_->Destroy();
    delete platformObj_;
    platform_ = nullptr;
  }
#if defined(WITH_COMPILER_LIB)
  if (compiler_ != nullptr) {
    amd::Hsail::CompilerFini(compiler_);
    compiler_ = nullptr;
  }
#endif  // defined(WITH_COMPILER_LIB)
}

Memory* Device::getGpuMemory(amd::Memory* mem) const {
  return static_cast<pal::Memory*>(mem->getDeviceMemory(*this));
}

const device::BlitManager& Device::xferMgr() const { return xferQueue_->blitMgr(); }

Pal::ChNumFormat Device::getPalFormat(const amd::Image::Format& format,
                                      Pal::ChannelMapping* channel) const {
  // Find PAL format
  for (uint i = 0; i < sizeof(MemoryFormatMap) / sizeof(MemoryFormat); ++i) {
    if ((format.image_channel_data_type == MemoryFormatMap[i].clFormat_.image_channel_data_type) &&
        (format.image_channel_order == MemoryFormatMap[i].clFormat_.image_channel_order)) {
      *channel = MemoryFormatMap[i].palChannel_;
      return MemoryFormatMap[i].palFormat_;
    }
  }
  assert(!"We didn't find PAL resource format!");
  *channel = MemoryFormatMap[0].palChannel_;
  return MemoryFormatMap[0].palFormat_;
}

// Create buffer without an owner (merge common code with createBuffer() ?)
pal::Memory* Device::createScratchBuffer(size_t size) const {
  // Create a memory object
  Memory* gpuMemory = new pal::Memory(*this, size);
  if (nullptr == gpuMemory || !gpuMemory->create(Resource::Local)) {
    delete gpuMemory;
    gpuMemory = nullptr;
  }

  return gpuMemory;
}

pal::Memory* Device::createBuffer(amd::Memory& owner, bool directAccess) const {
  size_t size = owner.getSize();
  pal::Memory* gpuMemory;

  // Create resource
  bool result = false;

  if (owner.getType() == CL_MEM_OBJECT_PIPE) {
    // directAccess isnt needed as Pipes shouldnt be host accessible for GPU
    directAccess = false;
  }

  if (nullptr != owner.parent()) {
    pal::Memory* gpuParent = getGpuMemory(owner.parent());
    if (nullptr == gpuParent) {
      LogError("Can't get the owner object for subbuffer allocation");
      return nullptr;
    }

    if ((nullptr != owner.parent()->getSvmPtr()) &&
        (owner.parent()->getContext().devices().size() > 1)) {
      amd::Memory* amdParent = owner.parent();
      {
        // Lock memory object, so only one commitment will occur
        amd::ScopedLock lock(amdParent->lockMemoryOps());
        amdParent->commitSvmMemory();
        amdParent->setHostMem(amdParent->getSvmPtr());
      }
      // Ignore a possible pinning error. Runtime will fallback to SW emulation
      bool ok = gpuParent->pinSystemMemory(amdParent->getHostMem(), amdParent->getSize());
    }
    return gpuParent->createBufferView(owner);
  }

  Resource::MemoryType type =
      (owner.forceSysMemAlloc() || (owner.getMemFlags() & CL_MEM_SVM_FINE_GRAIN_BUFFER))
          ? Resource::Remote
          : Resource::Local;

  // Check if runtime can force a tiny buffer into USWC memory
  if ((size <= (GPU_MAX_REMOTE_MEM_SIZE * Ki)) && (type == Resource::Local) &&
      (owner.getMemFlags() & CL_MEM_READ_ONLY)) {
    type = Resource::RemoteUSWC;
  }

  if (owner.getMemFlags() & CL_MEM_BUS_ADDRESSABLE_AMD) {
    type = Resource::BusAddressable;
  } else if (owner.getMemFlags() & CL_MEM_EXTERNAL_PHYSICAL_AMD) {
    type = Resource::ExternalPhysical;
  } else if (owner.getMemFlags() & CL_MEM_VA_RANGE_AMD) {
    type = Resource::VaRange;
  }

  // Use direct access if it's possible
  bool remoteAlloc = false;
  // Internal means VirtualDevice!=nullptr
  bool internalAlloc =
      ((owner.getMemFlags() & CL_MEM_USE_HOST_PTR) && (owner.getVirtualDevice() != nullptr))
          ? true
          : false;

  // Create a memory object
  gpuMemory = new pal::Buffer(*this, owner, owner.getSize());
  if (nullptr == gpuMemory) {
    return nullptr;
  }

  // Check if owner is interop memory
  if (owner.isInterop()) {
    result = gpuMemory->createInterop();
  } else if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
    // Attempt to allocate from persistent heap
    result = gpuMemory->create(Resource::Persistent);
    if (result) {
      // Disallow permanent map for Win7 only, since OS will move buffer to sysmem
      if (IS_LINUX ||
          // Or Win10
          (properties().gpuMemoryProperties.flags.supportPerSubmitMemRefs == false)) {
        void* address = gpuMemory->map(nullptr);
        CondLog(address == nullptr, "PAL failed lock of persistent memory!");
      }
    } else {
      delete gpuMemory;
      return nullptr;
    }
  } else if (directAccess || (type == Resource::Remote)) {
    // Check for system memory allocations
    if ((owner.getMemFlags() & (CL_MEM_ALLOC_HOST_PTR | CL_MEM_USE_HOST_PTR)) ||
        (settings().remoteAlloc_)) {
      // Allocate remote memory if AHP allocation and context has just 1 device
      if ((owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR) &&
          (owner.getContext().devices().size() == 1) &&
          (owner.getSize() < static_cast<size_t>(GPU_MAX_USWC_ALLOC_SIZE) * Mi)) {
        if (owner.getMemFlags() &
            (CL_MEM_READ_ONLY | CL_MEM_HOST_WRITE_ONLY | CL_MEM_HOST_NO_ACCESS)) {
          // GPU will be reading from this host memory buffer,
          // so assume Host write into it
          type = Resource::RemoteUSWC;
        } else {
          type = Resource::Remote;
        }
        remoteAlloc = true;
      }
      // Make sure owner has a valid hostmem pointer and it's not COPY
      if (!remoteAlloc && (owner.getHostMem() != nullptr)) {
        Resource::PinnedParams params;
        params.owner_ = &owner;
        params.gpu_ = reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());

        params.hostMemRef_ = owner.getHostMemRef();
        params.size_ = owner.getHostMemRef()->size();
        if (0 == params.size_) {
          params.size_ = owner.getSize();
        }
        // Create memory object
        result = gpuMemory->create(Resource::Pinned, &params);

        // If direct access failed
        if (!result) {
          // Don't use cached allocation if size is biger than max single alloc or it's HIP
          if (amd::IS_HIP || (owner.getSize() > info().maxMemAllocSize_)) {
            delete gpuMemory;
            return nullptr;
          }
        }
      }
    }
  }

  if (!result &&
      // Make sure it's not internal alloc
      !internalAlloc) {
    Resource::CreateParams params;
    params.owner_ = &owner;
    params.gpu_ = static_cast<VirtualGPU*>(owner.getVirtualDevice());
    params.svmBase_ = static_cast<Memory*>(owner.svmBase());
    if (owner.P2PAccess()) {
      params.svmBase_ = static_cast<Memory*>(owner.BaseP2PMemory());
      if (params.svmBase_ != nullptr) {
        type = Resource::P2PAccess;
      }
    }
    params.interprocess_ = (owner.getMemFlags() & ROCCLR_MEM_INTERPROCESS) ? true : false;
    // Disable interprocess for >3GBs local memory due to PAL failure.
    if ((type == Resource::Local || type == Resource::Persistent) && owner.getSize() > 3 * Gi) {
      params.interprocess_ = false;
    }
    if (owner.ipcShared()) {
      type = Resource::IpcMemory;
    }
    params.alignment_ = owner.getAlignment();
    // Create memory object
    result = gpuMemory->create(type, &params);

    // If allocation was successful
    if (result) {
      // Initialize if the memory is a pipe object
      if (owner.getType() == CL_MEM_OBJECT_PIPE) {
        // Pipe initialize in order read_idx, write_idx, end_idx. Refer clk_pipe_t structure.
        // Init with 3 DWORDS for 32bit addressing and 6 DWORDS for 64bit
        size_t pipeInit[3] = {0, 0, owner.asPipe()->getMaxNumPackets()};
        static_cast<const KernelBlitManager&>(xferMgr()).writeRawData(*gpuMemory, sizeof(pipeInit),
                                                                      pipeInit);
      }
      // If memory has direct access from host, then get CPU address
      if (gpuMemory->isHostMemDirectAccess() && (type != Resource::ExternalPhysical) &&
          (type != Resource::P2PAccess)) {
        void* address = gpuMemory->map(nullptr);
        if (address != nullptr) {
          // Copy saved memory
          // Note: UHP is an optional check if pinning failed and sysmem alloc was forced
          if (owner.getMemFlags() & (CL_MEM_COPY_HOST_PTR | CL_MEM_USE_HOST_PTR)) {
            memcpy(address, owner.getHostMem(), owner.getSize());
          }
          // It should be safe to change the host memory pointer,
          // because it's lock protected from the upper caller
          owner.setHostMem(address);
        } else {
          result = false;
        }
      }
      // An optimization for CHP. Copy memory and destroy sysmem allocation
      else if ((gpuMemory->memoryType() != Resource::Pinned) &&
               (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
               (owner.getContext().devices().size() == 1)) {
        amd::Coord3D origin(0, 0, 0);
        amd::Coord3D region(owner.getSize());
        static const bool Entire = true;
        if (xferMgr().writeBuffer(owner.getHostMem(), *gpuMemory, origin, region, Entire)) {
          // Clear CHP memory
          owner.setHostMem(nullptr);
        }
      }
    }
  }

  if (!result) {
    delete gpuMemory;
    return nullptr;
  }

  return gpuMemory;
}

pal::Memory* Device::createImage(amd::Memory& owner, bool directAccess) const {
  amd::Image& image = *owner.asImage();
  pal::Memory* gpuImage = nullptr;

  if ((nullptr != owner.parent()) && (owner.parent()->asImage() != nullptr)) {
    device::Memory* devParent = owner.parent()->getDeviceMemory(*this);
    if (nullptr == devParent) {
      LogError("Can't get the owner object for image view allocation");
      return nullptr;
    }
    // Create a view on the specified device
    gpuImage = (pal::Memory*)createView(owner, *devParent);
    if ((nullptr != gpuImage) && (gpuImage->owner() != nullptr)) {
      gpuImage->owner()->setHostMem((address)(owner.parent()->getHostMem()) +
                                    gpuImage->owner()->getOrigin());
    }
    return gpuImage;
  }

  gpuImage = new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                            image.getImageFormat(), image.getType(), image.getMipLevels());

  // Create resource
  if (nullptr != gpuImage) {
    const bool imageBuffer =
        ((owner.parent() != nullptr) && (owner.parent()->asBuffer() != nullptr));
    bool result = false;

    // Check if owner is interop memory
    if (owner.isInterop()) {
      result = gpuImage->createInterop();
    } else if (imageBuffer) {
      Resource::ImageBufferParams params;
      pal::Memory* buffer = reinterpret_cast<pal::Memory*>(image.parent()->getDeviceMemory(*this));
      if (buffer == nullptr) {
        LogError("Buffer creation for ImageBuffer failed!");
        delete gpuImage;
        return nullptr;
      }
      params.owner_ = &owner;
      params.resource_ = buffer;
      params.memory_ = buffer;
      // Create memory object
      result = gpuImage->create(amd::IS_HIP && owner.parent()->isInterop()
                                    ? Resource::ImageExternalBuffer
                                    : Resource::ImageBuffer,
                                &params);
    } else if (directAccess && (owner.getMemFlags() & CL_MEM_ALLOC_HOST_PTR)) {
      Resource::PinnedParams params;
      params.owner_ = &owner;
      params.hostMemRef_ = owner.getHostMemRef();
      params.size_ = owner.getHostMemRef()->size();

      // Create memory object
      result = gpuImage->create(Resource::Pinned, &params);
    } else {
      Resource::CreateParams params;
      params.owner_ = &owner;
      params.gpu_ = static_cast<VirtualGPU*>(owner.getVirtualDevice());
      params.svmBase_ = static_cast<Memory*>(owner.svmBase());
      Resource::MemoryType type = Resource::MemoryType::Empty;
      if (owner.P2PAccess()) {
        params.svmBase_ = static_cast<Memory*>(owner.BaseP2PMemory());
        if (params.svmBase_ != nullptr) {
          type = Resource::P2PAccess;
        }
      }
      if (type == Resource::P2PAccess) {
        // Create memory object
        result = gpuImage->create(type, &params);
      }
    }

    if (!result && !owner.isInterop()) {
      if (owner.getMemFlags() & CL_MEM_USE_PERSISTENT_MEM_AMD) {
        // Attempt to allocate from persistent heap
        result = gpuImage->create(Resource::Persistent);
      } else {
        Resource::MemoryType type =
            (owner.forceSysMemAlloc()) ? Resource::RemoteUSWC : Resource::Local;
        // Create memory object
        result = gpuImage->create(type);
      }
    }

    if (!result) {
      delete gpuImage;
      return nullptr;
    } else if ((gpuImage->memoryType() != Resource::Pinned) &&
               (owner.getMemFlags() & CL_MEM_COPY_HOST_PTR) &&
               (owner.getContext().devices().size() == 1)) {
      // Ignore copy for image1D_buffer, since it was already done for buffer
      if (imageBuffer) {
        // Clear CHP memory
        owner.setHostMem(nullptr);
      } else {
        amd::Coord3D origin(0, 0, 0);
        // Copy data with the original pitch values, since runtime doesn't perform
        // extra sysmem allocation for one device
        constexpr bool kEntire = true;
        if (xferMgr().writeImage(owner.getHostMem(), *gpuImage, origin, image.getRegion(),
                                 image.getRowPitch(), image.getSlicePitch(), kEntire)) {
          // Clear CHP memory
          owner.setHostMem(nullptr);
        }
      }
    }

    if (result) {
      size_t bytePitch = gpuImage->elementSize() * gpuImage->desc().width_;
      image.setBytePitch(bytePitch);
    }
  }

  return gpuImage;
}

// ================================================================================================
device::Memory* Device::createMemory(amd::Memory& owner) const {
  bool directAccess = false;
  pal::Memory* memory = nullptr;

  if (owner.asBuffer()) {
    directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemBuffer) ? true : false;
    memory = createBuffer(owner, directAccess);
  } else if (owner.asImage()) {
    directAccess = (settings().hostMemDirectAccess_ & Settings::HostMemImage) ? true : false;
    memory = createImage(owner, directAccess);
  } else {
    LogError("Unknown memory type!");
  }

  // Attempt to pin system memory if runtime didn't use direct access
  if ((memory != nullptr) && (memory->memoryType() != Resource::Pinned) &&
      (memory->memoryType() != Resource::Remote) &&
      (memory->memoryType() != Resource::RemoteUSWC) &&
      (memory->memoryType() != Resource::ExternalPhysical) &&
      ((owner.getHostMem() != nullptr) ||
       ((nullptr != owner.parent()) && (owner.getHostMem() != nullptr)))) {
    bool ok = memory->pinSystemMemory(owner.getHostMem(), (owner.getHostMemRef()->size())
                                                              ? owner.getHostMemRef()->size()
                                                              : owner.getSize());
    //! \note: Ignore the pinning result for now
  }

  return memory;
}

// ================================================================================================
device::Memory* Device::createMemory(size_t size, size_t alignment) const {
  auto buffer = new pal::Memory(*this, size);
  Resource::CreateParams params{};
  params.alignment_ = alignment;
  if ((buffer == nullptr) || !buffer->create(Resource::Local, &params)) {
    LogError("Couldn't allocate memory on device!");
    return nullptr;
  }
  return buffer;
}

// ================================================================================================
bool Device::createSampler(const amd::Sampler& owner, device::Sampler** sampler) const {
  *sampler = nullptr;
  Sampler* gpuSampler = new Sampler(*this);
  if ((nullptr == gpuSampler) || !gpuSampler->create(owner)) {
    delete gpuSampler;
    return false;
  }
  *sampler = gpuSampler;
  return true;
}

device::Memory* Device::createView(amd::Memory& owner, const device::Memory& parent) const {
  assert((owner.asImage() != nullptr) && "View supports images only");
  const amd::Image& image = *owner.asImage();
  pal::Memory* gpuImage =
      new pal::Image(*this, owner, image.getWidth(), image.getHeight(), image.getDepth(),
                     image.getImageFormat(), image.getType(), image.getMipLevels());

  // Create resource
  if (nullptr != gpuImage) {
    Resource::ImageViewParams params;
    const pal::Memory& gpuMem = static_cast<const pal::Memory&>(parent);

    params.owner_ = &owner;
    params.level_ = image.getBaseMipLevel();
    params.layer_ = 0;
    params.resource_ = &gpuMem;
    params.gpu_ = reinterpret_cast<VirtualGPU*>(owner.getVirtualDevice());
    params.memory_ = &gpuMem;

    // Create memory object
    bool result = gpuImage->create(Resource::ImageView, &params);
    if (!result) {
      delete gpuImage;
      return nullptr;
    }
  }

  return gpuImage;
}

//! Attempt to bind with external graphics API's device/context
bool Device::bindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                bool validateOnly) {
  assert(pDevice);

#ifdef _WIN32
  if (flags & amd::Context::Flags::D3D10DeviceKhr) {
    if (!associateD3D10Device(pDevice[amd::Context::DeviceFlagIdx::D3D10DeviceKhrIdx])) {
      LogError("Failed associateD3D10Device()");
      return false;
    }
  }

  if (flags & amd::Context::Flags::D3D11DeviceKhr) {
    if (!associateD3D11Device(pDevice[amd::Context::DeviceFlagIdx::D3D11DeviceKhrIdx])) {
      LogError("Failed associateD3D11Device()");
      return false;
    }
  }

  if (flags & amd::Context::Flags::D3D9DeviceKhr) {
    if (!associateD3D9Device(pDevice[amd::Context::DeviceFlagIdx::D3D9DeviceKhrIdx])) {
      LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
      return false;
    }
  }

  if (flags & amd::Context::Flags::D3D9DeviceEXKhr) {
    if (!associateD3D9Device(pDevice[amd::Context::DeviceFlagIdx::D3D9DeviceEXKhrIdx])) {
      LogWarning("D3D9<->OpenCL adapter mismatch or D3D9Associate() failure");
      return false;
    }
  }
#endif  //_WIN32

  if (flags & amd::Context::Flags::GLDeviceKhr) {
    // Attempt to associate PAL-OGL
    if (!glAssociate(pContext, pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx])) {
      if (!validateOnly) {
        LogError("Failed glAssociate()");
      }
      return false;
    }
  }

  return true;
}

bool Device::unbindExternalDevice(uint flags, void* const pDevice[], void* pContext,
                                  bool validateOnly) {
  if ((flags & amd::Context::Flags::GLDeviceKhr) == 0) {
    return true;
  }

  void* glDevice = pDevice[amd::Context::DeviceFlagIdx::GLDeviceKhrIdx];
  if (glDevice != nullptr) {
    // Dissociate PAL-OGL
    if (!glDissociate(pContext, glDevice)) {
      if (validateOnly) {
        LogWarning("Failed glDissociate()");
      }
      return false;
    }
  }
  return true;
}

bool Device::globalFreeMemory(size_t* freeMemory) const {
  constexpr uint32_t TotalFreeMemory = 0;
  constexpr uint32_t LargestFreeBlock = 1;

  // Initialization of heap and other resources because getMemInfo needs it.
  if (!(const_cast<Device*>(this)->initializeHeapResources())) {
    return false;
  }

  // First, runtime calculates per process memory usage

  // Don't report cached memory in runtime as allocated, since allocedMem tracked at PAL calls
  Pal::gpusize local = allocedMem[Pal::GpuHeapLocal] - resourceCache().persistentCacheSize();
  Pal::gpusize invisible = allocedMem[Pal::GpuHeapInvisible] - resourceCache().lclCacheSize();
  Pal::gpusize total_alloced = local + invisible;
  size_t cache_group_local = resourceCache().persistentCacheSize() + resourceCache().lclCacheSize();
  // Allocated system memory without cached allocations. Cache size contains all allocations, so
  // don't count persistent and local
  Pal::gpusize system_memory = allocedMem[Pal::GpuHeapGartCacheable] +
                               allocedMem[Pal::GpuHeapGartUswc] + cache_group_local -
                               resourceCache().cacheSize();

#if IS_WINDOWS
  // Second, query OS for overall memory usage on the system

  if (properties().osProperties.supportMemoryBudgetQuery) {
    Pal::GpuMemoryBudgetInfo mem_budget_info = {};
    // Query OS how much memory is available
    iDev()->QueryGpuMemoryBudgetInfo(&mem_budget_info);

    Pal::gpusize system_total_alloced = mem_budget_info.usage[Pal::GpuHeapGroupLocal];
    // Avoid possible negative values in case of alignments
    if (mem_budget_info.usage[Pal::GpuHeapGroupLocal] > cache_group_local) {
      system_total_alloced = mem_budget_info.usage[Pal::GpuHeapGroupLocal] - cache_group_local;
    }
    // System usage exceeds per process usage for device memory
    if (system_total_alloced > total_alloced) {
      total_alloced = system_total_alloced;
    }
    system_total_alloced = mem_budget_info.usage[Pal::GpuHeapGroupNonLocal];
    // Avoid possible negative values in case of extra alignments
    if (mem_budget_info.usage[Pal::GpuHeapGroupNonLocal] >
        (resourceCache().cacheSize() - cache_group_local)) {
      system_total_alloced = mem_budget_info.usage[Pal::GpuHeapGroupNonLocal] + cache_group_local -
                             resourceCache().cacheSize();
    }
    // System usage exceeds per process usage for system memory
    if (system_total_alloced > system_memory) {
      system_memory = system_total_alloced;
    }
  }
#endif
  // Third, finalize reported free memory

  // Fill free memory info
  freeMemory[TotalFreeMemory] =
      (total_alloced > info().globalMemSize_)
          ? 0
          : static_cast<size_t>((info().globalMemSize_ - total_alloced) / Ki);

  freeMemory[TotalFreeMemory] -=
      (freeMemory[TotalFreeMemory] > HIP_HIDDEN_FREE_MEM * Ki) ? HIP_HIDDEN_FREE_MEM * Ki : 0;

  Pal::gpusize largest_block = 0;
  if (settings().apuSystem_) {
    system_memory /= Ki;
    if (system_memory >= freeMemory[TotalFreeMemory]) {
      freeMemory[TotalFreeMemory] = 0;
    } else {
      freeMemory[TotalFreeMemory] -= system_memory;
    }
    if (system_memory < heaps_[Pal::GpuHeapGartUswc].logicalSize) {
      largest_block = heaps_[Pal::GpuHeapGartUswc].logicalSize - system_memory;
    }
  }

  if (invisible < heaps_[Pal::GpuHeapInvisible].logicalSize) {
    largest_block = std::max(largest_block, heaps_[Pal::GpuHeapInvisible].logicalSize - invisible);
  }
  if (local < heaps_[Pal::GpuHeapLocal].logicalSize) {
    largest_block = std::max(largest_block, heaps_[Pal::GpuHeapLocal].logicalSize - invisible);
  }

  largest_block /= Ki;
  freeMemory[LargestFreeBlock] =
      (largest_block > freeMemory[TotalFreeMemory]) ? freeMemory[TotalFreeMemory] : largest_block;

  return true;
}

// PAL Device file I/O stubs: PAL doesn't implement AIS file I/O yet.
// Provide definitions so the vtable and linker are satisfied.
// Replace with a real implementation if/when PAL supports file I/O.
bool Device::amdFileRead(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                      uint64_t* size_copied, int32_t* status) {
  if (size_copied) {
    *size_copied = 0;
  }
  LogError("PAL Device: amdFileRead not supported on this backend");
  return false;
}

bool Device::amdFileWrite(amd::Os::FileDesc handle, void* devicePtr, uint64_t size, int64_t file_offset,
                       uint64_t* size_copied, int32_t* status) {
  if (size_copied) {
    *size_copied = 0;
  }
  LogError("PAL Device: amdFileWrite not supported on this backend");
  return false;
}

amd::Memory* Device::findMapTarget(size_t size) const {
  // Must be serialised for access
  amd::ScopedLock lk(mapCacheOps_);

  amd::Memory* map = nullptr;
  size_t minSize = 0;
  size_t maxSize = 0;
  uint mapId = mapCache_->size();
  uint releaseId = mapCache_->size();

  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); i++) {
    if ((*mapCache_)[i] != nullptr) {
      // Requested size is smaller than the entry size
      if (size < (*mapCache_)[i]->getSize()) {
        if ((minSize == 0) || (minSize > (*mapCache_)[i]->getSize())) {
          minSize = (*mapCache_)[i]->getSize();
          mapId = i;
        }
      }
      // Requeted size matches the entry size
      else if (size == (*mapCache_)[i]->getSize()) {
        mapId = i;
        break;
      } else {
        // Find the biggest map target in the list
        if (maxSize < (*mapCache_)[i]->getSize()) {
          maxSize = (*mapCache_)[i]->getSize();
          releaseId = i;
        }
      }
    }
  }

  // Check if we found any map target
  if (mapId < mapCache_->size()) {
    map = (*mapCache_)[mapId];
    (*mapCache_)[mapId] = nullptr;
    Memory* gpuMemory = reinterpret_cast<Memory*>(map->getDeviceMemory(*this));

    // Get the base pointer for the map resource
    if ((gpuMemory == nullptr) || (nullptr == gpuMemory->map(nullptr))) {
      (*mapCache_)[mapId]->release();
      map = nullptr;
    }
  }
  // If cache is full, then release the biggest map target
  else if (releaseId < mapCache_->size()) {
    (*mapCache_)[releaseId]->release();
    (*mapCache_)[releaseId] = nullptr;
  }

  return map;
}

bool Device::addMapTarget(amd::Memory* memory) const {
  // Must be serialised for access
  amd::ScopedLock lk(mapCacheOps_);

  // the svm memory shouldn't be cached
  if (!memory->canBeCached()) {
    return false;
  }
  // Find if the list has a map target of appropriate size
  for (uint i = 0; i < mapCache_->size(); ++i) {
    if ((*mapCache_)[i] == nullptr) {
      (*mapCache_)[i] = memory;
      return true;
    }
  }

  // Add a new entry
  mapCache_->push_back(memory);

  return true;
}

Device::ScratchBuffer::~ScratchBuffer() { destroyMemory(); }

void Device::ScratchBuffer::destroyMemory() {
  // Release memory object
  delete memObj_;
  memObj_ = nullptr;
}

bool Device::allocScratch(uint regNum, const VirtualGPU* vgpu, uint vgprs) {
  if (regNum > 0 && vgprs > 0) {
    // Serialize the scratch buffer allocation code
    amd::ScopedLock lk(scratchAlloc_);
    uint sb = vgpu->hwRing();
    static const uint WaveSizeLimit = ((1 << 21) - 256);
    const uint threadSizeLimit = WaveSizeLimit / info().wavefrontWidth_;
    if (regNum > threadSizeLimit) {
      LogError("Requested private memory is bigger than HW supports!");
      regNum = threadSizeLimit;
    }

    // The algorithm below finds the most optimal size for the current execution.
    // PAL reprograms COMPUTE_TMPRING_SIZE.WAVESIZE and COMPUTE_TMPRING_SIZE.WAVES on
    // every dispatch and sync mode is enabled in runtime

    // Calculate the size of the scratch buffer for a queue
    uint32_t numTotalCUs = properties().gfxipProperties.shaderCore.numAvailableCus;
    // Find max waves based on VGPR per SIMD
    uint32_t numMaxWaves = properties().gfxipProperties.shaderCore.vgprsPerSimd / vgprs;
    // Find max waves per CU
    numMaxWaves *= properties().gfxipProperties.shaderCore.numSimdsPerCu;
    // Find max waves per device
    numMaxWaves = std::min(settings().numScratchWavesPerCu_, numMaxWaves);
    // Current private mem size
    uint32_t privateMemSize = regNum * sizeof(uint32_t);
    uint64_t newSize =
        static_cast<uint64_t>(info().wavefrontWidth_) * privateMemSize * numMaxWaves * numTotalCUs;

    // Check if the current buffer isn't big enough
    if (newSize > scratch_[sb]->size_) {
      // Stall all command queues, since runtime will reallocate memory
      ScopedLockVgpus lock(*this);

      scratch_[sb]->size_ = newSize;

      uint64_t size = 0;
      uint64_t offset = 0;

      // Destroy all views
      for (uint s = 0; s < scratch_.size(); ++s) {
        ScratchBuffer* scratchBuf = scratch_[s];
        if (scratchBuf->size_ > 0) {
          scratchBuf->destroyMemory();
          // Adjust the size for the current queue only
          if (s == sb) {
            scratchBuf->size_ = std::min(newSize, info().maxMemAllocSize_);
            scratchBuf->size_ = std::min(newSize, uint64_t(3 * Gi));
            // Note: Generic address space setup in HW requires 64KB alignment for scratch
            scratchBuf->size_ = amd::alignUp(newSize, 64 * Ki);
          }
          scratchBuf->offset_ = offset;
          size += scratchBuf->size_;
          offset += scratchBuf->size_;
        }
      }

      delete globalScratchBuf_;

      // Allocate new buffer.
      globalScratchBuf_ = new pal::Memory(*this, static_cast<size_t>(size));
      if ((globalScratchBuf_ == nullptr) || !globalScratchBuf_->create(Resource::Scratch)) {
        LogError("Couldn't allocate scratch memory");
        for (uint s = 0; s < scratch_.size(); ++s) {
          scratch_[s]->size_ = 0;
        }
        return false;
      }

      for (uint s = 0; s < scratch_.size(); ++s) {
        // Loop through all memory objects and reallocate them
        if (scratch_[s]->size_ > 0) {
          // Allocate new buffer
          scratch_[s]->memObj_ = new pal::Memory(*this, scratch_[s]->size_);
          Resource::ViewParams view;
          view.resource_ = globalScratchBuf_;
          view.offset_ = scratch_[s]->offset_;
          view.size_ = scratch_[s]->size_;
          if ((scratch_[s]->memObj_ == nullptr) ||
              !scratch_[s]->memObj_->create(Resource::View, &view)) {
            LogError("Couldn't allocate a scratch view");
            delete scratch_[s]->memObj_;
            scratch_[s]->size_ = 0;
            return false;
          }
        }
      }
    }
  }
  return true;
}

bool Device::validateKernel(const amd::Kernel& kernel, const device::VirtualDevice* vdev,
                            bool coop_groups) {
  // Find the number of scratch registers used in the kernel
  const device::Kernel* devKernel = kernel.getDeviceKernel(*this);
  uint32_t regNum = static_cast<uint32_t>(devKernel->workGroupInfo()->scratchRegs_);
  // OCL does not have API to set dynamic stack size i.e. hipDeviceSetLimit and hence there
  // is no need for OCL to refresh value here and even for HIP, Update should be only if
  // compiler notifies use of stack size.
  if (IS_HIP && (devKernel->workGroupInfo()->usedStackSize_ & 0x1) == 0x1) {
    regNum = std::max<uint32_t>(static_cast<uint32_t>(stack_size_) / sizeof(uint32_t), regNum);
  }
  const VirtualGPU* vgpu = static_cast<const VirtualGPU*>(vdev);

  if (!allocScratch(regNum, vgpu, devKernel->workGroupInfo()->usedVGPRs_)) {
    return false;
  }
  // Runtime plans to launch cooperative groups on the device queue, thus
  // validate the scratch buffer on that queue
  if (coop_groups) {
    vgpu = xferQueue();
    if (!allocScratch(regNum, vgpu, devKernel->workGroupInfo()->usedVGPRs_)) {
      return false;
    }
  }

  const HSAILKernel* hsaKernel = static_cast<const HSAILKernel*>(devKernel);
  if (hsaKernel->dynamicParallelism()) {
    if (settings().useDeviceQueue_) {
      amd::DeviceQueue* defQueue = kernel.program().context().defDeviceQueue(*this);
      if (defQueue != nullptr) {
        vgpu = static_cast<VirtualGPU*>(defQueue->vDev());
        if (!allocScratch(hsaKernel->prog().maxScratchRegs(), vgpu, hsaKernel->prog().maxVgprs())) {
          return false;
        }
      } else {
        return false;
      }
    } else {
      if (!allocScratch(hsaKernel->prog().maxScratchRegs(), vgpu, hsaKernel->prog().maxVgprs())) {
        return false;
      }
    }
  }

  return true;
}

void Device::destroyScratchBuffers() {
  if (globalScratchBuf_ != nullptr) {
    for (uint s = 0; s < scratch_.size(); ++s) {
      scratch_[s]->destroyMemory();
      scratch_[s]->size_ = 0;
    }
    delete globalScratchBuf_;
    globalScratchBuf_ = nullptr;
  }
}

void Device::fillHwSampler(uint32_t state, void* hwState, uint32_t hwStateSize,
                           const uint* addressMode, uint32_t mipFilter, float minLod,
                           float maxLod) const {
  Pal::SamplerInfo samplerInfo = {};
  assert(addressMode != nullptr);
  samplerInfo.borderColorType = Pal::BorderColorType::TransparentBlack;

  samplerInfo.filter.zFilter = Pal::XyFilterPoint;

  samplerInfo.flags.unnormalizedCoords = !(state & amd::Sampler::StateNormalizedCoordsMask);
  samplerInfo.maxLod = 4096.0f;

  state &= ~amd::Sampler::StateNormalizedCoordsMask;

  // Program texture filter mode
  if (state == amd::Sampler::StateFilterLinear) {
    samplerInfo.filter.magnification = Pal::XyFilterLinear;
    samplerInfo.filter.minification = Pal::XyFilterLinear;
    samplerInfo.filter.zFilter = Pal::ZFilterLinear;
  }

  if (mipFilter == CL_FILTER_NEAREST) {
    samplerInfo.filter.mipFilter = Pal::MipFilterPoint;
  } else if (mipFilter == CL_FILTER_LINEAR) {
    samplerInfo.filter.mipFilter = Pal::MipFilterLinear;
  }

  auto addessModeMap = [](const uint addreMode) {
    switch (addreMode) {
      case CL_ADDRESS_CLAMP_TO_EDGE:
        return Pal::TexAddressMode::Clamp;
      case CL_ADDRESS_REPEAT:
        return Pal::TexAddressMode::Wrap;
      case CL_ADDRESS_MIRRORED_REPEAT:
        return Pal::TexAddressMode::Mirror;
      case CL_ADDRESS_CLAMP:
      case CL_ADDRESS_NONE:
      default:
        return Pal::TexAddressMode::ClampBorder;
    }
  };
  samplerInfo.addressU = addessModeMap(addressMode[0]);
  samplerInfo.addressV = addessModeMap(addressMode[1]);
  samplerInfo.addressW = addessModeMap(addressMode[2]);

  iDev()->CreateSamplerSrds(1, &samplerInfo, hwState);
}

void* Device::hostAlloc(size_t size, size_t alignment, MemorySegment mem_seg,
                        const void* agentInfo) const {
  // for discrete gpu, we only reserve,no commit yet.
  return amd::Os::reserveMemory(nullptr, size, alignment, amd::Os::MEM_PROT_NONE);
}

bool Device::allowPeerAccess(device::Memory* memory) const {
  if (memory == nullptr) {
    return false;
  }
  Resource::CreateParams params;
  amd::Memory* owner = reinterpret_cast<amd::Memory*>(memory->owner());
  params.owner_ = owner;
  params.gpu_ = static_cast<VirtualGPU*>(owner->getVirtualDevice());
  params.svmBase_ = static_cast<Memory*>(owner->BaseP2PMemory());
  pal::Memory* gpuMemory = getGpuMemory(owner);
  bool result = gpuMemory->CreateP2PAccess(&params);
  if (result != true) {
    LogError("Allow p2p access");
    return false;
  }
  return true;
}

void Device::hostFree(void* ptr, size_t size) const {
  // If we allocate the host memory, we need free, or we have to release
  amd::Os::releaseMemory(ptr, size);
}

bool Device::deviceAllowAccess(void* ptr) const {
  std::lock_guard<std::mutex> lock(lockAllowAccess_);
  // Empty function for now.
  return true;
}

void* Device::svmAlloc(amd::Context& context, size_t size, size_t alignment, cl_svm_mem_flags flags,
                       void* svmPtr) const {
  constexpr bool kForceAllocation = true;
  alignment = std::max(alignment, static_cast<size_t>(info_.memBaseAddrAlign_));

  if (amd::IS_HIP && PAL_HIP_IPC_FLAG) {
    // set interprocess for IPC memory support
    flags |= ROCCLR_MEM_INTERPROCESS;
  }

  amd::Memory* mem = nullptr;
  freeCPUMem_ = false;
  if (nullptr == svmPtr) {
    if (isFineGrainedSystem()) {
      freeCPUMem_ = true;
      return amd::Os::alignedMalloc(size, alignment);
    }

    // create a hidden buffer, which will allocated on the device later
    mem = new (context) amd::Buffer(context, flags, size, reinterpret_cast<void*>(1));
    if (mem == nullptr) {
      LogError("failed to create a svm mem object!");
      return nullptr;
    }

    if (!mem->create(nullptr, false, false, kForceAllocation)) {
      LogError("failed to create a svm hidden buffer!");
      mem->release();
      return nullptr;
    }
    // if the device supports SVM FGS, return the committed CPU address directly.
    pal::Memory* gpuMem = getGpuMemory(mem);

    // add the information to context so that we can use it later.
    amd::MemObjMap::AddMemObj(mem->getSvmPtr(), mem);
    svmPtr = mem->getSvmPtr();

    if (settings().apuSystem_ && gpuMem->isHostMemDirectAccess()) {
      mem->commitSvmMemory();
    }
  } else {
    // find the existing amd::mem object
    mem = amd::MemObjMap::FindMemObj(svmPtr);
    if (nullptr == mem) {
      return nullptr;
    }
    // commit the CPU memory for FGS device.
    if (isFineGrainedSystem()) {
      mem->commitSvmMemory();
    } else {
      pal::Memory* gpuMem = getGpuMemory(mem);
    }
    svmPtr = mem->getSvmPtr();
  }
  return svmPtr;
}

void Device::svmFree(void* ptr) const {
  if (freeCPUMem_) {
    amd::Os::alignedFree(ptr);
  } else {
    amd::Memory* svmMem = amd::MemObjMap::FindMemObj(ptr);
    if (nullptr != svmMem) {
      amd::MemObjMap::RemoveMemObj(ptr);
      svmMem->release();
    }
  }
}

// ================================================================================================
void* Device::virtualAlloc(void* addr, size_t size, size_t alignment) {
  constexpr bool kParent = true;
  constexpr bool kForceAlloc = true;
  amd::Memory* mem = CreateVirtualBuffer(context(), addr, size, -1, -1, kParent, kForceAlloc);
  assert(mem != nullptr);
  return mem->getSvmPtr();
}

// ================================================================================================
bool Device::virtualFree(void* addr) {
  auto vaddr_mem_obj = amd::MemObjMap::FindVirtualMemObj(addr);
  if (vaddr_mem_obj == nullptr) {
    LogPrintfError("Cannot find any mem_obj for addr: 0x%x \n", addr);
    return false;
  }

  if (!vaddr_mem_obj->getContext().devices()[0]->DestroyVirtualBuffer(vaddr_mem_obj)) {
    LogPrintfError("Cannot destroy mem_obj:0x%x for addr: 0x%x \n", vaddr_mem_obj, addr);
    return false;
  }
  return true;
}

// ================================================================================================
bool Device::SetMemAccess(void* va_addr, size_t va_size, VmmAccess access_flags,
                          VmmLocationType access_location) {
  amd::Memory* amd_mem_obj = amd::MemObjMap::FindMemObj(va_addr);
  if (amd_mem_obj == nullptr) {
    // If the amd_mem_obj is null, the check if this is a valid va_addr, but not-mapped,
    // if not-mapped then print a different error message. (No functional change due to this check).
    amd_mem_obj = amd::MemObjMap::FindVirtualMemObj(va_addr);
    if (amd_mem_obj == nullptr) {
      LogPrintfError("Cannot find virtual address: 0x%x \n", va_addr);
      return false;
    }
    LogPrintfError("Virtual address present, but not mapped yet: 0x%x \n", va_addr);
  }

  // Check for valid size.
  if (va_size > amd_mem_obj->getSize()) {
    LogPrintfError("Given size: %u cannot be greater than mem_size: %u \n", va_size,
                   amd_mem_obj->getSize());
    return false;
  }

  device::Memory* dev_mem_obj = amd_mem_obj->getDeviceMemory(*this);
  dev_mem_obj->SetAccess(static_cast<device::Memory::MemAccess>(access_flags));

  return true;
}

// ================================================================================================
bool Device::ValidateMemAccess(amd::Memory& amd_mem_obj, bool read_write) const {
  device::Memory* dev_mem = amd_mem_obj.getDeviceMemory(*this);
  device::Memory::MemAccess mem_access = dev_mem->GetAccess();

  // If read_write flag is set, then only read_write is valid, else it could be a read or write.
  if (read_write && mem_access != device::Memory::MemAccess::kMemAccessReadWrite) {
    return false;
  } else if ((mem_access != device::Memory::MemAccess::kMemAccessRead) &&
             (mem_access != device::Memory::MemAccess::kMemAccessReadWrite)) {
    return false;
  }

  return true;
}

// ================================================================================================
bool Device::GetMemAccess(void* va_addr, VmmAccess* access_flags_ptr) const {
  amd::Memory* phys_mem_obj = amd::MemObjMap::FindMemObj(va_addr);
  if (phys_mem_obj == nullptr) {
    // If the phys_mem_obj is null, the check if this is a valid va_addr, but not-mapped,
    // if not-mapped then print a different error message. (No functional change due to this check).
    amd::Memory* vaddr_mem_obj = amd::MemObjMap::FindVirtualMemObj(va_addr);
    if (vaddr_mem_obj == nullptr) {
      LogPrintfError("Cannot find virtual address: 0x%x \n", va_addr);
      return false;
    }
    LogPrintfInfo("Virtual address present, but not mapped yet: 0x%x \n", va_addr);
    return false;
  }

  device::Memory* phys_dev_mem = phys_mem_obj->getDeviceMemory(*this);
  device::Memory::MemAccess mem_access = phys_dev_mem->GetAccess();
  *access_flags_ptr = static_cast<VmmAccess>(mem_access);

  return true;
}
// ================================================================================================
bool Device::AcquireExclusiveGpuAccess() {
  // Lock the virtual GPU list
  vgpusAccess().lock();

  // Find all available virtual GPUs and lock them
  // from the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().lock();
    // Make sure a wait is done
    vgpus()[idx]->WaitForIdleCompute();
  }

  return true;
}

void Device::ReleaseExclusiveGpuAccess(VirtualGPU& vgpu) const {
  vgpu.WaitForIdleCompute();
  // Find all available virtual GPUs and unlock them
  // for the execution of commands
  for (uint idx = 0; idx < vgpus().size(); ++idx) {
    vgpus()[idx]->execution().unlock();
  }

  // Unock the virtual GPU list
  vgpusAccess().unlock();
}

// ================================================================================================
void Device::HiddenHeapAlloc(const VirtualGPU& gpu) {
  auto HeapAlloc = [this, &gpu]() -> bool {
    // Allocate initial heap for device memory allocator
    static constexpr size_t HeapBufferSize = 128 * Ki;
    heap_buffer_ = createMemory(HeapBufferSize);
    if (initial_heap_size_ != 0) {
      initial_heap_size_ = amd::alignUp(initial_heap_size_, 2 * Mi);
      initial_heap_buffer_ = createMemory(initial_heap_size_, 2 * Mi);
    }
    if (heap_buffer_ == nullptr) {
      LogError("Heap buffer allocation failed!");
      return false;
    }
    bool result = static_cast<const KernelBlitManager&>(gpu.blitMgr())
                      .initHeap(heap_buffer_, initial_heap_buffer_, HeapBufferSize,
                                initial_heap_size_ / (2 * Mi));

    return result;
  };
  std::call_once(heap_initialized_, HeapAlloc);
}

// ================================================================================================
Device::SrdManager::~SrdManager() {
  for (uint i = 0; i < pool_.size(); ++i) {
    pool_[i].buf_->unmap(nullptr);
    delete pool_[i].buf_;
    delete pool_[i].flags_;
  }
}

bool Sampler::create(uint32_t oclSamplerState, const uint addressMode[3]) {
  hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
  if (0 == hwSrd_) {
    return false;
  }
  dev_.fillHwSampler(oclSamplerState, hwState_, HsaSamplerObjectSize, addressMode);
  return true;
}

bool Sampler::create(const amd::Sampler& owner) {
  hwSrd_ = dev_.srds().allocSrdSlot(&hwState_);
  if (0 == hwSrd_) {
    return false;
  }
  dev_.fillHwSampler(owner.state(), hwState_, HsaSamplerObjectSize, owner.addessMode(),
                     owner.mipFilter(), owner.minLod(), owner.maxLod());
  return true;
}

Sampler::~Sampler() { dev_.srds().freeSrdSlot(hwSrd_); }

uint64_t Device::SrdManager::allocSrdSlot(address* cpuAddr) {
  amd::ScopedLock lock(ml_);
  // Check all buffers in the pool of chunks
  for (uint i = 0; i < pool_.size(); ++i) {
    const Chunk& ch = pool_[i];
    // Search for an empty slot
    for (uint s = 0; s < numFlags_; ++s) {
      uint mask = ch.flags_[s];
      // Check if there is an empty slot in this group
      if (mask != 0) {
        uint idx;
        // Find the first empty index
        for (idx = 0; (mask & 0x1) == 0; mask >>= 1, ++idx);
        // Mark the slot as busy
        ch.flags_[s] &= ~(1 << idx);
        // Calculate SRD offset in the buffer
        uint offset = (s * MaskBits + idx) * srdSize_;
        *cpuAddr = ch.buf_->data() + offset;
        return ch.buf_->vmAddress() + offset;
      }
    }
  }
  // At this point the manager doesn't have empty slots
  // and has to allocate a new chunk
  Chunk chunk;
  chunk.flags_ = new uint[numFlags_];
  if (chunk.flags_ == nullptr) {
    return 0;
  }
  chunk.buf_ = new Memory(dev_, bufSize_);
  if (chunk.buf_ == nullptr || !chunk.buf_->create(Resource::Remote) ||
      (nullptr == chunk.buf_->map(nullptr))) {
    delete[] chunk.flags_;
    delete chunk.buf_;
    return 0;
  }
  // All slots in the chunk are in "free" state
  memset(chunk.flags_, 0xff, numFlags_ * sizeof(uint));
  // Take the first one...
  chunk.flags_[0] &= ~0x1;
  pool_.push_back(chunk);
  *cpuAddr = chunk.buf_->data();
  return chunk.buf_->vmAddress();
}

void Device::SrdManager::freeSrdSlot(uint64_t addr) {
  amd::ScopedLock lock(ml_);
  if (addr == 0) return;
  // Check all buffers in the pool of chunks
  for (uint i = 0; i < pool_.size(); ++i) {
    Chunk* ch = &pool_[i];
    // Find the offset
    int64_t offs = static_cast<int64_t>(addr) - static_cast<int64_t>(ch->buf_->vmAddress());
    // Check if the offset inside the chunk buffer
    if ((offs >= 0) && (offs < bufSize_)) {
      // Find the index in the chunk
      uint idx = offs / srdSize_;
      uint s = idx / MaskBits;
      // Free the slot
      ch->flags_[s] |= 1 << (idx % MaskBits);
      return;
    }
  }
  assert(false && "Wrong slot address!");
}

void Device::updateAllocedMemory(Pal::GpuHeap heap, Pal::gpusize size, bool free) const {
  if (free) {
    allocedMem[heap] -= size;
  } else {
    allocedMem[heap] += size;
  }
}

bool Device::createBlitProgram() {
  bool result = true;

  // Delayed compilation due to brig_loader memory allocation
  std::string extraBlits;
  std::string ocl20;
  if (amd::IS_HIP) {
    if (settings().gwsInitSupported_) {
      extraBlits = device::HipExtraSourceCode;
    } else {
      extraBlits = device::HipExtraSourceCodeNoGWS;
    }
  } else {
    if (settings().oclVersion_ >= OpenCL20) {
      extraBlits = iDev()->GetDispatchKernelSource();
      if (settings().useLightning_) {
        extraBlits.append(SchedulerSourceCode20);
      } else {
        extraBlits.append(SchedulerSourceCode);
      }
      ocl20 = "-cl-std=CL2.0";
    }
  }

  if (settings().useLightning_) {
    const std::string TrapHandlerAsm = TrapHandlerCode;
    // Create a program for trap handler
    // note: It's not critical for runtime functionality to fail trap handler initialization
    auto asm_program = new amd::Program(*context_, TrapHandlerAsm.c_str(), amd::Program::Assembly);
    if (asm_program != nullptr) {
      std::vector<amd::Device*> devices;
      devices.push_back(this);
      std::string opt = "-cl-internal-kernel ";
      if (auto retval =
              asm_program->build(devices, opt.c_str(), nullptr, nullptr, false) != CL_SUCCESS) {
        DevLogPrintfError("Build failed for trap handler with error code: %d\n", retval);
        asm_program->release();
      } else {
        if (asm_program->load()) {
          trap_handler_ = asm_program;
        } else {
          DevLogError("Could not load the trap handler \n");
          asm_program->release();
        }
      }
    } else {
      DevLogError("Trap handler creation failed\n");
    }
  }

  blitProgram_ = new BlitProgram(context_);
  // Create blit programs
  if (blitProgram_ == nullptr || !blitProgram_->create(this, extraBlits, ocl20)) {
    delete blitProgram_;
    blitProgram_ = nullptr;
    LogError("Couldn't create blit kernels!");
    result = false;
  }
  return result;
}

void Device::SrdManager::fillResourceList(VirtualGPU& gpu) {
  for (uint i = 0; i < pool_.size(); ++i) {
    gpu.addVmMemory(pool_[i].buf_);
  }
}

bool Device::SetClockMode(const cl_set_device_clock_mode_input_amd setClockModeInput,
                          cl_set_device_clock_mode_output_amd* pSetClockModeOutput) {
  Pal::SetClockModeInput setClockMode = {};
  Pal::DeviceClockMode palClockMode =
      static_cast<Pal::DeviceClockMode>(setClockModeInput.clock_mode);
  setClockMode.clockMode = palClockMode;
  bool result =
      (Pal::Result::Success ==
       (iDev()->SetClockMode(setClockMode,
                             reinterpret_cast<Pal::SetClockModeOutput*>(pSetClockModeOutput))))
          ? true
          : false;
  return result;
}

// ================================================================================================
bool Device::importExtSemaphore(void** extSemaphore, const amd::Os::FileDesc& handle,
                                amd::ExternalSemaphoreHandleType sem_handle_type) {
  Pal::ExternalQueueSemaphoreOpenInfo palOpenInfo = {};
  palOpenInfo.externalSemaphore = handle;
  palOpenInfo.flags.crossProcess = false;
  palOpenInfo.flags.isReference = true;
  palOpenInfo.flags.timeline = palOpenInfo.flags.timeline =
      (sem_handle_type == amd::ExternalSemaphoreHandleType::TimelineSemaphoreWin32 ||
       sem_handle_type == amd::ExternalSemaphoreHandleType::TimelineSemaphoreFd);
  palOpenInfo.flags.sharedViaNtHandle =
      (sem_handle_type == amd::ExternalSemaphoreHandleType::OpaqueWin32 ||
       sem_handle_type == amd::ExternalSemaphoreHandleType::D3D12Fence);
  Pal::Result result;

  size_t semaphoreSize = iDev()->GetExternalSharedQueueSemaphoreSize(palOpenInfo, &result);
  if (result != Pal::Result::Success) {
    return false;
  }
  void* mem = amd::Os::alignedMalloc(semaphoreSize, 16);
  result = iDev()->OpenExternalSharedQueueSemaphore(
      palOpenInfo, mem, reinterpret_cast<Pal::IQueueSemaphore**>(extSemaphore));
  if (result != Pal::Result::Success) {
    amd::Os::alignedFree(mem);
    return false;
  }
  return true;
}

// ================================================================================================
void Device::DestroyExtSemaphore(void* extSemaphore) {
  Pal::IQueueSemaphore* sem = reinterpret_cast<Pal::IQueueSemaphore*>(extSemaphore);
  sem->Destroy();
  amd::Os::alignedFree(extSemaphore);
}

// ================================================================================================
bool Device::ExportShareableVMMHandle(amd::Memory& amd_mem_obj, int flags, void* shareableHandle) {
  device::Memory* dev_mem = static_cast<device::Memory*>(amd_mem_obj.getDeviceMemory(*this));
  return dev_mem->ExportHandle(shareableHandle);
}

// ================================================================================================
amd::Memory* Device::ImportShareableVMMHandle(void* osHandle) {
  int flags = 0;
  size_t mem_offset = 0;
  size_t mem_size = 0;

  amd::Memory* amd_mem_obj =
      new (context()) amd::IpcBuffer(context(), flags, mem_offset, mem_size, osHandle);

  if (amd_mem_obj == nullptr) {
    LogError("failed to create a mem object!");
    return nullptr;
  }

  if (!amd_mem_obj->create(nullptr)) {
    LogError("failed to create a svm hidden buffer!");
    amd_mem_obj->release();
    return nullptr;
  }
  return amd_mem_obj;
}

}  // namespace amd::pal
