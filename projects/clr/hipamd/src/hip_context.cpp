/* Copyright (c) 2015 - 2024 Advanced Micro Devices, Inc.

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

#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "platform/runtime.hpp"
#include "rocclr/utils/flags.hpp"
#include "rocclr/utils/versions.hpp"
#include "rocclr/os/os.hpp"

#include <hip/amd_detail/hip_api_trace.hpp>
namespace hip {
const HipToolsDispatchTable* GetHipToolsDispatchTable();
}  // namespace hip

namespace hip {
std::once_flag g_ihipInitialized;

std::vector<hip::Device*> g_devices ROCCLR_INIT_PRIORITY(101);
thread_local TlsAggregator tls;
amd::Context* host_context = nullptr;

// init() is only to be called from the HIP_INIT macro only once
void init(bool* status) {
  amd::IS_HIP = true;
  GPU_NUM_MEM_DEPENDENCY = 0;
#if DISABLE_DIRECT_DISPATCH
  constexpr bool kDirectDispatch = false;
#else
#if defined(WITH_HSA_DEVICE)
  constexpr bool kDirectDispatch = true;
#else
  constexpr bool kDirectDispatch = false;
#endif
#endif
  AMD_DIRECT_DISPATCH = flagIsDefault(AMD_DIRECT_DISPATCH) ? kDirectDispatch : AMD_DIRECT_DISPATCH;
  if (!amd::Runtime::init()) {
    *status = false;
    return;
  }

  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Version: %d.%d.%d.%s, Direct Dispatch: %d",
          HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH, HIP_VERSION_GITHASH,
          AMD_DIRECT_DISPATCH);
  // Print the current path of the library
  amd::Os::PrintLibraryLocation();
  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);
  const size_t deviceCount = devices.size();
  g_devices.reserve(deviceCount);  // Pre-allocate space for better performance

  for (unsigned int i = 0; i < deviceCount; i++) {
    // Enable active wait on the device by default
    devices[i]->SetActiveWait(true);
    // use the eternal contexts that already exist for new hip::Device's here
    auto device = new Device(&devices[i]->context(), i);
    if ((device == nullptr) || !device->Create()) {
      *status = false;
      return;
    }
    g_devices.push_back(device);
    amd::RuntimeTearDown::RegisterObject(device);
  }

  if (hip::GetHipToolsDispatchTable()->__hipReportDevices_fn != nullptr) {
    size_t numDevices = g_devices.size();

    std::vector<hipUUID> uuids(numDevices);

    int i = 0;
    for (const auto& dev : g_devices) {
      auto* deviceHandle = dev->devices()[0];
      const auto& info = deviceHandle->info();
      memcpy(uuids[i].bytes, info.uuid_, sizeof(info.uuid_));
      // if assert fails, the memcpy bytes param needs to be addressed
      static_assert(sizeof(info.uuid_) == sizeof(uuids[0].bytes), "error ABI issue");
      ++i;
    }

    hip::GetHipToolsDispatchTable()->__hipReportDevices_fn(numDevices, uuids.data());
  }

  amd::Context* hContext = new amd::Context(devices, amd::Context::Info());
  if (!hContext) {
    *status = false;
    return;
  }

  if (CL_SUCCESS != hContext->create(nullptr)) {
    hContext->release();
  }
  host_context = hContext;
  amd::RuntimeTearDown::RegisterObject(hContext);

  PlatformState::instance().init();
  *status = true;
  return;
}

Device* getCurrentDevice() { return tls.device_; }

void setCurrentDevice(unsigned int index) {
  assert(index < g_devices.size());
  tls.device_ = g_devices[index];
  uint32_t preferredNumaNode = (tls.device_)->devices()[0]->getPreferredNumaNode();
  amd::Os::setPreferredNumaNode(preferredNumaNode);
}

hip::Stream* getStream(hipStream_t stream, bool wait) {
  if (stream == nullptr || stream == hipStreamLegacy) {
    return getNullStream(wait);
  } else {
    hip::Stream* hip_stream = reinterpret_cast<hip::Stream*>(stream);
    if (wait && !(hip_stream->Flags() & hipStreamNonBlocking)) {
      constexpr bool WaitNullStreamOnly = true;
      hip_stream->GetDevice()->WaitActiveStreams(hip_stream, WaitNullStreamOnly);
    }
    return hip_stream;
  }
}

// ================================================================================================
hip::Stream* getNullStream(amd::Context& ctx, bool wait) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->NullStream(wait);
    }
  }
  // If it's a pure SVM allocation with system memory access, then it shouldn't matter which device
  // runtime selects by default
  if (hip::host_context == &ctx) {
    // Return current...
    return getNullStream(wait);
  }
  return nullptr;
}

// ================================================================================================
int getDeviceID(amd::Context& ctx) {
  for (auto& it : g_devices) {
    if (it->asContext() == &ctx) {
      return it->deviceId();
    }
  }
  return -1;
}

// ================================================================================================
hip::Stream* getNullStream(bool wait) {
  Device* device = getCurrentDevice();
  if (device == nullptr) {
    LogError("Invalid device");
  }
  return device ? device->NullStream(wait) : nullptr;
}

hipError_t hipInit(unsigned int flags) {
  HIP_INIT_API(hipInit, flags);

  if (flags != 0) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxCreate(hipCtx_t* ctx, unsigned int flags, hipDevice_t device) {
  HIP_INIT_API(hipCtxCreate, ctx, flags, device);

  if (static_cast<size_t>(device) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *ctx = reinterpret_cast<hipCtx_t>(g_devices[device]);

  // Increment ref count for device primary context
  g_devices[device]->retain();
  g_devices[device]->setFlags(flags);
  tls.ctxt_stack_.push(g_devices[device]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxSetCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxSetCurrent, ctx);

  if (ctx == nullptr) {
    if (!tls.ctxt_stack_.empty()) {
      tls.ctxt_stack_.pop();
    }
  } else {
    hip::tls.device_ = reinterpret_cast<hip::Device*>(ctx);
    if (!tls.ctxt_stack_.empty()) {
      tls.ctxt_stack_.pop();
    }
    tls.ctxt_stack_.push(hip::getCurrentDevice());
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxGetCurrent, ctx);

  *ctx = reinterpret_cast<hipCtx_t>(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetSharedMemConfig(hipSharedMemConfig* pConfig) {
  HIP_INIT_API(hipCtxGetSharedMemConfig, pConfig);

  *pConfig = hipSharedMemBankSizeFourByte;

  HIP_RETURN(hipSuccess);
}

hipError_t hipRuntimeGetVersion(int* runtimeVersion) {
  HIP_INIT_API_NO_RETURN(hipRuntimeGetVersion, runtimeVersion);

  if (!runtimeVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // HIP_VERSION = HIP_VERSION_MAJOR*100 + HIP_MINOR_VERSION
  *runtimeVersion = HIP_VERSION;

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxDestroy(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxDestroy, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // Need to remove the ctx of calling thread if its the top one
  if (!tls.ctxt_stack_.empty() && tls.ctxt_stack_.top() == dev) {
    tls.ctxt_stack_.pop();
  }

  // Remove context from global context list
  for (unsigned int i = 0; i < g_devices.size(); i++) {
    if (g_devices[i] == dev) {
      // Decrement ref count for device primary context
      dev->release();
    }
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPopCurrent(hipCtx_t* ctx) {
  HIP_INIT_API(hipCtxPopCurrent, ctx);

  hip::Device** dev = reinterpret_cast<hip::Device**>(ctx);
  if (!tls.ctxt_stack_.empty()) {
    if (dev != nullptr) {
      *dev = tls.ctxt_stack_.top();
    }
    tls.ctxt_stack_.pop();
  } else {
    DevLogError("Context Stack empty");
    HIP_RETURN(hipErrorInvalidContext);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxPushCurrent(hipCtx_t ctx) {
  HIP_INIT_API(hipCtxPushCurrent, ctx);

  hip::Device* dev = reinterpret_cast<hip::Device*>(ctx);
  if (dev == nullptr) {
    HIP_RETURN(hipErrorInvalidContext);
  }

  hip::tls.device_ = dev;
  tls.ctxt_stack_.push(hip::getCurrentDevice());

  HIP_RETURN(hipSuccess);
}

hipError_t hipDriverGetVersion(int* driverVersion) {
  HIP_INIT_API_NO_RETURN(hipDriverGetVersion, driverVersion);

  if (!driverVersion) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  // HIP_VERSION = HIP_VERSION_MAJOR*100 + HIP_MINOR_VERSION
  *driverVersion = HIP_VERSION;

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxGetDevice(hipDevice_t* device) {
  HIP_INIT_API(hipCtxGetDevice, device);

  if (device != nullptr) {
    *device = hip::getCurrentDevice()->deviceId();
    HIP_RETURN(hipSuccess);
  } else {
    HIP_RETURN(hipErrorInvalidValue);
  }

  HIP_RETURN(hipErrorInvalidContext);
}

hipError_t hipCtxGetApiVersion(hipCtx_t ctx, unsigned int* apiVersion) {
  HIP_INIT_API(hipCtxGetApiVersion, apiVersion);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetCacheConfig(hipFuncCache_t* cacheConfig) {
  HIP_INIT_API(hipCtxGetCacheConfig, cacheConfig);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetCacheConfig(hipFuncCache_t cacheConfig) {
  HIP_INIT_API(hipCtxSetCacheConfig, cacheConfig);

  if (cacheConfig != hipFuncCachePreferNone && cacheConfig != hipFuncCachePreferShared &&
      cacheConfig != hipFuncCachePreferL1 && cacheConfig != hipFuncCachePreferEqual) {
    HIP_RETURN(hipErrorInvalidValue);
  }
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSetSharedMemConfig(hipSharedMemConfig config) {
  HIP_INIT_API(hipCtxSetSharedMemConfig, config);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxSynchronize(void) {
  HIP_INIT_API(hipCtxSynchronize, 1);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipCtxGetFlags(unsigned int* flags) {
  HIP_INIT_API(hipCtxGetFlags, flags);
  HIP_RETURN(hipErrorNotSupported);
}

hipError_t hipDevicePrimaryCtxGetState(hipDevice_t dev, unsigned int* flags, int* active) {
  HIP_INIT_API(hipDevicePrimaryCtxGetState, dev, flags, active);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  if (flags != nullptr) {
    *flags = 0;
  }

  if (active != nullptr) {
    *active = g_devices[dev]->GetActiveStatus() ? 1 : 0;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRelease(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRelease, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxRetain(hipCtx_t* pctx, hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxRetain, pctx, dev);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  }
  if (pctx == nullptr) {
    HIP_RETURN(hipErrorInvalidValue);
  }

  *pctx = reinterpret_cast<hipCtx_t>(g_devices[dev]);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxReset(hipDevice_t dev) {
  HIP_INIT_API(hipDevicePrimaryCtxReset, dev);

  HIP_RETURN(hipSuccess);
}

hipError_t hipDevicePrimaryCtxSetFlags(hipDevice_t dev, unsigned int flags) {
  HIP_INIT_API(hipDevicePrimaryCtxSetFlags, dev, flags);

  if (static_cast<unsigned int>(dev) >= g_devices.size()) {
    HIP_RETURN(hipErrorInvalidDevice);
  } else {
    HIP_RETURN(hipErrorContextAlreadyInUse);
  }
}
}  // namespace hip
