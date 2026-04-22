/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime.h>

#include <atomic>

#include "hip_internal.hpp"
#include "hip_platform.hpp"
#include "platform/runtime.hpp"
#include "rocclr/utils/flags.hpp"
#include "rocclr/utils/versions.hpp"
#include "rocclr/os/os.hpp"

#include <hip/amd_detail/hip_api_trace.hpp>

#if defined(HIP_ROCPROFILER_REGISTER) && HIP_ROCPROFILER_REGISTER > 0 && \
    defined(HIP_ENABLE_AQLMON_RUNTIME_CONTRACT) && HIP_ENABLE_AQLMON_RUNTIME_CONTRACT > 0
#include <aqlmon/runtime_contract.h>
#include <rocprofiler-register/rocprofiler-register.h>
#endif

namespace hip {
const HipToolsDispatchTable* GetHipToolsDispatchTable();
}  // namespace hip

namespace {

std::atomic<bool> g_hipAqlmonRuntimeProvidesKernelCompletionSignals{false};

#if defined(HIP_ROCPROFILER_REGISTER) && HIP_ROCPROFILER_REGISTER > 0 && \
    defined(HIP_ENABLE_AQLMON_RUNTIME_CONTRACT) && HIP_ENABLE_AQLMON_RUNTIME_CONTRACT > 0

constexpr uint32_t kHipAqlmonRequestedCapabilities =
    AQLMON_COMPLETION_SIGNAL_CAP_KERNEL_DISPATCH_SIGNALS;

const char* toolActivationModeString(rocprofiler_register_tool_activation_mode_t mode) {
  switch (mode) {
    case ROCP_REG_TOOL_ACTIVATION_STARTUP: return "startup";
    case ROCP_REG_TOOL_ACTIVATION_ATTACH: return "attach";
    case ROCP_REG_TOOL_ACTIVATION_NONE: return "none";
  }

  return "unknown";
}

void negotiateHipAqlmonCompletionSignals(rocprofiler_register_tool_activation_mode_t mode,
                                         void*) {
  auto request = aqlmon_runtime_negotiation_request_t{};
  request.size = sizeof(request);
  request.abi_version = AQLMON_RUNTIME_CONTRACT_ABI_VERSION;
  request.proposed_mode = AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED;
  request.proposed_capabilities = kHipAqlmonRequestedCapabilities;

  auto response = aqlmon_runtime_negotiation_response_t{};
  response.size = sizeof(response);
  response.abi_version = AQLMON_RUNTIME_CONTRACT_ABI_VERSION;

  const auto status = aqlmon_runtime_negotiate(&request, &response);
  const bool runtime_provided =
      status == AQLMON_STATUS_SUCCESS &&
      response.selected_mode == AQLMON_COMPLETION_SIGNAL_MODE_RUNTIME_PROVIDED &&
      (response.granted_capabilities & kHipAqlmonRequestedCapabilities) ==
          kHipAqlmonRequestedCapabilities;

  g_hipAqlmonRuntimeProvidesKernelCompletionSignals.store(runtime_provided,
                                                          std::memory_order_relaxed);

  ClPrint(amd::LOG_INFO, amd::LOG_INIT,
          "AQLMON completion-signal negotiation via %s: status=%u, selected_mode=%u, "
          "granted_capabilities=0x%x",
          toolActivationModeString(mode), static_cast<unsigned>(status),
          static_cast<unsigned>(response.selected_mode), response.granted_capabilities);
}

void registerHipAqlmonCompletionSignalCallback() {
  const auto status = rocprofiler_register_runtime_tool_activation_callback(
      "hip", &negotiateHipAqlmonCompletionSignals, nullptr);
  if (status != ROCP_REG_SUCCESS) {
    ClPrint(amd::LOG_INFO, amd::LOG_INIT,
            "AQLMON completion-signal callback registration failed: %u",
            static_cast<unsigned>(status));
  }
}

#else

void registerHipAqlmonCompletionSignalCallback() {}

#endif

}  // namespace

namespace hip {
std::once_flag g_ihipInitialized;

std::vector<hip::Device*> g_devices ROCCLR_INIT_PRIORITY(101);
thread_local TlsAggregator tls;
amd::Context* host_context = nullptr;

// ================================================================================================
// init() is only to be called from the HIP_INIT macro only once
void init(bool* status) {
  // Configure HIP runtime mode
  amd::IS_HIP = true;
  GPU_NUM_MEM_DEPENDENCY = 0;
  // Initialize AMD runtime - critical for all subsequent operations
  if (!amd::Runtime::init()) {
    *status = false;
    return;
  }

  registerHipAqlmonCompletionSignalCallback();

  ClPrint(amd::LOG_INFO, amd::LOG_INIT, "HIP Version: %d.%d.%d, Direct Dispatch: %d",
          HIP_VERSION_MAJOR, HIP_VERSION_MINOR, HIP_VERSION_PATCH, AMD_DIRECT_DISPATCH);
  // Print the current path of the library
  amd::Os::PrintLibraryLocation();
  // Enumerate and initialize GPU devices
  const std::vector<amd::Device*>& devices = amd::Device::getDevices(CL_DEVICE_TYPE_GPU, false);
  const size_t device_count = devices.size();
  g_devices.reserve(device_count);

  for (size_t i = 0; i < device_count; ++i) {
    amd::Device* const amd_device = devices[i];
    amd_device->SetActiveWait(true);
    // Use the eternal contexts that already exist in amd::Device for the new hip::Device
    auto* device = new Device(&amd_device->context(), static_cast<unsigned int>(i));
    if (!device || !device->Create()) {
      *status = false;
      if (device) {
        device->release();
      }
      return;
    }
    g_devices.push_back(device);
    amd::RuntimeTearDown::RegisterObject(device);
  }

  // Register tool dispatch table to profiler v3.
  // If app is attached by profiler, __hipTriggerReportDevices_fn() will be called
  // by profiler.
  const auto* tools_dispatch_table = hip::GetHipToolsDispatchTable();
  tools_dispatch_table->__hipTriggerReportDevices_fn();

  // Create and initialize host context
  host_context = new amd::Context(devices, amd::Context::Info());
  if (!host_context || CL_SUCCESS != host_context->create(nullptr)) {
    if (host_context) {
      host_context->release();
    }
    *status = false;
    return;
  }

  amd::RuntimeTearDown::RegisterObject(host_context);

  // Complete platform initialization
  PlatformState::Instance().Init();
  *status = true;
}

bool aqlmonRuntimeProvidesKernelCompletionSignals() {
  return g_hipAqlmonRuntimeProvidesKernelCompletionSignals.load(std::memory_order_relaxed);
}

// ================================================================================================
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
    ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_API, "Context Stack empty");
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

void __hipTriggerReportDevices() {
  const auto* tools_dispatch_table = hip::GetHipToolsDispatchTable();
  if (tools_dispatch_table->__hipReportDevices_fn) {
    // If app is started or attached by profiler, __hipReportDevices_fn must be valid
    std::vector<hipUUID> uuids;
    uuids.reserve(g_devices.size());

    for (const auto* dev : g_devices) {
      const auto& info = dev->devices()[0]->info();
      static_assert(sizeof(info.cuid_) == sizeof(hipUUID::bytes), "UUID size mismatch");
      uuids.emplace_back();
      std::copy(std::begin(info.cuid_), std::end(info.cuid_), std::begin(uuids.back().bytes));
    }
    tools_dispatch_table->__hipReportDevices_fn(g_devices.size(), uuids.data());
  }
}
}  // namespace hip
