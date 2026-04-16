//===-- HSAInterceptor.cpp - HSA Queue Interception -------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of HSA queue interception for dispatch modification.
///
/// The shared library is loaded via `LD_PRELOAD`, so we resolve the AMD HSA
/// extension table after the real `hsa_init()` completes and then interpose
/// `hsa_queue_create()`/`hsa_shut_down()` directly.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/HSAInterceptor.h"
#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/TracingEngine.h"
#include "llvm/Support/Error.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <rocprofiler-register/rocprofiler-register.h>
#endif

#include <atomic>
#include <cstring>
#include <dlfcn.h>
#include <string>

using namespace llvm;

namespace aegisbit {

#ifdef AEGISBIT_HAS_GPU

struct ApiTableVersion {
  uint32_t major_id;
  uint32_t minor_id;
  uint32_t step_id;
  uint32_t reserved;
};

constexpr size_t kQueueInterceptCreateIndex = 37;
constexpr size_t kQueueInterceptRegisterIndex = 38;
constexpr size_t kQueueCreateIndex = 8;
constexpr size_t kQueueInterceptCreateOffset =
    sizeof(ApiTableVersion) + (kQueueInterceptCreateIndex * sizeof(void*));
constexpr size_t kQueueInterceptRegisterOffset =
    sizeof(ApiTableVersion) + (kQueueInterceptRegisterIndex * sizeof(void*));
constexpr size_t kQueueCreateOffset =
    sizeof(ApiTableVersion) + (kQueueCreateIndex * sizeof(void*));

struct HsaApiTable {
  ApiTableVersion version;
  void* core_;
  void* amd_ext_;
  void* finalizer_ext_;
  void* image_ext_;
  void* tools_;
  void* pc_sampling_ext_;
};

// Callback type for writing modified packets back to the queue
typedef void (*hsa_amd_queue_intercept_packet_writer)(
    const void* pkts, uint64_t pkt_count);

// Callback type for intercepting packets (matches hsa_api_trace.h lines 114-117)
typedef void (*hsa_amd_queue_intercept_handler)(
    const void* pkts, uint64_t pkt_count,
    uint64_t user_pkt_index, void* data,
    hsa_amd_queue_intercept_packet_writer writer);

using hsa_amd_queue_intercept_register_fn_t = hsa_status_t (*)(
    hsa_queue_t* queue,
    hsa_amd_queue_intercept_handler callback,
    void* user_data);

using hsa_amd_queue_intercept_create_fn_t = hsa_status_t (*)(
    hsa_agent_t agent_handle,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue);

using hsa_queue_create_fn_t = hsa_status_t (*)(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue);

struct InterceptState {
  std::atomic<bool> Installed{false};
  std::atomic<bool> ExtensionTableReady{false};
  DispatchModifyCallback Callback;
  HSAInterceptor::Stats Statistics;
  std::mutex Mutex;
  std::vector<hsa_queue_t*> TrackedQueues;

  hsa_amd_queue_intercept_create_fn_t QueueInterceptCreateFn = nullptr;
  hsa_amd_queue_intercept_register_fn_t QueueInterceptRegisterFn = nullptr;
  hsa_queue_create_fn_t OriginalQueueCreateFn = nullptr;
  hsa_queue_create_fn_t* QueueCreateSlot = nullptr;
};

InterceptState& getInterceptState() {
  static InterceptState S;
  return S;
}

void packetInterceptHandler(const void* Pkts, uint64_t PktCount,
                            uint64_t UserPktIndex, void* Data,
                            hsa_amd_queue_intercept_packet_writer Writer) {
  (void)UserPktIndex;  // Currently unused
  (void)Data;          // Currently unused
  HSAInterceptor::handlePacketWrite(Pkts, PktCount, 0, nullptr,
                                    reinterpret_cast<void*>(Writer));
}

namespace {

bool initializeExtensionTable() {
  return false;
}

bool captureApiTable(void* TablePtr) {
  auto& S = getInterceptState();
  std::lock_guard<std::mutex> Lock(S.Mutex);

  if (!TablePtr) {
    return false;
  }

  auto* HsaTable = static_cast<HsaApiTable*>(TablePtr);
  if (!HsaTable->amd_ext_) {
    RuntimeConfig::getInstance().log("HSA API table is missing AMD extension table");
    return false;
  }

  S.QueueInterceptCreateFn =
      *reinterpret_cast<hsa_amd_queue_intercept_create_fn_t*>(
          static_cast<uint8_t*>(HsaTable->amd_ext_) + kQueueInterceptCreateOffset);
  S.QueueInterceptRegisterFn =
      *reinterpret_cast<hsa_amd_queue_intercept_register_fn_t*>(
          static_cast<uint8_t*>(HsaTable->amd_ext_) + kQueueInterceptRegisterOffset);

  if (HsaTable->core_) {
    auto* CoreTable = static_cast<uint8_t*>(HsaTable->core_);
    S.QueueCreateSlot = reinterpret_cast<hsa_queue_create_fn_t*>(
        CoreTable + kQueueCreateOffset);
    S.OriginalQueueCreateFn = *S.QueueCreateSlot;
  }

  if (!S.QueueInterceptCreateFn || !S.QueueInterceptRegisterFn) {
    RuntimeConfig::getInstance().log(
        "HSA AMD extension table is missing queue intercept functions");
    return false;
  }

  S.ExtensionTableReady.store(true);
  RuntimeConfig::getInstance().log("HSA queue interception enabled");
  return true;
}

bool captureApiTableForInterpose(void* TablePtr) {
  return captureApiTable(TablePtr);
}

} // anonymous namespace

//===----------------------------------------------------------------------===//
// HSAInterceptor implementation
//===----------------------------------------------------------------------===//

Error HSAInterceptor::install() {
  auto& S = getInterceptState();
  bool AlreadyInstalled = S.Installed.exchange(true);
  if (AlreadyInstalled) {
    if (!S.ExtensionTableReady.load()) {
      initializeExtensionTable();
    }
    return Error::success();
  }
  initializeExtensionTable();
  RuntimeConfig::getInstance().log("HSA interceptor armed");
  return Error::success();
}

void HSAInterceptor::uninstall() {
  auto& S = getInterceptState();

  if (!S.Installed.load()) {
    return;
  }

  {
    std::lock_guard<std::mutex> Lock(S.Mutex);
    S.Callback = nullptr;
    S.TrackedQueues.clear();
    S.QueueInterceptCreateFn = nullptr;
    S.QueueInterceptRegisterFn = nullptr;
    S.OriginalQueueCreateFn = nullptr;
    S.QueueCreateSlot = nullptr;
  }

  S.ExtensionTableReady.store(false);
  S.Installed.store(false);
  RuntimeConfig::getInstance().log("HSA interceptor uninstalled");
}

bool HSAInterceptor::isInstalled() {
  return getInterceptState().Installed.load();
}

void HSAInterceptor::setDispatchCallback(DispatchModifyCallback Callback) {
  auto& S = getInterceptState();
  std::lock_guard<std::mutex> Lock(S.Mutex);
  S.Callback = std::move(Callback);
}

void HSAInterceptor::clearDispatchCallback() {
  auto& S = getInterceptState();
  std::lock_guard<std::mutex> Lock(S.Mutex);
  S.Callback = nullptr;
}

HSAInterceptor::Stats HSAInterceptor::getStats() {
  auto& S = getInterceptState();
  std::lock_guard<std::mutex> Lock(S.Mutex);
  return S.Statistics;
}

void HSAInterceptor::resetStats() {
  auto& S = getInterceptState();
  std::lock_guard<std::mutex> Lock(S.Mutex);
  S.Statistics = Stats{};
}

Error HSAInterceptor::registerQueueIntercept(hsa_queue_t* Queue) {
  if (!Queue) {
    return createStringError(inconvertibleErrorCode(), "Null queue");
  }

  auto& S = getInterceptState();

  if (!S.ExtensionTableReady.load() || !S.QueueInterceptRegisterFn) {
    return createStringError(inconvertibleErrorCode(),
                             "Queue intercept API not available");
  }

  hsa_status_t Status = S.QueueInterceptRegisterFn(
      Queue,
      packetInterceptHandler,
      nullptr);

  if (Status != HSA_STATUS_SUCCESS) {
    return createStringError(inconvertibleErrorCode(),
                             "Failed to register queue intercept");
  }

  {
    std::lock_guard<std::mutex> Lock(S.Mutex);
    S.TrackedQueues.push_back(Queue);
  }

  RuntimeConfig::getInstance().log("Queue intercept registered");

  return Error::success();
}

void HSAInterceptor::handlePacketWrite(
    const void* Packets,
    uint64_t PacketCount,
    uint64_t /*UserData*/,
    void* /*CallbackData*/,
    void* WriterPtr) {

  auto& S = getInterceptState();
  auto Writer = reinterpret_cast<hsa_amd_queue_intercept_packet_writer>(WriterPtr);

  const auto* AQLPackets = static_cast<const hsa_kernel_dispatch_packet_t*>(Packets);

  // Process each packet
  for (uint64_t i = 0; i < PacketCount; ++i) {
    const hsa_kernel_dispatch_packet_t& Packet = AQLPackets[i];

    // Check if this is a kernel dispatch packet (type in header bits 0-7)
    uint8_t PacketType = Packet.header & 0xFF;
    if (PacketType != HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      // Not a kernel dispatch, pass through unchanged
      if (Writer) {
        Writer(&Packet, 1);
      }
      continue;
    }

    {
      std::lock_guard<std::mutex> Lock(S.Mutex);
      S.Statistics.TotalDispatches++;
    }

    // Get callback
    DispatchModifyCallback Callback;
    {
      std::lock_guard<std::mutex> Lock(S.Mutex);
      Callback = S.Callback;
    }

    if (!Callback) {
      // No callback, pass through unchanged
      if (Writer) {
        Writer(&Packet, 1);
      }
      continue;
    }

    // Make a modifiable copy of the packet
    hsa_kernel_dispatch_packet_t ModifiedPacket = Packet;

    // Extract kernel info from packet
    uint64_t KernelObject = Packet.kernel_object;
    void* Kernarg = reinterpret_cast<void*>(Packet.kernarg_address);
    uint32_t KernargSize = 0; // Will be looked up from kernel symbol

    // Call the callback
    bool Proceed = Callback(
        nullptr,  // Queue not available in this context
        &ModifiedPacket,
        KernelObject,
        Kernarg,
        KernargSize);

    if (Proceed) {
      // Write the (potentially modified) packet
      if (Writer) {
        Writer(&ModifiedPacket, 1);
      }
      {
        std::lock_guard<std::mutex> Lock(S.Mutex);
        // Check if packet was actually modified
        if (ModifiedPacket.kernel_object != Packet.kernel_object ||
            ModifiedPacket.kernarg_address != Packet.kernarg_address) {
          S.Statistics.ModifiedDispatches++;
        }
      }
    } else {
      // Skip this dispatch
      {
        std::lock_guard<std::mutex> Lock(S.Mutex);
        S.Statistics.SkippedDispatches++;
      }
    }
  }
}

#else // !AEGISBIT_HAS_GPU

// Stub implementations when GPU support is not available

Error HSAInterceptor::install() {
  return createStringError(inconvertibleErrorCode(),
                           "HSA support not available (built without GPU)");
}

void HSAInterceptor::uninstall() {}

bool HSAInterceptor::isInstalled() { return false; }

void HSAInterceptor::setDispatchCallback(DispatchModifyCallback) {}

void HSAInterceptor::clearDispatchCallback() {}

HSAInterceptor::Stats HSAInterceptor::getStats() { return Stats{}; }

void HSAInterceptor::resetStats() {}

Error HSAInterceptor::registerQueueIntercept(hsa_queue_t*) {
  return createStringError(inconvertibleErrorCode(),
                           "HSA support not available (built without GPU)");
}

void HSAInterceptor::handlePacketWrite(
    const void*, uint64_t, uint64_t, void*, void*) {}

#endif // AEGISBIT_HAS_GPU

} // namespace aegisbit

#ifdef AEGISBIT_HAS_GPU
using hsa_queue_create_t = hsa_status_t (*)(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue);

using hsa_init_t = hsa_status_t (*)();
using hsa_shut_down_t = hsa_status_t (*)();
using rocprofiler_register_library_api_table_t = decltype(&rocprofiler_register_library_api_table);

static hsa_queue_create_t getRealHsaQueueCreate() {
  static hsa_queue_create_t RealFn = nullptr;
  if (!RealFn) {
    RealFn = reinterpret_cast<hsa_queue_create_t>(dlsym(RTLD_NEXT, "hsa_queue_create"));
  }
  return RealFn;
}

static hsa_init_t getRealHsaInit() {
  static hsa_init_t RealFn = nullptr;
  if (!RealFn) {
    RealFn = reinterpret_cast<hsa_init_t>(dlsym(RTLD_NEXT, "hsa_init"));
  }
  return RealFn;
}

static hsa_shut_down_t getRealHsaShutDown() {
  static hsa_shut_down_t RealFn = nullptr;
  if (!RealFn) {
    RealFn = reinterpret_cast<hsa_shut_down_t>(dlsym(RTLD_NEXT, "hsa_shut_down"));
  }
  return RealFn;
}

static rocprofiler_register_library_api_table_t
getRealRegisterLibraryApiTable() {
  static rocprofiler_register_library_api_table_t RealFn = nullptr;
  if (!RealFn) {
    RealFn = reinterpret_cast<rocprofiler_register_library_api_table_t>(
        dlsym(RTLD_NEXT, "rocprofiler_register_library_api_table"));
  }
  return RealFn;
}

extern "C" {

hsa_status_t hsa_init() {
  auto RealFn = getRealHsaInit();
  if (!RealFn) {
    return HSA_STATUS_ERROR;
  }

  hsa_status_t Status = RealFn();
  if (Status == HSA_STATUS_SUCCESS) {
    if (auto Err = aegisbit::HSAInterceptor::install()) {
      llvm::consumeError(std::move(Err));
    }
  }
  return Status;
}

rocprofiler_register_error_code_t rocprofiler_register_library_api_table(
    const char* lib_name, rocprofiler_register_import_func_t import_func,
    uint32_t lib_version, void** api_tables, uint64_t api_table_length,
    rocprofiler_register_library_indentifier_t* register_id) {
  auto RealFn = getRealRegisterLibraryApiTable();

  if (lib_name && api_table_length > 0 &&
      std::string(lib_name).find("hsa") != std::string::npos) {
    aegisbit::captureApiTableForInterpose(api_tables[0]);
  }

  if (!RealFn) {
    return ROCP_REG_NO_TOOLS;
  }
  return RealFn(lib_name, import_func, lib_version, api_tables, api_table_length,
                register_id);
}

hsa_status_t hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue) {

  auto& Config = aegisbit::RuntimeConfig::getInstance();
  auto RealFn = getRealHsaQueueCreate();
  if (!RealFn) {
    return HSA_STATUS_ERROR;
  }

  auto& S = aegisbit::getInterceptState();
  if (!Config.Enabled || !S.Installed.load() || !S.ExtensionTableReady.load() ||
      !S.QueueInterceptCreateFn || !S.QueueInterceptRegisterFn) {
    return RealFn(agent, size, type, callback, data,
                  private_segment_size, group_segment_size, queue);
  }

  hsa_status_t Status = S.QueueInterceptCreateFn(
      agent, size, type, callback, data, private_segment_size,
      group_segment_size, queue);
  if (Status != HSA_STATUS_SUCCESS) {
    aegisbit::RuntimeConfig::getInstance().log(
        "hsa_amd_queue_intercept_create failed (status=" +
        std::to_string(Status) + "), falling back to hsa_queue_create");
    return RealFn(agent, size, type, callback, data,
                  private_segment_size, group_segment_size, queue);
  }

  Status = S.QueueInterceptRegisterFn(*queue, aegisbit::packetInterceptHandler,
                                      nullptr);
  if (Status != HSA_STATUS_SUCCESS) {
    aegisbit::RuntimeConfig::getInstance().log(
        "hsa_amd_queue_intercept_register failed (status=" +
        std::to_string(Status) + ")");
  } else {
    std::lock_guard<std::mutex> Lock(S.Mutex);
    S.TrackedQueues.push_back(*queue);
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t hsa_shut_down() {
  auto RealFn = getRealHsaShutDown();
  if (!RealFn) {
    return HSA_STATUS_ERROR;
  }

  aegisbit::DispatchInterceptor::finalize();
  return RealFn();
}

} // extern "C"

#endif // AEGISBIT_HAS_GPU
