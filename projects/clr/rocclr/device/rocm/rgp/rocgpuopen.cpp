/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/rocm/rgp/rocgpuopen.hpp"
#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocvirtual.hpp"
#include "device/rocm/rockernel.hpp"

#ifdef ROC_GPUOPEN

#include "devDriverServer.h"
#include "protocols/driverControlServer.h"
#include "protocols/rgpServer.h"
#include "protocols/rgpProtocol.h"
#include "ddRpcServer.h"
#include "device/rocm/roccounters.hpp"
#include "device/rocm/rocblit.hpp"
#include "device/rocm/rgp/roctracesession.hpp"
#include "device/rocm/rgp/rocubertracesvc.hpp"
#include "device/rocm/rgp/rocdriverutils.hpp"
#include "utils/flags.hpp"  // amd::IS_HIP
#include "hsa_ven_amd_aqlprofile.h"

#include <elfio/elf_types.hpp>  // Elf64_Ehdr/Shdr/Sym + ELF constants (amd::ELFIO namespace)

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#if defined(_WIN32)
#include <malloc.h>   // _aligned_malloc / _aligned_free
#include <windows.h>  // QueryPerformanceCounter
#endif

// RGP file format structs — mirror of pal/shared/inc/sqtt_file_format.h.
// Kept inline to avoid a PAL header dependency in the ROC build.
namespace RgpFile {

#define SQTT_FILE_MAGIC_NUMBER         0x50303042u
#define RGP_FILE_FORMAT_SPEC_MAJOR_VER 1
#define RGP_FILE_FORMAT_SPEC_MINOR_VER 6

struct ChunkVersionNumbers { uint16_t major, minor; };

// Chunk type enum (values must match the spec — RGP tool uses them verbatim).
enum ChunkType : uint8_t {
  kAsicInfo          = 0,
  kSqttDesc          = 1,
  kSqttData          = 2,
  kApiInfo           = 3,
  kReserved          = 4,
  kQueueEventTimings = 5,
  kClockCalibration  = 6,
  kCpuInfo           = 7,
};

// Per-chunk version table (matches RgpChunkVersionNumberLookup in sqtt_file_format.h).
static constexpr ChunkVersionNumbers kChunkVersion[] = {
  {0, 6}, // kAsicInfo
  {0, 2}, // kSqttDesc
  {0, 0}, // kSqttData
  {0, 2}, // kApiInfo
  {0, 0}, // kReserved
  {1, 1}, // kQueueEventTimings
  {0, 0}, // kClockCalibration
  {0, 0}, // kCpuInfo
};

struct ChunkIdentifier {
  uint32_t chunkType  :  8;
  uint32_t chunkIndex :  8;
  uint32_t reserved   : 16;
};

struct ChunkHeader {
  ChunkIdentifier chunkIdentifier;
  uint16_t        minorVersion;
  uint16_t        majorVersion;
  int32_t         sizeInBytes;
  int32_t         padding;
};

static ChunkHeader MakeChunkHeader(ChunkType type, int index, int32_t sizeInBytes) {
  ChunkHeader h = {};
  h.chunkIdentifier.chunkType  = static_cast<uint32_t>(type);
  h.chunkIdentifier.chunkIndex = static_cast<uint32_t>(index);
  h.majorVersion               = kChunkVersion[type].major;
  h.minorVersion               = kChunkVersion[type].minor;
  h.sizeInBytes                = sizeInBytes;
  return h;
}

struct FileHeaderFlags {
  union {
    struct { int32_t isSemaphoreQueueTimingETW : 1; int32_t noQueueSemaphoreTimeStamps : 1; int32_t reserved : 30; };
    uint32_t value;
  };
};

struct FileHeader {
  uint32_t        magicNumber;
  uint32_t        versionMajor;
  uint32_t        versionMinor;
  FileHeaderFlags flags;
  int32_t         chunkOffset;
  int32_t         second, minute, hour, dayInMonth, month, year, dayInWeek, dayInYear, isDaylightSavings;
};

struct CpuInfo {
  ChunkHeader header;
  uint32_t    vendorId[4];
  uint32_t    processorBrand[12];
  uint32_t    reserved[2];
  uint64_t    cpuTimestampFrequency;
  uint32_t    clockSpeed;
  uint32_t    numLogicalCores;
  uint32_t    numPhysicalCores;
  uint32_t    systemRamSize;
};

enum GpuType  : uint32_t { kGpuUnknown = 0, kGpuIntegrated = 1, kGpuDiscrete = 2, kGpuVirtual = 3 };
enum MemType  : uint32_t { kMemUnknown = 0, kMemHbm = 0x20, kMemHbm2 = 0x21, kMemHbm3 = 0x22,
                           kMemHbm3e = 0x23, kMemGddr6 = 0x13 };
enum GfxIpLevel : uint32_t {
  kGfxNone  = 0x0, kGfx10_1 = 0x7, kGfx10_3 = 0x9,
  kGfx11_0  = 0xC, kGfx11_5 = 0xD, kGfx12   = 0x10,
};
enum SqttVersion : uint32_t {
  kSqttNone = 0x0, kSqtt3_0 = 0x7, kSqtt3_2 = 0xB, kSqtt3_3 = 0xC,
};

static constexpr uint32_t kGpuNameMaxSize           = 256;
static constexpr uint32_t kMaxNumSE                 = 32;
static constexpr uint32_t kSaPerSE                  = 2;
static constexpr uint32_t kActivePixelPackerDwords   = 4;

struct AsicInfo {
  ChunkHeader header;
  uint64_t    flags;
  uint64_t    traceShaderCoreClock;
  uint64_t    traceMemoryClock;
  int32_t     deviceId;
  int32_t     deviceRevisionId;
  int32_t     vgprsPerSimd;
  int32_t     sgprsPerSimd;
  int32_t     shaderEngines;
  int32_t     computeUnitPerShaderEngine;
  int32_t     simdPerComputeUnit;
  int32_t     wavefrontsPerSimd;
  int32_t     minimumVgprAlloc;
  int32_t     vgprAllocGranularity;
  int32_t     minimumSgprAlloc;
  int32_t     sgprAllocGranularity;
  int32_t     hardwareContexts;
  GpuType     gpuType;
  GfxIpLevel  gfxIpLevel;
  int32_t     gpuIndex;
  int32_t     gdsSize;
  int32_t     gdsPerShaderEngine;
  int32_t     ceRamSize;
  int32_t     ceRamSizeGraphics;
  int32_t     ceRamSizeCompute;
  int32_t     maxNumberOfDedicatedCus;
  int64_t     vramSize;
  int32_t     vramBusWidth;
  int32_t     l2CacheSize;
  int32_t     l1CacheSize;
  int32_t     ldsSize;
  char        gpuName[kGpuNameMaxSize];
  float       aluPerClock;
  float       texturePerClock;
  float       primsPerClock;
  float       pixelsPerClock;
  uint64_t    gpuTimestampFrequency;
  uint64_t    maxShaderCoreClock;
  uint64_t    maxMemoryClock;
  uint32_t    memoryOpsPerClock;
  MemType     memoryChipType;
  uint32_t    ldsGranularity;
  uint16_t    cuMask[kMaxNumSE][kSaPerSE];
  char        reserved1[128];
  uint32_t    activePixelPackerMask[kActivePixelPackerDwords];
  char        reserved2[16];
  uint32_t    gl1CacheSize;
  uint32_t    instructionCacheSize;
  uint32_t    scalarCacheSize;
  uint32_t    mallCacheSize;
};

enum ApiType : uint32_t { kApiDx12 = 0, kApiVulkan = 1, kApiGeneric = 2, kApiOpenCl = 3, kApiHip = 5 };
enum ProfilingMode : uint32_t { kProfilingModePresent = 0, kProfilingModeUserMarkers = 1,
                                kProfilingModeIndex = 2, kProfilingModeTag = 3 };
enum InstructionTraceMode : uint32_t { kInstrTraceDisabled = 0, kInstrTraceFullFrame = 1, kInstrTraceApiPso = 2 };

struct ApiInfo {
  ChunkHeader header;
  ApiType     apiType;
  uint16_t    versionMajor;
  uint16_t    versionMinor;
  ProfilingMode        profilingMode;
  uint32_t             reserved;
  uint8_t              profilingModeData[512];  // SqttProfilingModeData — zero-filled is fine for dispatch mode
  InstructionTraceMode instructionTraceMode;
  uint32_t             reserved2;
  uint64_t             instructionTraceData;   // SqttInstructionTraceData — zero = disabled
};

struct SqttDesc {
  ChunkHeader  header;
  int32_t      shaderEngineIndex;
  SqttVersion  sqttVersion;
  int16_t      instrumentationSpecVersion;
  int16_t      instrumentationApiVersion;
  int32_t      computeUnitIndex;
};

struct SqttData {
  ChunkHeader header;
  int32_t     offset;  // byte offset from start of file to the raw SQTT bytes
  int32_t     size;    // byte count of raw SQTT data
};

// Queue event timings chunk — written with zero records for the HIP compute path
// (which has no queue semaphore timestamps).  The file header flag
// noQueueSemaphoreTimeStamps=1 signals this to the loader, but the chunk itself
// is still mandatory per RgpFileIsChunkNecessaryForLoading().
struct QueueEventTimings {
  ChunkHeader header;
  uint32_t    queueInfoTableRecordCount;  // number of SqttQueueInfoRecord entries (0)
  uint32_t    queueInfoTableSize;         // byte size of info records (0)
  uint32_t    queueEventTableRecordCount; // number of SqttQueueEventRecord entries (0)
  uint32_t    queueEventTableSize;        // byte size of event records (0)
};

// Clock calibration chunk — provides a CPU/GPU timestamp pair so RGP can correlate
// the two clock domains.  For the HIP legacy path we supply a real CPU timestamp
// (HSA_SYSTEM_INFO_TIMESTAMP, in the same clock domain as the trace) and leave
// the GPU timestamp at 0 (timeline correlation is best-effort for compute).
struct ClockCalibration {
  ChunkHeader header;
  uint64_t    cpuTimestamp;
  uint64_t    gpuTimestamp;
  uint64_t    reserved;
};

// Map gfx major.minor.stepping → SqttGfxIpLevel and SqttVersion.
static GfxIpLevel GfxVerToIpLevel(uint32_t maj, uint32_t min) {
  if (maj == 10 && min == 1) return kGfx10_1;
  if (maj == 10 && min == 3) return kGfx10_3;
  if (maj == 11 && min == 0) return kGfx11_0;
  if (maj == 11 && min == 5) return kGfx11_5;
  if (maj == 12)             return kGfx12;
  return kGfxNone;
}
static SqttVersion GfxVerToSqttVersion(uint32_t maj, uint32_t min) {
  if (maj == 10)             return kSqtt3_0;
  if (maj == 11)             return kSqtt3_2;
  if (maj == 12)             return kSqtt3_3;
  return kSqttNone;
}

} // namespace RgpFile

// PM4 packet encoding constants (from soc15d.h / GFX9+ architecture spec)
namespace {

// PM4 Type-3 header encoding: type=3, opcode in bits[8:15], count in bits[16:29].
// 'count' is the number of body DWORDs minus 1 (i.e. total_dwords - 2).
constexpr uint32_t kPm4Type3Shift    = 30;
constexpr uint32_t kPm4OpcodeShift   = 8;
constexpr uint32_t kPm4CountShift    = 16;
constexpr uint32_t kPacket3Nop       = 0x10;  // IT_NOP opcode
constexpr uint32_t kPacket3Indirect  = 0x3F;  // IT_INDIRECT_BUFFER opcode

inline uint32_t Pm4Header(uint32_t opcode, uint32_t body_dwords) {
  // count field = total_dwords - 2 = body_dwords - 1
  return (3u << kPm4Type3Shift) | (opcode << kPm4OpcodeShift) |
         ((body_dwords - 1) << kPm4CountShift);
}

// FNV-1a 64-bit hash — used in AddElfBinary() as a stable, dependency-free alternative to
// DevDriver::MetroHash::MetroHash64().  Produces the same role: a content-derived handle used
// for PSO/ELF correlation in the RGP trace stream.
inline uint64_t Fnv1a64(const void* data, size_t size) {
  constexpr uint64_t kFnvBasis = 0xcbf29ce484222325ull;
  constexpr uint64_t kFnvPrime = 0x100000001b3ull;
  uint64_t hash = kFnvBasis;
  const auto* bytes = reinterpret_cast<const uint8_t*>(data);
  for (size_t i = 0; i < size; ++i) {
    hash ^= static_cast<uint64_t>(bytes[i]);
    hash *= kFnvPrime;
  }
  return hash;
}

// Maximum PM4 payload (marker bytes) that fits directly in hsa_ext_amd_aql_pm4_packet_t.
// The packet has pm4_command[27] (27 × uint16 = 54 bytes). The NOP header consumes one
// DWORD, leaving 52 bytes (13 DWORDs) for marker payload.
constexpr size_t kMaxInlinePm4PayloadBytes = 52;

// Returns a monotonic timestamp in the same units as HSA_SYSTEM_INFO_TIMESTAMP.
// On Windows, HSA's ReadSystemClock() is unimplemented (aborts), so we use
// QueryPerformanceCounter directly — it is the same underlying clock.
inline uint64_t GetSystemTimestamp() {
#if defined(_WIN32)
  LARGE_INTEGER counter;
  QueryPerformanceCounter(&counter);
  return static_cast<uint64_t>(counter.QuadPart);
#else
  uint64_t ts = 0;
  Hsa::system_get_info(HSA_SYSTEM_INFO_TIMESTAMP, &ts);
  return ts;
#endif
}

// Number of slots in the marker IB ring buffer.  Each WriteMarker call uses the next slot
// (wrapping), so up to kMarkerRingSize submissions can be in-flight before a slot is reused.
// dispatchCounterAqlPacket(blocking=true) serialises callers, so in practice only one slot
// is ever in-flight; the ring provides safety margin against future non-blocking use.
constexpr uint32_t kMarkerRingSize = 16384;
constexpr size_t   kMarkerSlotBytes = sizeof(uint32_t) + kMaxInlinePm4PayloadBytes;

// ROCm-compatible allocator callbacks for DevDriverServer.
// Delegates directly to the global CRT heap; DevDriver manages its own lifetimes.
void* DevDriverAlloc(void* /*pUserdata*/, size_t size, size_t alignment, bool zero) {
#if defined(_WIN32)
  void* ptr = _aligned_malloc(size, alignment);
#else
  void* ptr = nullptr;
  if (posix_memalign(&ptr, alignment, size) != 0) {
    ptr = nullptr;
  }
#endif
  if (ptr != nullptr && zero) {
    memset(ptr, 0, size);
  }
  return ptr;
}

void DevDriverFree(void* /*pUserdata*/, void* pMemory) {
#if defined(_WIN32)
  _aligned_free(pMemory);
#else
  free(pMemory);
#endif
}

}  // namespace

namespace amd::roc {

// ================================================================================================
RocUberTraceCaptureMgr::RocUberTraceCaptureMgr(Device* device)
    : device_(device),
      dev_driver_server_(nullptr),
      rgp_server_(nullptr),
      roc_trace_session_(nullptr),
      uber_trace_svc_(nullptr),
      driver_utils_svc_(nullptr),
      rpc_server_(DD_API_INVALID_HANDLE),
      global_disp_count_(1),  // Must start from 1 per RGP spec
      current_event_id_(0),
      user_event_(nullptr),
      trace_running_(false),
      sqtt_api_({}),
      sqtt_profile_({}),
      sqtt_start_packet_({}),
      sqtt_stop_packet_({}),
      sqtt_read_packet_({}),
      sqtt_cmd_buf_(nullptr),
      sqtt_cmd_buf_size_(0),
      sqtt_output_(nullptr),
      sqtt_output_size_(0),
      marker_cmd_buf_(nullptr),
      marker_buf_idx_(0),
      prep_disp_count_(0),
      num_prep_frames_(2),
      sqtt_se_mask_(0),
      sqtt_instruction_tokens_(false),
      sqtt_capture_code_objects_(true),
      capture_start_index_(0),
      capture_stop_index_(0),
      capture_index_mode_(false),
      sqtt_state_(SqttState::Idle),
      trace_gpu_(nullptr),
      pending_abort_(false) {}

// ================================================================================================
RocUberTraceCaptureMgr::~RocUberTraceCaptureMgr() { Destroy(); }

// ================================================================================================
// Factory: creates and initialises the UberTrace capture manager for the ROCm device path.
RocUberTraceCaptureMgr* RocUberTraceCaptureMgr::Create(Device* device) {
  RocUberTraceCaptureMgr* mgr = new RocUberTraceCaptureMgr(device);

  if (mgr != nullptr && !mgr->Init()) {
    delete mgr;
    mgr = nullptr;
  }

  return mgr;
}

// ================================================================================================
bool RocUberTraceCaptureMgr::Init() {
  // Build the allocator callback using the ROCm-specific heap delegates defined above.
  DevDriver::AllocCb allocCb = {};
  allocCb.pUserdata = nullptr;
  allocCb.pfnAlloc  = DevDriverAlloc;
  allocCb.pfnFree   = DevDriverFree;

  // Configure the server: compute driver component, local named-pipe transport,
  // DriverControl + RGP + Event protocol servers enabled.
  DevDriver::ServerCreateInfo createInfo = {};
  createInfo.componentType     = DevDriver::Component::Driver;
  createInfo.createUpdateThread = true;
  // Use the same session name strings as PAL's GetClientApiStr() so that RGP/RDP
  // tools can identify the driver on the DevDriver bus.
  const char* clientDesc = amd::IS_HIP ? "AMD HIP Driver" : "AMD OpenCL Driver";
  strncpy(createInfo.clientDescription, clientDesc,
          sizeof(createInfo.clientDescription) - 1);

  // Mirror PAL's Platform::EarlyInitDevDriver(): default to the named-pipe transport.
  // kMessageBus (UWP/KMD message bus) is only used by PAL when DD_ENABLE_UWP_TRANSPORT
  // is defined and the named-pipe connection is unavailable — not applicable here.
  DevDriver::HostInfo hostInfo = DevDriver::kDefaultNamedPipe;

  // Bail early if no developer-tools host is listening, matching PAL's
  // IsConnectionAvailable() check before allocating the server.
  if (!DevDriver::DevDriverServer::IsConnectionAvailable(hostInfo)) {
    return false;
  }

  createInfo.connectionInfo = hostInfo;
  createInfo.servers.driverControl = 1;
  createInfo.servers.rgp           = 1;

  dev_driver_server_ = new DevDriver::DevDriverServer(allocCb, createInfo);
  if (dev_driver_server_ == nullptr) {
    return false;
  }

  if (dev_driver_server_->Initialize() != DevDriver::Result::Success) {
    delete dev_driver_server_;
    dev_driver_server_ = nullptr;
    return false;
  }

  // Mirror PAL's EarlyInitDevDriver(): verify a tool with DeveloperModeEnabled is present
  // on the bus before proceeding.  Without this, we race ahead of the tool and
  // StartEarlyDeviceInit() fires before the tool has sent EnableProfilingRequest.
  // If no tool is found, tear down and return false — same as PAL's early-exit path.
  {
    DevDriver::IMsgChannel* pMsgChannel = dev_driver_server_->GetMessageChannel();
    DevDriver::ClientId clientId = DevDriver::kBroadcastClientId;
    DevDriver::ClientMetadata filter = {};
    filter.clientType = DevDriver::Component::Tool;
    filter.status     = static_cast<DevDriver::StatusFlags>(
        DevDriver::ClientStatusFlags::DeveloperModeEnabled);

    if (pMsgChannel->FindFirstClient(filter, &clientId,
                                     DevDriver::kFindClientTimeout, &filter)
        != DevDriver::Result::Success) {
      delete dev_driver_server_;
      dev_driver_server_ = nullptr;
      return false;
    }

    // The FindFirstClient call above (DeveloperModeEnabled) already blocks up to
    // kFindClientTimeout ms, giving the tool's connection thread sufficient time to
    // open its DriverControl session and set PlatformHaltOnConnect before
    // StartEarlyDeviceInit() → DiscoverHaltRequests() runs.  A second FindFirstClient
    // here is redundant and causes a deadlock: it pumps messages on the receive thread
    // while m_sessionMutex may already be held by HandleReceivedSessionMessage, so a
    // transport error in Forward() → Disconnect() → ShutDownAllSessions() re-enters
    // the non-recursive mutex and hangs.
  }

  // Retrieve the RGP server.  It is owned by DevDriverServer; we hold a non-owning pointer.
  rgp_server_ = dev_driver_server_->GetRGPServer();
  if (rgp_server_ == nullptr) {
    delete dev_driver_server_;
    dev_driver_server_ = nullptr;
    return false;
  }

  user_event_ = new RgpSqttMarkerUserEventWithString;
  if (user_event_ == nullptr) {
    delete dev_driver_server_;
    dev_driver_server_ = nullptr;
    rgp_server_ = nullptr;
    return false;
  }

  // ── Mirror PAL's EarlyInitDevDriver() + RegisterRpcServices() sequence ────────
  // PAL NEVER calls DevDriverServer::Finalize().  That function sets m_isFinalized=true
  // on all protocol servers, permanently closing the RGP EnableProfilingRequest window.
  // Instead PAL registers the DriverControlServer directly on the message channel
  // (platform.cpp RegisterRpcServices() line 1258) and then calls StartEarlyDeviceInit().
  // The tool sends EnableProfilingRequest during the halt inside StartEarlyDeviceInit(),
  // while m_isFinalized is still false.  So we must follow the same pattern.

  // 1. Create and initialise RocTraceSession (RDF state machine).
  roc_trace_session_ = new ::roc::RocTraceSession();
  if (roc_trace_session_ == nullptr || !roc_trace_session_->Init()) {
    delete roc_trace_session_;
    roc_trace_session_ = nullptr;
    // Non-fatal: fall back to legacy RGP-only path.
  }

  // 2. Create RocUberTraceService + RocDriverUtilsService, register this as controller.
  if (roc_trace_session_ != nullptr) {
    uber_trace_svc_    = new ::roc::RocUberTraceService(roc_trace_session_);
    driver_utils_svc_  = new ::roc::RocDriverUtilsService(roc_trace_session_);
    // Register this capture manager as the trace controller so OnTraceRequested()
    // is invoked when a tool calls UberTrace::RequestTrace() via RPC.
    roc_trace_session_->RegisterController(this);
  }

  // 3. Create ddRpcServer and register UberTrace + DriverUtils RPC services.
  //    Also register the DriverControlServer on the message channel directly —
  //    mirrors PAL's RegisterRpcServices() (platform.cpp line 1258).
  // Register DriverControlServer on the message channel so it processes protocol messages
  // (StepDriver, ResumeDriver) during StartEarlyDeviceInit() halt.  Must happen regardless
  // of whether the RPC server creation succeeds.
  dev_driver_server_->GetMessageChannel()->RegisterProtocolServer(
      dev_driver_server_->GetDriverControlServer());

  DDRpcServerCreateInfo rpcInfo = {};
  rpcInfo.hConnection = reinterpret_cast<DDNetConnection>(
      dev_driver_server_->GetMessageChannel());
  DD_RESULT rpcResult = ddRpcServerCreate(&rpcInfo, &rpc_server_);
  fprintf(stderr, "[CLR-Init] ddRpcServerCreate result=%d rpc_server=%p\n",
          static_cast<int>(rpcResult), static_cast<void*>((void*)(uintptr_t)rpc_server_));
  if (rpcResult == DD_RESULT_SUCCESS) {
    if (uber_trace_svc_   != nullptr) {
      DD_RESULT regResult = UberTrace::RegisterService(rpc_server_, uber_trace_svc_);
      fprintf(stderr, "[CLR-Init] UberTrace service registered result=%d\n",
              static_cast<int>(regResult));
    }
    if (driver_utils_svc_ != nullptr) {
      DriverUtils::RegisterService(rpc_server_, driver_utils_svc_);
      fprintf(stderr, "[CLR-Init] DriverUtils service registered\n");
    }
  } else {
    LogError("RocUberTraceCaptureMgr: ddRpcServerCreate failed — UberTrace RPC unavailable");
    fprintf(stderr, "[CLR-Init] ddRpcServerCreate FAILED result=%d — UberTrace RPC unavailable\n",
            static_cast<int>(rpcResult));
  }
  fprintf(stderr, "[CLR-Init] roc_trace_session=%p controller=%s\n",
          static_cast<void*>(roc_trace_session_),
          (roc_trace_session_ && roc_trace_session_->HasController()) ? "registered" : "NOT registered");

  // 4. Enable RGP tracing BEFORE StartEarlyDeviceInit().
  //    The tool sends EnableProfilingRequest during the halt; the server accepts it
  //    only when (m_isFinalized==false && profilingStatus==Available).
  //    Since we never call DevDriverServer::Finalize(), m_isFinalized stays false.
  rgp_server_->EnableTraces();

  // 5. Signal DriverControlServer that early device init is starting.
  //    This may block if a tool requested HaltOnDeviceInit, giving it time to
  //    send EnableProfilingRequest and configure trace parameters.
  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();
  if (pDriverControlServer != nullptr) {
    pDriverControlServer->StartEarlyDeviceInit();
  }

  return true;
}


// ================================================================================================
void RocUberTraceCaptureMgr::Destroy() {
  delete user_event_;
  user_event_ = nullptr;

  if (rgp_server_ != nullptr) {
    // Abort any in-progress trace so the RGP server reaches a clean Idle state.
    if (rgp_server_->IsTraceRunning()) {
      rgp_server_->AbortTrace();
    }
    rgp_server_->DisableTraces();
    rgp_server_ = nullptr;  // Owned by dev_driver_server_; do not delete.
  }

  FreeSqttResources();

  // Tear down RPC services + TraceSession in reverse-init order.
  // Mirrors PAL's Platform::DestroyRpcServices() + DestroyTraceSession().
  if (roc_trace_session_ != nullptr) {
    roc_trace_session_->UnregisterController(this);
  }

  // Wait for any in-flight CollectTrace RPC callback to finish before destroying
  // the RPC server.  Mirrors PAL's m_traceInactiveEvent.Wait(m_maxDeviceTimeout).
  // m_collecting is set in RequestTrace() and cleared at the very end of CollectTrace()
  // (after pfnEnd returns), so ddRpcServerDestroy cannot close the socket while
  // pfnWriteBytes is still transferring data.
  if (uber_trace_svc_ != nullptr) {
    fprintf(stderr, "[CLR-Destroy] WaitForCollectDone...\n");
    const bool done = uber_trace_svc_->WaitForCollectDone(30000);
    fprintf(stderr, "[CLR-Destroy] WaitForCollectDone: %s\n", done ? "done" : "TIMEOUT");
  }

  if (rpc_server_ != DD_API_INVALID_HANDLE) {
    ddRpcServerDestroy(rpc_server_);
    rpc_server_ = DD_API_INVALID_HANDLE;
  }
  delete driver_utils_svc_;
  driver_utils_svc_ = nullptr;
  delete uber_trace_svc_;
  uber_trace_svc_ = nullptr;
  delete roc_trace_session_;
  roc_trace_session_ = nullptr;

  delete dev_driver_server_;
  dev_driver_server_ = nullptr;

  trace_running_ = false;
}

// ================================================================================================
// Mirrors PAL's Platform::IsTracingEnabled():
//   returns true if the tool enabled UberTrace tracing (via RPC) OR any trace is active.
// Falls back to legacy RGP server state if TraceSession is not available.
bool RocUberTraceCaptureMgr::IsTracingEnabled() const {
  if (roc_trace_session_ == nullptr) return false;
  const ::roc::RocTraceSessionState state = roc_trace_session_->GetState();
  // Tracing is enabled when the tool has called EnableTracing() via RPC,
  // OR when any trace is active (not Idle/Ready/Completed).
  return roc_trace_session_->IsTracingEnabled() ||
         (state != ::roc::RocTraceSessionState::Ready &&
          state != ::roc::RocTraceSessionState::Completed);
}

// ================================================================================================
bool RocUberTraceCaptureMgr::Update() {
  return true;
}

// ================================================================================================
// Signals DriverControlServer that late device init is starting.
// Mirrors PAL's DriverControlServer::StartLateDeviceInit() call site.
// Produces "[DriverControlServer] Driver starting late device initialization".
void RocUberTraceCaptureMgr::StartLateDeviceInit() {
  if (dev_driver_server_ == nullptr) return;
  auto* pDcs = dev_driver_server_->GetDriverControlServer();
  if (pDcs != nullptr) {
    pDcs->StartLateDeviceInit();
  }
}

// ================================================================================================
// Signals DriverControlServer that device init is fully complete.
// Mirrors PAL's DriverControlServer::FinishDeviceInit() call site.
// Produces "[DriverControlServer] Driver device initialization finished".
void RocUberTraceCaptureMgr::FinishDeviceInit() {
  if (dev_driver_server_ == nullptr) return;
  auto* pDcs = dev_driver_server_->GetDriverControlServer();
  if (pDcs != nullptr) {
    pDcs->FinishDeviceInit();
  }
}

// ================================================================================================
// Waits for the driver to be resumed if it has been paused by the developer tools host.
void RocUberTraceCaptureMgr::WaitForDriverResume() {
  auto* pDriverControlServer = dev_driver_server_->GetDriverControlServer();
  assert(pDriverControlServer != nullptr);
  pDriverControlServer->DriverTick();
}


// ================================================================================================
// IRocTraceController::OnTraceRequested — called by RocTraceSession when a tool calls
// UberTrace::RequestTrace() via RPC.  Accept the trace immediately; SQTT hardware setup
// happens in PreDispatch once the preparation dispatches have run.
bool RocUberTraceCaptureMgr::OnTraceRequested(::roc::RocTraceSession* pSession) {
  prep_disp_count_ = 0;

  fprintf(stderr, "[CLR-Ctrl] OnTraceRequested: session=%p uber_trace_svc=%p\n",
          static_cast<void*>(pSession), static_cast<void*>(uber_trace_svc_));

  if (uber_trace_svc_ == nullptr) {
    fprintf(stderr, "[CLR-Ctrl] OnTraceRequested: REJECTED (no uber_trace_svc — config unavailable)\n");
    return false;
  }

  // OnTraceRequested is only ever called via the UberTrace RPC path (RocUberTraceService::RequestTrace).
  // Parameters come from ConfigureTraceParams() JSON, stored in uber_trace_svc_->GetTraceConfig().
  // Do NOT use rgp_server_->QueryTraceParameters() here: that struct is populated only by the
  // legacy RGP DevDriver protocol and stays all-zeros when ubertrace is the active path.
  {
    const auto& cfg       = uber_trace_svc_->GetTraceConfig();
    num_prep_frames_      = cfg.numPrepDispatches;
    sqtt_output_size_     = static_cast<size_t>(cfg.sqttMemoryLimitInMb) * 1024u * 1024u;
    sqtt_se_mask_         = cfg.seMask;
    sqtt_instruction_tokens_  = cfg.enableInstructionTokens;
    sqtt_capture_code_objects_= cfg.captureCodeObjects;
    // Index-mode window: preparationStartIndex marks when prep counting begins;
    // captureDispatchCount drives the stop index (0 = unlimited).
    capture_index_mode_   = cfg.indexMode;
    capture_start_index_  = cfg.captureStartIndex;
    capture_stop_index_   = (cfg.indexMode && cfg.captureDispatchCount > 0)
                            ? cfg.captureStartIndex + cfg.numPrepDispatches + cfg.captureDispatchCount
                            : 0;

    fprintf(stderr, "[CLR-Ctrl] OnTraceRequested: numPrep=%u memMb=%u seMask=0x%x instrTokens=%d"
            " codeObj=%d indexMode=%d startIdx=%u stopIdx=%u\n",
            num_prep_frames_, cfg.sqttMemoryLimitInMb, sqtt_se_mask_,
            sqtt_instruction_tokens_, sqtt_capture_code_objects_,
            capture_index_mode_, capture_start_index_, capture_stop_index_);
  }

  sqtt_state_ = SqttState::Preparing;
  // AcceptTrace() opens the RDF chunk writer and advances state to Preparing.
  const bool accepted = pSession->AcceptTrace(this);
  fprintf(stderr, "[CLR-Ctrl] OnTraceRequested: AcceptTrace returned %s\n",
          accepted ? "true" : "false");
  return accepted;
}

// ================================================================================================
// Called by RocTraceSession on the DevDriver RPC thread when the tool cancels the trace.
// We cannot submit AQL stop packets here (wrong thread).  Set pending_abort_ so the
// next PostDispatch call (on the GPU dispatch thread) does the actual hardware stop.
// If SQTT was still Preparing (start packet not yet submitted), just reset state directly.
void RocUberTraceCaptureMgr::OnTraceCanceled() {
  if (sqtt_state_ == SqttState::Running) {
    // Signal the dispatch thread to stop hardware on the next PostDispatch.
    pending_abort_.store(true, std::memory_order_release);
  } else {
    // Preparing or Idle — no GPU work in flight yet; safe to reset directly.
    sqtt_state_    = SqttState::Idle;
    trace_running_ = false;
  }
}

// ================================================================================================
void RocUberTraceCaptureMgr::OnTraceFinished() {
  // SQTT stop + collect already done in PostDispatch/FinishRGPTrace before EndTrace().
  sqtt_state_    = SqttState::Idle;
  trace_running_ = false;
}

// ================================================================================================
// BeginSqttTrace — obtains aqlprofile extension table, sets up TRACE profile, allocates
// buffers, populates and submits the start AQL packet (blocking).
bool RocUberTraceCaptureMgr::BeginSqttTrace(VirtualGPU* gpu) {
  fprintf(stderr, "[CLR-Ctrl] BeginSqttTrace: gpu=%p disp=%llu memMb=%u seMask=0x%x\n",
          static_cast<void*>(gpu),
          static_cast<unsigned long long>(global_disp_count_),
          static_cast<uint32_t>(sqtt_output_size_ / (1024u * 1024u)),
          sqtt_se_mask_);

  // Guard: if output size is zero the config was never properly set — do not start SQTT,
  // which would leave trace_running_ false and prevent IsTraceRunning() returning true.
  if (sqtt_output_size_ == 0) {
    fprintf(stderr, "[CLR-Ctrl] BeginSqttTrace: REJECTED (sqtt_output_size_==0, config not set)\n");
    sqtt_state_ = SqttState::Idle;
    return false;
  }

  // Obtain the aqlprofile extension function table via the HSA extension mechanism.
  // Mirrors PerfCounterProfile::Create() — same extension, TRACE type instead of PMC.
  if (Hsa::system_get_major_extension_table(
          HSA_EXTENSION_AMD_AQLPROFILE, hsa_ven_amd_aqlprofile_VERSION_MAJOR,
          sizeof(hsa_ven_amd_aqlprofile_pfn_t), &sqtt_api_) != HSA_STATUS_SUCCESS) {
    LogError("SQTT: failed to obtain aqlprofile extension table");
    return false;
  }

  // Configure profile for SQTT (thread-trace) capture.
  memset(&sqtt_profile_, 0, sizeof(sqtt_profile_));
  sqtt_profile_.agent  = device_->getBackendDevice();
  sqtt_profile_.type   = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE;
  sqtt_profile_.events = nullptr;
  sqtt_profile_.event_count = 0;

  // Query required command buffer size (aqlprofile tells us how much PM4 space it needs).
  // ATT_BUFFER_SIZE parameter and output_buffer are set below once out_size_mb is known.
  uint32_t cmd_size = 0;
  sqtt_api_.hsa_ven_amd_aqlprofile_get_info(
      &sqtt_profile_, HSA_VEN_AMD_AQLPROFILE_INFO_COMMAND_BUFFER_SIZE, &cmd_size);
  if (cmd_size == 0) cmd_size = 4096;

  // Build aqlprofile parameter list from DevDriver trace configuration.
  // All parameters were read from QueryTraceParameters() in OnTraceRequested().
  const uint32_t out_size_mb = sqtt_output_size_ / (1024u * 1024u);

  // Maximum possible parameters: ATT_BUFFER_SIZE + SE_MASK + SIMD_SELECTION + OCCUPANCY_MODE.
  hsa_ven_amd_aqlprofile_parameter_t sqtt_params[4];
  uint32_t param_count = 0;

  // ATT_BUFFER_SIZE (MB) — always required; must match allocated output buffer.
  sqtt_params[param_count++] = { HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE,
                                  out_size_mb };

  // SE_MASK — restrict capture to specific shader engines (0 = all SEs, as set by tool).
  if (sqtt_se_mask_ != 0) {
    sqtt_params[param_count++] = { HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK,
                                    sqtt_se_mask_ };
  }

  // SIMD_SELECTION — all SIMDs when instruction tokens are enabled; SIMD 0 only otherwise.
  sqtt_params[param_count++] = { HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SIMD_SELECTION,
                                  sqtt_instruction_tokens_ ? 0xFu : 0x1u };

  // OCCUPANCY_MODE — suppress instruction token hardware capture when not requested.
  // aqlprofile defaults to full token capture (INST/INST_PC bits set in TOKEN_MASK);
  // OCCUPANCY_MODE=1 switches to a lightweight token mask that excludes instruction tokens.
  // Mirrors rocprofiler-sdk's no_detail_simd → OCCUPANCY_MODE=1 mapping.
  if (!sqtt_instruction_tokens_) {
    sqtt_params[param_count++] = { HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_OCCUPANCY_MODE, 1u };
  }

  sqtt_profile_.parameters      = sqtt_params;
  sqtt_profile_.parameter_count = param_count;

  // Command buffer: fine-grained host memory (CPU writes PM4, GPU/CP reads via DMA).
  sqtt_cmd_buf_ = device_->hostAlloc(cmd_size, /*alignment=*/4096,
                                     Device::MemorySegment::kAtomics);
  if (sqtt_cmd_buf_ == nullptr) {
    LogError("SQTT: failed to allocate command buffer");
    return false;
  }

  // Small fine-grained host buffer for WriteMarker NOP PM4.  The vendor-specific AQL
  // packet uses an IT_INDIRECT_BUFFER jump into GPU-accessible memory; inline stack
  // buffers are rejected by the MEC firmware (EC_QUEUE_PACKET_VENDOR_UNSUPPORTED).
  marker_cmd_buf_ = device_->hostAlloc(kMarkerRingSize * kMarkerSlotBytes,
                                       /*alignment=*/256, Device::MemorySegment::kNoAtomics);
  if (marker_cmd_buf_ == nullptr) {
    LogError("SQTT: failed to allocate marker command buffer");
    FreeSqttResources();
    return false;
  }
  sqtt_cmd_buf_size_            = cmd_size;
  sqtt_profile_.command_buffer.ptr  = sqtt_cmd_buf_;
  sqtt_profile_.command_buffer.size = cmd_size;

  // Output buffer: GPU device-local memory (VRAM). The GPU writes SQTT thread-trace data
  // directly here during capture. After the read packet completes, we copy to host for RDF.
  // Using deviceLocalAlloc (gpuvm_segment_) — no CPU atomics needed, coarse-grained is fine.
  Device::AllocationFlags devFlags = {};
  sqtt_output_ = device_->deviceLocalAlloc(sqtt_output_size_, devFlags);
  if (sqtt_output_ == nullptr) {
    LogError("SQTT: failed to allocate device-local output buffer; falling back to host memory");
    sqtt_output_ = device_->hostAlloc(sqtt_output_size_, /*alignment=*/4096,
                                      Device::MemorySegment::kAtomics);
  }
  if (sqtt_output_ == nullptr) {
    LogError("SQTT: failed to allocate output buffer");
    FreeSqttResources();
    return false;
  }
  sqtt_profile_.output_buffer.ptr  = sqtt_output_;
  sqtt_profile_.output_buffer.size = sqtt_output_size_;

  // Populate the start AQL packet (pm4_command[] only; header set by dispatchCounterAqlPacket).
  memset(&sqtt_start_packet_, 0, sizeof(sqtt_start_packet_));
  if (sqtt_api_.hsa_ven_amd_aqlprofile_start(&sqtt_profile_, &sqtt_start_packet_)
      != HSA_STATUS_SUCCESS) {
    LogError("SQTT: hsa_ven_amd_aqlprofile_start failed");
    FreeSqttResources();
    return false;
  }

  // Submit start packet — blocking so SQTT is active before the next dispatch.
  gpu->dispatchCounterAqlPacket(&sqtt_start_packet_, PerfCounter::ROC_GFX9,
                                /*blocking=*/true, nullptr);

  sqtt_state_    = SqttState::Running;
  trace_running_ = true;
  trace_gpu_     = gpu;  // remembered for OnTraceCanceled() deferred stop
  fprintf(stderr, "[CLR-Ctrl] BeginSqttTrace: SQTT started, state=Running\n");

  // For relative-mode ubertrace captures (captureMode == "relative"), the stop index is not
  // known until the trace actually starts.  Compute it now from the current dispatch count.
  // Index-mode captures already have capture_stop_index_ set in OnTraceRequested(); skip those.
  if (!capture_index_mode_ && capture_stop_index_ == 0) {
    const uint32_t count = uber_trace_svc_ != nullptr
                           ? uber_trace_svc_->GetTraceConfig().captureDispatchCount : 0;
    if (count > 0) {
      capture_stop_index_ = static_cast<uint32_t>(global_disp_count_) + count;
    }
  }
  // Notify RGP server if the tool used the legacy RGP protocol (IsTracePending() was true).
  // Never do this in the UberTrace path: rgp_server_->BeginTrace() sets IsTraceRunning()=true,
  // which triggers the legacy PostDispatch auto-stop in FinishRGPTrace, and the resulting
  // rgp_server_->EndTrace() message causes RDP to disconnect the UberTrace poll thread.
  if (roc_trace_session_ == nullptr && rgp_server_ != nullptr && rgp_server_->IsTracePending()) {
    rgp_server_->BeginTrace();
  }
  fprintf(stderr, "[CLR-Ctrl] BeginSqttTrace: rgp IsTracePending=%d IsTraceRunning=%d"
          " (ubertrace=%s)\n",
          (rgp_server_ != nullptr ? (int)rgp_server_->IsTracePending() : -1),
          (rgp_server_ != nullptr ? (int)rgp_server_->IsTraceRunning() : -1),
          roc_trace_session_ != nullptr ? "yes" : "no");
  // Advance UberTrace session: Preparing → Running.
  if (roc_trace_session_ != nullptr) {
    roc_trace_session_->BeginTrace();
  }
  return true;
}

// ================================================================================================
// EndSqttTrace — populates and submits the stop + read AQL packets (both blocking).
void RocUberTraceCaptureMgr::EndSqttTrace(VirtualGPU* gpu) {
  fprintf(stderr, "[CLR-Ctrl] EndSqttTrace: gpu=%p disp=%llu\n",
          static_cast<void*>(gpu),
          static_cast<unsigned long long>(global_disp_count_));
  if (sqtt_state_ != SqttState::Running) return;

  memset(&sqtt_stop_packet_, 0, sizeof(sqtt_stop_packet_));
  if (sqtt_api_.hsa_ven_amd_aqlprofile_stop(&sqtt_profile_, &sqtt_stop_packet_)
      == HSA_STATUS_SUCCESS) {
    gpu->dispatchCounterAqlPacket(&sqtt_stop_packet_, PerfCounter::ROC_GFX9,
                                  /*blocking=*/true, nullptr);
  }

  memset(&sqtt_read_packet_, 0, sizeof(sqtt_read_packet_));
  if (sqtt_api_.hsa_ven_amd_aqlprofile_read(&sqtt_profile_, &sqtt_read_packet_)
      == HSA_STATUS_SUCCESS) {
    gpu->dispatchCounterAqlPacket(&sqtt_read_packet_, PerfCounter::ROC_GFX9,
                                  /*blocking=*/true, nullptr);
  }

  sqtt_state_    = SqttState::WaitingForResults;
  trace_running_ = false;
}

// Per-SE SQTT buffer extent within the host-side copy of sqtt_output_.
struct SeEntry {
  size_t offset;  // byte offset from start of host_buf
  size_t size;    // bytes written by GPU for this SE
};

// ================================================================================================
// BuildRgpFileBlob — assembles a complete .rgp file blob from per-SE SQTT data.
// Mirrors what PAL's GpaSession::GetResults() produces so the legacy rgp_server_ protocol
// delivers a file the RGP tool can open directly.
//
// host_buf  — host-side copy of the full sqtt_output_ VRAM buffer.
// se_entries — per-SE {offset, size} within host_buf, one entry per shader engine,
//              in SE index order as reported by hsa_ven_amd_aqlprofile_iterate_data.
static std::vector<uint8_t> BuildRgpFileBlob(const Device* device,
                                              const void* host_buf,
                                              const std::vector<SeEntry>& se_entries) {
  using namespace RgpFile;

  const device::Info& di  = device->info();
  const amd::Isa&     isa = device->isa();
  const uint32_t      maj = isa.versionMajor();
  const uint32_t      min = isa.versionMinor();
  const int           num_se = static_cast<int>(se_entries.size());

  // File header
  FileHeader fh = {};
  fh.magicNumber  = SQTT_FILE_MAGIC_NUMBER;
  fh.versionMajor = RGP_FILE_FORMAT_SPEC_MAJOR_VER;
  fh.versionMinor = RGP_FILE_FORMAT_SPEC_MINOR_VER;
  fh.flags.noQueueSemaphoreTimeStamps = 1;  // no queue timing in the ROC SQTT path
  fh.chunkOffset  = static_cast<int32_t>(sizeof(FileHeader));
  {
    time_t now = time(nullptr);
    struct tm t = {};
#if defined(_WIN32)
    localtime_s(&t, &now);
#else
    localtime_r(&now, &t);
#endif
    fh.second = t.tm_sec; fh.minute = t.tm_min; fh.hour = t.tm_hour;
    fh.dayInMonth = t.tm_mday; fh.month = t.tm_mon; fh.year = t.tm_year;
    fh.dayInWeek = t.tm_wday; fh.dayInYear = t.tm_yday; fh.isDaylightSavings = t.tm_isdst;
  }

  // HSA agent properties shared across chunks
  uint64_t ts_freq = 0;
  Hsa::agent_get_info(device->getBackendDevice(),
                      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY),
                      &ts_freq);
  uint32_t mem_width = 0;
  Hsa::agent_get_info(device->getBackendDevice(),
                      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_WIDTH),
                      &mem_width);
  uint32_t asic_rev = 0;
  Hsa::agent_get_info(device->getBackendDevice(),
                      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                      &asic_rev);
  uint32_t max_waves_per_cu = 0;
  Hsa::agent_get_info(device->getBackendDevice(),
                      static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                      &max_waves_per_cu);

  // CPU info chunk — vendor/brand/cores left zero (not exposed via HSA for compute).
  CpuInfo cpu = {};
  cpu.header = MakeChunkHeader(kCpuInfo, 0, static_cast<int32_t>(sizeof(CpuInfo)));
  cpu.cpuTimestampFrequency = ts_freq;

  // ASIC info chunk
  AsicInfo asic = {};
  asic.header = MakeChunkHeader(kAsicInfo, 0, static_cast<int32_t>(sizeof(AsicInfo)));
  {
    const uint32_t waves_per_simd = (di.simdPerCU_ > 0)
                                    ? (max_waves_per_cu / di.simdPerCU_) : max_waves_per_cu;
    const uint64_t engine_mhz     = di.maxEngineClockFrequency_;
    const uint64_t mem_mhz        = di.maxMemoryClockFrequency_;
    // shaderEngines comes from the actual SE count reported by iterate_data.
    const uint32_t total_cu = di.maxComputeUnits_;
    const int32_t  se_count = (num_se > 0) ? num_se : 1;

    asic.traceShaderCoreClock       = engine_mhz * 1000000ULL;
    asic.traceMemoryClock           = mem_mhz    * 1000000ULL;
    asic.maxShaderCoreClock         = engine_mhz * 1000000ULL;
    asic.maxMemoryClock             = mem_mhz    * 1000000ULL;
    asic.gpuTimestampFrequency      = ts_freq;
    asic.deviceId                   = static_cast<int32_t>(di.pcieDeviceId_);
    asic.deviceRevisionId           = static_cast<int32_t>(asic_rev);
    asic.vgprsPerSimd               = static_cast<int32_t>(di.vgprsPerSimd_);
    asic.minimumVgprAlloc           = static_cast<int32_t>(di.vgprAllocGranularity_);
    asic.vgprAllocGranularity       = static_cast<int32_t>(di.vgprAllocGranularity_);
    asic.sgprsPerSimd               = static_cast<int32_t>(di.sgprsPerSimd_);
    asic.simdPerComputeUnit         = static_cast<int32_t>(di.simdPerCU_);
    asic.wavefrontsPerSimd          = static_cast<int32_t>(waves_per_simd);
    asic.ldsSize                    = static_cast<int32_t>(isa.localMemSizePerCU());
    asic.gfxIpLevel                 = GfxVerToIpLevel(maj, min);
    asic.gpuType                    = kGpuDiscrete;
    asic.memoryChipType             = kMemHbm2;
    asic.vramBusWidth               = static_cast<int32_t>(mem_width);
    asic.vramSize                   = static_cast<int64_t>(di.globalMemSize_);
    asic.l2CacheSize                = static_cast<int32_t>(di.l2CacheSize_);
    // LDS allocation granularity in bytes.  GFX10.3 uses 256 DWORDs (1024 B);
    // all other supported GFX (9/10.1/11/12) use 128 DWORDs (512 B).
    asic.ldsGranularity             = (maj == 10 && min >= 3) ? 1024u : 512u;
    asic.shaderEngines              = se_count;
    asic.computeUnitPerShaderEngine = (se_count > 0)
                                      ? static_cast<int32_t>(total_cu / se_count) : 0;
    strncpy(asic.gpuName, di.boardName_, sizeof(asic.gpuName) - 1);

    // Fill cuMask[se][sa]: HSA has no per-SE/SA mask API, so derive from topology.
    // All CUs are assumed active (no partial harvesting visibility through HSA).
    uint32_t sa_per_se = 1;
    Hsa::agent_get_info(device->getBackendDevice(),
                        static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE),
                        &sa_per_se);
    if (sa_per_se < 1) sa_per_se = 1;

    const uint32_t cu_per_sa = (se_count > 0 && sa_per_se > 0)
                                ? (total_cu / (static_cast<uint32_t>(se_count) * sa_per_se)) : 0;
    // Build a mask with all cu_per_sa low bits set (max 16 bits per the cuMask field width).
    const uint16_t active_cu_mask = (cu_per_sa >= 16) ? 0xFFFFu
                                  : (cu_per_sa > 0)   ? static_cast<uint16_t>((1u << cu_per_sa) - 1u)
                                  : 0u;
    for (uint32_t se = 0; se < static_cast<uint32_t>(se_count) && se < kMaxNumSE; ++se) {
      for (uint32_t sa = 0; sa < sa_per_se && sa < kSaPerSE; ++sa) {
        asic.cuMask[se][sa] = active_cu_mask;
      }
    }
  }

  // API info chunk
  ApiInfo api = {};
  api.header        = MakeChunkHeader(kApiInfo, 0, static_cast<int32_t>(sizeof(ApiInfo)));
  api.apiType       = amd::IS_HIP ? kApiHip : kApiOpenCl;
  api.profilingMode = kProfilingModePresent;

  // Queue event timings chunk — empty (no queue semaphore timestamps in HIP compute path).
  // The chunk is mandatory for RGP to load the file.
  QueueEventTimings qt = {};
  qt.header = MakeChunkHeader(kQueueEventTimings, 0, static_cast<int32_t>(sizeof(QueueEventTimings)));
  // All record counts and sizes remain zero.

  // Clock calibration chunk — CPU timestamp from HSA, GPU timestamp left 0.
  ClockCalibration cc = {};
  cc.header = MakeChunkHeader(kClockCalibration, 0, static_cast<int32_t>(sizeof(ClockCalibration)));
  cc.cpuTimestamp = GetSystemTimestamp();

  // Compute total blob size: fixed chunks + per-SE (SqttDesc + SqttData + raw bytes).
  size_t total = sizeof(FileHeader) + sizeof(CpuInfo) + sizeof(AsicInfo) + sizeof(ApiInfo)
               + sizeof(QueueEventTimings) + sizeof(ClockCalibration);
  for (const auto& se : se_entries) {
    total += sizeof(SqttDesc) + sizeof(SqttData) + se.size;
  }

  std::vector<uint8_t> blob(total, 0);
  uint8_t* p          = blob.data();
  size_t   cur_offset = 0;  // tracks current write position (== absolute file offset)

  auto append = [&](const void* src, size_t n) {
    memcpy(p, src, n);
    p          += n;
    cur_offset += n;
  };

  append(&fh,   sizeof(fh));
  append(&cpu,  sizeof(cpu));
  append(&asic, sizeof(asic));
  append(&api,  sizeof(api));
  append(&qt,   sizeof(qt));
  append(&cc,   sizeof(cc));

  // One SqttDesc + SqttData + raw-bytes triple per shader engine — mirrors gpaSession.cpp:4860.
  const SqttVersion sqtt_ver = GfxVerToSqttVersion(maj, min);
  for (int i = 0; i < num_se; i++) {
    const size_t se_size = se_entries[i].size;

    SqttDesc desc = {};
    desc.header            = MakeChunkHeader(kSqttDesc, i, static_cast<int32_t>(sizeof(SqttDesc)));
    desc.shaderEngineIndex = i;
    desc.sqttVersion       = sqtt_ver;
    append(&desc, sizeof(desc));

    // data.offset is the absolute file offset of the raw SQTT bytes for this SE,
    // which immediately follow the SqttData header.
    SqttData data = {};
    data.header = MakeChunkHeader(kSqttData, i,
                                  static_cast<int32_t>(sizeof(SqttData) + se_size));
    data.offset = static_cast<int32_t>(cur_offset + sizeof(SqttData));
    data.size   = static_cast<int32_t>(se_size);
    append(&data, sizeof(data));

    append(static_cast<const uint8_t*>(host_buf) + se_entries[i].offset, se_size);
  }

  return blob;
}

// ================================================================================================
// CollectSqttResults — reads per-SE trace extents via iterate_data, DMA-copies the full
// device-local buffer to host, builds a complete .rgp file blob, and delivers it.
void RocUberTraceCaptureMgr::CollectSqttResults(VirtualGPU* /*gpu*/) {
  fprintf(stderr, "[CLR-Ctrl] CollectSqttResults: state=%d output=%p size=%zu\n",
          static_cast<int>(sqtt_state_), sqtt_output_, sqtt_output_size_);
  if (sqtt_state_ != SqttState::WaitingForResults || sqtt_output_ == nullptr) return;

  // Collect per-SE {offset_from_base, size} pairs.
  // iterate_data fires once per shader engine for TRACE_DATA; sample_id is the SE index.
  // trace_data.ptr is an absolute pointer into sqtt_output_ (VRAM).
  struct IterCtx {
    void*                base;
    std::vector<SeEntry> entries;
  };
  IterCtx iter_ctx = { sqtt_output_, {} };

  // Snapshot the allocated size NOW, before iterate_data may call into the aqlprofile
  // runtime and corrupt sqtt_output_size_ via sqtt_profile_ write-back.
  // (aqlprofile encodes a packed GPU descriptor into output_buffer.size after the read
  //  packet completes, which on Windows overwrites sqtt_output_size_ if it is adjacent.)
  const size_t allocated_size = sqtt_output_size_;

  sqtt_api_.hsa_ven_amd_aqlprofile_iterate_data(
      &sqtt_profile_,
      [](hsa_ven_amd_aqlprofile_info_type_t type,
         hsa_ven_amd_aqlprofile_info_data_t* data, void* ud) -> hsa_status_t {
        if (type == HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA) {
          auto* ctx = reinterpret_cast<IterCtx*>(ud);
          const size_t offset = static_cast<uint8_t*>(data->trace_data.ptr)
                              - static_cast<uint8_t*>(ctx->base);
          ctx->entries.push_back({offset, data->trace_data.size});
          fprintf(stderr, "[CLR-Ctrl] CollectSqttResults: SE entry offset=%zu size=%zu\n",
                  offset, data->trace_data.size);
        }
        return HSA_STATUS_SUCCESS;
      },
      &iter_ctx);

  // Compute the actual data extent from per-SE entries: max(offset + size).
  // This avoids relying on sqtt_output_size_ which may be corrupted after EndSqttTrace
  // by the aqlprofile runtime writing a packed GPU descriptor back into the profile struct.
  size_t data_extent = 0;
  for (const auto& e : iter_ctx.entries) {
    data_extent = std::max(data_extent, e.offset + e.size);
  }

  // Use data_extent if valid; fall back to the snapshotted allocated_size only if
  // iterate_data returned nothing (unexpected — means no SQTT data was captured).
  const size_t copy_size = (data_extent > 0 && data_extent <= allocated_size)
                           ? data_extent : allocated_size;
  // Restore sqtt_output_size_ to the correct allocated value so FreeSqttResources
  // passes the right size to hostFree.
  sqtt_output_size_ = allocated_size;

  fprintf(stderr, "[CLR-Ctrl] CollectSqttResults: %zu SE entries, extent=%zu copy_size=%zu\n",
          iter_ctx.entries.size(), data_extent, copy_size);

  if (copy_size == 0) {
    FreeSqttResources();
    sqtt_state_ = SqttState::Idle;
    return;
  }

  void* host_buf = device_->hostAlloc(copy_size, /*alignment=*/4096,
                                      Device::MemorySegment::kAtomics);
  if (host_buf != nullptr) {
    Hsa::memory_copy(host_buf, sqtt_output_, copy_size);

    // If iterate_data yielded no per-SE entries treat the whole buffer as one SE.
    if (iter_ctx.entries.empty()) {
      iter_ctx.entries.push_back({0, copy_size});
    }

    // Legacy RGP path: build a complete .rgp file blob and stream to the tool.
    // Skip when uber trace session is active — it handles delivery via RDF chunks.
    if (rgp_server_ != nullptr && roc_trace_session_ == nullptr) {
      std::vector<uint8_t> blob = BuildRgpFileBlob(device_, host_buf, iter_ctx.entries);
      rgp_server_->WriteTraceData(blob.data(), blob.size());
    }

    // UberTrace path: write one "SqttData" RDF chunk per shader engine, each with a
    // SqttDataHeader chunk-header.  Mirrors GpuPerfExperimentTraceSource::WriteSqttDataChunks().
    if (roc_trace_session_ != nullptr) {
      // Chunk identifier: "SqttData" (16 bytes, NUL-padded).
      static const char kSqttChunkId[kRdfIdentifierSize] = "SqttData";
      // Version 5 matches SqttDataChunkVersion in PAL's gpuPerfExperimentTraceSource.h.
      static constexpr uint32_t kSqttDataChunkVersion = 5;

      const device::Info& di = device_->info();
      const uint32_t gfx_maj = device_->isa().versionMajor();
      const uint32_t gfx_min = device_->isa().versionMinor();
      const uint32_t sqtt_ver =
          static_cast<uint32_t>(RgpFile::GfxVerToSqttVersion(gfx_maj, gfx_min));

      for (uint32_t se_idx = 0; se_idx < static_cast<uint32_t>(iter_ctx.entries.size()); ++se_idx) {
        const SeEntry& se = iter_ctx.entries[se_idx];

        // SqttDataHeader: mirrors PAL's sqttDataHeader initialiser in WriteSqttDataChunks().
        struct SqttDataHeader {
          uint32_t pciId;
          uint32_t shaderEngine;
          uint32_t sqttVersion;
          uint32_t instrumentationVersionSpec;
          uint32_t instrumentationVersionApi;
          uint32_t wgpIndex;
          uint64_t traceBufferSize;
          uint32_t instructionTimingEnabled : 1;
          uint32_t execPopTokensEnabled     : 1;
          uint32_t reserved                 : 30;
        };

        const bool instr_enabled =
            sqtt_instruction_tokens_ &&
            (sqtt_se_mask_ == 0 || (sqtt_se_mask_ & (1u << se_idx)) != 0);

        SqttDataHeader hdr = {};
        hdr.pciId                     = di.pcieDeviceId_;
        hdr.shaderEngine              = se_idx;
        hdr.sqttVersion               = sqtt_ver;
        hdr.instrumentationVersionSpec = 1;  // instrumentation spec version (API-agnostic)
        hdr.instrumentationVersionApi  = 0;  // API-specific version (OpenCL = 0)
        hdr.wgpIndex                  = 0;   // not tracked per-dispatch in HSA path
        hdr.traceBufferSize           = static_cast<uint64_t>(sqtt_output_size_);
        hdr.instructionTimingEnabled  = instr_enabled ? 1u : 0u;
        hdr.execPopTokensEnabled      = 0;

        const uint8_t* se_data = static_cast<const uint8_t*>(host_buf) + se.offset;
        roc_trace_session_->WriteDataChunk(kSqttChunkId, kSqttDataChunkVersion,
                                           &hdr, sizeof(hdr),
                                           se_data, se.size);
      }

      // Write "ApiInfo" RDF chunk so RGP can identify the API type.
      // The RDF format uses different numeric values than the legacy .rgp format:
      //   kOpenCl = 6, kHip = 8  (matches RgpRdfTraceApiType in rgp_rdf_chunks.h).
      // Without this chunk RGP's RDF loader never sets api_type and displays "Unknown".
      // Mirrors PAL's ApiInfoTraceSource::OnTraceFinished() → WriteDataChunk("ApiInfo").
      static const char kApiInfoChunkId[kRdfIdentifierSize] = "ApiInfo";
      static constexpr uint32_t kApiInfoChunkVersion = 2;  // TraceChunk::ApiChunkVersion
      struct RdfApiInfo {
        uint32_t api_type;           // RgpRdfTraceApiType
        uint16_t api_version_major;
        uint16_t api_version_minor;
      };
      // RDF ApiType values: kOpenCl=6, kHip=8 (matches RgpRdfTraceApiType).
      static constexpr uint32_t kRdfApiTypeOpenCl = 6;
      static constexpr uint32_t kRdfApiTypeHip    = 8;
      RdfApiInfo api_info = {};
      api_info.api_type          = amd::IS_HIP ? kRdfApiTypeHip : kRdfApiTypeOpenCl;
      api_info.api_version_major = 0;
      api_info.api_version_minor = 0;
      roc_trace_session_->WriteDataChunk(kApiInfoChunkId, kApiInfoChunkVersion,
                                         nullptr, 0,
                                         &api_info, sizeof(api_info));

      // Write "AsicInfo" RDF chunk — mandatory for RGP to open the file.
      // Mirrors PAL's AsicInfoTraceSource::OnTraceFinished() → WriteDataChunk("AsicInfo").
      // Layout must match TraceChunk::AsicInfo in pal/src/gpuUtil/asicInfoTraceSource.h exactly.
      // Version 3 matches AsicInfoChunkVersion in that header.
      {
        static const char kAsicInfoChunkId[kRdfIdentifierSize] = "AsicInfo";
        static constexpr uint32_t kAsicInfoChunkVersion = 3;

        // Mirror of TraceChunk::AsicInfo.  Explicit _pad fields reproduce the implicit
        // padding the compiler inserts in the PAL struct so both sides agree on layout.
        struct RdfAsicInfo {
          uint32_t pciId;
          uint32_t _pad0;                    // align uint64 fields to 8 bytes
          uint64_t shaderCoreClockFrequency;
          uint64_t memoryClockFrequency;
          uint64_t gpuTimestampFrequency;
          uint64_t maxShaderCoreClock;
          uint64_t maxMemoryClock;
          int32_t  deviceId;
          int32_t  deviceRevisionId;
          int32_t  vgprsPerSimd;
          int32_t  sgprsPerSimd;
          int32_t  shaderEngines;
          int32_t  computeUnitPerShaderEngine;
          int32_t  simdPerComputeUnit;
          int32_t  wavefrontsPerSimd;
          int32_t  minimumVgprAlloc;
          int32_t  vgprAllocGranularity;
          int32_t  minimumSgprAlloc;
          int32_t  sgprAllocGranularity;
          int32_t  hardwareContexts;
          uint32_t gpuType;                  // TraceGpuType: Discrete=2
          struct { uint16_t major, minor, stepping; } gfxIpLevel;
          uint16_t _pad1;                    // align int32 gpuIndex to 4 bytes
          int32_t  gpuIndex;
          int32_t  ceRamSize;
          int32_t  ceRamSizeGraphics;
          int32_t  ceRamSizeCompute;
          int32_t  maxNumberOfDedicatedCus;
          uint32_t _pad2;                    // align int64 vramSize to 8 bytes
          int64_t  vramSize;
          int32_t  vramBusWidth;
          int32_t  l2CacheSize;
          int32_t  l1CacheSize;
          int32_t  ldsSize;
          char     gpuName[256];
          float    aluPerClock;
          float    texturePerClock;
          float    primsPerClock;
          float    pixelsPerClock;
          uint32_t memoryOpsPerClock;
          uint32_t memoryChipType;           // TraceMemoryType: Hbm2=10
          uint32_t ldsGranularity;
          uint16_t cuMask[32][2];
          uint32_t pixelPackerMask[4];
          uint32_t gl1CacheSize;
          uint32_t instCacheSize;
          uint32_t scalarCacheSize;
          uint32_t mallCacheSize;
        };
        static_assert(sizeof(RdfAsicInfo) == 608,
                      "RdfAsicInfo size mismatch with TraceChunk::AsicInfo");

        uint64_t ai_ts_freq = 0;
        Hsa::agent_get_info(device_->getBackendDevice(),
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_TIMESTAMP_FREQUENCY),
                            &ai_ts_freq);
        uint32_t ai_mem_width = 0;
        Hsa::agent_get_info(device_->getBackendDevice(),
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MEMORY_WIDTH),
                            &ai_mem_width);
        uint32_t ai_asic_rev = 0;
        Hsa::agent_get_info(device_->getBackendDevice(),
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_ASIC_REVISION),
                            &ai_asic_rev);
        uint32_t ai_max_waves = 0;
        Hsa::agent_get_info(device_->getBackendDevice(),
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_MAX_WAVES_PER_CU),
                            &ai_max_waves);
        uint32_t ai_sa_per_se = 1;
        Hsa::agent_get_info(device_->getBackendDevice(),
                            static_cast<hsa_agent_info_t>(HSA_AMD_AGENT_INFO_NUM_SHADER_ARRAYS_PER_SE),
                            &ai_sa_per_se);
        if (ai_sa_per_se < 1) ai_sa_per_se = 1;

        const device::Info& ai_di    = device_->info();
        const amd::Isa&     ai_isa   = device_->isa();
        const uint32_t ai_gfx_maj    = ai_isa.versionMajor();
        const uint32_t ai_gfx_min    = ai_isa.versionMinor();
        const int32_t  ai_se_count   = static_cast<int32_t>(
                                         iter_ctx.entries.size() > 0
                                         ? iter_ctx.entries.size() : 1u);
        const uint32_t ai_total_cu   = ai_di.maxComputeUnits_;
        const uint64_t ai_engine_mhz = ai_di.maxEngineClockFrequency_;
        const uint64_t ai_mem_mhz    = ai_di.maxMemoryClockFrequency_;
        const uint32_t ai_wpsimd     = (ai_di.simdPerCU_ > 0)
                                       ? (ai_max_waves / ai_di.simdPerCU_) : ai_max_waves;
        const uint32_t ai_cu_per_sa  = (ai_se_count > 0 && ai_sa_per_se > 0)
                                       ? (ai_total_cu / (static_cast<uint32_t>(ai_se_count)
                                                         * ai_sa_per_se)) : 0;
        const uint16_t ai_cu_mask    = (ai_cu_per_sa >= 16) ? 0xFFFFu
                                     : (ai_cu_per_sa > 0)
                                       ? static_cast<uint16_t>((1u << ai_cu_per_sa) - 1u) : 0u;

        RdfAsicInfo ai = {};
        ai.pciId                      = ai_di.pcieDeviceId_;
        ai.shaderCoreClockFrequency   = ai_engine_mhz * 1000000ULL;
        ai.memoryClockFrequency       = ai_mem_mhz    * 1000000ULL;
        ai.gpuTimestampFrequency      = ai_ts_freq;
        ai.maxShaderCoreClock         = ai_engine_mhz * 1000000ULL;
        ai.maxMemoryClock             = ai_mem_mhz    * 1000000ULL;
        ai.deviceId                   = static_cast<int32_t>(ai_di.pcieDeviceId_);
        ai.deviceRevisionId           = static_cast<int32_t>(ai_asic_rev);
        ai.vgprsPerSimd               = static_cast<int32_t>(ai_di.vgprsPerSimd_);
        ai.sgprsPerSimd               = static_cast<int32_t>(ai_di.sgprsPerSimd_);
        ai.shaderEngines              = ai_se_count;
        ai.computeUnitPerShaderEngine = (ai_se_count > 0)
                                        ? static_cast<int32_t>(
                                            ai_total_cu / static_cast<uint32_t>(ai_se_count))
                                        : 0;
        ai.simdPerComputeUnit         = static_cast<int32_t>(ai_di.simdPerCU_);
        ai.wavefrontsPerSimd          = static_cast<int32_t>(ai_wpsimd);
        ai.minimumVgprAlloc           = static_cast<int32_t>(ai_di.vgprAllocGranularity_);
        ai.vgprAllocGranularity       = static_cast<int32_t>(ai_di.vgprAllocGranularity_);
        ai.gpuType                    = 2;  // TraceGpuType::Discrete
        ai.gfxIpLevel.major           = static_cast<uint16_t>(ai_gfx_maj);
        ai.gfxIpLevel.minor           = static_cast<uint16_t>(ai_gfx_min);
        ai.gfxIpLevel.stepping        = static_cast<uint16_t>(ai_isa.versionStepping());
        ai.vramSize                   = static_cast<int64_t>(ai_di.globalMemSize_);
        ai.vramBusWidth               = static_cast<int32_t>(ai_mem_width);
        ai.l2CacheSize                = static_cast<int32_t>(ai_di.l2CacheSize_);
        ai.ldsSize                    = static_cast<int32_t>(ai_isa.localMemSizePerCU());
        strncpy(ai.gpuName, ai_di.boardName_, sizeof(ai.gpuName) - 1);
        ai.memoryChipType             = 10;  // PAL TraceMemoryType::Hbm2 (matches what AsicInfoTraceSource writes)
        ai.ldsGranularity             = (ai_gfx_maj == 10 && ai_gfx_min >= 3) ? 1024u : 512u;
        // GL1 cache size per shader array — mirrors PAL gfx9Device.cpp hardcoded values.
        // gfx12+ have no GL1 cache (0). gfx11 = 256 KiB, gfx10 = 128 KiB.
        if (ai_gfx_maj == 11)       { ai.gl1CacheSize = 256u * 1024u; }
        else if (ai_gfx_maj == 10)  { ai.gl1CacheSize = 128u * 1024u; }
        else                        { ai.gl1CacheSize = 0u; }
        for (uint32_t se = 0; se < static_cast<uint32_t>(ai_se_count) && se < 32u; ++se) {
          for (uint32_t sa = 0; sa < ai_sa_per_se && sa < 2u; ++sa) {
            ai.cuMask[se][sa] = ai_cu_mask;
          }
        }

        roc_trace_session_->WriteDataChunk(kAsicInfoChunkId, kAsicInfoChunkVersion,
                                           nullptr, 0,
                                           &ai, sizeof(ai));
      }

      // Write "ClockCalibration" RDF chunk — provides a CPU/GPU timestamp pair for
      // timeline correlation.  Mirrors PAL's ClockCalibrationTraceSource::OnTraceFinished().
      // The chunk ID is exactly 16 bytes with no null terminator.
      // Version 2 matches ClockCalibChunkVersion in pal/src/gpuUtil/clockCalibTraceSource.h.
      {
        static const char kClockCalibChunkId[kRdfIdentifierSize] =
            {'C','l','o','c','k','C','a','l','i','b','r','a','t','i','o','n'};
        static constexpr uint32_t kClockCalibChunkVersion = 2;

        // Mirror of TraceChunkClockCalibration in clockCalibTraceSource.h.
        struct RdfClockCalib {
          uint32_t pciId;
          // 4 bytes implicit padding (uint64 alignment)
          uint64_t cpuTimestamp;
          uint64_t gpuTimestamp;
        };
        static_assert(sizeof(RdfClockCalib) == 24,
                      "RdfClockCalib size mismatch with TraceChunkClockCalibration");

        RdfClockCalib cc = {};
        cc.pciId        = device_->info().pcieDeviceId_;
        cc.cpuTimestamp = GetSystemTimestamp();
        cc.gpuTimestamp = 0;  // GPU timestamp not available on this path

        roc_trace_session_->WriteDataChunk(kClockCalibChunkId, kClockCalibChunkVersion,
                                           nullptr, 0,
                                           &cc, sizeof(cc));
      }

      // Write "TraceConfig" RDF chunk — mandatory for RGP to open the file.
      // Contains a JSON string describing the trace controller configuration.
      // Mirrors PAL's TraceConfigTraceSource::OnTraceFinished() → WriteDataChunk("TraceConfig").
      // Version 1 matches the only accepted version in rgp_rdf_file_loader.cpp.
      {
        static const char kTraceConfigChunkId[kRdfIdentifierSize] = "TraceConfig";
        // Minimal valid JSON: controller "renderop" (HIP/OpenCL compute path) with
        // renderOpMode "dispatch".  RGP's GetProfileConfig() requires both fields.
        static const char kTraceConfigJson[] =
            R"({"controller":{"name":"renderop","config":{"renderOpMode":"dispatch"}}})";
        roc_trace_session_->WriteDataChunk(kTraceConfigChunkId, 1,
                                           nullptr, 0,
                                           kTraceConfigJson,
                                           sizeof(kTraceConfigJson) - 1);  // exclude NUL
      }

      // Write "QueueInfo" and "QueueEvent" RDF chunks — both mandatory for RGP.
      // RGP uses GetChunkCount() (not just ContainsChunk) for these chunks and iterates
      // over each instance reading one struct per chunk.  Writing a 0-byte chunk creates
      // one instance whose data is uninitialized, causing the queue_id lookup to fail
      // with kRgpErrorMalformedData.  We must write one properly-initialized entry each:
      //   QueueInfo:  describes the HSA compute queue (queue_id=0, Compute engine).
      //   QueueEvent: one CmdBufSubmit record referencing queue_id=0.
      // The HIP path has no GPU-side queue semaphore timestamps; gpu_timestamp_* are 0.
      {
        static const char kQueueInfoChunkId[kRdfIdentifierSize]  = "QueueInfo";
        static const char kQueueEventChunkId[kRdfIdentifierSize] = "QueueEvent";
        const uint32_t qi_pci_id = device_->info().pcieDeviceId_;

        // One QueueInfo entry: compute queue, queue_id=0 (no OS context).
        struct RdfQueueInfo {
          uint32_t pciId;
          uint32_t _pad;       // align uint64 queue_id
          uint64_t queueId;
          uint64_t queueContext;
          uint8_t  queueType;  // RgpRdfQueueType::kCompute = 2
          uint8_t  engineType; // RgpRdfHwEngineType::kCompute = 2
        };
        static_assert(sizeof(RdfQueueInfo) == 32, "RdfQueueInfo size mismatch");
        const RdfQueueInfo qi = { qi_pci_id, 0, /*queueId=*/0, /*ctx=*/0, 2, 2 };
        roc_trace_session_->WriteDataChunk(kQueueInfoChunkId, 1,
                                           nullptr, 0, &qi, sizeof(qi));

        // One QueueEvent chunk per dispatch — RGP iterates by chunk index
        // (GetChunkCount + ReadChunkDataToBuffer(id, i, &record)), so each dispatch
        // must be its own chunk.  gpu_timestamp_* = 0 (no HW timestamps in HIP path).
        struct RdfQueueEvent {
          uint32_t pciId;
          uint32_t _pad;        // align uint64 queue_id
          uint64_t queueId;
          uint32_t eventType;   // RgpRdfQueueEventType::kCmdBufSubmit = 0
          uint32_t sqttCmdBufId;
          uint64_t frameIndex;
          uint32_t submitSubIndex;
          uint32_t _pad2;
          uint64_t apiEventId;
          uint64_t cpuTimestamp;
          uint64_t gpuTimestamp1;
          uint64_t gpuTimestamp2;
        };
        static_assert(sizeof(RdfQueueEvent) == 72, "RdfQueueEvent size mismatch");

        if (pending_queue_events_.empty()) {
          // Fallback: at least one entry so RGP doesn't reject the file.
          const RdfQueueEvent qe = { qi_pci_id, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
          roc_trace_session_->WriteDataChunk(kQueueEventChunkId, 1,
                                             nullptr, 0, &qe, sizeof(qe));
        } else {
          for (const PendingQueueEvent& pqe : pending_queue_events_) {
            RdfQueueEvent qe = {};
            qe.pciId         = qi_pci_id;
            qe.queueId       = 0;
            qe.eventType     = 0;  // kCmdBufSubmit
            qe.sqttCmdBufId  = pqe.sqttCmdBufId;
            qe.frameIndex    = 0;
            qe.submitSubIndex= pqe.submitSubIndex;
            qe.apiEventId    = pqe.apiEventId;
            qe.cpuTimestamp  = pqe.cpuTimestamp;
            roc_trace_session_->WriteDataChunk(kQueueEventChunkId, 1,
                                               nullptr, 0, &qe, sizeof(qe));
          }
          pending_queue_events_.clear();
        }
      }
    }

    device_->hostFree(host_buf, copy_size);
  }

  // Write all buffered ELF data (CodeObject + PsoCorrelation) and COLoadEvent entries.
  // Programs are loaded before the trace starts so these are deferred to here.
  // Must happen before EndTrace() seals the session.
  FlushPendingTraceData();

  // Free device-side SQTT buffers now — all SDMA copies from sqtt_output_ are done.
  // sqtt_output_ is no longer referenced after this point.
  FreeSqttResources();
  sqtt_state_ = SqttState::Idle;

  // Signal the UberTrace RPC layer that data is ready for CollectTrace().
  // Placed AFTER hostFree and FreeSqttResources so that state=Completed is only visible
  // to the polling thread once the SDMA copy and all RDF chunk writes are fully committed.
  if (roc_trace_session_ != nullptr) {
    const bool ended = roc_trace_session_->EndTrace();
    fprintf(stderr, "[CLR-Ctrl] CollectSqttResults: EndTrace returned %s, session state=%d\n",
            ended ? "true" : "false",
            static_cast<int>(roc_trace_session_->GetState()));
  }
}

// ================================================================================================
void RocUberTraceCaptureMgr::FreeSqttResources() {
  if (sqtt_cmd_buf_ != nullptr) {
    // Command buffer was allocated with hostAlloc (fine-grained system memory).
    device_->hostFree(sqtt_cmd_buf_, sqtt_cmd_buf_size_);
    sqtt_cmd_buf_      = nullptr;
    sqtt_cmd_buf_size_ = 0;
  }
  if (marker_cmd_buf_ != nullptr) {
    device_->hostFree(marker_cmd_buf_, kMarkerRingSize * kMarkerSlotBytes);
    marker_cmd_buf_ = nullptr;
  }
  if (sqtt_output_ != nullptr) {
    // Output buffer may be device-local (VRAM) or host fallback — hostFree() calls
    // Hsa::memory_pool_free() which works for both HSA pool types.
    device_->hostFree(sqtt_output_, sqtt_output_size_);
    sqtt_output_      = nullptr;
    sqtt_output_size_ = 0;
  }
  sqtt_profile_.command_buffer = {};
  sqtt_profile_.output_buffer  = {};
  trace_gpu_ = nullptr;
  pending_queue_events_.clear();
}

// ================================================================================================
// PreDispatch: called before every kernel submission.
// Drives the SQTT hardware capture state machine (UberTrace-driven) and writes SQTT
// correlation markers into the HSA queue when capture is active.
void RocUberTraceCaptureMgr::PreDispatch(VirtualGPU* gpu, const Kernel& kernel, size_t x,
                                         size_t y, size_t z) {
  WaitForDriverResume();

  // ── SQTT state machine ────────────────────────────────────────────────────────────────────
  // Entry point 1: UberTrace RPC path — OnTraceRequested() already set sqtt_state_ = Preparing.
  // Entry point 2: legacy RGP protocol — tool called EnableProfilingRequest; poll IsTracePending().
  if (sqtt_state_ == SqttState::Idle && rgp_server_ != nullptr && rgp_server_->IsTracePending()) {
    // Read trace parameters and enter Preparing state (mirrors PAL's PrepareRGPTrace).
    const auto params         = rgp_server_->QueryTraceParameters();
    num_prep_frames_          = (params.numPreparationFrames > 0) ? params.numPreparationFrames : 2;
    const uint32_t mb         = (params.gpuMemoryLimitInMb > 0) ? (params.gpuMemoryLimitInMb / 2) : 4;
    sqtt_output_size_         = mb * 1024u * 1024u;
    sqtt_se_mask_             = params.seMask;
    sqtt_instruction_tokens_  = (params.flags.enableInstructionTokens != 0);
    sqtt_capture_code_objects_= (params.flags.captureDriverCodeObjects != 0);
    capture_index_mode_       = (params.captureMode ==
                                 DevDriver::RGPProtocol::CaptureTriggerMode::Index);
    capture_start_index_      = capture_index_mode_ ? params.captureStartIndex : 0;
    capture_stop_index_       = capture_index_mode_ ? params.captureStopIndex  : 0;
    prep_disp_count_          = 0;
    sqtt_state_               = SqttState::Preparing;
  }

  // Count prep dispatches, then begin SQTT hardware capture.
  // In Index mode, prep counting deferred until captureStartIndex is reached.
  if (sqtt_state_ == SqttState::Preparing) {
    if (!capture_index_mode_ || global_disp_count_ >= capture_start_index_) {
      if (++prep_disp_count_ >= num_prep_frames_) {
        BeginSqttTrace(gpu);  // advances sqtt_state_ to Running, sets trace_running_
      }
    }
  }

  if (IsTraceRunning()) {
    RgpSqttMarkerEventType apiEvent = RgpSqttMarkerEventType::CmdNDRangeKernel;

    if (kernel.isInternalKernel()) {
      static constexpr RgpSqttMarkerEventType kBlitApiEvents[KernelBlitManager::BlitTotal] = {
          RgpSqttMarkerEventType::CmdCopyImage,
          RgpSqttMarkerEventType::CmdCopyImage,
          RgpSqttMarkerEventType::CmdCopyImageToBuffer,
          RgpSqttMarkerEventType::CmdCopyBufferToImage,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdCopyBuffer,
          RgpSqttMarkerEventType::CmdFillBuffer,
          RgpSqttMarkerEventType::CmdFillImage,
          RgpSqttMarkerEventType::CmdScheduler,
      };
      for (uint32_t i = 0; i < KernelBlitManager::BlitTotal; ++i) {
        if (kernel.name().compare(BlitName[i]) == 0) {
          apiEvent = kBlitApiEvents[i];
          break;
        }
      }
    }

    WriteCbStartMarker(gpu);
    WriteComputeBindMarker(gpu, kernel.ApiHash());
    WriteUserEventMarker(gpu, RgpSqttMarkerUserEventObjectName, kernel.name());
    WriteEventWithDimsMarker(gpu, apiEvent, static_cast<uint32_t>(x), static_cast<uint32_t>(y),
                             static_cast<uint32_t>(z));
    WriteCbEndMarker(gpu);

    // Record a QueueEvent for this dispatch.  RGP reads one QueueEvent chunk per dispatch
    // (GetChunkCount + indexed ReadChunkDataToBuffer), so we accumulate here and write
    // one chunk per entry in CollectSqttResults.
    // current_event_id_ was just incremented by WriteEventWithDimsMarker(); subtract 1 to
    // get the cmdID that was written into the Event marker for this dispatch.
    PendingQueueEvent qe = {};
    qe.cpuTimestamp   = GetSystemTimestamp();
    qe.sqttCmdBufId   = (gpu->gpu_queue() != nullptr)
                            ? static_cast<uint32_t>(gpu->gpu_queue()->doorbell_signal.handle & 0xFFFFFu)
                            : 0u;
    qe.submitSubIndex = 0;
    qe.apiEventId     = current_event_id_ - 1;
    pending_queue_events_.push_back(qe);
  }

  global_disp_count_++;
}

// ================================================================================================
// PostDispatch: advance the SQTT capture state machine after each dispatch.
//
// Three stop paths:
//  1. pending_abort_ — set by OnTraceCanceled() on the RPC thread; triggers aborted finish.
//  2. Index mode auto-stop — when global_disp_count_ reaches captureStopIndex, finish normally.
//  3. Legacy RGP path auto-stop — after one capture dispatch, mirrors PAL's one-frame capture.
//     UberTrace path (roc_trace_session_ != nullptr, non-index) relies on CancelTrace() instead.
void RocUberTraceCaptureMgr::PostDispatch(VirtualGPU* gpu) {
  if (sqtt_state_ != SqttState::Running) return;

  // RPC-driven abort (tool called CancelTrace() — deferred from the RPC thread).
  if (pending_abort_.load(std::memory_order_acquire)) {
    pending_abort_.store(false, std::memory_order_relaxed);
    FinishRGPTrace(gpu, /*aborted=*/true);
    return;
  }

  // Dispatch-count auto-stop: used by both index-mode and relative-mode ubertrace captures.
  // capture_stop_index_ is set to (captureStartIndex + numPrep + captureDispatchCount) for
  // index mode, or to (global_disp_count_at_start + captureDispatchCount) for relative mode
  // (set in BeginSqttTrace when capture starts).  0 means unlimited — tool calls CancelTrace.
  if (capture_stop_index_ != 0 && global_disp_count_ >= capture_stop_index_) {
    FinishRGPTrace(gpu, /*aborted=*/false);
    return;
  }

  // Legacy RGP auto-stop: when not using ubertrace dispatch count and the trace was started
  // via the legacy IsTracePending() path, finish after one capture dispatch — mirrors PAL's
  // one-frame capture boundary.
  if (capture_stop_index_ == 0 && rgp_server_ != nullptr && rgp_server_->IsTraceRunning()) {
    FinishRGPTrace(gpu, /*aborted=*/false);
  }
}

// ================================================================================================
// FinishRGPTrace: called when the tool wants to end the trace (either normally or aborted).
// Stops SQTT hardware, reads back the data, writes the RDF chunk, and advances the session.
void RocUberTraceCaptureMgr::FinishRGPTrace(VirtualGPU* gpu, bool aborted) {
  if (aborted) {
    // Abort: stop hardware if running, discard data, reset state.
    if (sqtt_state_ == SqttState::Running) {
      EndSqttTrace(gpu);
    }
    if (rgp_server_ != nullptr && rgp_server_->IsTraceRunning()) {
      rgp_server_->AbortTrace();  // notifies tool (legacy RGP path only)
    }
    if (roc_trace_session_ != nullptr) {
      roc_trace_session_->CancelTrace();
    }
    FreeSqttResources();
    sqtt_state_    = SqttState::Idle;
    trace_running_ = false;
    return;
  }

  // Normal finish: stop SQTT, collect results, notify tool via RGP server.
  if (sqtt_state_ == SqttState::Running) {
    EndSqttTrace(gpu);
  }
  if (sqtt_state_ == SqttState::WaitingForResults) {
    CollectSqttResults(gpu);  // writes RDF chunk and calls roc_trace_session_->EndTrace()
    // Close the DevDriver RGP session regardless of which delivery path was used.
    // The legacy RGP condition (roc_trace_session_ == nullptr) was previously guarding
    // this, leaving the RGP server stuck in TraceRunning in the UberTrace path — the
    // tool waits for this session closure before writing the output file.
    if (rgp_server_ != nullptr && rgp_server_->IsTraceRunning()) {
      rgp_server_->EndTrace();
    }
  }
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteBarrierStartMarker(const VirtualGPU* gpu,
                                                     uint32_t reason) const {
  if (!IsTraceRunning()) {
    return;
  }

  std::lock_guard<std::mutex> traceLock(trace_mutex_);

  RgpSqttMarkerBarrierStart marker = {};
  marker.identifier = RgpSqttMarkerIdentifierBarrierStart;
  marker.extDwords  = 0;
  marker.internal = true;
  marker.driverReason = reason;

  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteBarrierEndMarker(const VirtualGPU* gpu) const {
  if (!IsTraceRunning()) {
    return;
  }

  std::lock_guard<std::mutex> traceLock(trace_mutex_);

  RgpSqttMarkerBarrierEnd marker = {};
  marker.identifier = RgpSqttMarkerIdentifierBarrierEnd;
  marker.extDwords  = 0;

  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
bool RocUberTraceCaptureMgr::RegisterTimedQueue(uint32_t queue_id, hsa_queue_t* queue,
                                                bool* debug_vmid) const {
  *debug_vmid = false;

  if (queue == nullptr) {
    return false;
  }

  // Store the queue pointer keyed by the caller-supplied queue_id so that
  // TimedQueueSubmit() can locate the HSA queue handle when needed.
  std::lock_guard<std::mutex> lock(timed_queues_mutex_);
  timed_queues_.emplace(queue_id, queue);
  return true;
}

// ================================================================================================
bool RocUberTraceCaptureMgr::TimedQueueSubmit(hsa_queue_t* /*queue*/, uint64_t /*cmd_id*/,
                                              hsa_signal_t /*completion_signal*/) const {
  // HSA does not expose a direct equivalent of PAL's QueueTimingsTraceSource::TimedSubmit().
  // Return false so the caller falls back to its normal HSA packet-ring submission path.
  // Queue timing data will be correlated via the SQTT timestamp stream instead.
  return false;
}

// ================================================================================================
uint64_t RocUberTraceCaptureMgr::AddElfBinary(const void* /*exe_binary*/, size_t /*exe_binary_size*/,
                                              const void* elf_binary, size_t elf_binary_size,
                                              uint64_t /*gpu_addr*/) {
  if (elf_binary == nullptr || elf_binary_size == 0) return 0;

  // Compute stable hash of the device ELF — used as apiPsoHash in SQTT PipelineBind markers,
  // as code_object_hash in CodeObject/COLoadEvent chunks, and as apiPsoHash in PsoCorrelation.
  const uint64_t original_hash = Fnv1a64(elf_binary, elf_binary_size);

  // Buffer the ELF binary for deferred writing at trace-end.
  // Programs are typically loaded BEFORE the trace starts (trace_running_=false), so we
  // cannot write RDF chunks here — the RDF session is not yet open or has not started.
  // FlushPendingTraceData() writes CodeObject + PsoCorrelation when the session IS open.
  // Deduplication by hash avoids writing the same ELF twice if the program is reloaded.
  std::lock_guard<std::mutex> lk(trace_mutex_);
  const bool already_buffered = std::any_of(pending_elfs_.begin(), pending_elfs_.end(),
      [original_hash](const PendingElfData& p) { return p.original_hash == original_hash; });
  if (!already_buffered) {
    PendingElfData entry;
    entry.elf.assign(static_cast<const uint8_t*>(elf_binary),
                     static_cast<const uint8_t*>(elf_binary) + elf_binary_size);
    entry.original_hash = original_hash;
    entry.pci_id        = device_->info().pcieDeviceId_;
    pending_elfs_.push_back(std::move(entry));
    fprintf(stderr, "[CLR-Uber] AddElfBinary: buffered ELF hash=0x%llx size=%zu\n",
            static_cast<unsigned long long>(original_hash), elf_binary_size);
  }

  return original_hash;
}

// ================================================================================================
// Minimal ELF64 symbol table lookup: returns the st_value of the first symbol whose name
// matches kernel_name, or 0 if not found.  Handles both SHT_SYMTAB and SHT_DYNSYM.
// Uses only standard ELF structs from <elf.h> / the ELF spec — no external libs required.
static uint64_t FindElfSymbolValue(const void* elf_binary, size_t elf_binary_size,
                                   const char* kernel_name) {
  using namespace amd::ELFIO;
  if (elf_binary == nullptr || elf_binary_size < sizeof(Elf64_Ehdr) || kernel_name == nullptr) {
    return 0;
  }
  const uint8_t* base = static_cast<const uint8_t*>(elf_binary);
  const Elf64_Ehdr* ehdr = reinterpret_cast<const Elf64_Ehdr*>(base);

  // Validate ELF magic and 64-bit class.
  if (ehdr->e_ident[EI_MAG0] != ELFMAG0 || ehdr->e_ident[EI_MAG1] != ELFMAG1 ||
      ehdr->e_ident[EI_MAG2] != ELFMAG2 || ehdr->e_ident[EI_MAG3] != ELFMAG3 ||
      ehdr->e_ident[EI_CLASS] != ELFCLASS64) {
    return 0;
  }

  if (ehdr->e_shoff == 0 || ehdr->e_shentsize < sizeof(Elf64_Shdr)) return 0;

  // Bounds-check section header table.
  const uint64_t shdr_end = ehdr->e_shoff +
      static_cast<uint64_t>(ehdr->e_shnum) * ehdr->e_shentsize;
  if (shdr_end > elf_binary_size) return 0;

  const Elf64_Shdr* shdrs = reinterpret_cast<const Elf64_Shdr*>(base + ehdr->e_shoff);

  // Walk sections looking for SHT_SYMTAB or SHT_DYNSYM.
  for (uint16_t i = 0; i < ehdr->e_shnum; ++i) {
    const Elf64_Shdr& shdr = shdrs[i];
    if (shdr.sh_type != SHT_SYMTAB && shdr.sh_type != SHT_DYNSYM) continue;
    if (shdr.sh_entsize < sizeof(Elf64_Sym)) continue;
    if (shdr.sh_offset + shdr.sh_size > elf_binary_size) continue;
    if (shdr.sh_link >= ehdr->e_shnum) continue;

    // Associated string table section.
    const Elf64_Shdr& strtab_shdr = shdrs[shdr.sh_link];
    if (strtab_shdr.sh_offset + strtab_shdr.sh_size > elf_binary_size) continue;
    const char* strtab = reinterpret_cast<const char*>(base + strtab_shdr.sh_offset);
    const size_t strtab_size = static_cast<size_t>(strtab_shdr.sh_size);

    const Elf64_Sym* syms = reinterpret_cast<const Elf64_Sym*>(base + shdr.sh_offset);
    const uint64_t nsyms = shdr.sh_size / shdr.sh_entsize;
    const size_t kname_len = strlen(kernel_name);

    // Two-pass: prefer the ".kd" symbol (HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT
    // returns the kernel descriptor VA, so base_address must be computed against the
    // ".kd" symbol offset to get the correct code-segment load base).
    // Fall back to the bare kernel name (COv2 code objects without ".kd").
    uint64_t kd_value   = 0;
    uint64_t bare_value = 0;
    bool found_kd   = false;
    bool found_bare = false;
    for (uint64_t j = 0; j < nsyms; ++j) {
      const Elf64_Sym& sym = syms[j];
      if (sym.st_name >= strtab_size) continue;
      const char* sym_name = strtab + sym.st_name;
      if (!found_kd &&
          strncmp(sym_name, kernel_name, kname_len) == 0 &&
          strcmp(sym_name + kname_len, ".kd") == 0) {
        kd_value = sym.st_value;
        found_kd = true;
      }
      if (!found_bare && strcmp(sym_name, kernel_name) == 0) {
        bare_value = sym.st_value;
        found_bare = true;
      }
      if (found_kd && found_bare) break;
    }
    if (found_kd)   return kd_value;
    if (found_bare) return bare_value;
  }
  return 0;
}

// ================================================================================================
// Emits a COLoadEvent RDF chunk for a single kernel once its GPU VA is known.
// Called from rocprogram.cpp after HSA resolves each kernel's KERNEL_OBJECT address.
//
// base_address = kernel_gpu_va - elf_sym_value(kernel_name)
// This ensures that: base_address + elf_sym_value == kernel_gpu_va
// which is exactly what RGP's loader event address map uses to look up dispatched shaders.
void RocUberTraceCaptureMgr::AddKernelLoadEvent(uint64_t api_hash,
                                                const void* elf_binary, size_t elf_binary_size,
                                                const char* kernel_name,
                                                uint64_t kernel_gpu_va) {
  // Always accumulate regardless of trace_running_ state: PAL records code objects
  // throughout the session lifetime (pre-trace programs included) and writes them all
  // when the trace finishes.  sqtt_capture_code_objects_ is checked only at flush time.
  if (elf_binary == nullptr || elf_binary_size == 0 || kernel_name == nullptr) return;

  // Find the ELF symbol offset for this kernel to compute the code segment base address.
  const uint64_t sym_value = FindElfSymbolValue(elf_binary, elf_binary_size, kernel_name);
  const uint64_t base_address = (sym_value <= kernel_gpu_va) ? (kernel_gpu_va - sym_value) : 0;

  fprintf(stderr, "[CLR-Uber] AddKernelLoadEvent: kernel=%s gpu_va=0x%llx sym_value=0x%llx"
          " base_addr=0x%llx api_hash=0x%llx\n",
          kernel_name ? kernel_name : "(null)",
          static_cast<unsigned long long>(kernel_gpu_va),
          static_cast<unsigned long long>(sym_value),
          static_cast<unsigned long long>(base_address),
          static_cast<unsigned long long>(api_hash));

  // Accumulate — all entries are flushed as a single COLoadEvent chunk in
  // FlushKernelLoadEvents() because RGP only reads one COLoadEvent chunk per file.
  std::lock_guard<std::mutex> traceLock(trace_mutex_);
  const uint32_t pci_id = device_->info().pcieDeviceId_;
  pending_load_events_.push_back({pci_id, 0u, base_address, api_hash, 0u,
                                  GetSystemTimestamp()});
}

// ================================================================================================
// Writes all buffered ELF data (CodeObject + PsoCorrelation) and accumulated COLoadEvent
// entries as a single set of RDF chunks.  Called from CollectSqttResults() before EndTrace().
//
// Programs are loaded before the trace starts, so all three chunk types are deferred here
// rather than written at program-load time.  RGP needs all three to resolve kernel names:
//   CodeObject   → contains the device ELF; RGP parses its .symtab for symbol offsets
//   PsoCorrelation → links apiPsoHash (from PipelineBind marker) to code_object_hash
//   COLoadEvent  → maps (base_address + sym_offset) == kernelCodeHandle_ for address lookup
void RocUberTraceCaptureMgr::FlushPendingTraceData() {
  if (roc_trace_session_ == nullptr || !sqtt_capture_code_objects_) {
    pending_elfs_.clear();
    pending_load_events_.clear();
    return;
  }

  // ── CodeObject + PsoCorrelation chunks (one per unique ELF) ─────────────────────────────
  struct CodeObjectHeader {
    uint32_t pciId;
    uint64_t hashLower;  // Rgp128bitHash::low  = original_hash
    uint64_t hashUpper;  // Rgp128bitHash::high = 0 (FNV1a is 64-bit)
  };
  struct PsoCorrelationHeader { uint32_t count; };
  struct PsoCorrelation {
    uint32_t pciId;
    uint64_t apiPsoHash;
    uint64_t internalHashStable;  // Rgp128bitHash::low  — must match COLoadEvent hashLower
    uint64_t internalHashUnique;  // Rgp128bitHash::high — must match COLoadEvent hashUpper (0)
    char     apiLevelObjectName[64];
  };
  static const char kCodeObjChunkId[kRdfIdentifierSize]  = "CodeObject";
  static const char kPsoCorrelChunkId[kRdfIdentifierSize] = "PsoCorrelation";

  fprintf(stderr, "[CLR-Uber] FlushPendingTraceData: %zu ELFs, %zu COLoadEvents\n",
          pending_elfs_.size(), pending_load_events_.size());

  // ── CodeObject chunks (one per unique ELF) ───────────────────────────────────────────────
  // PAL writes one CodeObject chunk per code object; RGP builds a database with one entry
  // per chunk.  Write these first so the database is populated before the correlation chunks.
  for (const PendingElfData& elf_entry : pending_elfs_) {
    const CodeObjectHeader co_hdr = { elf_entry.pci_id, elf_entry.original_hash, 0u };
    roc_trace_session_->WriteDataChunk(kCodeObjChunkId, /*version=*/2,
                                       &co_hdr, sizeof(co_hdr),
                                       elf_entry.elf.data(), elf_entry.elf.size());
  }

  // ── PsoCorrelation chunk (single chunk with ALL records) ─────────────────────────────────
  // PAL writes one PsoCorrelation chunk containing all records (count=N).
  // RGP reads only the first PsoCorrelation chunk and expects count == number of CodeObject
  // database entries.  Writing one chunk per ELF (count=1 each) causes a count mismatch and
  // a crash when RGP tries to index into the correlation array by code-object index.
  if (!pending_elfs_.empty()) {
    std::vector<PsoCorrelation> pso_records;
    pso_records.reserve(pending_elfs_.size());
    for (const PendingElfData& elf_entry : pending_elfs_) {
      PsoCorrelation pc = {};
      pc.pciId              = elf_entry.pci_id;
      pc.apiPsoHash         = elf_entry.original_hash;
      pc.internalHashStable = elf_entry.original_hash;
      pc.internalHashUnique = 0;
      pso_records.push_back(pc);
    }
    const PsoCorrelationHeader pc_hdr = {
        static_cast<uint32_t>(pso_records.size()) };
    roc_trace_session_->WriteDataChunk(kPsoCorrelChunkId, /*version=*/3,
                                       &pc_hdr, sizeof(pc_hdr),
                                       pso_records.data(),
                                       pso_records.size() * sizeof(PsoCorrelation));
  }

  pending_elfs_.clear();

  // ── COLoadEvent chunk (single chunk with all kernel entries) ─────────────────────────────
  // RGP reads only the first COLoadEvent chunk, so all entries must be in one chunk.
  if (!pending_load_events_.empty()) {
    struct CodeObjectLoadEventHeader { uint32_t count; };
    static const char kLoaderEventChunkId[kRdfIdentifierSize] = "COLoadEvent";
    const CodeObjectLoadEventHeader le_hdr = {
        static_cast<uint32_t>(pending_load_events_.size()) };
    roc_trace_session_->WriteDataChunk(kLoaderEventChunkId, /*version=*/3,
                                       &le_hdr, sizeof(le_hdr),
                                       pending_load_events_.data(),
                                       pending_load_events_.size() * sizeof(KernelLoadEntry));
    pending_load_events_.clear();
  }
}

// ================================================================================================
// Injects SQTT thread-trace marker data into the HSA compute queue.
//
// PAL writes SQTT markers by issuing SET_UCONFIG_REG packets to SQ_THREAD_TRACE_USERDATA_2/3
// (mmSQ_THREAD_TRACE_USERDATA_2 = 0xC342, USERDATA_3 = 0xC343).  The SQTT hardware
// intercepts these register writes and records each dword as a ThreadTraceMarker token in
// the SQTT stream.  IT_NOP does NOT generate ThreadTraceMarker tokens — it is silently
// discarded by the SQTT HW.
//
// Marker dwords are written in pairs via IT_SET_UCONFIG_REG packets (opcode 0x79):
//   [PM4_HDR(0x79, body=N)] [reg_offset] [data[0]] [data[1]] ...
// where reg_offset = mmSQ_THREAD_TRACE_USERDATA_2 - UCONFIG_SPACE_START = 0xC342 - 0xC000 = 0x342.
// Each pair writes USERDATA_2 then USERDATA_3 consecutively.  An odd trailing dword uses
// a 3-dword packet (body=1) targeting USERDATA_2 alone.
//
// The MEC firmware requires an IT_INDIRECT_BUFFER jump into GPU-accessible memory for
// vendor AQL packets; inline PM4 in pm4_command[] is rejected (EC_QUEUE_PACKET_VENDOR_UNSUPPORTED).
// So the SET_UCONFIG_REG body lives in marker_cmd_buf_ (fine-grained host memory) and
// is reached via an IT_INDIRECT_BUFFER in the AQL packet's pm4_command[] field.
//
// Callers must ensure data_size ≤ kMaxInlinePm4PayloadBytes (enforced by assert).
void RocUberTraceCaptureMgr::WriteMarker(const VirtualGPU* gpu, const void* data,
                                         size_t data_size) const {
  assert((data_size % sizeof(uint32_t)) == 0);
  assert(data_size > 0 && data_size <= kMaxInlinePm4PayloadBytes);
  if (gpu == nullptr || data == nullptr || data_size == 0) {
    return;
  }

  if (marker_cmd_buf_ == nullptr) {
    return;
  }

  // Pick the next slot in the marker IB ring buffer.
  const uint32_t slot_idx = marker_buf_idx_.fetch_add(1, std::memory_order_relaxed)
                            % kMarkerRingSize;
  auto* cmd = reinterpret_cast<uint32_t*>(
      reinterpret_cast<uintptr_t>(marker_cmd_buf_) + slot_idx * kMarkerSlotBytes);

  // Build SET_UCONFIG_REG packets for one marker slot.
  // IT_SET_UCONFIG_REG opcode = 0x79; UCONFIG_SPACE_START = 0xC000;
  // mmSQ_THREAD_TRACE_USERDATA_2 = 0xC342 → reg_offset = 0x342.
  // Mirrors PAL's CmdInsertRgpTraceMarker: raw struct bytes, pairs to USERDATA_2+3, no GRBM write.
  constexpr uint32_t kSetUconfigRegOpcode = 0x79;
  constexpr uint32_t kUserdataReg2Offset  = 0x342;  // USERDATA_2 - UCONFIG_SPACE_START

  const uint32_t  data_dwords = static_cast<uint32_t>(data_size / sizeof(uint32_t));
  const uint32_t* src         = static_cast<const uint32_t*>(data);
  uint32_t*       out         = cmd;

  for (uint32_t i = 0; i < data_dwords; ) {
    const uint32_t pair = std::min(data_dwords - i, 2u);
    *out++ = Pm4Header(kSetUconfigRegOpcode, /*body_dwords=*/1 + pair);
    *out++ = kUserdataReg2Offset;
    *out++ = src[i];
    if (pair == 2) *out++ = src[i + 1];
    i += pair;
  }
  const uint32_t cmd_dwords = static_cast<uint32_t>(out - cmd);

  // Build the amd_aql_pm4_ib vendor-specific AQL packet:
  //   pm4_command[0]   = ven_hdr = AMD_AQL_FORMAT_PM4_IB (0x1)
  //   pm4_command[1..8] = IT_INDIRECT_BUFFER (4-dword PM4 packet)
  //   pm4_command[9..]  = reserved (zeroed)
  constexpr uint16_t kAmdAqlFormatPm4Ib = 0x1;
  const uintptr_t buf_addr = reinterpret_cast<uintptr_t>(cmd);

  // IT_INDIRECT_BUFFER header: type=3, opcode=0x3F, count=2 (4-dword packet, count=pkt_size-2)
  const uint32_t ib_hdr = Pm4Header(kPacket3Indirect, /*body_dwords=*/3);
  // DW1: IB_BASE_LO — bits[31:2] of byte address (hardware shifts left 2).
  const uint32_t ib_lo  = static_cast<uint32_t>(buf_addr & 0xFFFFFFFFu) & ~3u;
  // DW2: IB_BASE_HI — upper 16 bits of address.
  const uint32_t ib_hi  = static_cast<uint32_t>(buf_addr >> 32) & 0xFFFFu;
  // DW3: IB_SIZE[19:0] | IB_VALID[23].  Matches HSA runtime ExecutePM4 exactly.
  const uint32_t ib_ctrl = (cmd_dwords & 0xFFFFFu) | (1u << 23);

  hsa_ext_amd_aql_pm4_packet_t aql = {};
  auto* p = reinterpret_cast<uint16_t*>(aql.pm4_command);
  p[0] = kAmdAqlFormatPm4Ib;                       // ven_hdr
  auto* ib = reinterpret_cast<uint32_t*>(&p[1]);    // ib_jump_cmd[0..3]
  ib[0] = ib_hdr;
  ib[1] = ib_lo;
  ib[2] = ib_hi;
  ib[3] = ib_ctrl;
  constexpr uint32_t kAmdAqlPm4IbDwCountRemain = 0xA;
  ib[4] = kAmdAqlPm4IbDwCountRemain;                // dw_cnt_remain: remaining DWORDs in AQL slot after ib_jump_cmd

  VirtualGPU* mutable_gpu = const_cast<VirtualGPU*>(gpu);
  mutable_gpu->dispatchCounterAqlPacket(&aql, PerfCounter::ROC_GFX9,
                                        /*blocking=*/true, nullptr);
}

// ================================================================================================
// WriteCbStartMarker / WriteCbEndMarker — wrap each dispatch group with command-buffer
// start/end markers so RGP's instrumentation processor can track command-buffer boundaries.
//
// device_id (64-bit) must be identical in CbStart and CbEnd.  We derive it from the HSA
// agent handle — a stable, unique-per-device value for the lifetime of the process.
// cbID uses the low 20 bits of the queue doorbell handle, matching WriteComputeBindMarker().
// queueIndex=0 and queueFlags=0: single compute queue, no semaphore-timing capability.
void RocUberTraceCaptureMgr::WriteCbStartMarker(const VirtualGPU* gpu) const {
  RgpSqttMarkerCbStart marker = {};
  marker.identifier = RgpSqttMarkerIdentifierCbStart;
  marker.extDwords  = 0;

  if (gpu->gpu_queue() != nullptr) {
    marker.cbID = static_cast<uint32_t>(
        gpu->gpu_queue()->doorbell_signal.handle & 0xFFFFFu);
  }
  marker.queueIndex = 0;

  // Derive a stable 64-bit device_id from the HSA agent handle.
  const uint64_t agent_handle = device_->getBackendDevice().handle;
  marker.deviceIdLow  = static_cast<uint32_t>(agent_handle & 0xFFFFFFFFu);
  marker.deviceIdHigh = static_cast<uint32_t>(agent_handle >> 32);
  marker.queueFlags   = 0;  // compute queue — no SQTT semaphore timing

  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteCbEndMarker(const VirtualGPU* gpu) const {
  RgpSqttMarkerCbEnd marker = {};
  marker.identifier = RgpSqttMarkerIdentifierCbEnd;
  marker.extDwords  = 0;

  if (gpu->gpu_queue() != nullptr) {
    marker.cbID = static_cast<uint32_t>(
        gpu->gpu_queue()->doorbell_signal.handle & 0xFFFFFu);
  }

  const uint64_t agent_handle = device_->getBackendDevice().handle;
  marker.deviceIdLow  = static_cast<uint32_t>(agent_handle & 0xFFFFFFFFu);
  marker.deviceIdHigh = static_cast<uint32_t>(agent_handle >> 32);

  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteComputeBindMarker(const VirtualGPU* gpu,
                                                    uint64_t api_hash) const {
  RgpSqttMarkerPipelineBind marker = {};
  marker.identifier = RgpSqttMarkerIdentifierBindPipeline;
  marker.extDwords  = 0;
  marker.bindPoint = 1;  // Compute bind point

  // Use the low 20 bits of the HSA queue doorbell signal handle as a command-buffer ID proxy.
  // This is unique per queue and stable for the lifetime of the queue, analogous to PAL's
  // gpu->queue(MainEngine).cmdBufId().
  if (gpu->gpu_queue() != nullptr) {
    marker.cbID = static_cast<uint32_t>(
        gpu->gpu_queue()->doorbell_signal.handle & 0xFFFFFu);
  }

  memcpy(marker.apiPsoHash, &api_hash, sizeof(api_hash));

  WriteMarker(gpu, &marker, sizeof(marker));
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteEventWithDimsMarker(const VirtualGPU* gpu,
                                                      RgpSqttMarkerEventType api_type,
                                                      uint32_t x, uint32_t y,
                                                      uint32_t z) const {
  assert(api_type != RgpSqttMarkerEventType::Invalid);

  RgpSqttMarkerEventWithDims eventWithDims = {};
  eventWithDims.event.identifier  = RgpSqttMarkerIdentifierEvent;
  eventWithDims.event.extDwords   = 0;  // threadX/Y/Z handled via has_thread_dimensions bit
  eventWithDims.event.apiType     = static_cast<uint32_t>(api_type);
  eventWithDims.event.hasThreadDims = 1;
  eventWithDims.event.cmdID       = current_event_id_++;
  eventWithDims.threadX = x;
  eventWithDims.threadY = y;
  eventWithDims.threadZ = z;

  // Same doorbell-signal-handle cbID proxy as WriteComputeBindMarker().
  if (gpu->gpu_queue() != nullptr) {
    eventWithDims.event.cbID = static_cast<uint32_t>(
        gpu->gpu_queue()->doorbell_signal.handle & 0xFFFFFu);
  }

  WriteMarker(gpu, &eventWithDims, sizeof(eventWithDims));
}

// ================================================================================================
void RocUberTraceCaptureMgr::WriteUserEventMarker(const VirtualGPU* gpu,
                                                  RgpSqttMarkerUserEventType event_type,
                                                  const std::string& name) const {
  memset(user_event_, 0, sizeof(RgpSqttMarkerUserEventWithString));

  user_event_->header.identifier = RgpSqttMarkerIdentifierUserEvent;
  user_event_->header.dataType = event_type;

  size_t markerSize = sizeof(user_event_->header);

  if (event_type != RgpSqttMarkerUserEventPop) {
    size_t strLength =
        std::min(name.size(), RgpSqttMaxUserEventStringLengthInDwords * sizeof(uint32_t));
    for (size_t i = 0; i < strLength; ++i) {
      uint32_t c = static_cast<uint32_t>(static_cast<unsigned char>(name[i]));
      user_event_->stringData[i / 4] |= (c << (8 * (i % 4)));
    }
    user_event_->stringLength = static_cast<uint32_t>(strLength);
    const size_t strDwords = (strLength + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    markerSize += sizeof(uint32_t);         // stringLength field
    markerSize += sizeof(uint32_t) * strDwords;
  }

  WriteMarker(gpu, user_event_, markerSize);
}

}  // namespace amd::roc

#endif  // ROC_GPUOPEN
