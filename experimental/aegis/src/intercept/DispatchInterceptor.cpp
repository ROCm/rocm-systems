//===-- DispatchInterceptor.cpp - Dispatch Interception --------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of dispatch/code-object interception using public HSA APIs.
///
/// We interpose `hsa_executable_freeze`, then use the AMD loader extension and
/// `hsa_executable_iterate_agent_symbols` to capture loaded code objects and
/// kernel descriptors without depending on rocprofiler-sdk.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/DispatchInterceptor.h"
#include "aegisbit/CodeObjectHandler.h"
#include "aegisbit/DescriptorUpdater.h"
#include "aegisbit/HSAInterceptor.h"
#include "aegisbit/RuntimeConfig.h"
#include "aegisbit/TracingEngine.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"

#ifdef AEGISBIT_HAS_GPU
#include <hsa/hsa.h>
#include <hsa/hsa_ven_amd_loader.h>
#endif

#include <dlfcn.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace llvm;

namespace {

bool shouldLog() {
  return aegisbit::RuntimeConfig::getInstance().LogEnabled;
}

std::string trimKernelDescriptorSuffix(std::string Name) {
  if (Name.size() >= 3 && Name.compare(Name.size() - 3, 3, ".kd") == 0) {
    Name.resize(Name.size() - 3);
  }
  return Name;
}

} // namespace

namespace aegisbit {

DispatchInterceptorImpl& DispatchInterceptor::getInstance() {
  static DispatchInterceptorImpl Instance;
  return Instance;
}

#ifdef AEGISBIT_HAS_GPU
namespace {

using hsa_executable_freeze_fn_t =
    hsa_status_t (*)(hsa_executable_t executable, const char* options);

hsa_executable_freeze_fn_t getRealHsaExecutableFreeze() {
  static auto* RealFn = reinterpret_cast<hsa_executable_freeze_fn_t>(
      dlsym(RTLD_NEXT, "hsa_executable_freeze"));
  return RealFn;
}

bool queryLoaderTable(hsa_ven_amd_loader_1_01_pfn_t& Table) {
  std::memset(&Table, 0, sizeof(Table));
  return hsa_system_get_major_extension_table(
             HSA_EXTENSION_AMD_LOADER, 1, sizeof(Table), &Table) ==
             HSA_STATUS_SUCCESS &&
         Table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects &&
         Table.hsa_ven_amd_loader_loaded_code_object_get_info;
}

std::string percentDecode(StringRef Encoded) {
  std::string Result;
  Result.reserve(Encoded.size());
  for (size_t I = 0; I < Encoded.size(); ++I) {
    if (Encoded[I] == '%' && I + 2 < Encoded.size() &&
        std::isxdigit(static_cast<unsigned char>(Encoded[I + 1])) &&
        std::isxdigit(static_cast<unsigned char>(Encoded[I + 2]))) {
      auto hexValue = [](char C) -> uint8_t {
        if (C >= '0' && C <= '9')
          return static_cast<uint8_t>(C - '0');
        C = static_cast<char>(std::tolower(static_cast<unsigned char>(C)));
        return static_cast<uint8_t>(10 + (C - 'a'));
      };
      Result.push_back(static_cast<char>((hexValue(Encoded[I + 1]) << 4) |
                                         hexValue(Encoded[I + 2])));
      I += 2;
      continue;
    }
    Result.push_back(Encoded[I]);
  }
  return Result;
}

bool parseUriInteger(StringRef Query, StringRef Key, uint64_t& Value) {
  size_t KeyPos = Query.find(Key);
  if (KeyPos == StringRef::npos)
    return false;
  KeyPos += Key.size();
  size_t End = Query.find('&', KeyPos);
  StringRef Number = Query.substr(KeyPos, End == StringRef::npos ? StringRef::npos
                                                                  : End - KeyPos);
  unsigned Base = 10;
  if (Number.starts_with_insensitive("0x")) {
    Number = Number.drop_front(2);
    Base = 16;
  }
  return !Number.getAsInteger(Base, Value);
}

bool loadBytesFromUri(StringRef URI, std::vector<uint8_t>& Bytes) {
  if (!URI.starts_with("file://"))
    return false;

  StringRef Payload = URI.drop_front(strlen("file://"));
  size_t Marker = Payload.find_first_of("#?");
  StringRef PathPart = Marker == StringRef::npos ? Payload : Payload.substr(0, Marker);
  StringRef Query = Marker == StringRef::npos ? StringRef() : Payload.substr(Marker + 1);

  uint64_t Offset = 0;
  uint64_t Size = 0;
  bool HasOffset = parseUriInteger(Query, "offset=", Offset);
  bool HasSize = parseUriInteger(Query, "size=", Size);
  if (!HasSize)
    return false;

  std::ifstream Input(percentDecode(PathPart), std::ios::binary);
  if (!Input)
    return false;

  Input.seekg(static_cast<std::streamoff>(HasOffset ? Offset : 0));
  Bytes.resize(static_cast<size_t>(Size));
  Input.read(reinterpret_cast<char*>(Bytes.data()), static_cast<std::streamsize>(Size));
  return Input.good() || Input.gcount() == static_cast<std::streamsize>(Size);
}

std::vector<hsa_agent_t> getGpuAgents() {
  std::vector<hsa_agent_t> Agents;
  hsa_iterate_agents(
      [](hsa_agent_t Agent, void* Data) -> hsa_status_t {
        hsa_device_type_t DeviceType = HSA_DEVICE_TYPE_CPU;
        if (hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType) !=
            HSA_STATUS_SUCCESS) {
          return HSA_STATUS_SUCCESS;
        }
        if (DeviceType == HSA_DEVICE_TYPE_GPU) {
          static_cast<std::vector<hsa_agent_t>*>(Data)->push_back(Agent);
        }
        return HSA_STATUS_SUCCESS;
      },
      &Agents);
  return Agents;
}

// Cache of HSA_AGENT_INFO_NAME keyed by hsa_agent_t::handle. Populated lazily
// the first time a symbol is registered for an agent.
std::string lookupAgentGfxName(hsa_agent_t Agent) {
  static std::mutex CacheMutex;
  static std::unordered_map<uint64_t, std::string> Cache;
  {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    auto It = Cache.find(Agent.handle);
    if (It != Cache.end())
      return It->second;
  }

  char Name[64] = {0};
  std::string Result;
  if (hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, Name) == HSA_STATUS_SUCCESS) {
    Result.assign(Name, strnlen(Name, sizeof(Name)));
    auto Colon = Result.find(':');
    if (Colon != std::string::npos)
      Result.resize(Colon);
  }

  {
    std::lock_guard<std::mutex> Lock(CacheMutex);
    Cache.emplace(Agent.handle, Result);
  }
  return Result;
}

// Translate a device-side kernel_object pointer into a host-readable pointer
// to the 64-byte kernel descriptor. Mirrors rocprofiler-sdk's
// get_kernel_descriptor(): uses hsa_ven_amd_loader_query_host_address when
// available, falls back to reinterpreting the device pointer as host (common
// on APUs / unified memory where the loader allocates KDs in system RAM).
const uint8_t* queryKernelDescriptorHost(
    const hsa_ven_amd_loader_1_01_pfn_t& LoaderTable, uint64_t KernelObject) {
  if (KernelObject == 0)
    return nullptr;

  const void* HostAddr = nullptr;
  if (LoaderTable.hsa_ven_amd_loader_query_host_address) {
    if (LoaderTable.hsa_ven_amd_loader_query_host_address(
            reinterpret_cast<const void*>(KernelObject), &HostAddr) ==
        HSA_STATUS_SUCCESS) {
      return static_cast<const uint8_t*>(HostAddr);
    }
  }
  // Fallback: treat kernel_object as host-readable directly.
  return reinterpret_cast<const uint8_t*>(KernelObject);
}

// Decode VGPR / SGPR counts for a kernel by reading its loaded kernel
// descriptor in place. No ELF re-parse; O(1) per symbol.
void enrichKernelRegisterCounts(
    CapturedKernelSymbol& KS, hsa_agent_t Agent,
    const hsa_ven_amd_loader_1_01_pfn_t& LoaderTable) {
  const uint8_t* KDBytes = queryKernelDescriptorHost(LoaderTable, KS.KernelObject);
  if (!KDBytes)
    return;

  std::string GfxName = lookupAgentGfxName(Agent);
  llvm::ArrayRef<uint8_t> KDView(KDBytes, DescriptorUpdater::DESCRIPTOR_SIZE);
  auto KDOrErr = DescriptorUpdater::parse(KDView, GfxName);
  if (!KDOrErr) {
    consumeError(KDOrErr.takeError());
    return;
  }
  const auto& KD = *KDOrErr;
  KS.SGPRCount = KD.SGPRCount;
  KS.VGPRCount = KD.VGPRCount;
}

const CapturedCodeObject* lookupCodeObjectForKernelObjectLocked(
    DispatchInterceptorImpl& Impl, uint64_t KernelObject) {
  for (auto& [Id, CO] : Impl.CodeObjects) {
    if (CO.LoadBase == 0 || CO.LoadSize == 0)
      continue;
    if (KernelObject >= CO.LoadBase && KernelObject < CO.LoadBase + CO.LoadSize) {
      return &CO;
    }
  }
  return nullptr;
}

void enumerateExecutableState(hsa_executable_t Executable) {
  hsa_ven_amd_loader_1_01_pfn_t LoaderTable{};
  if (!queryLoaderTable(LoaderTable)) {
    RuntimeConfig::getInstance().log(
        "AMD loader extension table unavailable during executable freeze");
    return;
  }

  LoaderTable.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
      Executable,
      [](hsa_executable_t Exec, hsa_loaded_code_object_t LoadedCodeObject,
         void*) -> hsa_status_t {
        DispatchInterceptor::handleCodeObjectLoad(Exec, LoadedCodeObject);
        return HSA_STATUS_SUCCESS;
      },
      nullptr);

  for (hsa_agent_t Agent : getGpuAgents()) {
    hsa_executable_iterate_agent_symbols(
        Executable, Agent,
        [](hsa_executable_t Exec, hsa_agent_t Agent,
           hsa_executable_symbol_t Symbol, void*) -> hsa_status_t {
          DispatchInterceptor::handleKernelSymbolRegister(Exec, Agent, Symbol);
          return HSA_STATUS_SUCCESS;
        },
        nullptr);
  }
}

hsa_executable_freeze_fn_t getRealHsaExecutableFreezeForInterpose() {
  return getRealHsaExecutableFreeze();
}

void handleExecutableFreeze(hsa_executable_t Executable) {
  enumerateExecutableState(Executable);
}

} // namespace
#endif

void DispatchInterceptor::handleCodeObjectLoad(hsa_executable_t /*Executable*/,
                                               hsa_loaded_code_object_t LoadedCodeObject) {
#ifdef AEGISBIT_HAS_GPU
  auto& Impl = getInstance();
  hsa_ven_amd_loader_1_01_pfn_t LoaderTable{};
  if (!queryLoaderTable(LoaderTable))
    return;

  CapturedCodeObject CO;
  CO.CodeObjectId = LoadedCodeObject.handle;

  uint32_t URILength = 0;
  LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
      LoadedCodeObject, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH,
      &URILength);
  if (URILength > 0) {
    std::vector<char> URI(static_cast<size_t>(URILength) + 1, '\0');
    if (LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
            LoadedCodeObject, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI,
            URI.data()) == HSA_STATUS_SUCCESS) {
      CO.URI = URI.data();
    }
  }

  LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
      LoadedCodeObject, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE,
      &CO.LoadBase);
  LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
      LoadedCodeObject, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE,
      &CO.LoadSize);

  uint32_t StorageType = 0;
  LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
      LoadedCodeObject,
      HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_TYPE,
      &StorageType);

  if (StorageType == HSA_VEN_AMD_LOADER_CODE_OBJECT_STORAGE_TYPE_MEMORY) {
    uint64_t MemoryBase = 0;
    uint64_t MemorySize = 0;
    if (LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
            LoadedCodeObject,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_BASE,
            &MemoryBase) == HSA_STATUS_SUCCESS &&
        LoaderTable.hsa_ven_amd_loader_loaded_code_object_get_info(
            LoadedCodeObject,
            HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_CODE_OBJECT_STORAGE_MEMORY_SIZE,
            &MemorySize) == HSA_STATUS_SUCCESS &&
        MemoryBase != 0 && MemorySize != 0) {
      const auto* BytesBase = reinterpret_cast<const uint8_t*>(MemoryBase);
      CO.Bytes.assign(BytesBase, BytesBase + MemorySize);
    }
  } else if (!CO.URI.empty()) {
    loadBytesFromUri(CO.URI, CO.Bytes);
  }

  CodeObjectLoadCallback Callback;
  CapturedCodeObject StoredCO;
  bool IsNew = false;
  {
    std::lock_guard<std::mutex> Lock(Impl.Mutex);
    auto [It, Inserted] = Impl.CodeObjects.insert_or_assign(CO.CodeObjectId, CO);
    IsNew = Inserted;
    StoredCO = It->second;
    Callback = Impl.OnCodeObjectLoad;
  }

  if (shouldLog()) {
    std::cerr << "[aegisbit] Code object loaded: id=" << CO.CodeObjectId
              << " load_base=0x" << std::hex << CO.LoadBase
              << " load_size=0x" << CO.LoadSize << std::dec
              << " uri=" << (CO.URI.empty() ? "(unknown)" : CO.URI) << "\n";
  }

  if (IsNew && Callback) {
    Callback(StoredCO);
  }
#else
  (void)LoadedCodeObject;
#endif
}

void DispatchInterceptor::handleKernelSymbolRegister(hsa_executable_t /*Executable*/,
                                                     hsa_agent_t Agent,
                                                     hsa_executable_symbol_t Symbol) {
#ifdef AEGISBIT_HAS_GPU
  auto& Impl = getInstance();

  hsa_symbol_kind_t SymbolKind = HSA_SYMBOL_KIND_VARIABLE;
  if (hsa_executable_symbol_get_info(Symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE,
                                     &SymbolKind) != HSA_STATUS_SUCCESS ||
      SymbolKind != HSA_SYMBOL_KIND_KERNEL) {
    return;
  }

  uint32_t NameLength = 0;
  if (hsa_executable_symbol_get_info(Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH,
                                     &NameLength) != HSA_STATUS_SUCCESS ||
      NameLength == 0) {
    return;
  }

  std::string KernelName(NameLength, '\0');
  if (hsa_executable_symbol_get_info(Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME,
                                     KernelName.data()) != HSA_STATUS_SUCCESS) {
    return;
  }
  if (!KernelName.empty() && KernelName.back() == '\0') {
    KernelName.pop_back();
  }

  CapturedKernelSymbol KS;
  KS.KernelId = Symbol.handle;
  KS.KernelName = trimKernelDescriptorSuffix(KernelName);

  hsa_executable_symbol_get_info(Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                 &KS.KernelObject);
  hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE,
      &KS.KernargSegmentSize);
  hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
      &KS.GroupSegmentSize);
  hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
      &KS.PrivateSegmentSize);

  // Enrich register counts via the loader's in-memory kernel descriptor
  // (mirrors rocprofiler-sdk). Done outside the impl mutex — no shared state.
  hsa_ven_amd_loader_1_01_pfn_t LoaderTable{};
  if (queryLoaderTable(LoaderTable)) {
    enrichKernelRegisterCounts(KS, Agent, LoaderTable);
  }

  KernelSymbolCallback Callback;
  CapturedKernelSymbol StoredKS;
  bool IsNew = false;
  {
    std::lock_guard<std::mutex> Lock(Impl.Mutex);
    if (const auto* CO = lookupCodeObjectForKernelObjectLocked(Impl, KS.KernelObject)) {
      KS.CodeObjectId = CO->CodeObjectId;
    }
    auto [It, Inserted] = Impl.KernelSymbols.insert_or_assign(KS.KernelId, KS);
    IsNew = Inserted;
    if (!KS.KernelName.empty()) {
      Impl.KernelNameToId[KS.KernelName] = KS.KernelId;
    }
    StoredKS = It->second;
    Callback = Impl.OnKernelSymbol;
  }

  if (shouldLog()) {
    std::cerr << "[aegisbit] Kernel registered: " << KS.KernelName
              << " code_object=" << KS.CodeObjectId
              << " vgpr=" << KS.VGPRCount << " sgpr=" << KS.SGPRCount << "\n";
  }

  if (IsNew && Callback) {
    Callback(StoredKS);
  }
#else
  (void)Symbol;
#endif
}

Error DispatchInterceptor::initialize() {
  RuntimeConfig::initialize();
  RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  auto& Impl = getInstance();

  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  if (Impl.Initialized) {
    return Error::success();
  }

  Impl.Initialized = true;

  if (Cfg.Enabled) {
    if (auto Err = TracingEngine::getInstance().initialize()) {
      std::cerr << "AegisBit: Failed to initialize tracing engine: "
                << llvm::toString(std::move(Err)) << "\n";
    }

    HSAInterceptor::setDispatchCallback(
        [](hsa_queue_t* Queue, hsa_kernel_dispatch_packet_t* Packet,
           uint64_t OriginalKernelObject, void* OriginalKernarg,
           uint32_t OriginalKernargSize) -> bool {
          return TracingEngine::getInstance().onDispatch(
              Queue, Packet, OriginalKernelObject, OriginalKernarg,
              OriginalKernargSize);
        });

    if (auto Err = HSAInterceptor::install()) {
      std::cerr << "AegisBit: Failed to install HSA interceptor: "
                << llvm::toString(std::move(Err)) << "\n";
    }
  }

  Cfg.log("Dispatch interceptor initialized");
  return Error::success();
}

void DispatchInterceptor::finalize() {
  RuntimeConfig& Cfg = RuntimeConfig::getInstance();
  auto& Impl = getInstance();

  {
    std::lock_guard<std::mutex> Lock(Impl.Mutex);
    if (!Impl.Initialized) {
      return;
    }

    Impl.CodeObjects.clear();
    Impl.KernelSymbols.clear();
    Impl.KernelNameToId.clear();
    Impl.OnCodeObjectLoad = nullptr;
    Impl.OnKernelSymbol = nullptr;
    Impl.OnKernelDispatch = nullptr;
    Impl.Initialized = false;
  }

  HSAInterceptor::clearDispatchCallback();
  HSAInterceptor::uninstall();
  if (Cfg.Enabled) {
    TracingEngine::getInstance().finalize();
  }
}

bool DispatchInterceptor::isInitialized() {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  return Impl.Initialized;
}

void DispatchInterceptor::setCodeObjectLoadCallback(CodeObjectLoadCallback Callback) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  Impl.OnCodeObjectLoad = std::move(Callback);
}

void DispatchInterceptor::setKernelSymbolCallback(KernelSymbolCallback Callback) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  Impl.OnKernelSymbol = std::move(Callback);
}

void DispatchInterceptor::setKernelDispatchCallback(KernelDispatchCallback Callback) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  Impl.OnKernelDispatch = std::move(Callback);
}

const CapturedCodeObject* DispatchInterceptor::getCodeObject(uint64_t CodeObjectId) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  auto It = Impl.CodeObjects.find(CodeObjectId);
  return (It != Impl.CodeObjects.end()) ? &It->second : nullptr;
}

const CapturedKernelSymbol* DispatchInterceptor::getKernelSymbol(uint64_t KernelId) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  auto It = Impl.KernelSymbols.find(KernelId);
  return (It != Impl.KernelSymbols.end()) ? &It->second : nullptr;
}

const CapturedKernelSymbol* DispatchInterceptor::getKernelSymbolByName(
    const std::string& Name) {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  auto It = Impl.KernelNameToId.find(Name);
  if (It == Impl.KernelNameToId.end()) {
    return nullptr;
  }
  auto SymIt = Impl.KernelSymbols.find(It->second);
  return (SymIt != Impl.KernelSymbols.end()) ? &SymIt->second : nullptr;
}

std::vector<const CapturedCodeObject*> DispatchInterceptor::getAllCodeObjects() {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  std::vector<const CapturedCodeObject*> Result;
  Result.reserve(Impl.CodeObjects.size());
  for (const auto& [Id, CO] : Impl.CodeObjects) {
    (void)Id;
    Result.push_back(&CO);
  }
  return Result;
}

std::vector<CapturedCodeObject> DispatchInterceptor::getAllCodeObjectsCopy() {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  std::vector<CapturedCodeObject> Result;
  Result.reserve(Impl.CodeObjects.size());
  for (const auto& [Id, CO] : Impl.CodeObjects) {
    (void)Id;
    Result.push_back(CO);
  }
  return Result;
}

std::vector<const CapturedKernelSymbol*> DispatchInterceptor::getAllKernelSymbols() {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  std::vector<const CapturedKernelSymbol*> Result;
  Result.reserve(Impl.KernelSymbols.size());
  for (const auto& [Id, KS] : Impl.KernelSymbols) {
    (void)Id;
    Result.push_back(&KS);
  }
  return Result;
}

std::vector<CapturedKernelSymbol> DispatchInterceptor::getAllKernelSymbolsCopy() {
  auto& Impl = getInstance();
  std::lock_guard<std::mutex> Lock(Impl.Mutex);
  std::vector<CapturedKernelSymbol> Result;
  Result.reserve(Impl.KernelSymbols.size());
  for (const auto& [Id, KS] : Impl.KernelSymbols) {
    (void)Id;
    Result.push_back(KS);
  }
  return Result;
}

} // namespace aegisbit

#ifdef AEGISBIT_HAS_GPU
extern "C" hsa_status_t hsa_executable_freeze(hsa_executable_t executable,
                                              const char* options) {
  auto RealFn = aegisbit::getRealHsaExecutableFreezeForInterpose();
  if (!RealFn) {
    return HSA_STATUS_ERROR;
  }

  hsa_status_t Status = RealFn(executable, options);
  if (Status == HSA_STATUS_SUCCESS) {
    aegisbit::handleExecutableFreeze(executable);
  }
  return Status;
}

__attribute__((constructor)) static void aegisbitDispatchConstructor() {
  if (auto Err = aegisbit::DispatchInterceptor::initialize()) {
    std::cerr << "AegisBit: Dispatch interceptor init failed: "
              << llvm::toString(std::move(Err)) << "\n";
  }
}

__attribute__((destructor)) static void aegisbitDispatchDestructor() {
  aegisbit::DispatchInterceptor::finalize();
}
#endif
