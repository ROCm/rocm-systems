//===-- KernelLauncher.cpp - Kernel Launch Wrapper --------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Implementation of kernel launcher using HSA runtime.
///
/// Uses HSA API for:
/// - Loading code objects into executables
/// - Creating kernel dispatch packets
/// - Managing queue submission
///
//===----------------------------------------------------------------------===//

#include "aegisbit/KernelLauncher.h"
#include "llvm/Support/Error.h"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hip/hip_runtime.h>

#include <cstring>

using namespace llvm;

namespace aegisbit {

namespace {

/// Convert HSA status to LLVM error
Error hsaErrorToLLVMError(hsa_status_t Status, const char* Context) {
  if (Status == HSA_STATUS_SUCCESS) {
    return Error::success();
  }
  const char* ErrorMsg = nullptr;
  hsa_status_string(Status, &ErrorMsg);
  return createStringError(inconvertibleErrorCode(),
                           "%s: %s", Context, ErrorMsg ? ErrorMsg : "unknown HSA error");
}

/// Callback for agent iteration
hsa_status_t findGPUAgentCallback(hsa_agent_t Agent, void* Data) {
  auto* Agents = static_cast<std::vector<uint64_t>*>(Data);

  hsa_device_type_t DeviceType;
  hsa_status_t Status = hsa_agent_get_info(Agent, HSA_AGENT_INFO_DEVICE, &DeviceType);
  if (Status != HSA_STATUS_SUCCESS) {
    return Status;
  }

  if (DeviceType == HSA_DEVICE_TYPE_GPU) {
    Agents->push_back(Agent.handle);
  }

  return HSA_STATUS_SUCCESS;
}

/// Callback for executable symbol iteration
struct SymbolSearchData {
  const char* TargetName;
  hsa_executable_symbol_t Symbol;
  bool Found;
};

hsa_status_t findSymbolCallback(hsa_executable_t /*Executable*/,
                                 hsa_executable_symbol_t Symbol,
                                 void* Data) {
  auto* SearchData = static_cast<SymbolSearchData*>(Data);

  uint32_t NameLength = 0;
  hsa_status_t Status = hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &NameLength);
  if (Status != HSA_STATUS_SUCCESS) {
    return Status;
  }

  std::string Name(NameLength, '\0');
  Status = hsa_executable_symbol_get_info(
      Symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, Name.data());
  if (Status != HSA_STATUS_SUCCESS) {
    return Status;
  }

  if (Name == SearchData->TargetName) {
    SearchData->Symbol = Symbol;
    SearchData->Found = true;
    return HSA_STATUS_INFO_BREAK;
  }

  return HSA_STATUS_SUCCESS;
}

} // anonymous namespace

std::vector<uint64_t> findAMDGPUAgents() {
  std::vector<uint64_t> Agents;

  // Ensure HSA is initialized
  hsa_status_t Status = hsa_init();
  if (Status != HSA_STATUS_SUCCESS && Status != HSA_STATUS_ERROR_NOT_INITIALIZED) {
    return Agents;
  }

  hsa_iterate_agents(findGPUAgentCallback, &Agents);
  return Agents;
}

uint64_t getDefaultGPUAgent() {
  auto Agents = findAMDGPUAgents();
  return Agents.empty() ? 0 : Agents[0];
}

Expected<std::unique_ptr<KernelLauncher>>
KernelLauncher::create(uint64_t AgentHandle) {
  if (AgentHandle == 0) {
    return createStringError(inconvertibleErrorCode(),
                             "Invalid agent handle (0)");
  }

  auto Launcher = std::unique_ptr<KernelLauncher>(new KernelLauncher());
  Launcher->AgentHandle = AgentHandle;

  hsa_agent_t Agent = {AgentHandle};

  // Get GPU name
  char Name[64] = {0};
  hsa_status_t Status = hsa_agent_get_info(Agent, HSA_AGENT_INFO_NAME, Name);
  if (Status == HSA_STATUS_SUCCESS) {
    Launcher->GPUName = Name;
  }

  // Create a queue for dispatch
  uint32_t QueueSize = 0;
  Status = hsa_agent_get_info(Agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &QueueSize);
  if (Status != HSA_STATUS_SUCCESS) {
    return hsaErrorToLLVMError(Status, "Failed to get queue size");
  }

  hsa_queue_t* Queue = nullptr;
  Status = hsa_queue_create(Agent, QueueSize, HSA_QUEUE_TYPE_SINGLE,
                            nullptr, nullptr, UINT32_MAX, UINT32_MAX, &Queue);
  if (Status != HSA_STATUS_SUCCESS) {
    return hsaErrorToLLVMError(Status, "Failed to create queue");
  }

  Launcher->QueueHandle = reinterpret_cast<uint64_t>(Queue);
  Launcher->GPUAvailable = true;

  return Launcher;
}

KernelLauncher::~KernelLauncher() {
  // Destroy loaded executables
  for (uint64_t ExecHandle : LoadedExecutables) {
    hsa_executable_t Exec = {ExecHandle};
    hsa_executable_destroy(Exec);
  }

  // Destroy queue
  if (QueueHandle != 0) {
    hsa_queue_t* Queue = reinterpret_cast<hsa_queue_t*>(QueueHandle);
    hsa_queue_destroy(Queue);
  }
}

Expected<LoadedKernel>
KernelLauncher::loadKernel(ArrayRef<uint8_t> PatchedELF,
                           StringRef KernelName,
                           uint32_t OriginalKernargSize) {
  if (!GPUAvailable) {
    return createStringError(inconvertibleErrorCode(),
                             "GPU not available");
  }

  hsa_agent_t Agent = {AgentHandle};

  // Create code object reader from memory
  hsa_code_object_reader_t Reader;
  hsa_status_t Status = hsa_code_object_reader_create_from_memory(
      PatchedELF.data(), PatchedELF.size(), &Reader);
  if (Status != HSA_STATUS_SUCCESS) {
    return hsaErrorToLLVMError(Status, "Failed to create code object reader");
  }

  // Create executable
  hsa_executable_t Executable;
  Status = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                      HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                      nullptr, &Executable);
  if (Status != HSA_STATUS_SUCCESS) {
    hsa_code_object_reader_destroy(Reader);
    return hsaErrorToLLVMError(Status, "Failed to create executable");
  }

  // Load code object into executable
  Status = hsa_executable_load_agent_code_object(Executable, Agent, Reader,
                                                  nullptr, nullptr);
  if (Status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(Executable);
    hsa_code_object_reader_destroy(Reader);
    return hsaErrorToLLVMError(Status, "Failed to load code object");
  }

  // Freeze executable
  Status = hsa_executable_freeze(Executable, nullptr);
  if (Status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(Executable);
    hsa_code_object_reader_destroy(Reader);
    return hsaErrorToLLVMError(Status, "Failed to freeze executable");
  }

  // Clean up reader (no longer needed after freeze)
  hsa_code_object_reader_destroy(Reader);

  // Find kernel symbol
  // HSA exposes the descriptor symbol (.kd suffix), not the function symbol
  std::string DescriptorSymbolName = KernelName.str() + ".kd";
  SymbolSearchData SearchData;
  SearchData.TargetName = DescriptorSymbolName.c_str();
  SearchData.Found = false;

  Status = hsa_executable_iterate_symbols(Executable, findSymbolCallback, &SearchData);
  if (!SearchData.Found) {
    hsa_executable_destroy(Executable);
    return createStringError(inconvertibleErrorCode(),
                             "Kernel symbol '%s' not found", KernelName.data());
  }

  // Get kernel object (for dispatch)
  uint64_t KernelObject = 0;
  Status = hsa_executable_symbol_get_info(
      SearchData.Symbol,
      HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
      &KernelObject);
  if (Status != HSA_STATUS_SUCCESS) {
    hsa_executable_destroy(Executable);
    return hsaErrorToLLVMError(Status, "Failed to get kernel object");
  }

  // Track executable for cleanup
  LoadedExecutables.push_back(Executable.handle);

  LoadedKernel Result;
  Result.ExecutableHandle = Executable.handle;
  Result.KernelSymbol = KernelObject;
  Result.KernelName = KernelName.str();
  Result.OriginalKernargSize = OriginalKernargSize;

  return Result;
}

} // namespace aegisbit
