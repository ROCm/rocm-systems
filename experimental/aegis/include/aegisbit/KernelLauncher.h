//===-- aegisbit/KernelLauncher.h - Kernel Launch Wrapper -------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Kernel launcher for loading instrumented code objects onto the GPU.
///
/// Wraps the HSA runtime to:
/// 1. Load the instrumented code object
/// 2. Resolve kernel symbols for dispatch
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_KERNEL_LAUNCHER_H
#define AEGISBIT_KERNEL_LAUNCHER_H

#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace aegisbit {

/// Information about a loaded instrumented kernel
struct LoadedKernel {
  uint64_t CodeObjectHandle;     ///< HSA code object handle
  uint64_t ExecutableHandle;     ///< HSA executable handle
  uint64_t KernelSymbol;         ///< Kernel symbol for dispatch
  std::string KernelName;        ///< Original kernel name
  uint32_t OriginalKernargSize;  ///< Original kernel argument size
};

/// Dispatch parameters for a kernel launch
struct DispatchParams {
  uint32_t WorkgroupSizeX = 1;   ///< Workgroup size in X dimension
  uint32_t WorkgroupSizeY = 1;   ///< Workgroup size in Y dimension
  uint32_t WorkgroupSizeZ = 1;   ///< Workgroup size in Z dimension
  uint32_t GridSizeX = 1;        ///< Grid size in X dimension
  uint32_t GridSizeY = 1;        ///< Grid size in Y dimension
  uint32_t GridSizeZ = 1;        ///< Grid size in Z dimension
  uint32_t DynamicLDSSize = 0;   ///< Dynamic LDS allocation size
};

/// Result of a kernel launch including trace data
struct LaunchResult {
  bool Success = false;          ///< Whether the kernel completed successfully
  double ExecutionTimeMs = 0.0;  ///< Kernel execution time in milliseconds
  std::vector<uint8_t> TraceData;///< Raw trace data from buffer
  uint64_t BytesWritten = 0;     ///< Actual bytes written to trace buffer
  bool Overflowed = false;       ///< Whether trace buffer overflowed
};

/// Kernel launcher for loading instrumented GPU kernels.
///
/// This class manages the lifecycle of instrumented kernels:
/// 1. Load patched code object into HSA runtime
/// 2. Resolve kernel symbols for dispatch by TracingEngine
///
/// Usage:
/// \code
///   auto LauncherOrErr = KernelLauncher::create(gpuAgentHandle);
///   auto& Launcher = *LauncherOrErr;
///
///   auto KernelOrErr = Launcher.loadKernel(patchedCodeObject, "kernelName",
///                                          originalKernargSize);
/// \endcode
class KernelLauncher {
public:
  /// Create a kernel launcher for the specified GPU agent.
  /// \param AgentHandle HSA agent handle for the target GPU
  /// \return KernelLauncher instance or error
  static llvm::Expected<std::unique_ptr<KernelLauncher>>
  create(uint64_t AgentHandle);

  /// Destructor - cleans up loaded executables
  ~KernelLauncher();

  /// Load an instrumented kernel from a patched code object.
  /// \param PatchedELF The instrumented ELF code object bytes
  /// \param KernelName Name of the kernel to load
  /// \param OriginalKernargSize Original size of kernel arguments
  /// \return LoadedKernel info or error
  llvm::Expected<LoadedKernel> loadKernel(llvm::ArrayRef<uint8_t> PatchedELF,
                                           llvm::StringRef KernelName,
                                           uint32_t OriginalKernargSize);

  /// Get the HSA agent handle.
  uint64_t getAgentHandle() const { return AgentHandle; }

  /// Check if GPU is available and initialized.
  bool isGPUAvailable() const { return GPUAvailable; }

  /// Get GPU agent name (e.g., "gfx942").
  const std::string& getGPUName() const { return GPUName; }

private:
  KernelLauncher() = default;

  uint64_t AgentHandle = 0;        ///< HSA agent handle
  uint64_t QueueHandle = 0;        ///< HSA queue for dispatch
  bool GPUAvailable = false;       ///< Whether GPU is usable
  std::string GPUName;             ///< GPU architecture name

  /// Loaded executables for cleanup
  std::vector<uint64_t> LoadedExecutables;

  // Non-copyable
  KernelLauncher(const KernelLauncher&) = delete;
  KernelLauncher& operator=(const KernelLauncher&) = delete;
};

/// Utility to find available AMD GPU agents.
/// \return Vector of agent handles or empty if none found
std::vector<uint64_t> findAMDGPUAgents();

/// Get the default GPU agent (first available).
/// \return Agent handle or 0 if none available
uint64_t getDefaultGPUAgent();

} // namespace aegisbit

#endif // AEGISBIT_KERNEL_LAUNCHER_H
