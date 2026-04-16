//===-- aegisbit/DispatchInterceptor.h - Dispatch Interception ---*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Dispatch interception using HSA loader and executable callbacks.
/// Captures code objects before kernel dispatch for instrumentation.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_DISPATCH_INTERCEPTOR_H
#define AEGISBIT_DISPATCH_INTERCEPTOR_H

#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// Forward declarations for HSA types
struct hsa_agent_s;
typedef struct hsa_agent_s hsa_agent_t;
struct hsa_executable_s;
typedef struct hsa_executable_s hsa_executable_t;
struct hsa_executable_symbol_s;
typedef struct hsa_executable_symbol_s hsa_executable_symbol_t;
struct hsa_loaded_code_object_s;
typedef struct hsa_loaded_code_object_s hsa_loaded_code_object_t;

namespace aegisbit {

/// Information about a captured code object
struct CapturedCodeObject {
  uint64_t CodeObjectId;             ///< Unique code object identifier
  std::vector<uint8_t> Bytes;        ///< Raw ELF bytes
  std::string URI;                   ///< Original URI (file path or memory)
  uint64_t LoadBase;                 ///< Base address where loaded
  uint64_t LoadSize;                 ///< Size of loaded code object
};

/// Information about a kernel symbol
struct CapturedKernelSymbol {
  uint64_t KernelId;                 ///< Unique kernel identifier
  uint64_t CodeObjectId;             ///< Parent code object
  std::string KernelName;            ///< Kernel name
  uint64_t KernelObject;             ///< Kernel object handle for dispatch
  uint32_t KernargSegmentSize;       ///< Size of kernel arguments
  uint32_t GroupSegmentSize;         ///< LDS size
  uint32_t PrivateSegmentSize;       ///< Scratch size
  uint32_t SGPRCount;                ///< Scalar register count
  uint32_t VGPRCount;                ///< Vector register count (arch + accum)
};

/// Callback type for code object load events
using CodeObjectLoadCallback = std::function<void(const CapturedCodeObject&)>;

/// Callback type for kernel symbol registration
using KernelSymbolCallback = std::function<void(const CapturedKernelSymbol&)>;

/// Callback type for kernel dispatch (before launch)
/// Return true to proceed with original dispatch, false to skip
using KernelDispatchCallback = std::function<bool(const CapturedKernelSymbol&,
                                                   void* KernArgs,
                                                   uint32_t WorkgroupSizeX,
                                                   uint32_t WorkgroupSizeY,
                                                   uint32_t WorkgroupSizeZ,
                                                   uint32_t GridSizeX,
                                                   uint32_t GridSizeY,
                                                   uint32_t GridSizeZ)>;

//===----------------------------------------------------------------------===//
// DispatchInterceptorImpl - Internal implementation class
//===----------------------------------------------------------------------===//

/// Internal implementation class holding interceptor state.
/// This is separate from the public interface to allow for better testability
/// and encapsulation of the static state required by the HSA callbacks.
class DispatchInterceptorImpl {
public:
  DispatchInterceptorImpl() = default;
  ~DispatchInterceptorImpl() = default;

  // Non-copyable, non-movable
  DispatchInterceptorImpl(const DispatchInterceptorImpl&) = delete;
  DispatchInterceptorImpl& operator=(const DispatchInterceptorImpl&) = delete;
  DispatchInterceptorImpl(DispatchInterceptorImpl&&) = delete;
  DispatchInterceptorImpl& operator=(DispatchInterceptorImpl&&) = delete;

  /// Internal state
  bool Initialized = false;
  mutable std::mutex Mutex;
  std::unordered_map<uint64_t, CapturedCodeObject> CodeObjects;
  std::unordered_map<uint64_t, CapturedKernelSymbol> KernelSymbols;
  std::unordered_map<std::string, uint64_t> KernelNameToId;

  /// User callbacks
  CodeObjectLoadCallback OnCodeObjectLoad;
  KernelSymbolCallback OnKernelSymbol;
  KernelDispatchCallback OnKernelDispatch;

  /// Clear all state (for testing or reset)
  void reset() {
    std::lock_guard<std::mutex> Lock(Mutex);
    CodeObjects.clear();
    KernelSymbols.clear();
    KernelNameToId.clear();
    OnCodeObjectLoad = nullptr;
    OnKernelSymbol = nullptr;
    OnKernelDispatch = nullptr;
    Initialized = false;
  }
};

//===----------------------------------------------------------------------===//
// DispatchInterceptor - Public interface
//===----------------------------------------------------------------------===//

/// Dispatch interceptor using public HSA runtime hooks.
///
/// This class uses HSA runtime hooks to:
/// 1. Capture code objects (GPU binaries) when they are loaded
/// 2. Track kernel symbols for identification
/// 3. Let `HSAInterceptor` modify dispatches before execution
///
/// The shared library initializes this interceptor at load time and refreshes
/// its captured executable state after each successful `hsa_executable_freeze`.
/// The static methods delegate to a global `DispatchInterceptorImpl` instance.
///
/// Usage:
/// \code
///   DispatchInterceptor::initialize();
///   DispatchInterceptor::setCodeObjectLoadCallback([](const CapturedCodeObject& co) {
///     // Process code object, instrument it
///   });
///   DispatchInterceptor::setKernelDispatchCallback([](const CapturedKernelSymbol& ks, ...) {
///     // Intercept dispatch, replace with instrumented version
///     return false; // Skip original dispatch
///   });
/// \endcode
class DispatchInterceptor {
public:
  /// Initialize the dispatch interceptor.
  /// Must be called before any HIP/ROCm operations.
  /// Safe to call multiple times - subsequent calls are no-ops.
  /// \return Error on failure
  static llvm::Error initialize();

  /// Finalize the dispatch interceptor.
  /// Should be called during application shutdown.
  /// Safe to call multiple times - subsequent calls are no-ops.
  static void finalize();

  /// Check if the interceptor is initialized.
  static bool isInitialized();

  /// Set callback for code object load events.
  /// Called when a code object (GPU binary) is loaded.
  static void setCodeObjectLoadCallback(CodeObjectLoadCallback Callback);

  /// Set callback for kernel symbol registration.
  /// Called when a kernel symbol is registered.
  static void setKernelSymbolCallback(KernelSymbolCallback Callback);

  /// Set callback for kernel dispatch interception.
  /// Called before each kernel dispatch.
  static void setKernelDispatchCallback(KernelDispatchCallback Callback);

  /// Get captured code object by ID.
  /// \param CodeObjectId Code object identifier
  /// \return Pointer to code object or nullptr if not found
  static const CapturedCodeObject* getCodeObject(uint64_t CodeObjectId);

  /// Get captured kernel symbol by ID.
  /// \param KernelId Kernel identifier
  /// \return Pointer to kernel symbol or nullptr if not found
  static const CapturedKernelSymbol* getKernelSymbol(uint64_t KernelId);

  /// Get kernel symbol by name.
  /// \param Name Kernel name
  /// \return Pointer to kernel symbol or nullptr if not found
  static const CapturedKernelSymbol* getKernelSymbolByName(const std::string& Name);

  /// Get all captured code objects (pointers valid only while lock is held).
  /// Prefer getAllCodeObjectsCopy() for thread-safe access.
  static std::vector<const CapturedCodeObject*> getAllCodeObjects();

  /// Get all captured kernel symbols (pointers valid only while lock is held).
  /// Prefer getAllKernelSymbolsCopy() for thread-safe access.
  static std::vector<const CapturedKernelSymbol*> getAllKernelSymbols();

  /// Get copies of all captured code objects (thread-safe — no dangling pointers).
  static std::vector<CapturedCodeObject> getAllCodeObjectsCopy();

  /// Get copies of all captured kernel symbols (thread-safe — no dangling pointers).
  static std::vector<CapturedKernelSymbol> getAllKernelSymbolsCopy();

  /// Get the singleton implementation instance.
  /// For advanced use cases and testing only.
  static DispatchInterceptorImpl& getInstance();

  /// Internal callback handlers used by the HSA interposition layer.
  static void handleCodeObjectLoad(hsa_executable_t Executable,
                                   hsa_loaded_code_object_t LoadedCodeObject);
  static void handleKernelSymbolRegister(hsa_executable_t Executable,
                                         hsa_agent_t Agent,
                                         hsa_executable_symbol_t Symbol);

private:
  DispatchInterceptor() = delete;
};

} // namespace aegisbit

#endif // AEGISBIT_DISPATCH_INTERCEPTOR_H
