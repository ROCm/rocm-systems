//===-- aegisbit/HSAInterceptor.h - HSA Queue Interception ------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// HSA queue interception for kernel dispatch modification.
/// Uses the AMD HSA extension table to create interceptable queues and
/// replace dispatches with instrumented versions.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_HSA_INTERCEPTOR_H
#define AEGISBIT_HSA_INTERCEPTOR_H

#include "llvm/Support/Error.h"
#include <cstdint>
#include <functional>

// Forward declarations for HSA types
struct hsa_queue_s;
typedef struct hsa_queue_s hsa_queue_t;
struct hsa_kernel_dispatch_packet_s;
typedef struct hsa_kernel_dispatch_packet_s hsa_kernel_dispatch_packet_t;

namespace aegisbit {

/// Callback type for dispatch interception.
/// Called for each kernel dispatch, allowing modification of the dispatch.
///
/// \param Queue The HSA queue
/// \param Packet The dispatch packet (can be modified)
/// \param OriginalKernelObject Original kernel object handle
/// \param OriginalKernarg Original kernel arguments pointer
/// \param OriginalKernargSize Original kernel arguments size
/// \return true to proceed with (potentially modified) dispatch, false to skip
using DispatchModifyCallback = std::function<bool(
    hsa_queue_t* Queue,
    hsa_kernel_dispatch_packet_t* Packet,
    uint64_t OriginalKernelObject,
    void* OriginalKernarg,
    uint32_t OriginalKernargSize)>;

/// HSA queue interceptor for modifying kernel dispatches.
///
/// This class intercepts kernel dispatches at the HSA queue level, allowing
/// the tracing engine to:
/// 1. Replace the kernel object with an instrumented version
/// 2. Extend kernel arguments with trace buffer pointers
/// 3. Install completion handlers for trace collection
///
/// The interceptor is armed during library initialization and resolves the AMD
/// extension table after `hsa_init()` succeeds. Queue creation is intercepted
/// via `LD_PRELOAD` so HIP and direct HSA callers both flow through the same
/// packet handler.
///
/// Usage:
/// \code
///   HSAInterceptor::install();
///   HSAInterceptor::setDispatchCallback([](queue, packet, ...) {
///     // Modify packet with instrumented kernel
///     return true;  // Proceed with dispatch
///   });
///   // ... run HIP program ...
///   HSAInterceptor::uninstall();
/// \endcode
class HSAInterceptor {
public:
  /// Install the HSA interceptor.
  /// Sets up queue interception for all existing and future queues.
  /// \return Error on failure
  static llvm::Error install();

  /// Uninstall the HSA interceptor.
  /// Restores original HSA queue behavior.
  static void uninstall();

  /// Check if interceptor is installed.
  static bool isInstalled();

  /// Set the dispatch modification callback.
  /// Called for each kernel dispatch to allow modification.
  static void setDispatchCallback(DispatchModifyCallback Callback);

  /// Clear the dispatch callback.
  static void clearDispatchCallback();

  /// Get statistics about intercepted dispatches.
  struct Stats {
    uint64_t TotalDispatches = 0;       ///< Total dispatches seen
    uint64_t ModifiedDispatches = 0;    ///< Dispatches that were modified
    uint64_t SkippedDispatches = 0;     ///< Dispatches that were skipped
    uint64_t ErrorDispatches = 0;       ///< Dispatches with errors
  };
  static Stats getStats();

  /// Reset statistics.
  static void resetStats();

  /// Internal queue write interception handler
  /// Called by the HSA queue intercept callback for each batch of packets.
  /// \param Packets Pointer to array of AQL packets
  /// \param PacketCount Number of packets in the array
  /// \param UserData User data passed to hsa_amd_queue_intercept_register
  /// \param CallbackData Reserved for future use
  /// \param WriterPtr Pointer to packet writer callback for submitting packets
  static void handlePacketWrite(
      const void* Packets,
      uint64_t PacketCount,
      uint64_t UserData,
      void* CallbackData,
      void* WriterPtr);

  /// Register interception for a specific queue
  static llvm::Error registerQueueIntercept(hsa_queue_t* Queue);

private:
  HSAInterceptor() = delete;
};

} // namespace aegisbit

#endif // AEGISBIT_HSA_INTERCEPTOR_H
