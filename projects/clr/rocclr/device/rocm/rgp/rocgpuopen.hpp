/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "device/rocm/rocrctx.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <atomic>
#include <mutex>

#ifdef ROC_GPUOPEN
#include "devDriverServer.h"
#include "protocols/rgpServer.h"
#include "ddRpcServer.h"
#include "hsa_ven_amd_aqlprofile.h"
#include "device/rocm/rgp/roctracesession.hpp"  // IRocTraceController full definition

// Forward declarations for UberTrace / DriverUtils layer (global ::roc namespace)
namespace roc {
class RocUberTraceService;
class RocDriverUtilsService;
} // namespace roc
#endif

namespace amd::roc {

class Device;
class VirtualGPU;
class Kernel;

// ================================================================================================
// RGP SQTT instrumentation marker identifier codes (Table 1, RGP SQTT Instrumentation Spec).
// These are hardware-level binary format definitions — not tied to any particular GPU driver API.
enum RgpSqttMarkerIdentifier : uint32_t {
  RgpSqttMarkerIdentifierEvent            = 0x0,
  RgpSqttMarkerIdentifierCbStart          = 0x1,
  RgpSqttMarkerIdentifierCbEnd            = 0x2,
  RgpSqttMarkerIdentifierBarrierStart     = 0x3,
  RgpSqttMarkerIdentifierBarrierEnd       = 0x4,
  RgpSqttMarkerIdentifierUserEvent        = 0x5,
  RgpSqttMarkerIdentifierGeneralApi       = 0x6,
  RgpSqttMarkerIdentifierSync             = 0x7,
  RgpSqttMarkerIdentifierPresent          = 0x8,
  RgpSqttMarkerIdentifierLayoutTransition = 0x9,
  RgpSqttMarkerIdentifierRenderPass       = 0xA,
  RgpSqttMarkerIdentifierBindPipeline     = 0xC,
};

// ================================================================================================
// API type codes used in per-dispatch event markers.
enum class RgpSqttMarkerEventType : uint32_t {
  CmdNDRangeKernel    = 0,
  CmdScheduler        = 1,
  CmdCopyBuffer       = 2,
  CmdCopyImageToBuffer = 3,
  CmdCopyBufferToImage = 4,
  CmdFillBuffer       = 5,
  CmdCopyImage        = 6,
  CmdFillImage        = 7,
  CmdPipelineBarrier  = 8,
  InternalUnknown     = 26,
  Invalid             = 0xffffffff
};

// ================================================================================================
// RgpSqttMarkerEvent — "Event (Per-draw/dispatch)" marker (Table 4).
struct RgpSqttMarkerEvent {
  union {
    struct {
      uint32_t identifier  : 4;
      uint32_t extDwords   : 3;
      uint32_t apiType     : 24;
      uint32_t hasThreadDims : 1;
    };
    uint32_t dword01;
  };
  union {
    struct {
      uint32_t cbID               : 20;
      uint32_t vertexOffsetRegIdx : 4;
      uint32_t instanceOffsetRegIdx : 4;
      uint32_t drawIndexRegIdx    : 4;
    };
    uint32_t dword02;
  };
  union {
    uint32_t cmdID;
    uint32_t dword03;
  };
};

// ================================================================================================
// RgpSqttMarkerEventWithDims — per-dispatch marker including workgroup dimensions.
struct RgpSqttMarkerEventWithDims {
  RgpSqttMarkerEvent event;
  uint32_t threadX;
  uint32_t threadY;
  uint32_t threadZ;
};

// ================================================================================================
// RgpSqttMarkerBarrierStart — "Barrier Start" marker (Table 5).
struct RgpSqttMarkerBarrierStart {
  union {
    struct {
      uint32_t identifier : 4;
      uint32_t extDwords  : 3;
      uint32_t cbId       : 20;
      uint32_t reserved   : 5;
    };
    uint32_t dword01;
  };
  union {
    struct {
      uint32_t driverReason : 31;
      uint32_t internal     : 1;
    };
    uint32_t dword02;
  };
};

// ================================================================================================
// RgpSqttMarkerBarrierEnd — "Barrier End" marker (Table 6).
struct RgpSqttMarkerBarrierEnd {
  union {
    struct {
      uint32_t identifier     : 4;
      uint32_t extDwords      : 3;
      uint32_t cbId           : 20;
      uint32_t waitOnEopTs    : 1;
      uint32_t vsPartialFlush : 1;
      uint32_t psPartialFlush : 1;
      uint32_t csPartialFlush : 1;
      uint32_t pfpSyncMe      : 1;
    };
    uint32_t dword01;
  };
  union {
    struct {
      uint32_t syncCpDma          : 1;
      uint32_t invalTcp           : 1;
      uint32_t invalSqI           : 1;
      uint32_t invalSqK           : 1;
      uint32_t flushTcc           : 1;
      uint32_t invalTcc           : 1;
      uint32_t flushCb            : 1;
      uint32_t invalCb            : 1;
      uint32_t flushDb            : 1;
      uint32_t invalDb            : 1;
      uint32_t numLayoutTransitions : 16;
      uint32_t reserved           : 6;
    };
    uint32_t dword02;
  };
};

// ================================================================================================
// RgpSqttMarkerCbStart — "Command Buffer Start" marker (identifier=0x1).
// 4 DWORDs: dword01 (header), dword02 (device_id_lo), dword03 (device_id_hi), dword04 (flags).
// device_id is a 64-bit identifier that must match the CbEnd for the assert in RGP to pass.
// queue_flags: 0 for compute queues (no SQTT semaphore timing).
struct RgpSqttMarkerCbStart {
  union {
    struct {
      uint32_t identifier     : 4;   // bits[3:0]  = RgpSqttMarkerIdentifierCbStart (0x1)
      uint32_t extDwords      : 3;   // bits[6:4]  = 0 (no extra dwords)
      uint32_t cbID           : 20;  // bits[26:7] = command-buffer ID (queue doorbell low 20 bits)
      uint32_t queueIndex     : 5;   // bits[31:27] = API queue index (0 for single-queue compute)
    };
    uint32_t dword01;
  };
  uint32_t deviceIdLow;   // low  32 bits of a stable 64-bit device identifier
  uint32_t deviceIdHigh;  // high 32 bits of a stable 64-bit device identifier
  uint32_t queueFlags;    // queue-capability flags (0 for compute; no semaphore timing)
};

// ================================================================================================
// RgpSqttMarkerCbEnd — "Command Buffer End" marker (identifier=0x2).
// 3 DWORDs: dword01 (header), dword02 (device_id_lo), dword03 (device_id_hi).
// device_id must match the corresponding CbStart.
struct RgpSqttMarkerCbEnd {
  union {
    struct {
      uint32_t identifier : 4;   // bits[3:0]  = RgpSqttMarkerIdentifierCbEnd (0x2)
      uint32_t extDwords  : 3;   // bits[6:4]  = 0
      uint32_t cbID       : 20;  // bits[26:7] = command-buffer ID
      uint32_t reserved   : 5;   // bits[31:27]
    };
    uint32_t dword01;
  };
  uint32_t deviceIdLow;   // must match the CbStart device_id low 32 bits
  uint32_t deviceIdHigh;  // must match the CbStart device_id high 32 bits
};

// ================================================================================================
// RgpSqttMarkerPipelineBind — pipeline-bind marker (Table 12).
struct RgpSqttMarkerPipelineBind {
  union {
    struct {
      uint32_t identifier : 4;
      uint32_t extDwords  : 3;
      uint32_t bindPoint  : 1;
      uint32_t cbID       : 20;
      uint32_t reserved   : 4;
    };
    uint32_t dword01;
  };
  union {
    uint32_t apiPsoHash[2];
    struct {
      uint32_t dword02;
      uint32_t dword03;
    };
  };
};

// ================================================================================================
// User event marker type codes.
enum RgpSqttMarkerUserEventType : uint32_t {
  RgpSqttMarkerUserEventTrigger    = 0x0,
  RgpSqttMarkerUserEventPop        = 0x1,
  RgpSqttMarkerUserEventPush       = 0x2,
  RgpSqttMarkerUserEventObjectName = 0x3,
};

// ================================================================================================
// RgpSqttMarkerUserEvent — user event marker header.
union RgpSqttMarkerUserEvent {
  struct {
    uint32_t identifier : 4;
    uint32_t extDwords  : 8;
    uint32_t dataType   : 8;
    uint32_t reserved   : 12;
  };
  uint32_t dword01;
};

static constexpr size_t RgpSqttMaxUserEventStringLengthInDwords = 1024;

// ================================================================================================
// RgpSqttMarkerUserEventWithString — user event marker with embedded UTF-8 string payload.
struct RgpSqttMarkerUserEventWithString {
  RgpSqttMarkerUserEvent header;
  uint32_t stringLength;
  uint32_t stringData[RgpSqttMaxUserEventStringLengthInDwords];
};

// ================================================================================================
// Barrier reason codes used when writing SQTT barrier markers.
enum class RgpSqttBarrierReason : uint32_t {
  Invalid           = 0,
  MemDependency     = 0xC0000000,
  ProfilingControl  = 0xC0000001,
  SignalSubmit      = 0xC0000002,
  PostDeviceEnqueue = 0xC0000003,
  Unknown           = 0xffffffff
};

// ================================================================================================
// Abstract capture manager interface for the ROCm device path. Uses HSA types in place of
// PAL types (hsa_queue_t* instead of Pal::IQueue*, hsa_signal_t instead of Pal::IFence, etc.)
class ICaptureMgr {
 public:
  virtual ~ICaptureMgr() = default;

  virtual bool Update() = 0;

  // DriverControlServer device-init lifecycle — mirrors PAL's three-stage sequence:
  //   StartEarlyDeviceInit() → StartLateDeviceInit() → FinishDeviceInit()
  // Called from rocdevice.cpp at the corresponding device init milestones.
  virtual void StartLateDeviceInit() = 0;
  virtual void FinishDeviceInit() = 0;

  virtual void PreDispatch(VirtualGPU* gpu, const Kernel& kernel, size_t x, size_t y,
                           size_t z) = 0;
  virtual void PostDispatch(VirtualGPU* gpu) = 0;

  virtual void FinishRGPTrace(VirtualGPU* gpu, bool aborted) = 0;

  virtual void WriteBarrierStartMarker(const VirtualGPU* gpu, uint32_t reason) const = 0;
  virtual void WriteBarrierEndMarker(const VirtualGPU* gpu) const = 0;

  virtual bool RegisterTimedQueue(uint32_t queue_id, hsa_queue_t* queue,
                                  bool* debug_vmid) const = 0;

  virtual bool TimedQueueSubmit(hsa_queue_t* queue, uint64_t cmd_id,
                                hsa_signal_t completion_signal) const = 0;

  virtual uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size,
                                const void* elf_binary, size_t elf_binary_size,
                                uint64_t gpu_addr) = 0;

  virtual void AddKernelLoadEvent(uint64_t api_hash, const void* elf_binary,
                                  size_t elf_binary_size, const char* kernel_name,
                                  uint64_t kernel_gpu_va) {}
};

#ifdef ROC_GPUOPEN

// ================================================================================================
// SQTT hardware capture state (UberTrace-driven, replaces legacy RGP SqttState).
enum class SqttState : uint32_t {
  Idle              = 0,  ///< No SQTT capture in flight
  Preparing         = 1,  ///< Counting preparation dispatches before capture starts
  Running           = 2,  ///< SQTT hardware capture active
  WaitingForResults = 3,  ///< Stop submitted; waiting for GPU to finish
};

// ================================================================================================
// Implements the UberTrace capture manager for the ROCm/HSA device path.
// Mirrors amd::pal::UberTraceCaptureMgr but uses HSA primitives instead of PAL types.
// Owns RocTraceSession (state machine + RDF serialisation) and RocUberTraceService
// (UberTrace RPC bridge), mirroring PAL's Platform::CreateUberTraceService() +
// TraceSession infrastructure.
//
// Also implements IRocTraceController so it can register with RocTraceSession and drive
// the SQTT hardware start/stop in response to UberTrace RequestTrace() calls.
class RocUberTraceCaptureMgr final : public ICaptureMgr,
                                     public ::roc::IRocTraceController {
 public:
  ~RocUberTraceCaptureMgr();

  static RocUberTraceCaptureMgr* Create(Device* device);

  // Returns true if the tool has enabled tracing via the UberTrace RPC protocol
  // OR if any trace state is active (Requested/Preparing/Running).
  // Mirrors PAL's Platform::IsTracingEnabled() which ORs TraceSession::IsTracingEnabled()
  // with any in-flight trace state.
  // Distinct from connection availability (IsConnectionAvailable), checked in Init().
  bool IsTracingEnabled() const;

  bool Update() override;

  void StartLateDeviceInit() override;
  void FinishDeviceInit() override;

  void PreDispatch(VirtualGPU* gpu, const Kernel& kernel, size_t x, size_t y, size_t z) override;
  void PostDispatch(VirtualGPU* gpu) override;
  void FinishRGPTrace(VirtualGPU* gpu, bool aborted) override;

  void WriteBarrierStartMarker(const VirtualGPU* gpu, uint32_t reason) const override;
  void WriteBarrierEndMarker(const VirtualGPU* gpu) const override;

  bool RegisterTimedQueue(uint32_t queue_id, hsa_queue_t* queue, bool* debug_vmid) const override;
  bool TimedQueueSubmit(hsa_queue_t* queue, uint64_t cmd_id,
                        hsa_signal_t completion_signal) const override;

  uint64_t AddElfBinary(const void* exe_binary, size_t exe_binary_size, const void* elf_binary,
                        size_t elf_binary_size, uint64_t gpu_addr) override;

  void AddKernelLoadEvent(uint64_t api_hash, const void* elf_binary, size_t elf_binary_size,
                          const char* kernel_name, uint64_t kernel_gpu_va);

 // ── IRocTraceController (drives SQTT hardware via UberTrace session) ──────────────────────
  const char* GetName()    const override { return "RocSqttController"; }
  uint32_t    GetVersion() const override { return 1; }
  bool OnTraceRequested(::roc::RocTraceSession* pSession) override;
  void OnTraceCanceled() override;
  void OnTraceFinished() override;

 private:
  explicit RocUberTraceCaptureMgr(Device* device);

  bool Init();
  void Destroy();

  void WaitForDriverResume();
  bool IsTraceRunning() const { return trace_running_; }

  // SQTT hardware capture via aqlprofile extension (HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE).
  // Uses the same aqlprofile API table as PerfCounterProfile — start/stop/read populate AQL
  // packets which are submitted via dispatchCounterAqlPacket, letting the HSA runtime/KFD
  // enable SQTT register access through its normal profiling-mode ioctl path.
  bool BeginSqttTrace(VirtualGPU* gpu);
  void EndSqttTrace(VirtualGPU* gpu);
  void CollectSqttResults(VirtualGPU* gpu);
  void FreeSqttResources();

  void WriteMarker(const VirtualGPU* gpu, const void* data, size_t data_size) const;
  void WriteCbStartMarker(const VirtualGPU* gpu) const;
  void WriteCbEndMarker(const VirtualGPU* gpu) const;
  void WriteComputeBindMarker(const VirtualGPU* gpu, uint64_t api_hash) const;
  void WriteEventWithDimsMarker(const VirtualGPU* gpu, RgpSqttMarkerEventType api_type,
                                uint32_t x, uint32_t y, uint32_t z) const;
  void WriteUserEventMarker(const VirtualGPU* gpu, RgpSqttMarkerUserEventType event_type,
                            const std::string& name) const;

  void FlushPendingTraceData();

  // ── Pending ELF / kernel-load / queue-event accumulators ────────────────────────────────
  struct PendingElfData {
    std::vector<uint8_t> elf;
    uint64_t original_hash;
    uint32_t pci_id;
  };

  // RDF COLoadEvent record — must match palCodeObjectTraceSource.h layout.
  struct KernelLoadEntry {
    uint32_t pciId;
    uint32_t _pad;
    uint64_t baseAddress;
    uint64_t apiPsoHash;
    uint64_t _reserved;
    uint64_t cpuTimestamp;
  };

  struct PendingQueueEvent {
    uint64_t cpuTimestamp;
    uint32_t sqttCmdBufId;
    uint32_t submitSubIndex;
    uint64_t apiEventId;
  };

  std::vector<PendingElfData>    pending_elfs_;
  std::vector<KernelLoadEntry>   pending_load_events_;
  std::vector<PendingQueueEvent> pending_queue_events_;

  Device* device_;
  DevDriver::DevDriverServer* dev_driver_server_;  //!< Owned DevDriver server instance
  DevDriver::RGPProtocol::RGPServer* rgp_server_;  //!< Non-owning; for DevDriver bus only

  // UberTrace + DriverUtils layer — mirrors PAL's RPC service infrastructure.
  // Note: ::roc namespace (global) vs amd::roc (this class's namespace).
  ::roc::RocTraceSession*       roc_trace_session_;  //!< Owned RDF trace state machine
  ::roc::RocUberTraceService*   uber_trace_svc_;     //!< Owned UberTrace RPC service
  ::roc::RocDriverUtilsService* driver_utils_svc_;   //!< Owned DriverUtils RPC service
  DDRpcServer                   rpc_server_;          //!< DevDriver RPC server handle
  uint64_t                      global_disp_count_;
  mutable uint32_t              current_event_id_;
  mutable std::mutex            trace_mutex_;
  RgpSqttMarkerUserEventWithString* user_event_;

  // Lightweight registry of HSA queues registered for timed submission.
  mutable std::mutex                               timed_queues_mutex_;
  mutable std::unordered_map<uint32_t, hsa_queue_t*> timed_queues_;

  // trace_running_ is a cached copy of (SQTT state == Running).
  // Written in PreDispatch; read in IsTraceRunning() on the hot marker path.
  mutable bool trace_running_;

  // ── SQTT hardware capture state ────────────────────────────────────────────
  // aqlprofile extension function table — populated once via hsa_system_get_major_extension_table.
  hsa_ven_amd_aqlprofile_1_00_pfn_t sqtt_api_;

  // Profile context for SQTT (EVENT_TYPE_TRACE); buffers set during BeginSqttTrace().
  hsa_ven_amd_aqlprofile_profile_t  sqtt_profile_;

  // AQL PM4 packets — populated by hsa_ven_amd_aqlprofile_start/stop/read
  // and submitted via dispatchCounterAqlPacket.
  hsa_ext_amd_aql_pm4_packet_t sqtt_start_packet_;
  hsa_ext_amd_aql_pm4_packet_t sqtt_stop_packet_;
  hsa_ext_amd_aql_pm4_packet_t sqtt_read_packet_;

  // Host-accessible (fine-grained coherent) buffers for SQTT command + output data.
  void*    sqtt_cmd_buf_;       //!< PM4 command buffer (GPU writes commands here)
  uint32_t sqtt_cmd_buf_size_;  //!< Size of sqtt_cmd_buf_
  void*    sqtt_output_;        //!< SQTT output data buffer (GPU writes trace here)
  uint32_t sqtt_output_size_;   //!< Size of sqtt_output_
  void*    marker_cmd_buf_;       //!< Fine-grained host ring buffer for SQTT NOP marker PM4
  mutable std::atomic<uint32_t> marker_buf_idx_;  //!< Next ring slot index (wraps mod kMarkerRingSize)

  // Number of "warm-up" dispatches to run before starting actual SQTT capture.
  // Mirrors PAL's RenderOpTraceController prep-dispatch count.
  uint32_t prep_disp_count_;
  uint32_t num_prep_frames_;

  // Detailed SQTT capture options read from DevDriver trace parameters in OnTraceRequested().
  uint32_t sqtt_se_mask_;                //!< Shader engine mask (0 = all SEs)
  bool     sqtt_instruction_tokens_;     //!< true = full instruction-level tokens
  bool     sqtt_capture_code_objects_;   //!< true = include ELF binaries as RDF chunks

  // Dispatch-index capture window (CaptureTriggerMode::Index).
  // When captureMode == Index the tool specifies an inclusive [start, stop] dispatch range.
  // captureStartIndex is compared against global_disp_count_ to decide when to begin SQTT;
  // captureStopIndex is compared in PostDispatch to auto-stop without waiting for a RPC cancel.
  // For other modes (Present, Markers) these remain at sentinel values and are not checked.
  uint32_t capture_start_index_;   //!< First dispatch index to capture (inclusive)
  uint32_t capture_stop_index_;    //!< Last dispatch index to capture (inclusive); 0 = unlimited
  bool     capture_index_mode_;    //!< true when captureMode == Index

  SqttState sqtt_state_;  //!< Current SQTT hardware capture state

  // The VirtualGPU that started the current SQTT capture.  Set in BeginSqttTrace(),
  // cleared in FreeSqttResources().  Used by OnTraceCanceled() to stop hardware from
  // the UberTrace RPC callback path (deferred to the next PostDispatch on the GPU thread).
  VirtualGPU* trace_gpu_;

  // Set by OnTraceCanceled() (on the RPC thread) to signal PostDispatch (on the GPU thread)
  // to stop SQTT capture.  Avoids submitting AQL packets from the RPC thread.
  std::atomic<bool> pending_abort_;

  RocUberTraceCaptureMgr(const RocUberTraceCaptureMgr&) = delete;
  RocUberTraceCaptureMgr& operator=(const RocUberTraceCaptureMgr&) = delete;
};

#else  // ROC_GPUOPEN

// Stub capture manager used when ROC_GPUOPEN is not defined — all methods are no-ops.
class RocUberTraceCaptureMgr : public ICaptureMgr {
 public:
  static RocUberTraceCaptureMgr* Create(Device* /*device*/) { return nullptr; }

  bool Update() override { return true; }
  void StartLateDeviceInit() override {}
  void FinishDeviceInit() override {}
  void PreDispatch(VirtualGPU*, const Kernel&, size_t, size_t, size_t) override {}
  void PostDispatch(VirtualGPU*) override {}
  void FinishRGPTrace(VirtualGPU*, bool) override {}
  void WriteBarrierStartMarker(const VirtualGPU*, uint32_t) const override {}
  void WriteBarrierEndMarker(const VirtualGPU*) const override {}
  bool RegisterTimedQueue(uint32_t, hsa_queue_t*, bool*) const override { return true; }
  bool TimedQueueSubmit(hsa_queue_t*, uint64_t, hsa_signal_t) const override { return true; }
  uint64_t AddElfBinary(const void*, size_t, const void*, size_t, uint64_t) override { return 0; }
};

#endif  // ROC_GPUOPEN

}  // namespace amd::roc
