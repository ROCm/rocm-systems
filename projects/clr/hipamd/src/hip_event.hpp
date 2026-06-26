/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_EVENT_H
#define HIP_EVENT_H

#include "hip_internal.hpp"
#include "thread/monitor.hpp"

#if !defined(_MSC_VER)
#include <sys/mman.h>
#endif

// Internal structure for stream callback handler
namespace hip {
class StreamCallback {
 protected:
  void* userData_;  //!< User data passed to callback function

 public:
  explicit StreamCallback(void* userData) : userData_(userData) {}

  virtual void CL_CALLBACK callback() = 0;

  virtual ~StreamCallback() = default;
};

class StreamAddCallback : public StreamCallback {
  hipStreamCallback_t callBack_;  //!< Stream callback function pointer
  hipStream_t stream_;            //!< Stream associated with the callback

 public:
  StreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData)
      : StreamCallback(userData), stream_(stream), callBack_(callback) {}

  void CL_CALLBACK callback() override {
    hipError_t status = hipSuccess;
    callBack_(stream_, status, userData_);
  }
};

class LaunchHostFuncCallback : public StreamCallback {
  hipHostFn_t callBack_;  //!< Host function callback pointer

 public:
  LaunchHostFuncCallback(hipHostFn_t callback, void* userData)
      : StreamCallback(userData), callBack_(callback) {}

  void CL_CALLBACK callback() override { callBack_(userData_); }
};

void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data);

#define IPC_SIGNALS_PER_EVENT 32

// Optimized IPC event shared memory structure
// Note: All atomics use relaxed memory ordering where safe for performance
typedef struct ihipIpcEventShmem_s {
  // Reference counting for shared memory lifecycle
  std::atomic<int> owners;
  // Metadata: only written once during initialization, relaxed ordering safe
  std::atomic<int> owners_device_id;
  std::atomic<int> owners_process_id;
  // Ring buffer indices: requires acquire-release ordering for synchronization
  std::atomic<int> read_index;
  std::atomic<int> write_index;
  // Signal array: GPU-accessible memory for event signaling
  // Using uint32_t for GPU compatibility
  alignas(64) uint32_t signal[IPC_SIGNALS_PER_EVENT];
} ihipIpcEventShmem_t;

// Deferred cleanup of emulated-IPC-event shared memory.
//
// Physical IPC cleanup (ihipHostUnregister -> munmap -> shm_unlink) is moved out
// of ~IPCEventEmulated() so that hipEventDestroy() does not block on unrelated
// GPU work. ihipHostUnregister() internally calls device->SyncAllStreams(),
// which drains ALL pending streams on the device within the current context;
// calling it from the destroy path makes hipEventDestroy() latency proportional
// to whatever unrelated work happens to be queued. Instead we enqueue the
// cleanup and drain it at points where a device-wide wait is already expected.
struct DeferredIpcEventCleanup {
  ihipIpcEventShmem_t* shmem;  //!< mapped IPC shared memory to unregister/unmap
  std::string ipc_name;        //!< shm object name (unlinked when owners == 0)
  int owners_after_decrement;  //!< owners count after this process released
  int device_id;               //!< device whose streams the cleanup will drain
};

// Enqueue an emulated-IPC-event cleanup item. O(1) and non-blocking.
void enqueueDeferredIpcEventCleanup(const DeferredIpcEventCleanup& item);

// Drain pending emulated-IPC-event cleanups. Must only be called where a
// device-wide blocking wait is already semantically expected (e.g.
// hipDeviceSynchronize / hipDeviceReset / runtime teardown), because the
// physical cleanup calls ihipHostUnregister() -> SyncAllStreams(). Passing
// device_id < 0 drains items for all devices.
void drainDeferredIpcEventCleanup(int device_id = -1);

class EventMarker : public amd::Marker {
 public:
  EventMarker(amd::HostQueue& stream, bool disableFlush, bool markerTs = false,
              int32_t scope = amd::Device::kCacheStateInvalid, bool batch_flush = true,
              bool enable_profiling = true)
      : amd::Marker(stream, disableFlush) {
    profilingInfo_.enabled_ = enable_profiling;
    profilingInfo_.marker_ts_ = markerTs;
    profilingInfo_.batch_flush_ = batch_flush;
    profilingInfo_.clear();
    setCommandEntryScope(scope);
  }
};

class Event {
  /// Capture stream where event is recorded
  hipStream_t captureStream_ = nullptr;
  /// Previous captured nodes before event record
  std::vector<hip::GraphNode*> nodesPrevToRecorded_;

 protected:
  bool CheckHwEvent() {
    const amd::SyncPolicy policy =
       (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;
    return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, false, policy);
  }

 public:
  // Flushes CPU command batch in direct dispatch mode
  static constexpr bool kBatchFlush = true;

  explicit Event(uint32_t flags) : flags_(flags), event_(nullptr) {
    device_id_ = hip::getCurrentDevice()->deviceId();
  }

  virtual ~Event() {
    if (event_ != nullptr) {
      event_->release();
    }
  }

  virtual hipError_t query();
  virtual hipError_t synchronize();
  hipError_t elapsedTime(Event& eStop, float& ms);

  virtual hipError_t streamWaitCommand(amd::Command*& command, hip::Stream* stream);
  virtual hipError_t streamWait(hip::Stream* stream, uint flags);

  virtual hipError_t recordCommand(amd::Command*& command, amd::HostQueue* stream,
                                   uint32_t flags = 0, bool batch_flush = true);
  virtual hipError_t enqueueRecordCommand(hip::Stream* stream, amd::Command* command);
  hipError_t addMarker(hip::Stream* stream, amd::Command* command, bool batch_flush = true);

  uint32_t flags() const { return flags_; }

  void BindCommand(amd::Command& command) {
    std::scoped_lock lock(lock_);
    if (event_ != nullptr) {
      event_->release();
    }
    event_ = &command.event();
    command.retain();
  }

  std::recursive_mutex& lock() { return lock_; }
  const int deviceId() const { return device_id_; }
  void setDeviceId(int id) { device_id_ = id; }
  amd::Event* event() { return event_; }

  /// Get capture stream where event is recorded
  hipStream_t GetCaptureStream() const { return captureStream_; }
  /// Set capture stream where event is recorded
  void SetCaptureStream(hipStream_t stream) { captureStream_ = stream; }
  /// Returns previous captured nodes before event record
  const std::vector<hip::GraphNode*>& GetNodesPrevToRecorded() const {
    return nodesPrevToRecorded_;
  }
  /// Set last captured graph node before event record
  void SetNodesPrevToRecorded(const std::vector<hip::GraphNode*>& graphNode) {
    nodesPrevToRecorded_ = graphNode;
  }
  virtual hipError_t GetHandle(ihipIpcEventHandle_t* handle) {
    return hipErrorInvalidConfiguration;
  }
  virtual hipError_t OpenHandle(ihipIpcEventHandle_t* handle) {
    return hipErrorInvalidConfiguration;
  }
  virtual bool awaitEventCompletion();
  virtual bool ready();
  virtual int64_t time(bool getStartTs) const;

 protected:
  uint32_t flags_;             //!< Flags associated with the event
  std::recursive_mutex lock_;  //!< Mutex for thread-safe access to event state
  amd::Event* event_;          //!< Underlying ROCclr event object for GPU synchronization
  int device_id_;              //!< Device ID where this event was created
  std::atomic<bool> synced_since_last_record_{false};  //!< Set by hipEventSynchronize, cleared by hipEventRecord
  uint64_t coalesce_id_ = 0;  //!< 0 = unassigned; non-zero = unique coalesce identity

 public:
  void MarkSynced() { synced_since_last_record_.store(true, std::memory_order_release); }
  bool WasSyncedSinceLastRecord() {
    return synced_since_last_record_.exchange(false, std::memory_order_acq_rel);
  }

 private:
  static uint64_t GenerateCoalesceId() {
    static std::atomic<uint64_t> nextId{0};  // First id is 1; 0 remains the sentinel
    return ++nextId;
  }
};

class EventDD : public Event {
 public:
  explicit EventDD(uint32_t flags) : Event(flags) {}
  ~EventDD() override = default;

  bool awaitEventCompletion() override;
  bool ready() override;
  int64_t time(bool getStartTs) const override;
};

/// Emulated IPC event using POSIX shared memory + stream write/wait value.
/// Used on PAL/Windows path where ROCr IPC signals are unavailable.
class IPCEventEmulated : public Event {
  /// IPC event metadata structure
  struct ihipIpcEvent_t {
    std::string ipc_name_;                //!< Name of the shared memory object for IPC
    ihipIpcEventShmem_t* ipc_shmem_;      //!< Pointer to mapped IPC shared memory structure

    ihipIpcEvent_t() : ipc_shmem_(nullptr) {
      ipc_name_.reserve(32);  // Reserve space for typical IPC name "/hip_<pid>_<counter>"
    }
  };
  ihipIpcEvent_t ipc_evt_;

 public:
  explicit IPCEventEmulated(uint32_t flags = hipEventInterprocess) : Event(flags) {}
  ~IPCEventEmulated() override {
    if (ipc_evt_.ipc_shmem_) {
      int owners = --ipc_evt_.ipc_shmem_->owners;
      // Poll the IPC signal (this event's own completion). Cheap when the event
      // is already complete; otherwise it only waits for this event's recorded
      // work -- it does NOT drain unrelated streams.
      hipError_t status = synchronize();
      (void)status;
#if !defined(_MSC_VER)
      // POSIX: defer the physical IPC cleanup (ihipHostUnregister ->
      // SyncAllStreams, munmap, shm_unlink) out of the destroy path so
      // hipEventDestroy() does not block on unrelated device work (the #7520
      // stall). Drained later at a safe device-wide sync point.
      hip::ihipIpcEventShmem_t* shmem = ipc_evt_.ipc_shmem_;
      ipc_evt_.ipc_shmem_ = nullptr;  // detach the user-facing handle
      hip::enqueueDeferredIpcEventCleanup(
          {shmem, ipc_evt_.ipc_name_, owners, deviceId()});
#else
      // Windows/PAL: keep the original inline cleanup. The destroy-latency issue
      // (#7520) was reported and validated only on Linux, so the emulated path
      // here is left unchanged to avoid an untested behavior shift on a path
      // that has no ROCr-IPC-signal alternative.
      status = ihipHostUnregister(&ipc_evt_.ipc_shmem_->signal);
      if (!amd::Os::MemoryUnmapFile(ipc_evt_.ipc_shmem_, sizeof(hip::ihipIpcEventShmem_t))) {
        // print hipErrorInvalidHandle;
      }
      if (owners == 0) {
        amd::Os::shm_unlink(ipc_evt_.ipc_name_);
      }
#endif
    }
    // NOTE: the previous unconditional POSIX shm_unlink() that ran here (outside
    // the ipc_shmem_ guard, under #if !defined(_MSC_VER)) was removed. It fired
    // even when owners > 0 (peers still hold the shmem) or when ipc_shmem_ was
    // null / already unlinked, racing peers that still needed to open the shm by
    // name. The amd::Os::shm_unlink in the owners == 0 path is the correct and
    // sufficient unlink.
  }
  bool createIpcEventShmemIfNeeded();
  hipError_t GetHandle(ihipIpcEventHandle_t* handle) override;
  hipError_t OpenHandle(ihipIpcEventHandle_t* handle) override;
  hipError_t synchronize() override;
  hipError_t query() override;

  hipError_t streamWait(hip::Stream* stream, uint flags) override;

  hipError_t recordCommand(amd::Command*& command, amd::HostQueue* queue, uint32_t flags = 0,
                           bool batch_flush = true) override;
  hipError_t enqueueRecordCommand(hip::Stream* stream, amd::Command* command) override;
};

/// Callback data for IPC event stream wait operations
struct CallbackData {
  const int previous_read_index;               //!< Snapshot of read index for synchronization
  hip::ihipIpcEventShmem_t* const shmem;       //!< IPC shared memory for event signaling
};

/// True IPC event backed by a device::Signal with IPC capability.
/// On ROCm, this uses ROCr IPC signals
/// Record dispatches a standard barrier with an internal tracking signal, then
/// registers an async handler that sets the IPC signal to 0 when work completes.
/// StreamWait dispatches a barrier with the IPC signal as dep_signal;
/// the GPU waits until the signal reaches 0 before proceeding.
class IPCEvent : public Event {
  amd::device::Signal* ipc_signal_;

 public:
  explicit IPCEvent(uint32_t flags = hipEventInterprocess)
      : Event(flags), ipc_signal_(nullptr) {}
  ~IPCEvent() override;

  hipError_t GetHandle(ihipIpcEventHandle_t* handle) override;
  hipError_t OpenHandle(ihipIpcEventHandle_t* handle) override;
  hipError_t synchronize() override;
  hipError_t query() override;
  hipError_t streamWait(hip::Stream* stream, uint flags) override;
  hipError_t recordCommand(amd::Command*& command, amd::HostQueue* queue, uint32_t flags = 0,
                           bool batch_flush = true) override;
  hipError_t enqueueRecordCommand(hip::Stream* stream, amd::Command* command) override;

 private:
  hipError_t createIpcSignalIfNeeded();
};
}  // namespace hip

#endif  // HIP_EVEMT_H
