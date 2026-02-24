/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

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

class EventMarker : public amd::Marker {
 public:
  EventMarker(amd::HostQueue& stream, bool disableFlush, bool markerTs = false,
              int32_t scope = amd::Device::kCacheStateInvalid, bool batch_flush = true)
      : amd::Marker(stream, disableFlush) {
    profilingInfo_.enabled_ = true;
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

  explicit Event(uint32_t flags)
      : flags_(flags), lock_(true), event_(nullptr) {
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
    amd::ScopedLock lock(lock_);
    if (event_ != nullptr) {
      event_->release();
    }
    event_ = &command.event();
    command.retain();
  }

  amd::Monitor& lock() { return lock_; }
  int deviceId() const { return device_id_; }
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
  uint32_t flags_;         //!< Flags associated with the event
  amd::Monitor lock_;      //!< Mutex for thread-safe access to event state
  amd::Event* event_;      //!< Underlying ROCclr event object for GPU synchronization
  int device_id_;          //!< Device ID where this event was created
};

class EventDD : public Event {
 public:
  explicit EventDD(uint32_t flags) : Event(flags) {}
  ~EventDD() override = default;

  bool awaitEventCompletion() override;
  bool ready() override;
  int64_t time(bool getStartTs) const override;
};

class IPCEvent : public Event {
  /// IPC event metadata structure
  struct ihipIpcEvent_t {
    std::string ipc_name_;                //!< Name of the shared memory object for IPC
    ihipIpcEventShmem_t* ipc_shmem_;      //!< Pointer to mapped IPC shared memory structure

    ihipIpcEvent_t() : ipc_shmem_(nullptr) {
      ipc_name_.reserve(32);  // Reserve space for typical IPC name "/hip_<pid>_<counter>"
    }
    void setipcname(const char* name) { ipc_name_ = name; }
  };
  ihipIpcEvent_t ipc_evt_;

 public:
  explicit IPCEvent(uint32_t flags = hipEventInterprocess) : Event(flags) {}
  ~IPCEvent() override {
    if (ipc_evt_.ipc_shmem_) {
      int owners = --ipc_evt_.ipc_shmem_->owners;
      // Make sure event is synchronized
      hipError_t status = synchronize();
      status = ihipHostUnregister(&ipc_evt_.ipc_shmem_->signal);
      if (!amd::Os::MemoryUnmapFile(ipc_evt_.ipc_shmem_, sizeof(hip::ihipIpcEventShmem_t))) {
        // print hipErrorInvalidHandle;
      }
      if (owners == 0) {
        amd::Os::shm_unlink(ipc_evt_.ipc_name_);
      }
    }
#if !defined(_MSC_VER)
    // Clean up the POSIX shared memory object
    if (!ipc_evt_.ipc_name_.empty()) {
      shm_unlink(ipc_evt_.ipc_name_.c_str());
    }
#endif
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
}  // namespace hip

#endif  // HIP_EVEMT_H
