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
  void* userData_;

 public:
  StreamCallback(void* userData) : userData_(userData) {}

  virtual void CL_CALLBACK callback() = 0;

  virtual ~StreamCallback() {};
};

class StreamAddCallback : public StreamCallback {
  hipStreamCallback_t callBack_;
  hipStream_t stream_;

 public:
  StreamAddCallback(hipStream_t stream, hipStreamCallback_t callback, void* userData)
      : StreamCallback(userData) {
    stream_ = stream;
    callBack_ = callback;
  }

  void CL_CALLBACK callback() {
    hipError_t status = hipSuccess;
    callBack_(stream_, status, userData_);
  }
};

class LaunchHostFuncCallback : public StreamCallback {
  hipHostFn_t callBack_;

 public:
  LaunchHostFuncCallback(hipHostFn_t callback, void* userData) : StreamCallback(userData) {
    callBack_ = callback;
  }

  void CL_CALLBACK callback() { callBack_(userData_); }
};

void CL_CALLBACK ihipStreamCallback(cl_event event, cl_int command_exec_status, void* user_data);


#define IPC_SIGNALS_PER_EVENT 32
typedef struct ihipIpcEventShmem_s {
  std::atomic<int> owners;
  std::atomic<int> owners_device_id;
  std::atomic<int> owners_process_id;
  std::atomic<int> read_index;
  std::atomic<int> write_index;
  uint32_t signal[IPC_SIGNALS_PER_EVENT];
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
  /// capture stream where event is recorded
  hipStream_t captureStream_ = nullptr;
  /// Previous captured nodes before event record
  std::vector<hip::GraphNode*> nodesPrevToRecorded_;

 protected:
  bool CheckHwEvent() {
    amd::SyncPolicy policy =
        (flags_ == hipEventBlockingSync) ? amd::SyncPolicy::Blocking : amd::SyncPolicy::Auto;
    return g_devices[deviceId()]->devices()[0]->IsHwEventReady(*event_, false, policy);
  }

 public:
  constexpr static bool kBatchFlush = true;  //!< Flushes CPU command batch in direct dispatch mode

  Event(uint32_t flags)
      : flags_(flags), lock_(true) /* hipEvent_t lock*/, event_(nullptr), stream_(nullptr) {
    device_id_ = hip::getCurrentDevice()->deviceId();  // Created in current device ctx
  }

  virtual ~Event() {
    if (event_ != nullptr) {
      event_->release();
    }
  }
  uint32_t flags_;  //!< flags associated with the event

  virtual hipError_t query();
  virtual hipError_t synchronize();
  hipError_t elapsedTime(Event& eStop, float& ms);

  virtual hipError_t streamWaitCommand(amd::Command*& command, hip::Stream* stream);
  virtual hipError_t streamWait(hip::Stream* stream, uint flags);

  virtual hipError_t recordCommand(amd::Command*& command, amd::HostQueue* stream,
                                   uint32_t flags = 0, bool batch_flush = true);
  virtual hipError_t enqueueRecordCommand(hip::Stream* stream, amd::Command* command);
  hipError_t addMarker(hip::Stream* stream, amd::Command* command, bool batch_flush = true);

  void BindCommand(amd::Command& command) {
    amd::ScopedLock lock(lock_);
    if (event_ != nullptr) {
      event_->release();
    }
    event_ = &command.event();
    command.retain();
  }

  amd::Monitor& lock() { return lock_; }
  const int deviceId() const { return device_id_; }
  void setDeviceId(int id) { device_id_ = id; }
  amd::Event* event() { return event_; }
  /// Get capture stream where event is recorded
  hipStream_t GetCaptureStream() const { return captureStream_; }
  /// Set capture stream where event is recorded
  void SetCaptureStream(hipStream_t stream) { captureStream_ = stream; }
  /// Returns previous captured nodes before event record
  std::vector<hip::GraphNode*> GetNodesPrevToRecorded() const { return nodesPrevToRecorded_; }
  /// Set last captured graph node before event record
  void SetNodesPrevToRecorded(std::vector<hip::GraphNode*>& graphNode) {
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
  amd::Monitor lock_;
  hip::Stream* stream_;
  amd::Event* event_;
  int device_id_;
};

class EventDD : public Event {
 public:
  EventDD(unsigned int flags) : Event(flags) {}
  virtual ~EventDD() {}

  virtual bool awaitEventCompletion();
  virtual bool ready();
  virtual int64_t time(bool getStartTs) const;
};

class IPCEvent : public Event {
  // IPC Events
  struct ihipIpcEvent_t {
    std::string ipc_name_;
    int ipc_fd_;
    ihipIpcEventShmem_t* ipc_shmem_;
    ihipIpcEvent_t() : ipc_name_("dummy"), ipc_fd_(0), ipc_shmem_(nullptr) {}
    void setipcname(const char* name) { ipc_name_ = std::string(name); }
  };
  ihipIpcEvent_t ipc_evt_;

 public:
  ~IPCEvent() {
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
    if (!ipc_evt_.ipc_name_.empty() && ipc_evt_.ipc_name_ != "dummy") {
      shm_unlink(ipc_evt_.ipc_name_.c_str());
    }
#endif
  }
  IPCEvent() : Event(hipEventInterprocess) {}
  bool createIpcEventShmemIfNeeded();
  hipError_t GetHandle(ihipIpcEventHandle_t* handle);
  hipError_t OpenHandle(ihipIpcEventHandle_t* handle);
  hipError_t synchronize();
  hipError_t query();

  hipError_t streamWait(hip::Stream* stream, uint flags);

  hipError_t recordCommand(amd::Command*& command, amd::HostQueue* queue, uint32_t flags = 0,
                           bool batch_flush = true) override;
  hipError_t enqueueRecordCommand(hip::Stream* stream, amd::Command* command);
};


struct CallbackData {
  int previous_read_index;
  hip::ihipIpcEventShmem_t* shmem;
};
}  // namespace hip

#endif  // HIP_EVEMT_H
