/* Copyright (c) 2015 - 2022 Advanced Micro Devices, Inc.

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

#ifndef HIP_SRC_HIP_INTERNAL_H
#define HIP_SRC_HIP_INTERNAL_H

#include "vdi_common.hpp"
#include "hip_prof_api.h"
#include "trace_helper.h"
#include "rocclr/utils/debug.hpp"
#include "hip_graph_capture.hpp"

#include <unordered_set>
#include <thread>
#include <stack>
#include <mutex>
#include <iterator>
#ifdef _WIN32
#include <process.h>
#else
#include <unistd.h>
#endif

#define KNRM "\x1B[0m"
#define KRED "\x1B[31m"
#define KGRN "\x1B[32m"
#define KYEL "\x1B[33m"
#define KBLU "\x1B[34m"
#define KMAG "\x1B[35m"
#define KCYN "\x1B[36m"
#define KWHT "\x1B[37m"

namespace hip{
  extern std::once_flag g_ihipInitialized;
}
typedef struct hipArray {
    void* data;  // FIXME: generalize this
    struct hipChannelFormatDesc desc;
    unsigned int type;
    unsigned int width;
    unsigned int height;
    unsigned int depth;
    enum hipArray_Format Format;
    unsigned int NumChannels;
    bool isDrv;
    unsigned int textureType;
    unsigned int flags;
}hipArray;

namespace hip {
enum MemcpyType {
  /// Memcpy from host to host
  hipHostToHost,
  /// Memcpy from host to device
  hipWriteBuffer,
  /// Memcpy from device to host
  hipReadBuffer,
  /// Memcpy from device A to device A
  /// Memcpy from pinned host buffer to device/device to pinned host buffer
  hipCopyBuffer,
  /// Memcpy from device A to device A, user forced to SDMA
  hipCopyBufferSDMA,
  /// Memcpy from device A to device B
  hipCopyBufferP2P,
};
struct Graph;
struct GraphNode;
struct GraphExec;
struct UserObject;
class Stream;

#define IHIP_IPC_EVENT_HANDLE_SIZE 32
#define IHIP_IPC_EVENT_RESERVED_SIZE LP64_SWITCH(28,24)
typedef struct ihipIpcEventHandle_st {
    //hsa_amd_ipc_signal_t ipc_handle;  ///< ipc signal handle on ROCr
    //char ipc_handle[IHIP_IPC_EVENT_HANDLE_SIZE];
    //char reserved[IHIP_IPC_EVENT_RESERVED_SIZE];
    char shmem_name[IHIP_IPC_EVENT_HANDLE_SIZE];
}ihipIpcEventHandle_t;

const char* ihipGetErrorName(hipError_t hip_error);
}

#define HIP_INIT(noReturn)                                                                         \
  {                                                                                                \
    bool status = true;                                                                            \
    std::call_once(hip::g_ihipInitialized, hip::init, &status);                                    \
    if (!status && !noReturn) {                                                                    \
      HIP_RETURN(hipErrorInvalidDevice);                                                           \
    }                                                                                              \
    if (hip::tls.device_ == nullptr && hip::g_devices.size() > 0) {                                \
      hip::tls.device_ = hip::g_devices[0];                                                        \
      amd::Os::setPreferredNumaNode(hip::g_devices[0]->devices()[0]->getPreferredNumaNode());      \
    }                                                                                              \
  }

#define HIP_INIT_VOID()                                                                            \
  {                                                                                                \
    bool status = true;                                                                            \
    std::call_once(hip::g_ihipInitialized, hip::init, &status);                                    \
    if (hip::tls.device_ == nullptr && hip::g_devices.size() > 0) {                                \
      hip::tls.device_ = hip::g_devices[0];                                                        \
      amd::Os::setPreferredNumaNode(hip::g_devices[0]->devices()[0]->getPreferredNumaNode());      \
    }                                                                                              \
  }


#define HIP_API_PRINT(...)                                          \
  uint64_t startTimeUs = 0;                                         \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,       \
                  "%s %s ( %s ) %s", KGRN,                          \
                  __func__, ToString( __VA_ARGS__ ).c_str(), KNRM);

#define HIP_ERROR_PRINT(err, ...)                                                  \
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s : %s",                     \
          __func__, hip::ihipGetErrorName(err), ToString( __VA_ARGS__ ).c_str());

#define HIP_INIT_API_INTERNAL(noReturn, cid, ...)                                                  \
  HIP_INIT(noReturn)                                                                               \
  HIP_API_PRINT(__VA_ARGS__)                                                                       \
  HIP_CB_SPAWNER_OBJECT(cid);

// This macro should be called at the beginning of every HIP API.
#define HIP_INIT_API(cid, ...)                                                                     \
  if (amd::Device::IsGPUInError()) {                                                              \
    HIP_RETURN(ConvertCLErrorIntoHIPError(amd::Device::GetGPUError()));                            \
  }                                                                                                \
  HIP_INIT_API_INTERNAL(0, cid, __VA_ARGS__)                                                       \
  if (hip::g_devices.size() == 0) {                                                                \
    HIP_RETURN(hipErrorNoDevice);                                                                  \
  }                                                                                                \

#define HIP_INIT_API_NO_RETURN(cid, ...)                                                           \
  HIP_INIT_API_INTERNAL(1, cid, __VA_ARGS__)

#define HIP_RETURN_DURATION(ret, ...)                                                              \
  hip::tls.last_command_error_ = ret;                                                              \
  if (amd::Device::IsGPUInError()) {                                                               \
    hipError_t hip_error = ConvertCLErrorIntoHIPError(amd::Device::GetGPUError());                 \
    hip::tls.last_error_ = hip_error;                                                              \
    hip::tls.last_command_error_ = hip_error;                                                      \
  } else {                                                                                         \
    if (hip::tls.last_command_error_ != hipSuccess &&                                              \
           hip::tls.last_command_error_ != hipErrorNotReady) {                                     \
      hip::tls.last_error_ = hip::tls.last_command_error_;                                         \
    }                                                                                              \
  }                                                                                                \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%s: Returned %s : %s", __func__,    \
                   hip::ihipGetErrorName(hip::tls.last_command_error_),                            \
                   ToString(__VA_ARGS__).c_str());                                                 \
  return hip::tls.last_command_error_;

#define HIP_RETURN(ret, ...)                                                                       \
  hip::tls.last_command_error_ = ret;                                                              \
  if (amd::Device::IsGPUInError()) {                                                               \
    hipError_t hip_error = ConvertCLErrorIntoHIPError(amd::Device::GetGPUError());                 \
    hip::tls.last_error_ = hip_error;                                                              \
    hip::tls.last_command_error_ = hip_error;                                                      \
  } else {                                                                                         \
    if (hip::tls.last_command_error_ != hipSuccess &&                                              \
           hip::tls.last_command_error_ != hipErrorNotReady) {                                     \
      hip::tls.last_error_ = hip::tls.last_command_error_;                                         \
    }                                                                                              \
  }                                                                                                \
  HIP_ERROR_PRINT(hip::tls.last_command_error_, __VA_ARGS__)                                       \
  return hip::tls.last_command_error_;

#define HIP_RETURN_ONFAIL(func)          \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      HIP_RETURN(herror);                \
    }                                    \
  } while (0);

// Cannot be use in place of HIP_RETURN.
// Refrain from using for external HIP APIs
#define IHIP_RETURN_ONFAIL(func)         \
  do {                                   \
    hipError_t herror = (func);          \
    if (herror != hipSuccess) {          \
      return herror;                     \
    }                                    \
  } while (0);

// During stream capture some actions, such as a call to hipMalloc, may be unsafe and prohibited
// during capture. It is allowed only in relaxed mode.
#define CHECK_STREAM_CAPTURE_SUPPORTED()                                                           \
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeThreadLocal) {                          \
    if (hip::tls.capture_streams_.size() != 0) {                                                   \
      for (auto stream : hip::tls.capture_streams_) {                                              \
        stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                               \
      }                                                                                            \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
  } else if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeGlobal) {                        \
    if (hip::tls.capture_streams_.size() != 0) {                                                   \
      for (auto stream : hip::tls.capture_streams_) {                                              \
        stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                               \
      }                                                                                            \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
    if (g_captureStreams.size() != 0) {                                                            \
      for (auto stream : g_captureStreams) {                                                       \
        stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                               \
      }                                                                                            \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
  }

// Device sync is not supported during capture
#define CHECK_SUPPORTED_DURING_CAPTURE()                                                           \
  if (!g_allCapturingStreams.empty()) {                                                            \
    for (auto stream : g_allCapturingStreams) {                                                    \
      stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                                 \
    }                                                                                              \
    return hipErrorStreamCaptureUnsupported;                                                       \
  }

// Sync APIs like hipMemset, hipMemcpy etc.. cannot be called when stream capture is active
// for all capture modes hipStreamCaptureModeGlobal, hipStreamCaptureModeThreadLocal and
// hipStreamCaptureModeRelaxed
#define CHECK_STREAM_CAPTURING()                                                                   \
  if (!g_allCapturingStreams.empty()) {                                                            \
    for (auto stream : g_allCapturingStreams) {                                                    \
      stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                                 \
    }                                                                                              \
    return hipErrorStreamCaptureImplicit;                                                          \
  }

#define STREAM_CAPTURE(name, stream, ...)                                                          \
  hip::getStreamPerThread(stream);                                                                 \
  if (stream != nullptr && stream != hipStreamLegacy &&                                            \
      reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus() ==                                \
          hipStreamCaptureStatusActive) {                                                          \
    hipError_t status = hip::capture##name(stream, ##__VA_ARGS__);                                 \
    return status;                                                                                 \
  } else if (stream != nullptr && stream != hipStreamLegacy &&                                     \
             reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus() ==                         \
                 hipStreamCaptureStatusInvalidated) {                                              \
    return hipErrorStreamCaptureInvalidated;                                                       \
  } else if (stream == nullptr || stream == hipStreamLegacy) {                                     \
    CHECK_STREAM_CAPTURING()                                                                       \
  }

#define PER_THREAD_DEFAULT_STREAM(stream)                                                         \
  if (stream == nullptr || stream == hipStreamLegacy) {                                           \
    stream = getPerThreadDefaultStream();                                                         \
  }

namespace hc {
class accelerator;
class accelerator_view;
};

struct ihipExec_t {
  dim3 gridDim_;
  dim3 blockDim_;
  size_t sharedMem_;
  hipStream_t hStream_;
  std::vector<char> arguments_;
};

namespace hip {
class stream_per_thread {
private:
  std::vector<hipStream_t> m_streams;
public:
  stream_per_thread();
  stream_per_thread(const stream_per_thread& ) = delete;
  void operator=(const stream_per_thread& ) = delete;
  ~stream_per_thread();
  hipStream_t get();
  void clear_spt();
};

  class Device;
  class MemoryPool;
  class Event;
  class Stream : public amd::HostQueue {
  public:
    enum Priority : int { High = -1, Normal = 0, Low = 1 };

  private:
    mutable amd::Monitor lock_;
    Device* device_;
    Priority priority_;
    unsigned int flags_;
    bool null_;
    const std::vector<uint32_t> cuMask_;
    uint64_t stream_id_;

    /// Stream capture related parameters

    /// Current capture status of the stream
    hipStreamCaptureStatus captureStatus_;
    /// Graph that is constructed with capture
    hip::Graph* pCaptureGraph_;
    /// Based on mode stream capture places restrictions on API calls that can be made within or
    /// concurrently
    hipStreamCaptureMode captureMode_{hipStreamCaptureModeGlobal};
    bool originStream_;
    /// Origin sream has no parent. Parent stream for the derived captured streams with event
    /// dependencies
    hipStream_t parentStream_ = nullptr;
    /// Last graph node captured in the stream
    std::vector<hip::GraphNode*> lastCapturedNodes_;
    /// dependencies removed via API hipStreamUpdateCaptureDependencies
    std::vector<hip::GraphNode*> removedDependencies_;
    /// Derived streams/Paralell branches from the origin stream
    std::vector<hipStream_t> parallelCaptureStreams_;
    /// Capture events
    std::unordered_set<hipEvent_t> captureEvents_;
    unsigned long long captureID_;

    static inline CommandQueue::Priority convertToQueuePriority(Priority p) {
      return p == Priority::High ? amd::CommandQueue::Priority::High : p == Priority::Low ?
                    amd::CommandQueue::Priority::Low : amd::CommandQueue::Priority::Normal;
    }
    /// Generates unique stream Id for the lifetime of the process
    uint64_t GenerateStreamId() {
      static std::atomic<uint64_t> uniqueId{0};
      return ++uniqueId;
    }

  public:
    Stream(Device* dev, Priority p = Priority::Normal, unsigned int f = 0, bool null_stream = false,
           const std::vector<uint32_t>& cuMask = {},
           hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone);

    /// Creates the hip stream object, including AMD host queue
    bool Create();
    /// Get device ID associated with the current stream;
    int DeviceId() const;
    /// Get HIP device associated with the stream
    Device* GetDevice() const { return device_; }
    /// Get device ID associated with a stream;
    static int DeviceId(const hipStream_t hStream);
    /// Returns if stream is null stream
    bool Null() const { return null_; }
    /// Returns the lock object for the current stream
    amd::Monitor& Lock() const { return lock_; }
    /// Returns the creation flags for the current stream
    unsigned int Flags() const { return flags_; }
    /// Returns the priority for the current stream
    Priority GetPriority() const { return priority_; }
    /// Returns the CU mask for the current stream
    const std::vector<uint32_t> GetCUMask() const { return cuMask_; }

    /// Fetch the stream Id
    uint64_t GetStreamId() const { return stream_id_; }
    /// Check whether any blocking stream running
    static bool StreamCaptureBlocking();

    static void Destroy(hip::Stream* stream, bool forceDestroy = false);

    virtual bool terminate();

    /// Check Stream Capture status to make sure it is done
    static bool StreamCaptureOngoing(hipStream_t hStream);

    /// Returns capture status of the current stream
    hipStreamCaptureStatus GetCaptureStatus() const { return captureStatus_; }
    /// Returns capture mode of the current stream
    hipStreamCaptureMode GetCaptureMode() const { return captureMode_; }
    /// Returns if stream is origin stream
    bool IsOriginStream() const { return originStream_; }
    void SetOriginStream() { originStream_ = true; }
    /// Returns captured graph
    hip::Graph* GetCaptureGraph() const { return pCaptureGraph_; }
    /// Returns last captured graph node
    const std::vector<hip::GraphNode*>& GetLastCapturedNodes() const { return lastCapturedNodes_; }
    /// Set last captured graph node
    void SetLastCapturedNode(hip::GraphNode* graphNode) {
      lastCapturedNodes_.clear();
      lastCapturedNodes_.push_back(graphNode);
    }
    /// returns updated dependencies removed
    const std::vector<hip::GraphNode*>& GetRemovedDependencies() {
      return removedDependencies_;
    }
    /// Append captured node via the wait event cross stream
    void AddCrossCapturedNode(std::vector<hip::GraphNode*> graphNodes, bool replace = false) {
      // replace dependencies as per flag hipStreamSetCaptureDependencies
      if (replace == true) {
        for (auto node : lastCapturedNodes_) {
          removedDependencies_.push_back(node);
        }
        lastCapturedNodes_.clear();
      }
      for (auto node : graphNodes) {
        if (std::find(lastCapturedNodes_.begin(), lastCapturedNodes_.end(), node) ==
            lastCapturedNodes_.end()) {
          lastCapturedNodes_.push_back(node);
        }
      }
    }
    /// Set graph that is being captured
    void SetCaptureGraph(hip::Graph* pGraph) {
      pCaptureGraph_ = pGraph;
      captureStatus_ = hipStreamCaptureStatusActive;
    }
    /// Release graph when capture is invalidated
    void ReleaseCaptureGraph();
    void SetCaptureId() {
      // ID is generated in Begin Capture i.e.. when capture status is active
      captureID_ = GenerateCaptureID();
    }
    void SetCaptureId(unsigned long long captureId) {
      // ID is given from parent stream
      captureID_ = captureId;
    }
    /// reset capture parameters
    hipError_t EndCapture();
    /// Set capture status
    void SetCaptureStatus(hipStreamCaptureStatus captureStatus) { captureStatus_ = captureStatus; }
    /// Set capture mode
    void SetCaptureMode(hipStreamCaptureMode captureMode) { captureMode_ = captureMode; }
    /// Set parent stream
    void SetParentStream(hipStream_t parentStream) { parentStream_ = parentStream; }
    /// Get parent stream
    hipStream_t GetParentStream() const { return parentStream_; }
    /// Generate ID for stream capture unique over the lifetime of the process
    static unsigned long long GenerateCaptureID() {
      static std::atomic<unsigned long long> uid(0);
      return ++uid;
    }
    /// Get Capture ID
    unsigned long long GetCaptureID() { return captureID_; }
    void SetCaptureEvent(hipEvent_t e) {
      amd::ScopedLock lock(lock_);
      captureEvents_.emplace(e); }
    bool IsEventCaptured(hipEvent_t e) {
      amd::ScopedLock lock(lock_);
      auto it = captureEvents_.find(e);
      if (it != captureEvents_.end()) {
        return true;
      }
      return false;
    }
    void EraseCaptureEvent(hipEvent_t e) {
      amd::ScopedLock lock(lock_);
      auto it = captureEvents_.find(e);
      if (it != captureEvents_.end()) {
        captureEvents_.erase(it);
      }
    }
    void SetParallelCaptureStream(hipStream_t s) {
      auto it = std::find(parallelCaptureStreams_.begin(), parallelCaptureStreams_.end(), s);
      if (it == parallelCaptureStreams_.end()) {
        parallelCaptureStreams_.push_back(s);
      }
    }
    void EraseParallelCaptureStream(hipStream_t s) {
      auto it = std::find(parallelCaptureStreams_.begin(), parallelCaptureStreams_.end(), s);
      if (it != parallelCaptureStreams_.end()) {
        parallelCaptureStreams_.erase(it);
      }
    }

    /// The stream should be destroyed via release() rather than delete
    private:
      ~Stream() {};
  };

  /// HIP Device class
  class Device : public amd::ReferenceCountedObject {
    // Device lock
    amd::Monitor lock_{true};
    // Guards device stream set
    std::shared_mutex streamSetLock;
    std::unordered_set<hip::Stream*> streamSet;
    /// ROCclr context
    amd::Context* context_;
    /// Device's ID
    /// Store it here so we don't have to loop through the device list every time
    int deviceId_;
    /// ROCclr host queue for default streams
    Stream* null_stream_  = nullptr;
    /// Store device flags
    unsigned int flags_;
    /// Maintain list of user enabled peers
    std::list<int> userEnabledPeers;

    /// True if this device is active
    bool isActive_;


    MemoryPool* default_mem_pool_;  //!< Default memory pool for this device
    MemoryPool* current_mem_pool_;
    MemoryPool* graph_mem_pool_;    //!< Memory pool, associated with graphs for this device

    std::set<MemoryPool*> mem_pools_;

  public:
    Device(amd::Context* ctx, int devId): context_(ctx),
        deviceId_(devId),
         flags_(hipDeviceScheduleSpin),
        isActive_(false),
        default_mem_pool_(nullptr),
        current_mem_pool_(nullptr),
        graph_mem_pool_(nullptr)
        { assert(ctx != nullptr); }
    ~Device();

    bool Create();
    amd::Context* asContext() const { return context_; }
    int deviceId() const { return deviceId_; }
    void retain() const { context_->retain(); }
    void release() const { context_->release(); }
    const std::vector<amd::Device*>& devices() const { return context_->devices(); }
    hipError_t EnablePeerAccess(int peerDeviceId){
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        return hipErrorPeerAccessAlreadyEnabled;
      }
      userEnabledPeers.push_back(peerDeviceId);
      return hipSuccess;
    }
    hipError_t DisablePeerAccess(int peerDeviceId) {
      amd::ScopedLock lock(lock_);
      bool found = (std::find(userEnabledPeers.begin(), userEnabledPeers.end(), peerDeviceId) != userEnabledPeers.end());
      if (found) {
        userEnabledPeers.remove(peerDeviceId);
        return hipSuccess;
      } else {
        return hipErrorPeerAccessNotEnabled;
      }
    }
    unsigned int getFlags() const { return flags_; }
    void setFlags(unsigned int flags) { flags_ = flags; }
    void Reset();

    hip::Stream* NullStream(bool wait = true);
    Stream* GetNullStream() const {return null_stream_;};

    void SetActiveStatus() {
      isActive_ = true;
    }

    bool GetActiveStatus() {
      amd::ScopedLock lock(lock_);
      /// Either stream is active or device is active
      if (isActive_) return true;
      if (existsActiveStreamForDevice()) {
        isActive_ = true;
        return true;
      }
      return false;
    }

    /// Set the current memory pool on the device
    void SetCurrentMemoryPool(MemoryPool* pool = nullptr) {
      current_mem_pool_ = (pool == nullptr) ? default_mem_pool_ : pool;
    }

    /// Get the current memory pool on the device
    MemoryPool* GetCurrentMemoryPool() const { return current_mem_pool_; }

    /// Get the default memory pool on the device
    MemoryPool* GetDefaultMemoryPool() const { return default_mem_pool_; }

    /// Get the graph memory pool on the device
    MemoryPool* GetGraphMemoryPool() const { return graph_mem_pool_; }

    /// Add memory pool to the device
    void AddMemoryPool(MemoryPool* pool);

    /// Remove memory pool from the device
    void RemoveMemoryPool(MemoryPool* pool);

    /// Free memory from the device
    bool FreeMemory(amd::Memory* memory, Stream* stream, Event* event = nullptr);

    /// Release freed memory from all pools on the current device
    void ReleaseFreedMemory();

    /// Removes a destroyed stream from the safe list of memory pools
    void RemoveStreamFromPools(Stream* stream);

    /// Add safe streams into the memppools for reuse
    void AddSafeStream(Stream* event_stream, Stream* wait_stream);

    /// Returns true if memory pool is valid on this device
    bool IsMemoryPoolValid(MemoryPool* pool);
    void AddStream(Stream* stream);

    void RemoveStream(Stream* stream);

    bool StreamExists(Stream* stream);

    void destroyAllStreams();

    void SyncAllStreams( bool cpu_wait = true, bool wait_blocking_streams_only = false);

    bool StreamCaptureBlocking();

    bool existsActiveStreamForDevice();
  /// Wait all active streams on the blocking queue. The method enqueues a wait command and
  /// doesn't stall the current thread
    void WaitActiveStreams(hip::Stream* blocking_stream, bool wait_null_stream = false);
  };

  /// Thread Local Storage Variables Aggregator Class
  class TlsAggregator {
  public:
    Device* device_;
    std::stack<Device*> ctxt_stack_;
    hipError_t last_error_, last_command_error_;
    std::vector<hip::Stream*> capture_streams_;
    hipStreamCaptureMode stream_capture_mode_;
    std::stack<ihipExec_t> exec_stack_;
    stream_per_thread stream_per_thread_obj_;
    bool isSetDeviceCalled;

    TlsAggregator(): device_(nullptr),
      last_error_(hipSuccess),
      last_command_error_(hipSuccess),
      stream_capture_mode_(hipStreamCaptureModeGlobal),
      isSetDeviceCalled(false) {
    }
    ~TlsAggregator() {
    }
  };
  extern thread_local TlsAggregator tls;

  /// Device representing the host - for pinned memory
  extern amd::Context* host_context;

  extern void init(bool* status);

  extern Device* getCurrentDevice();

  extern void setCurrentDevice(unsigned int index);

  /// Get ROCclr queue associated with hipStream
  /// Note: This follows the CUDA spec to sync with default streams
  ///       and Blocking streams
  extern hip::Stream* getStream(hipStream_t stream, bool wait = true);
  /// Get default stream associated with the ROCclr context
  extern hip::Stream* getNullStream(amd::Context&, bool wait = true);
  /// Get default stream of the thread
  extern hip::Stream* getNullStream(bool wait = true);
  /// Get device ID associated with the ROCclr context
  int getDeviceID(amd::Context& ctx);
  /// Check if stream is valid
  extern bool isValid(hipStream_t& stream);
  extern bool isValid(hipEvent_t event);
  extern amd::Monitor hipArraySetLock;
  extern std::unordered_set<hipArray*> hipArraySet;

  extern void WaitThenDecrementSignal(hipStream_t stream, hipError_t status, void* user_data);

  extern std::vector<hip::Device*> g_devices;
  extern hipError_t ihipDeviceGetCount(int* count);
  extern int ihipGetDevice();

  extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
  extern hipError_t ihipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
  extern amd::Memory* getMemoryObject(const void* ptr, size_t& offset, size_t size = 0);
  extern amd::Memory* getMemoryObjectWithOffset(const void* ptr, const size_t size = 0);
  extern void getStreamPerThread(hipStream_t& stream);
  extern hipStream_t getPerThreadDefaultStream();
  extern hipError_t ihipUnbindTexture(textureReference* texRef);
  extern hipError_t ihipHostRegister(void* hostPtr, size_t sizeBytes, unsigned int flags);
  extern hipError_t ihipHostUnregister(void* hostPtr);
  extern hipError_t ihipGetDeviceProperties(hipDeviceProp_t* props, hipDevice_t device);

  extern hipError_t ihipDeviceGet(hipDevice_t* device, int deviceId);
  extern hipError_t ihipStreamOperation(hipStream_t stream, cl_command_type cmdType, void* ptr,
                                        uint64_t value, uint64_t mask, unsigned int flags,
                                        size_t sizeBytes);
  hipError_t ihipMemcpy(void* dst, const void* src, size_t sizeBytes, hipMemcpyKind kind,
                        hip::Stream& stream, bool isHostAsync = false, bool isGPUAsync = true);
  hipError_t ihipMemcpy3D(const hipMemcpy3DParms* p, hipStream_t stream = nullptr,
                          bool isAsync = false);
  constexpr bool kOptionChangeable = true;
  constexpr bool kNewDevProg = false;

  constexpr bool kMarkerDisableFlush = true;  //!< Avoids command batch flush in ROCclr

  extern std::vector<hip::Stream*> g_captureStreams;
  extern amd::Monitor g_captureStreamsLock;
  extern amd::Monitor g_streamSetLock;
  extern std::unordered_set<hip::Stream*> g_allCapturingStreams;
} // namespace hip
#endif  // HIP_SRC_HIP_INTERNAL_H
