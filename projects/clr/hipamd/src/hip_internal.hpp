/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
#include <algorithm>
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

template <typename T> T ReturnPtrValue(T* ptr) { return (ptr != nullptr) ? *ptr : nullptr; }

typedef struct hipArray {
    void* data;                        //!< Raw device pointer to array storage
    struct hipChannelFormatDesc desc;  //!< Channel format descriptor (bit depths and format)
    unsigned int type;                 //!< Array type (e.g. 1D, 2D, 3D)
    unsigned int width;                //!< Width of the array in elements
    unsigned int height;               //!< Height of the array in elements (1 for 1D)
    unsigned int depth;                //!< Depth of the array in elements (1 for 1D/2D)
    enum hipArray_Format Format;       //!< Element format (e.g. float, half, int)
    unsigned int NumChannels;          //!< Number of channels per element (1–4)
    bool isDrv;                        //!< True if allocated via the driver API
    unsigned int textureType;          //!< Texture type (1D, 2D, 3D, cubemap, etc.)
    unsigned int flags;                //!< Creation flags (e.g. hipArraySurfaceLoadStore)
} hipArray;

namespace hip{
extern std::once_flag g_ihipInitialized;
enum MemcpyType {
  hipHostToHost,      //!< Memcpy from host to host
  hipWriteBuffer,     //!< Memcpy from host to device
  hipReadBuffer,      //!< Memcpy from device to host
  hipCopyBuffer,      //!< Memcpy within same device or pinned host buffer <-> device
  hipCopyBufferSDMA,  //!< Memcpy within same device, user forced to SDMA
  hipCopyBufferP2P,   //!< Memcpy from device A to device B
};

struct Graph;
struct GraphNode;
struct GraphExec;
struct UserObject;
class Stream;

#define IHIP_IPC_EVENT_HANDLE_SIZE 32
#define IHIP_IPC_EVENT_RESERVED_SIZE LP64_SWITCH(28,24)
typedef struct ihipIpcEventHandle_st {
    //hsa_amd_ipc_signal_t ipc_handle;  //!< ipc signal handle on ROCr
    //char ipc_handle[IHIP_IPC_EVENT_HANDLE_SIZE];
    //char reserved[IHIP_IPC_EVENT_RESERVED_SIZE];
    char shmem_name[IHIP_IPC_EVENT_HANDLE_SIZE];
} ihipIpcEventHandle_t;

const char* ihipGetErrorName(hipError_t hip_error);

} // namespace hip

// Helper: set up TLS device pointer on first use.
#define HIP_INIT_TLS_DEVICE()                                                                      \
  if (hip::tls.device_ == nullptr && !hip::g_devices.empty()) {                                    \
    hip::tls.device_ = hip::g_devices[0];                                                          \
    amd::Os::setPreferredNumaNode(hip::g_devices[0]->devices()[0]->getPreferredNumaNode());        \
  }

#define HIP_INIT(noReturn)                                                                         \
  {                                                                                                \
    bool status = true;                                                                            \
    std::call_once(hip::g_ihipInitialized, hip::init, &status);                                    \
    if (!status && !noReturn) {                                                                    \
      HIP_RETURN(hipErrorInvalidDevice);                                                           \
    }                                                                                              \
    HIP_INIT_TLS_DEVICE()                                                                          \
  }

#define HIP_INIT_VOID()                                                                            \
  {                                                                                                \
    bool status = true;                                                                            \
    std::call_once(hip::g_ihipInitialized, hip::init, &status);                                    \
    HIP_INIT_TLS_DEVICE()                                                                          \
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
  if (amd::Device::IsGPUInError()) {                                                               \
    HIP_RETURN(ConvertCLErrorIntoHIPError(amd::Device::GetGPUError()));                            \
  }                                                                                                \
  HIP_INIT_API_INTERNAL(0, cid, __VA_ARGS__)                                                       \
  if (hip::g_devices.empty()) {                                                                    \
    HIP_RETURN(hipErrorNoDevice);                                                                  \
  }                                                                                                \

#define HIP_INIT_API_NO_RETURN(cid, ...)                                                           \
  HIP_INIT_API_INTERNAL(1, cid, __VA_ARGS__)

// Helper: update thread-local error state from a return code.
// Overrides with GPU error if the device is in an error state.
#define HIP_UPDATE_ERROR_STATE(ret)                                                                \
  hip::tls.last_command_error_ = ret;                                                              \
  if (amd::Device::IsGPUInError()) {                                                               \
    hipError_t hip_error = ConvertCLErrorIntoHIPError(amd::Device::GetGPUError());                 \
    hip::tls.last_error_ = hip_error;                                                              \
    hip::tls.last_command_error_ = hip_error;                                                      \
  } else if (hip::tls.last_command_error_ != hipSuccess &&                                         \
             hip::tls.last_command_error_ != hipErrorNotReady) {                                   \
    hip::tls.last_error_ = hip::tls.last_command_error_;                                           \
  }

#define HIP_RETURN_DURATION(ret, ...)                                                              \
  HIP_UPDATE_ERROR_STATE(ret)                                                                      \
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%s: Returned %s : %s", __func__,    \
                   hip::ihipGetErrorName(hip::tls.last_command_error_),                            \
                   ToString(__VA_ARGS__).c_str());                                                 \
  return hip::tls.last_command_error_;

#define HIP_RETURN(ret, ...)                                                                       \
  HIP_UPDATE_ERROR_STATE(ret)                                                                      \
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
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeThreadLocal ||                          \
      hip::tls.stream_capture_mode_ == hipStreamCaptureModeGlobal) {                               \
    if (!hip::tls.capture_streams_.empty()) {                                                      \
      for (auto stream : hip::tls.capture_streams_) {                                              \
        stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                               \
      }                                                                                            \
      HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                \
    }                                                                                              \
  }                                                                                                \
  if (hip::tls.stream_capture_mode_ == hipStreamCaptureModeGlobal &&                               \
      !g_captureStreams.empty()) {                                                                 \
    for (auto stream : g_captureStreams) {                                                         \
      stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                                 \
    }                                                                                              \
    HIP_RETURN(hipErrorStreamCaptureUnsupported);                                                  \
  }

// Helper: invalidate all capturing streams and return an error code.
#define INVALIDATE_ALL_CAPTURING_AND_RETURN(err)                                                   \
  if (!g_allCapturingStreams.empty()) {                                                            \
    for (auto stream : g_allCapturingStreams) {                                                    \
      stream->SetCaptureStatus(hipStreamCaptureStatusInvalidated);                                 \
    }                                                                                              \
    return err;                                                                                    \
  }

// Device sync is not supported during capture.
#define CHECK_SUPPORTED_DURING_CAPTURE()                                                           \
  INVALIDATE_ALL_CAPTURING_AND_RETURN(hipErrorStreamCaptureUnsupported)

// Sync APIs (hipMemset, hipMemcpy, etc.) cannot be called when stream capture is active
// for any capture mode (Global, ThreadLocal, or Relaxed).
#define CHECK_STREAM_CAPTURING()                                                                   \
  INVALIDATE_ALL_CAPTURING_AND_RETURN(hipErrorStreamCaptureImplicit)

#define STREAM_CAPTURE(name, stream, ...)                                                          \
  hip::getStreamPerThread(stream);                                                                 \
  if (stream != nullptr && stream != hipStreamLegacy) {                                            \
    auto captureStatus = reinterpret_cast<hip::Stream*>(stream)->GetCaptureStatus();               \
    if (captureStatus == hipStreamCaptureStatusActive) {                                           \
      return hip::capture##name(stream, ##__VA_ARGS__);                                            \
    } else if (captureStatus == hipStreamCaptureStatusInvalidated) {                               \
      return hipErrorStreamCaptureInvalidated;                                                     \
    }                                                                                              \
  }

#define PER_THREAD_DEFAULT_STREAM(stream)                                                         \
  if (stream == nullptr || stream == hipStreamLegacy) {                                           \
    stream = getPerThreadDefaultStream();                                                         \
  }

/// Stores the kernel launch configuration set by hipConfigureCall /
/// __hipPushCallConfiguration and consumed by hipLaunchByPtr.
/// Instances are managed on a per-thread stack (TlsAggregator::exec_stack_).
struct ihipExec_t {
  dim3 gridDim_;                   //!< Grid dimensions (number of blocks)
  dim3 blockDim_;                  //!< Block dimensions (threads per block)
  size_t sharedMem_;               //!< Dynamic shared memory size in bytes
  hipStream_t hStream_;            //!< Stream on which the kernel will be launched
  std::vector<char> arguments_;    //!< Raw kernel arguments accumulated via hipSetupArgument
};

namespace hip {
  /// Manages a per-thread default stream for each device.
  /// Each thread lazily creates one stream per device on first use via get().
  /// Owned by TlsAggregator::stream_per_thread_obj_ (thread-local).
  class StreamPerThread {
  public:
    StreamPerThread();
    ~StreamPerThread();

    StreamPerThread(const StreamPerThread&) = delete;
    StreamPerThread& operator=(const StreamPerThread&) = delete;

    /// Returns (or lazily creates) the per-thread default stream for the current device.
    hipStream_t Get();

    /// Clears the per-thread stream for the current device (used by hipDeviceReset).
    void Clear();

  private:
    std::vector<hipStream_t> streams_;  //!< One stream handle per device (indexed by device ID)
  };

  class Device;
  class MemoryPool;
  class Event;
  class Stream : public amd::HostQueue {
  public:
    enum Priority : int { High = -1, Normal = 0, Low = 1 };

    Stream(Device* dev, Priority p = Priority::Normal, unsigned int f = 0, bool null_stream = false,
           const std::vector<uint32_t>& cuMask = {},
           hipStreamCaptureStatus captureStatus = hipStreamCaptureStatusNone);

    // --- Core stream operations ---

    /// Creates the hip stream object, including AMD host queue
    bool Create();
    /// Get device ID associated with the current stream
    int DeviceId() const;
    /// Get HIP device associated with the stream
    Device* GetDevice() const { return device_; }
    /// Get device ID associated with a stream
    static int DeviceId(const hipStream_t hStream);
    /// Returns true if this is the null (default legacy) stream
    bool Null() const { return null_; }
    /// Returns the creation flags for the current stream
    unsigned int Flags() const { return flags_; }
    /// Returns the priority for the current stream
    Priority GetPriority() const { return priority_; }
    /// Returns the CU mask for the current stream
    const std::vector<uint32_t>& GetCUMask() const { return cuMask_; }
    /// Fetch the stream Id
    uint64_t GetStreamId() const { return stream_id_; }

    static void Destroy(hip::Stream* stream, bool forceDestroy = false);
    virtual bool terminate();

    // --- Stream capture ---

    /// Check whether any blocking stream running
    static bool StreamCaptureBlocking();
    /// Check Stream Capture status to make sure it is done
    static bool StreamCaptureOngoing(hipStream_t hStream);
    /// Generate ID for stream capture unique over the lifetime of the process
    static uint64_t GenerateCaptureID() {
      static std::atomic<uint64_t> uid(0);
      return ++uid;
    }

    /// Returns capture status of the current stream
    hipStreamCaptureStatus GetCaptureStatus() const { return captureStatus_; }
    /// Returns capture mode of the current stream
    hipStreamCaptureMode GetCaptureMode() const { return captureMode_; }
    /// Returns if stream is origin stream
    bool IsOriginStream() const { return originStream_; }
    /// Mark this stream as the origin of a capture
    void SetOriginStream() { originStream_ = true; }
    /// Returns captured graph
    hip::Graph* GetCaptureGraph() const { return pCaptureGraph_; }
    /// Returns last captured graph node
    const std::vector<hip::GraphNode*>& GetLastCapturedNodes() const { return lastCapturedNodes_; }
    /// Set last captured graph node
    void SetLastCapturedNode(hip::GraphNode* graphNode) { lastCapturedNodes_ = {graphNode}; }
    /// Returns dependencies removed during capture
    const std::vector<hip::GraphNode*>& GetRemovedDependencies() const {
      return removedDependencies_;
    }
    /// Append captured node via the wait event cross stream
    void AddCrossCapturedNode(const std::vector<hip::GraphNode*>& graphNodes,
                              bool replace = false);
    /// Set graph that is being captured
    void SetCaptureGraph(hip::Graph* pGraph) {
      pCaptureGraph_ = pGraph;
      captureStatus_ = hipStreamCaptureStatusActive;
    }
    /// Release graph when capture is invalidated
    void ReleaseCaptureGraph();
    /// Generate and assign a new capture ID (used at BeginCapture)
    void SetCaptureID() { captureID_ = GenerateCaptureID(); }
    /// Inherit capture ID from the parent stream
    void SetCaptureID(uint64_t captureId) { captureID_ = captureId; }
    /// Reset capture parameters
    hipError_t EndCapture();
    /// Set capture status
    void SetCaptureStatus(hipStreamCaptureStatus captureStatus) { captureStatus_ = captureStatus; }
    /// Set capture mode
    void SetCaptureMode(hipStreamCaptureMode captureMode) { captureMode_ = captureMode; }
    /// Set parent stream
    void SetParentStream(hipStream_t parentStream) { parentStream_ = parentStream; }
    /// Get parent stream
    hipStream_t GetParentStream() const { return parentStream_; }
    /// Get capture ID
    uint64_t GetCaptureID() const { return captureID_; }
    /// Associate an event with the current capture
    void SetCaptureEvent(hipEvent_t e) {
      std::scoped_lock lock(lock_);
      captureEvents_.emplace(e);
    }
    /// Returns true if the event is part of this capture
    bool IsEventCaptured(hipEvent_t e) const {
      std::scoped_lock lock(lock_);
      return captureEvents_.count(e) != 0;
    }
    /// Remove an event from the current capture
    void EraseCaptureEvent(hipEvent_t e) {
      std::scoped_lock lock(lock_);
      captureEvents_.erase(e);
    }
    /// Register a parallel (forked) capture stream
    void SetParallelCaptureStream(hipStream_t s) { parallelCaptureStreams_.insert(s); }
    /// Remove a parallel capture stream
    void EraseParallelCaptureStream(hipStream_t s) { parallelCaptureStreams_.erase(s); }

  private:
    ~Stream() = default;

    mutable std::recursive_mutex lock_;      //!< Guards captureEvents_ bookkeeping
    Device* device_;                         //!< Device that owns this stream
    Priority priority_;                      //!< Scheduling priority (High / Normal / Low)
    unsigned int flags_;                     //!< Creation flags (e.g. hipStreamNonBlocking)
    bool null_;                              //!< True for the null (default legacy) stream
    const std::vector<uint32_t> cuMask_;     //!< CU mask restricting which CUs may be used
    uint64_t stream_id_;                     //!< Process-unique monotonic stream identifier

    // ----- Stream capture state -----
    hipStreamCaptureStatus captureStatus_{hipStreamCaptureStatusNone}; //!< Current capture status
    hip::Graph* pCaptureGraph_ = nullptr;                 //!< Graph being constructed by capture
    hipStreamCaptureMode captureMode_{hipStreamCaptureModeGlobal}; //!< API restriction mode
    bool originStream_ = false;                           //!< True if this stream started capture
    hipStream_t parentStream_ = nullptr;                  //!< Parent stream (null for origin)
    std::vector<hip::GraphNode*> lastCapturedNodes_;      //!< Last graph node(s) captured
    std::vector<hip::GraphNode*> removedDependencies_;    //!< Deps removed via UpdateCaptureDeps
    std::unordered_set<hipStream_t> parallelCaptureStreams_; //!< Forked parallel capture branches
    std::unordered_set<hipEvent_t> captureEvents_;        //!< Events tied to this capture
    uint64_t captureID_ = 0;                              //!< Unique ID for this capture sequence

    static CommandQueue::Priority convertToQueuePriority(Priority p) {
      return p == Priority::High  ? amd::CommandQueue::Priority::High
             : p == Priority::Low ? amd::CommandQueue::Priority::Low
                                  : amd::CommandQueue::Priority::Normal;
    }
    /// Generates unique stream Id for the lifetime of the process
    static uint64_t GenerateStreamId() {
      static std::atomic<uint64_t> uniqueId{0};
      return ++uniqueId;
    }
  };

  /// Thread-safe registry for tracking objects of type T (e.g. graphics resources).
  template <typename T>
  class ObjectRegistry {
  public:
    /// Adds the object to the set. Returns true if newly inserted.
    bool add(T object) {
      if (object == nullptr) return false;
      std::scoped_lock lock(mtx_);
      return objects_.insert(object).second;
    }
    /// Removes the object from the set. Returns true if it was present.
    bool remove(T object) {
      std::scoped_lock lock(mtx_);
      return objects_.erase(object) > 0;
    }
    /// Returns true if the set contains the object.
    bool isValid(T object) const {
      if (object == nullptr) return false;
      std::scoped_lock lock(mtx_);
      return objects_.count(object) != 0;
    }
    /// Removes all tracked objects.
    void clear() {
      std::scoped_lock lock(mtx_);
      objects_.clear();
    }

  private:
    mutable std::mutex mtx_;              //!< Guards all access to objects_
    std::unordered_set<T> objects_;       //!< Tracked object set
  };

  /// HIP Device class
  class Device : public amd::ReferenceCountedObject {
  public:
    Device(amd::Context* ctx, int devId)
        : context_(ctx),
          deviceId_(devId),
          flags_(hipDeviceScheduleSpin) {
      assert(ctx != nullptr);
    }
    ~Device();

    bool Create();
    void Reset();

    // --- Accessors ---

    amd::Context* asContext() const { return context_; }
    int deviceId() const { return deviceId_; }
    void retain() const { context_->retain(); }
    void release() const { context_->release(); }
    const std::vector<amd::Device*>& devices() const { return context_->devices(); }
    unsigned int getFlags() const { return flags_; }
    void setFlags(unsigned int flags) { flags_ = flags; }

    // --- Peer access ---

    /// Enable peer access from this device to peerDeviceId
    hipError_t EnablePeerAccess(int peerDeviceId);
    /// Disable peer access from this device to peerDeviceId
    hipError_t DisablePeerAccess(int peerDeviceId);

    // --- Stream management ---

    hip::Stream* NullStream(bool wait = true);
    Stream* GetNullStream() const { return null_stream_; }
    /// Register a stream with this device
    void AddStream(Stream* stream);
    /// Unregister a stream from this device
    void RemoveStream(Stream* stream);
    /// Returns true if stream belongs to this device
    bool StreamExists(const Stream* stream);
    /// Synchronize all streams (optionally only blocking ones)
    void SyncAllStreams(bool cpu_wait = true, bool wait_blocking_streams_only = false);
    /// Returns true if any stream on this device is in capture mode
    bool StreamCaptureBlocking();
    /// Wait all active streams on the blocking queue. The method enqueues a wait command and
    /// doesn't stall the current thread.
    void WaitActiveStreams(hip::Stream* blocking_stream, bool wait_null_stream = false);

    // --- Device activity ---

    void SetActiveStatus() { isActive_.store(true, std::memory_order_release); }
    bool GetActiveStatus();

    // --- Memory pools ---

    /// Set the current memory pool on the device
    void SetCurrentMemoryPool(MemoryPool* pool = nullptr) {
      current_mem_pool_ = (pool == nullptr) ? default_mem_pool_ : pool;
    }
    MemoryPool* GetCurrentMemoryPool() const { return current_mem_pool_; }
    MemoryPool* GetDefaultMemoryPool() const { return default_mem_pool_; }
    MemoryPool* GetGraphMemoryPool() const { return graph_mem_pool_; }
    void SetCurrentManagedMemoryPool(MemoryPool* pool) { current_managed_mem_pool_ = pool; }
    MemoryPool* GetCurrentManagedMemoryPool() const { return current_managed_mem_pool_; }
    MemoryPool* GetDefaultManagedMemoryPool() const { return default_managed_mem_pool_; }
    void AddMemoryPool(MemoryPool* pool);
    void RemoveMemoryPool(MemoryPool* pool);
    bool FreeMemory(amd::Memory* memory, Stream* stream, Event* event = nullptr);
    void ReleaseFreedMemory();
    void RemoveStreamFromPools(Stream* stream);
    void AddSafeStream(Stream* event_stream, Stream* wait_stream);
    bool IsMemoryPoolValid(MemoryPool* pool);

    // --- Graphics resource tracking ---

    ObjectRegistry<hipGraphicsResource_t>& registeredGraphics() {
      return registeredGraphicsResources_;
    }
    ObjectRegistry<hipGraphicsResource_t>& mappedGraphics() {
      return mappedGraphicsResources_;
    }

  private:
    /// Destroy all streams on this device (called by Reset)
    void destroyAllStreams();

    std::recursive_mutex lock_;                //!< Device-wide lock
    std::shared_mutex streamSetLock_;          //!< Guards streamSet_ (reader-writer)
    std::unordered_set<hip::Stream*> streamSet_; //!< Active streams on this device
    amd::Context* context_;                    //!< ROCclr context
    int deviceId_;                             //!< Cached device ID (avoids linear scan)
    Stream* null_stream_ = nullptr;            //!< Default (null/legacy) stream
    unsigned int flags_;                       //!< Device flags (e.g. hipDeviceScheduleSpin)
    std::vector<int> userEnabledPeers_;        //!< Peer device IDs enabled via hipEnablePeerAccess
    std::atomic<bool> isActive_{false};        //!< True once any work has been submitted

    // ----- Memory pool state -----
    MemoryPool* default_mem_pool_ = nullptr;          //!< Default memory pool
    MemoryPool* current_mem_pool_ = nullptr;          //!< Currently active memory pool
    MemoryPool* graph_mem_pool_ = nullptr;            //!< Memory pool for graph allocations
    MemoryPool* current_managed_mem_pool_ = nullptr;  //!< Active managed-memory pool
    MemoryPool* default_managed_mem_pool_ = nullptr;  //!< Default managed-memory pool
    std::set<MemoryPool*> mem_pools_;                 //!< All pools associated with this device

    // ----- Graphics resource tracking -----
    ObjectRegistry<hipGraphicsResource_t> registeredGraphicsResources_;
    ObjectRegistry<hipGraphicsResource_t> mappedGraphicsResources_;
  };

  /// Per-thread state aggregator for HIP runtime (one instance per thread via thread_local).
  class TlsAggregator {
  public:
    Device* device_ = nullptr;                 //!< Current device for this thread
    std::stack<Device*> ctxt_stack_;            //!< CUDA-style context stack
    hipError_t last_error_ = hipSuccess;       //!< Sticky error (persists until queried)
    hipError_t last_command_error_ = hipSuccess;//!< Last per-command error
    std::vector<hip::Stream*> capture_streams_;//!< Streams currently capturing on this thread
    hipStreamCaptureMode stream_capture_mode_ = hipStreamCaptureModeGlobal; //!< Active capture mode
    std::stack<ihipExec_t> exec_stack_;        //!< Pending kernel launch configurations
    StreamPerThread stream_per_thread_obj_;    //!< Per-thread default streams (one per device)
    bool isSetDeviceCalled = false;            //!< True after explicit hipSetDevice call

    TlsAggregator() = default;
    ~TlsAggregator() = default;
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
  /// Check if stream is valid
  extern bool isValid(hipStream_t& stream);
  extern bool isValid(hipEvent_t event);
  extern amd::Monitor hipArraySetLock;
  extern std::unordered_set<hipArray*> hipArraySet;

  extern std::vector<hip::Device*> g_devices;
  extern hipError_t ihipDeviceGetCount(int* count);
  extern int ihipGetDevice();

  extern hipError_t ihipMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
  extern hipError_t ihipHostMalloc(void** ptr, size_t sizeBytes, unsigned int flags);
  extern hipError_t ihipMemGetInfo(size_t* free, size_t* total);
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

  constexpr bool kMarkerDisableFlush = true;  //!< Avoids command batch flush in ROCclr

  extern std::vector<hip::Stream*> g_captureStreams;
  extern amd::Monitor g_captureStreamsLock;
  extern amd::Monitor g_streamSetLock;
  extern std::unordered_set<hip::Stream*> g_allCapturingStreams;
} // namespace hip
#endif  // HIP_SRC_HIP_INTERNAL_H
