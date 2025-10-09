/* Copyright (c) 2010 - 2025 Advanced Micro Devices, Inc.

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

#ifndef COMMAND_HPP_
#define COMMAND_HPP_

#include "top.hpp"
#include "thread/monitor.hpp"
#include "thread/thread.hpp"
#include "platform/agent.hpp"
#include "platform/object.hpp"
#include "platform/context.hpp"
#include "platform/ndrange.hpp"
#include "platform/kernel.hpp"
#include "device/device.hpp"
#include "utils/concurrent.hpp"
#include "platform/memory.hpp"
#include "platform/perfctr.hpp"
#include "platform/threadtrace.hpp"
#include "platform/activity.hpp"
#include "platform/command_utils.hpp"

#include "CL/cl_ext.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <vector>

namespace amd {

/*! \addtogroup Runtime
 *  @{
 *
 *  \addtogroup Commands Event, Commands and Command-Queue
 *  @{
 */

class Command;
class HostQueue;
union ComputeCommand;

/*! \brief Encapsulates the status of a command.
 *
 *  \details An event object encapsulates the status of a Command
 *  it is associated with and can be used to synchronize operations
 *  in a Context.
 */
class Event : public RuntimeObject {
  typedef void(CL_CALLBACK* CallBackFunction)(cl_event event, int32_t command_exec_status,
                                              void* user_data);

  struct CallBackEntry : public HeapObject {
    struct CallBackEntry* next_;  //!< the next entry in the callback list.

    std::atomic<CallBackFunction> callback_;  //!< callback function pointer.
    void* data_;                              //!< user data passed to the callback function.
    int32_t status_;                          //!< execution status triggering the callback.
    bool blocking_;                           //!< TRUE if callback is blocking
    CallBackEntry(int32_t status, CallBackFunction callback, void* data, bool blocking)
        : callback_(callback), data_(data), status_(status), blocking_(blocking) {}
  };

 public:
  typedef std::vector<Event*> EventWaitList;

 private:
  Monitor lock_;
  Monitor notify_lock_;  //!< Lock used for notification with direct dispatch only

  std::atomic<CallBackEntry*> callbacks_;  //!< linked list of callback entries.
  std::atomic<int32_t> status_;            //!< current execution status.
  std::atomic_flag notified_;              //!< Command queue was notified

  void* hw_event_;        //!< HW event ID associated with SW event
  Event* notify_event_;   //!< Notify event, which should contain HW signal
  const Device* device_;  //!< Device, this event associated with

  std::atomic<int32_t> event_entry_scope_;  //!< Command entry scope
                                            //!< 2 - system scope, 1 - device scope,
                                            //!< 0 - ignore, -1 - invalid

 protected:
  static const EventWaitList nullWaitList;

  struct ProfilingInfo {
    ProfilingInfo(bool enabled = false) : enabled_(enabled), marker_ts_(false) {
      if (enabled) {
        clear();
        correlation_id_ = amd::activity_prof::correlation_id;
      }
    }

    uint64_t queued_;
    uint64_t submitted_;
    uint64_t start_;
    uint64_t end_;

    uint64_t correlation_id_;
    bool enabled_;             //!< Profiling enabled for the wave limiter
    bool marker_ts_;           //!< TS marker
    bool batch_flush_ = true;  //!< Command can flush the batch in direct dispatch mode

    void clear() {
      queued_ = 0ULL;
      submitted_ = 0ULL;
      start_ = 0ULL;
      end_ = 0ULL;
      correlation_id_ = 0ULL;
    }
  } profilingInfo_;

  //! Construct a new event.
  Event();

  //! Construct a new event associated to the given command \a queue.
  Event(HostQueue& queue, bool profilingEnabled = false);

  //! Destroy the event.
  virtual ~Event();

  //! Release the resources associated with this event.
  virtual void releaseResources() {}

  //! Record the profiling info for the given change of \a status.
  //  If the given \a timeStamp is 0 and profiling is enabled,
  //  use the current host clock time instead.
  uint64_t recordProfilingInfo(int32_t status, uint64_t timeStamp = 0);

  //! Process the callbacks for the given \a status change.
  void processCallbacks(int32_t status) const;

  //! Enable profiling for this command
  void EnableProfiling() {
    profilingInfo_.enabled_ = true;
    profilingInfo_.clear();
    profilingInfo_.correlation_id_ = amd::activity_prof::correlation_id;
  }

 public:
  //! Use profiling info to force a tracking signal on command
  void SetProfiling() {
    EnableProfiling();
    profilingInfo_.marker_ts_ = true;
  }

  //! Return the context for this event.
  virtual const Context& context() const = 0;

  //! Return the command this event is associated with.
  inline Command& command();
  inline const Command& command() const;

  //! Return the profiling info.
  const ProfilingInfo& profilingInfo() const { return profilingInfo_; }

  //! Return this command's execution status.
  int32_t status() const { return status_.load(std::memory_order_relaxed); }

  //! Insert the given \a callback into the callback stack.
  bool setCallback(int32_t status, CallBackFunction callback, void* data, bool blocking = true);

  /*! \brief Set the event status.
   *
   *  \details If the status becomes CL_COMPLETE, notify all threads
   *  awaiting this command's completion.  If the given \a timeStamp is 0
   *  and profiling is enabled, use the current host clock time instead.
   *
   *  \see amd::Event::awaitCompletion
   */
  bool setStatus(int32_t status, uint64_t timeStamp = 0);

  //! Reset the status of the command for reuse
  bool resetStatus(int32_t status);

  //! Signal all threads waiting on this event.
  void signal() {
    ScopedLock lock(lock_);  // Unnecessary
    lock_.notifyAll();
  }

  /*! \brief Suspend the current thread until the status of the Command
   *  associated with this event changes to CL_COMPLETE. Return true if the
   *  command successfully completed.
   */
  virtual bool awaitCompletion();

  /*! \brief Notifies current command queue about execution status
   */
  bool notifyCmdQueue(bool cpu_wait = false);

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeEvent; }

  //! Returns the callback for this event
  const CallBackEntry* Callback() const { return callbacks_; }

  //! Saves HW event, associated with the current command
  void SetHwEvent(void* hw_event) { hw_event_ = hw_event; }

  //! Returns HW event, associated with the current command
  void* HwEvent() const { return hw_event_; }

  //! Returns notify even associated with the current command
  Event* NotifyEvent() const { return notify_event_; }

  //! Get entry scope of the event
  int32_t getCommandEntryScope() const {
    return event_entry_scope_.load(std::memory_order_relaxed);
  }

  //! Set entry scope for the event
  void setCommandEntryScope(int32_t scope) {
    event_entry_scope_.store(scope, std::memory_order_relaxed);
  }
};

union CopyMetadata {
  enum CopyEnginePreference { NONE = 0, BLIT = 1, SDMA = 2, CPDMA = 3 };

  struct {
    uint32_t isAsync_ : 1;
    uint32_t copyEnginePreference_ : 2;
  };
  uint32_t flags_;
  CopyMetadata() : flags_(0) {}
  CopyMetadata(bool isAsync, CopyEnginePreference copyEnginePreference)
      : isAsync_(isAsync), copyEnginePreference_(copyEnginePreference) {}
};

// Interface to callback to allocate kernel args from the graph kernel arg pool.
class GraphKernelArgManager {
 public:
  virtual address AllocKernArg(size_t size, size_t alignment, int devId) = 0;
};

/*! \brief An operation that is submitted to a command queue.
 *
 *  %Command is the abstract base type of all OpenCL operations
 *  submitted to a HostQueue for execution. Classes derived from
 *  %Command must implement the submit() function.
 *
 */
class Command : public Event {
 private:
  static SysmemPool<ComputeCommand>* command_pool_;  //!< Pool of active commands
  HostQueue* queue_;               //!< The command queue this command is enqueue into
  Command* next_;                  //!< Next GPU command in the queue list
  Command* batch_head_ = nullptr;  //!< The head of the batch commands
  cl_command_type type_;           //!< This command's OpenCL type.
  std::vector<void*> data_;
  const Event* waitingEvent_;  //!< Waiting event associated with the marker

  bool packetCapturing_ = false;       //!< Flag to enable/disable graph gpu packet capture
  std::vector<uint8_t*>* gpuPackets_;  //!< GPU packets captured when graph capturing is enabled
  GraphKernelArgManager* graphKernArgMgr_ = nullptr;  //!< KernelMgr for graph
  address kernArgOffset_ = nullptr;  //!< KernelArg buffer to used when graph capturing is enabled
  std::string* capturedKernelName_ = nullptr;  //!< Kenrnel under capture
 protected:
  bool cpu_wait_ = false;  //!< If true, then the command was issued for CPU/GPU sync

  //! The Events that need to complete before this command is submitted.
  EventWaitList eventWaitList_;

  //! Force await completion of previous command
  //! 0x1 - wait before enqueue, 0x2 - wait after, 0x3 - wait both
  uint32_t commandWaitBits_;

  //! Construct a new command of the given OpenCL type.
  Command(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList = nullWaitList,
          uint32_t commandWaitBits = 0, const Event* waitingEvent = nullptr);

  //! Construct a new command of the given OpenCL type.
  Command(cl_command_type type)
      : Event(),
        queue_(nullptr),
        next_(nullptr),
        type_(type),
        waitingEvent_(nullptr),
        eventWaitList_(nullWaitList),
        commandWaitBits_(0) {}

  virtual bool terminate() {
    if (IS_HIP) {
      releaseResources();
    }
    if (Agent::shouldPostEventEvents() && type() != 0) {
      Agent::postEventFree(as_cl(static_cast<Event*>(this)));
    }
    return true;
  }

 public:
  //! Returns AQL buffer state
  static void ReleaseSysmemPool() {
    if (command_pool_ != nullptr) {
      delete command_pool_;
      command_pool_ = nullptr;
    }
  }
  bool getPktCapturingState() const { return packetCapturing_; }

  //! Sets AQL capture state, aql packet to capture and where to copy kernArgs
  void setPktCapturingState(bool state, std::vector<uint8_t*>* packet,
                            amd::GraphKernelArgManager* graphKernArgMgr,
                            std::string* capturedKernelName) {
    packetCapturing_ = state;
    gpuPackets_ = packet;
    graphKernArgMgr_ = graphKernArgMgr;
    capturedKernelName_ = capturedKernelName;
  }

  //! Updates kernel name with the captured kernel name
  void SetKernelName(const std::string& kernelName) {
    if (capturedKernelName_ != nullptr) {
      *capturedKernelName_ = kernelName;
    }
  }

  //! Returns the graph executable object command belongs to.
  const uint8_t* getAqlPacket() const {
    uint8_t* packet = new uint8_t[64];
    gpuPackets_->push_back(packet);
    return packet;
  }

  address getGraphKernArg(int size, int alignment, int devId) {
    return graphKernArgMgr_->AllocKernArg(size, alignment, devId);
  }

  //! Overload new/delete for fast commands allocation/destruction
  void* operator new(size_t size);
  void operator delete(void* ptr);

  //! Return the queue this command is enqueued into.
  HostQueue* queue() const { return queue_; }

  //! Enqueue this command into the associated command queue.
  void enqueue();

  //! Return the event encapsulating this command's status.
  const Event& event() const { return *this; }
  Event& event() { return *this; }

  //! Return the list of events this command needs to wait on before dispatch
  const EventWaitList& eventWaitList() const { return eventWaitList_; }

  //! Update with the list of events this command needs to wait on before dispatch
  void updateEventWaitList(const EventWaitList& waitList) {
    for (auto event : waitList) {
      event->retain();
      eventWaitList_.push_back(event);
    }
  }

  //! Return this command's OpenCL type.
  cl_command_type type() const { return type_; }

  //! Return the opaque, device specific data vector for this command.
  std::vector<void*>& data() { return data_; }


  /*! \brief The execution engine for this command.
   *
   *  \details All derived class must implement this virtual function.
   *
   *  \note This function will execute in the command queue thread.
   */
  virtual void submit(device::VirtualDevice& device) = 0;

  //! Release the resources associated with this event.
  virtual void releaseResources();

  //! Empty function for pinned memory check
  virtual bool IsMemoryPinned() const { return false; }

  //! Empty function for release of pinned memory
  virtual void ReleasePinnedMemory() {}

  //! Empty function for adding pinned memory
  virtual void AddPinnedMemory(Memory* pinned) {}

  //! Set the next GPU command
  void setNext(Command* next) { next_ = next; }

  //! Get the next GPU command
  Command* getNext() const { return next_; }

  //! Return the context for this event.
  virtual const Context& context() const;

  //! Get command wait bits
  uint32_t getWaitBits() const { return commandWaitBits_; }

  void OverrrideCommandType(cl_command_type type) { type_ = type; }

  //! Updates the batch head, associated with this command(marker)
  void SetBatchHead(Command* command) { batch_head_ = command; }

  //! Returns the current batch head
  Command* GetBatchHead() const { return batch_head_; }

  const Event* waitingEvent() const { return waitingEvent_; }

  //! Check if this command(should be a marker) requires CPU wait
  bool CpuWaitRequested() const { return cpu_wait_; }
};

class UserEvent : public Command {
  const Context& context_;            //!< OCL context associated with the event
  std::vector<Command*> dependents_;  //!< Commands, which depends on this user event

 public:
  UserEvent(Context& context) : Command(CL_COMMAND_USER), context_(context) {
    setStatus(CL_SUBMITTED);
  }

  //! Creates a user event in the backend layer
  bool Create() {
    if (AMD_DIRECT_DISPATCH) {
      return context_.devices()[0]->CreateUserEvent(this);
    } else {
      return true;
    }
  }

  //! Sets the execution status of the user event
  bool SetExecutionStatus(cl_int status) {
    if (AMD_DIRECT_DISPATCH) {
      // If it's invalid status, then mark dependent commands as invalid
      if (status < CL_COMPLETE) {
        for (auto it : dependents_) {
          it->setStatus(CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST);
        }
      }
      dependents_.clear();
      context_.devices()[0]->SetUserEvent(this);
    }
    return setStatus(status);
  }

  //! Adds dependent commands for the user event
  void AddDependent(Command* command) { dependents_.push_back(command); }

  virtual void submit(device::VirtualDevice& device) { ShouldNotCallThis(); }

  const Context& context() const { return context_; }
};

class ClGlEvent : public Command {
 private:
  const Context& context_;
  bool waitForFence();

 public:
  ClGlEvent(Context& context) : Command(CL_COMMAND_GL_FENCE_SYNC_OBJECT_KHR), context_(context) {
    setStatus(CL_SUBMITTED);
  }

  virtual void submit(device::VirtualDevice& device) { ShouldNotCallThis(); }

  bool awaitCompletion() { return waitForFence(); }

  const Context& context() const { return context_; }
};

inline Command& Event::command() { return *static_cast<Command*>(this); }

inline const Command& Event::command() const { return *static_cast<const Command*>(this); }

class Kernel;
class NDRangeContainer;

//! A memory command that holds a single memory object reference.
//
class OneMemoryArgCommand : public Command {
 protected:
  Memory* memory_;
  std::vector<Memory*> pinned_memory_;  //!< Pinned memory object

 public:
  OneMemoryArgCommand(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList,
                      Memory& memory)
      : Command(queue, type, eventWaitList, AMD_SERIALIZE_COPY), memory_(&memory) {
    memory_->retain();
  }

  virtual void releaseResources() {
    memory_->release();
    DEBUG_ONLY(memory_ = NULL);
    Command::releaseResources();
    ReleasePinnedMemory();
  }

  //! Release all pinned memory for this command
  virtual void ReleasePinnedMemory() {
    for (auto it : pinned_memory_) {
      it->release();
    }
    pinned_memory_.clear();
  }
  //! Release all pinned memory for this command
  virtual bool IsMemoryPinned() const { return !pinned_memory_.empty(); }

  //! Adds pinned memory, used in this command for later release
  virtual void AddPinnedMemory(Memory* pinned) override { pinned_memory_.push_back(pinned); }
  bool validateMemory();
  bool validatePeerMemory();
};

//! A memory command that holds a single memory object reference.
//
class TwoMemoryArgsCommand : public Command {
 protected:
  Memory* memory1_;
  Memory* memory2_;

 public:
  TwoMemoryArgsCommand(HostQueue& queue, cl_command_type type, const EventWaitList& eventWaitList,
                       Memory& memory1, Memory& memory2)
      : Command(queue, type, eventWaitList, AMD_SERIALIZE_COPY),
        memory1_(&memory1),
        memory2_(&memory2) {
    memory1_->retain();
    memory2_->retain();
  }

  virtual void releaseResources() {
    memory1_->release();
    memory2_->release();
    DEBUG_ONLY(memory1_ = memory2_ = NULL);
    Command::releaseResources();
  }

  bool validateMemory();
  bool validatePeerMemory();
};

/*!  \brief     A generic read memory command.
 *
 *   \details   Used for operations on both buffers and images. Backends
 *              are expected to handle any required translation. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 *
 *   @todo Find a cleaner way of merging the row and slice pitch concepts at this level.
 *
 */

class ReadMemoryCommand : public OneMemoryArgCommand {
 private:
  Coord3D origin_;     //!< Origin of the region to read.
  Coord3D size_;       //!< Size of the region to read.
  void* hostPtr_;      //!< The host pointer destination.
  size_t rowPitch_;    //!< Row pitch (for image operations)
  size_t slicePitch_;  //!< Slice pitch (for image operations)

  BufferRect bufRect_;   //!< Buffer rectangle information
  BufferRect hostRect_;  //!< Host memory rectangle information
  amd::CopyMetadata copyMetadata_;

 public:
  //! Construct a new ReadMemoryCommand
  ReadMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, Coord3D origin, Coord3D size, void* hostPtr,
                    size_t rowPitch = 0, size_t slicePitch = 0,
                    amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(rowPitch),
        slicePitch_(slicePitch),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  //! Construct a new ReadMemoryCommand
  ReadMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, Coord3D origin, Coord3D size, void* hostPtr,
                    const BufferRect& bufRect, const BufferRect& hostRect,
                    amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(0),
        slicePitch_(0),
        bufRect_(bufRect),
        hostRect_(hostRect),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitReadMemory(*this); }

  //! Return the memory object to read from.
  Memory& source() const { return *memory_; }
  //! Return the host memory to write to
  void* destination() const { return hostPtr_; }

  //! Return the origin of the region to read
  const Coord3D& origin() const { return origin_; }
  //! Return the size of the region to read
  const Coord3D& size() const { return size_; }
  //! Return the row pitch
  size_t rowPitch() const { return rowPitch_; }
  //! Return the slice pitch
  size_t slicePitch() const { return slicePitch_; }

  //! Return the buffer rectangle information
  const BufferRect& bufRect() const { return bufRect_; }
  //! Return the host rectangle information
  const BufferRect& hostRect() const { return hostRect_; }
  //! Return the copy MetaData
  amd::CopyMetadata copyMetadata() const { return copyMetadata_; }
  //! Updates the host memory to read from
  void setSource(Memory& memory) { memory_ = &memory; }
  //! Updates the host memory to write to
  void setDestination(void* hostPtr) { hostPtr_ = hostPtr; }

  //! Updates the origin of the region to read
  void setOrigin(const Coord3D& origin) { origin_ = origin; }
  //! Updates the size of the region to read
  void setSize(const Coord3D& size) { size_ = size; }
  //! Updates the row pitch
  void setRowPitch(const size_t rowPitch) { rowPitch_ = rowPitch; }
  //! Updates the slice pitch
  void setSlicePitch(const size_t slicePitch) { slicePitch_ = slicePitch; }

  //! Updates the buffer rectangle information
  void setBufRect(const BufferRect& bufRect) { bufRect_ = bufRect; }
  //! Updates the host rectangle information
  void setHostRect(const BufferRect& hostRect) { hostRect_ = hostRect; }

  //! Updates command parameters
  void setParams(Memory& memory, Coord3D origin, Coord3D size, void* hostPtr,
                 const BufferRect& bufRect, const BufferRect& hostRect) {
    memory_ = &memory;
    origin_ = origin;
    size_ = size;
    hostPtr_ = hostPtr;
    bufRect_ = bufRect;
    hostRect_ = hostRect;
  }
  //! Updates command parameters
  void setParams(Memory& memory, Coord3D origin, Coord3D size, void* hostPtr, size_t rowPitch = 0,
                 size_t slicePitch = 0) {
    memory_ = &memory;
    origin_ = origin;
    size_ = size;
    hostPtr_ = hostPtr;
    rowPitch_ = rowPitch;
    slicePitch_ = slicePitch;
  }

  //! Return true if the entire memory object is read.
  bool isEntireMemory() const;
};

/*! \brief      A generic write memory command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 */

class WriteMemoryCommand : public OneMemoryArgCommand {
 private:
  Coord3D origin_;       //!< Origin of the region to write to.
  Coord3D size_;         //!< Size of the region to write to.
  const void* hostPtr_;  //!< The host pointer source.
  size_t rowPitch_;      //!< Row pitch (for image operations)
  size_t slicePitch_;    //!< Slice pitch (for image operations)

  BufferRect bufRect_;   //!< Buffer rectangle information
  BufferRect hostRect_;  //!< Host memory rectangle information
  amd::CopyMetadata copyMetadata_;

 public:
  WriteMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                     size_t rowPitch = 0, size_t slicePitch = 0,
                     amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(rowPitch),
        slicePitch_(slicePitch),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  WriteMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                     const BufferRect& bufRect, const BufferRect& hostRect,
                     amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        hostPtr_(hostPtr),
        rowPitch_(0),
        slicePitch_(0),
        bufRect_(bufRect),
        hostRect_(hostRect),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(hostPtr != NULL && "hostPtr cannot be null");
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitWriteMemory(*this); }

  //! Return the host memory to read from
  const void* source() const { return hostPtr_; }
  //! Return the memory object to write to.
  Memory& destination() const { return *memory_; }

  //! Return the region origin
  const Coord3D& origin() const { return origin_; }
  //! Return the region size
  const Coord3D& size() const { return size_; }
  //! Return the row pitch
  size_t rowPitch() const { return rowPitch_; }
  //! Return the slice pitch
  size_t slicePitch() const { return slicePitch_; }

  //! Return the buffer rectangle information
  const BufferRect& bufRect() const { return bufRect_; }
  //! Return the host rectangle information
  const BufferRect& hostRect() const { return hostRect_; }
  //! Return the copy MetaData
  amd::CopyMetadata copyMetadata() const { return copyMetadata_; }
  //! Updates the host memory to read from
  void setSource(const void* hostPtr) { hostPtr_ = hostPtr; }
  //! Updates the host memory to write to
  void setDestination(Memory& memory) { memory_ = &memory; }

  //! Updates the origin of the region to read
  void setOrigin(const Coord3D& origin) { origin_ = origin; }
  //! Updates the size of the region to read
  void setSize(const Coord3D& size) { size_ = size; }
  //! Updates the row pitch
  void setRowPitch(const size_t rowPitch) { rowPitch_ = rowPitch; }
  //! Updates the slice pitch
  void setSlicePitch(const size_t slicePitch) { slicePitch_ = slicePitch; }

  //! Updates the buffer rectangle information
  void setBufRect(const BufferRect& bufRect) { bufRect_ = bufRect; }
  //! Updates the host rectangle information
  void setHostRect(const BufferRect& hostRect) { hostRect_ = hostRect; }

  //! Updates command parameters
  void setParams(Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                 const BufferRect& bufRect, const BufferRect& hostRect) {
    memory_ = &memory;
    origin_ = origin;
    size_ = size;
    hostPtr_ = hostPtr;
    bufRect_ = bufRect;
    hostRect_ = hostRect;
  }
  //! Updates command parameters
  void setParams(Memory& memory, Coord3D origin, Coord3D size, const void* hostPtr,
                 size_t rowPitch = 0, size_t slicePitch = 0) {
    memory_ = &memory;
    origin_ = origin;
    size_ = size;
    hostPtr_ = hostPtr;
    rowPitch_ = rowPitch;
    slicePitch_ = slicePitch;
  }

  //! Return true if the entire memory object is written.
  bool isEntireMemory() const;
};

/*! \brief      A generic fill memory command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations. Buffers
 *              are treated as 1D structures so origin_[0] and size_[0]
 *              are equivalent to offset_ and count_ respectively.
 */

class FillMemoryCommand : public OneMemoryArgCommand {
 public:
  static constexpr size_t MaxFillPatterSize = sizeof(double[16]);

 private:
  Coord3D origin_;                   //!< Origin of the region to write to.
  Coord3D size_;                     //!< Size of the region to write to.
  Coord3D surface_;                  //!< Total surface
  char pattern_[MaxFillPatterSize];  //!< The fill pattern
  size_t patternSize_;               //!< Pattern size

 public:
  FillMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& memory, const void* pattern, size_t patternSize, const Coord3D& origin,
                    const Coord3D& size, const Coord3D& surface)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        origin_(origin),
        size_(size),
        surface_(surface),
        patternSize_(patternSize) {
    // Sanity checks
    assert(pattern != NULL && "pattern cannot be null");
    assert(size.c[0] > 0 && "invalid");
    memcpy(pattern_, pattern, patternSize);
  }

  virtual void submit(device::VirtualDevice& device) { device.submitFillMemory(*this); }

  //! Return the pattern memory to fill with
  const void* pattern() const { return reinterpret_cast<const void*>(pattern_); }
  //! Return the pattern size
  const size_t patternSize() const { return patternSize_; }
  //! Return the memory object to write to.
  Memory& memory() const { return *memory_; }

  //! Return the region origin
  const Coord3D& origin() const { return origin_; }
  //! Return the region size
  const Coord3D& size() const { return size_; }

  //! Return the surface
  const Coord3D& surface() const { return surface_; }

  //! Updates the pattern memory to fill with and pattern size
  void setPattern(const void* pattern, const size_t patternSize) {
    assert(pattern != NULL && "pattern cannot be null");
    memcpy(pattern_, pattern, patternSize);
    patternSize_ = patternSize;
  }

  //! Updates the memory object to write to.
  void setMemory(Memory& memory) { memory_ = &memory; }

  //! Updates the region origin
  void setOrigin(const Coord3D& origin) { origin_ = origin; }
  //! Updates the region size
  void setSize(const Coord3D& size) { size_ = size; }

  //! Updates the surface
  void setSurface(const Coord3D& surface) { surface_ = surface; }

  //! Updates command parameters
  void setParams(Memory& memory, const void* pattern, size_t patternSize, const Coord3D& origin,
                 const Coord3D& size, const Coord3D& surface) {
    memory_ = &memory;
    assert(pattern != NULL && "pattern cannot be null");
    assert(size.c[0] > 0 && "invalid");
    memcpy(pattern_, pattern, patternSize);
    origin_ = origin;
    size_ = size;
    surface_ = surface;
  }

  //! Return true if the entire memory object is written.
  bool isEntireMemory() const;
};

/*! \brief      A stream operation command.
 *
 *  \details    Used to perform a stream wait or strem write operations.
 *              Wait: All the commands issued after stream wait are not executed until the wait
 *              condition is true.
 *              Write: Writes a 32 or 64 bit vaue to the memeory using a GPU Blit.
 */

class StreamOperationCommand : public OneMemoryArgCommand {
 private:
  uint64_t value_;      // !< Value to Wait on or to Write.
  uint64_t mask_;       // !< Mask to be applied on signal value for Wait operation.
  unsigned int flags_;  // !< Flags defining the Wait condition.
  size_t offset_;       // !< Offset into memory for Write
  size_t sizeBytes_;    // !< Size in bytes to Write.

  // NOTE: mask_ is only used for wait operation and
  // offset and sizeBytes are only used for write.

 public:
  StreamOperationCommand(HostQueue& queue, cl_command_type cmdType,
                         const EventWaitList& eventWaitList, Memory& memory, const uint64_t value,
                         const uint64_t mask, unsigned int flags, size_t offset, size_t sizeBytes)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        value_(value),
        mask_(mask),
        flags_(flags),
        offset_(offset),
        sizeBytes_(sizeBytes) {
    // Sanity check
    assert(((cmdType == ROCCLR_COMMAND_STREAM_WRITE_VALUE) ||
            (cmdType == ROCCLR_COMMAND_STREAM_WAIT_VALUE) ||
            ((cmdType == ROCCLR_COMMAND_STREAM_WAIT_VALUE) && GPU_STREAMOPS_CP_WAIT &&
             (memory_->getMemFlags() & ROCCLR_MEM_HSA_SIGNAL_MEMORY))) &&
           "Invalid Stream Operation");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitStreamOperation(*this); }

  //! Returns the value
  const uint64_t value() const { return value_; }
  //! Returns the wait mask
  const uint64_t mask() const { return mask_; }
  //! Return the wait flags
  const unsigned int flags() const { return flags_; }
  //! Return the memory object.
  Memory& memory() const { return *memory_; }
  //! Return the write offset.
  const size_t offset() const { return offset_; }
  //! Return the write size.
  const size_t sizeBytes() const { return sizeBytes_; }
};

/*! \brief      A batch memory operation command.
 *
 *  \details    Batch operations to synchronize the stream via memory operations
 *              Operations are either 32-bit stream wait or write.
 *              Wait: All the commands issued after stream wait are not executed
 *              until the wait condition is true.
 *              Write: Writes a 32 or 64 bit vaue to the memory using a GPU Blit.
 *              The operations are enqueued in the order they appear in the array.
 */

class BatchMemoryOperationCommand : public Command {
 public:
  BatchMemoryOperationCommand(HostQueue& queue, cl_command_type cmdType, uint32_t count,
                              uint32_t flags, EventWaitList& eventWaitList, const void* paramArray,
                              size_t paramSize)
      : Command(queue, cmdType, eventWaitList),
        count_(count),
        paramArray_(paramArray),
        flags_(flags),
        paramSize_(paramSize) {
    // Sanity check
    assert(((cmdType == ROCCLR_COMMAND_BATCH_STREAM)) && "Invalid batch memory operation");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitBatchMemoryOperation(*this); }

  //! Returns the value
  const uint64_t count() const { return count_; }
  //! Return the pointer to the paramList
  const void* getParamPtr() { return paramArray_; }
  //! Return the size of a single mem op param in bytes
  const size_t paramSize() const { return paramSize_; }

 private:
  uint32_t count_;          // !< The number of operations in the array.
  uint32_t flags_;          // !< Reserved for future expansion. Must be 0.
  const void* paramArray_;  // !< Pointer to the array of individual operations
  size_t paramSize_;        // !< size in bytes of the param array passed
};

/*! \brief      A generic copy memory command
 *
 *  \details    Used for both buffers and images. Backends are expected
 *              to handle any required translation. Buffers are treated
 *              as 1D structures so origin_[0] and size_[0] are
 *              equivalent to offset_ and count_ respectively.
 */

class CopyMemoryCommand : public TwoMemoryArgsCommand {
 private:
  Coord3D srcOrigin_;  //!< Origin of the source region.
  Coord3D dstOrigin_;  //!< Origin of the destination region.
  Coord3D size_;       //!< Size of the region to copy.

  BufferRect srcRect_;  //!< Source buffer rectangle information
  BufferRect dstRect_;  //!< Destination buffer rectangle information
  amd::CopyMetadata copyMetadata_;

 public:
  CopyMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                    Coord3D size, amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : TwoMemoryArgsCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory),
        srcOrigin_(srcOrigin),
        dstOrigin_(dstOrigin),
        size_(size),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  CopyMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                    Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                    Coord3D size, const BufferRect& srcRect, const BufferRect& dstRect,
                    amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : TwoMemoryArgsCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory),
        srcOrigin_(srcOrigin),
        dstOrigin_(dstOrigin),
        size_(size),
        srcRect_(srcRect),
        dstRect_(dstRect),
        copyMetadata_(copyMetadata) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitCopyMemory(*this); }

  //! Return the host memory to read from
  Memory& source() const { return *memory1_; }
  //! Return the memory object to write to.
  Memory& destination() const { return *memory2_; }

  //! Return the source origin
  const Coord3D& srcOrigin() const { return srcOrigin_; }
  //! Return the offset in bytes in the destination.
  const Coord3D& dstOrigin() const { return dstOrigin_; }
  //! Return the number of bytes to copy.
  const Coord3D& size() const { return size_; }

  //! Return the source buffer rectangle information
  const BufferRect& srcRect() const { return srcRect_; }
  //! Return the destination buffer rectangle information
  const BufferRect& dstRect() const { return dstRect_; }
  //! Return the copy MetaData
  amd::CopyMetadata copyMetadata() const { return copyMetadata_; }
  //! Updates copy MetaData
  void SetCopyMetadata(amd::CopyMetadata copyMetadata) { copyMetadata_ = copyMetadata; }
  //! Updates the host memory to read from
  void setSource(Memory& srcMemory) { memory1_ = &srcMemory; }
  //! Updates the memory object to write to.
  void setDestination(Memory& dstMemory) { memory2_ = &dstMemory; }

  //! Updates the source origin
  void setSrcOrigin(const Coord3D srcOrigin) { srcOrigin_ = srcOrigin; }
  //! Updates the offset in bytes in the destination.
  void setDstOrigin(const Coord3D dstOrigin) { dstOrigin_ = dstOrigin; }
  //! Updates the number of bytes to copy.
  void setSize(const Coord3D size) { size_ = size; }

  //! Updates the source buffer rectangle information
  void setSrcRect(const BufferRect srcRect) { srcRect_ = srcRect; }
  //! Updates the destination buffer rectangle information
  void setDstRect(const BufferRect dstRect) { dstRect_ = dstRect; }

  //! Updates command parameters
  void setParams(Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                 Coord3D size) {
    memory1_ = &srcMemory;
    memory2_ = &dstMemory;
    srcOrigin_ = srcOrigin;
    dstOrigin_ = dstOrigin;
    size_ = size;
  }
  //! Updates command parameters
  void setParams(Memory& srcMemory, Memory& dstMemory, Coord3D srcOrigin, Coord3D dstOrigin,
                 Coord3D size, const BufferRect& srcRect, const BufferRect& dstRect) {
    memory1_ = &srcMemory;
    memory2_ = &dstMemory;
    srcOrigin_ = srcOrigin;
    dstOrigin_ = dstOrigin;
    size_ = size;
    srcRect_ = srcRect;
    dstRect_ = dstRect;
  }

  //! Return true if the both memories are is read/written in their entirety.
  bool isEntireMemory() const;
};

/*! \brief  A generic map memory command. Makes a memory object accessible to the host.
 *
 * @todo:dgladdin   Need to think more about how the pitch parameters operate in
 *                  the context of unified buffer/image commands.
 */

class MapMemoryCommand : public OneMemoryArgCommand {
 private:
  cl_map_flags mapFlags_;  //!< Flags controlling the map.
  bool blocking_;          //!< True for blocking maps
  Coord3D origin_;         //!< Origin of the region to map.
  Coord3D size_;           //!< Size of the region to map.
  const void* mapPtr_;     //!< Host-space pointer that the object is currently mapped at

 public:
  //! Construct a new MapMemoryCommand
  MapMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                   Memory& memory, cl_map_flags mapFlags, bool blocking, Coord3D origin,
                   Coord3D size, size_t* imgRowPitch = nullptr, size_t* imgSlicePitch = nullptr,
                   void* mapPtr = nullptr)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        mapFlags_(mapFlags),
        blocking_(blocking),
        origin_(origin),
        size_(size),
        mapPtr_(mapPtr) {
    // Sanity checks
    assert(size.c[0] > 0 && "invalid");
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMapMemory(*this); }

  //! Read the memory object
  Memory& memory() const { return *memory_; }
  //! Read the map control flags
  cl_map_flags mapFlags() const { return mapFlags_; }
  //! Read the origin
  const Coord3D& origin() const { return origin_; }
  //! Read the size
  const Coord3D& size() const { return size_; }
  //! Read the blocking flag
  bool blocking() const { return blocking_; }
  //! Returns true if the entire memory object is mapped
  bool isEntireMemory() const;
  //! Read the map pointer
  const void* mapPtr() const { return mapPtr_; }
};


/*! \brief  A generic unmap memory command.
 *
 * @todo:dgladdin   Need to think more about how the pitch parameters operate in
 *                  the context of unified buffer/image commands.
 */

class UnmapMemoryCommand : public OneMemoryArgCommand {
 private:
  //! Host-space pointer that the object is currently mapped at
  void* mapPtr_;

 public:
  //! Construct a new MapMemoryCommand
  UnmapMemoryCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                     Memory& memory, void* mapPtr)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory), mapPtr_(mapPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitUnmapMemory(*this); }

  virtual void releaseResources();

  //! Read the memory object
  Memory& memory() const { return *memory_; }
  //! Read the map pointer
  void* mapPtr() const { return mapPtr_; }
};

/*! \brief      Migrate memory objects command.
 *
 *  \details    Used for operations on both buffers and images. Backends
 *              are expected to handle any required translations.
 */
class MigrateMemObjectsCommand : public Command {
 private:
  cl_mem_migration_flags migrationFlags_;  //!< Migration flags
  std::vector<amd::Memory*> memObjects_;   //!< The list of memory objects

 public:
  //! Construct a new AcquireExtObjectsCommand
  MigrateMemObjectsCommand(HostQueue& queue, cl_command_type type,
                           const EventWaitList& eventWaitList,
                           const std::vector<amd::Memory*>& memObjects,
                           cl_mem_migration_flags flags)
      : Command(queue, type, eventWaitList), migrationFlags_(flags) {
    for (const auto& it : memObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMigrateMemObjects(*this); }

  //! Release all resources associated with this command
  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  //! Returns the migration flags
  cl_mem_migration_flags migrationFlags() const { return migrationFlags_; }
  //! Returns the number of memory objects in the command
  uint32_t numMemObjects() const { return (uint32_t)memObjects_.size(); }
  //! Returns a pointer to the memory objects
  const std::vector<amd::Memory*>& memObjects() const { return memObjects_; }

  bool validateMemory();
};

//! To execute a kernel on a specific device.
class NDRangeKernelCommand : public Command {
 private:
  Kernel& kernel_;
  NDRangeContainer sizes_;
  address parameters_;  //!< Pointer to the kernel argumets
  // The below fields are specific to the HIP functionality
  uint32_t sharedMemBytes_;  //!< Size of reserved shared memory
  uint32_t extraParam_;      //!< Extra flags for the kernel launch
  uint32_t gridId_;          //!< Grid ID in the multi GPU kernel launch
  uint32_t numGrids_;        //!< Total number of grids in multi GPU launch
  uint64_t prevGridSum_;     //!< A sum of previous grids to the current launch
  uint64_t allGridSum_;      //!< A sum of all grids in multi GPU launch
  uint32_t firstDevice_;     //!< Device index of the first device in the gridc
  uint32_t numWorkgroups_;   //!< Total number of workgroups in the current launch

 public:
  enum {
    CooperativeGroups = 0x01,
    CooperativeMultiDeviceGroups = 0x02,
    AnyOrderLaunch = 0x04,
  };

  //! Construct an ExecuteKernel command
  NDRangeKernelCommand(HostQueue& queue, const EventWaitList& eventWaitList, Kernel& kernel,
                       const NDRangeContainer& sizes, uint32_t sharedMemBytes = 0,
                       uint32_t extraParam = 0, uint32_t gridId = 0, uint32_t numGrids = 0,
                       uint64_t prevGridSum = 0, uint64_t allGridSum = 0, uint32_t firstDevice = 0,
                       bool forceProfiling = false);

  virtual void submit(device::VirtualDevice& device) { device.submitKernel(*this); }

  //! Release all resources associated with this command (
  void releaseResources();

  //! Return the kernel.
  const Kernel& kernel() const { return kernel_; }

  //! Return the parameters given to this kernel.
  const_address parameters() const { return parameters_; }

  //! Return the kernel NDRange.
  const NDRangeContainer& sizes() const { return sizes_; }

  //! updates kernel NDRange.
  void setSizes(const size_t* globalWorkOffset, const size_t* globalWorkSize,
                const size_t* localWorkSize) {
    sizes_.update(3, globalWorkOffset, globalWorkSize, localWorkSize);
  }

  //! Return the shared memory size
  uint32_t sharedMemBytes() const { return sharedMemBytes_; }

  //! updates shared memory size
  void setSharedMemBytes(uint32_t sharedMemBytes) { sharedMemBytes_ = sharedMemBytes; }

  //! Return the cooperative groups mode
  bool cooperativeGroups() const { return (extraParam_ & CooperativeGroups) ? true : false; }

  //! Return the cooperative multi device groups mode
  bool cooperativeMultiDeviceGroups() const {
    return (extraParam_ & CooperativeMultiDeviceGroups) ? true : false;
  }

  //! Returns extra Param, set when using anyorder launch
  bool getAnyOrderLaunchFlag() const { return (extraParam_ & AnyOrderLaunch) ? true : false; }

  //! Return the current grid ID for multidevice launch
  uint32_t gridId() const { return gridId_; }

  //! Return the number of launched grids
  uint32_t numGrids() const { return numGrids_; }

  //! Return the total workload size for up to the current
  uint64_t prevGridSum() const { return prevGridSum_; }

  //! Return the total workload size for all GPUs
  uint64_t allGridSum() const { return allGridSum_; }

  //! Return the index of the first device in multi GPU launch
  uint64_t firstDevice() const { return firstDevice_; }

  uint32_t numWorkgroups() const { return numWorkgroups_; }

  //! Set the local work size.
  void setLocalWorkSize(const NDRange& local) { sizes_.local() = local; }

  //! Set the number of workgroups
  void setNumWorkgroups() {
    uint32_t numWorkgroups = 1;
    for (uint i = 0; i < sizes().dimensions(); i++) {
      if (sizes().local()[i] != 0) {
        numWorkgroups *= (sizes().global()[i] / sizes().local()[i]);
      }
    }
    numWorkgroups_ = numWorkgroups;
  }

  // Capture kernel parameters and validate
  int32_t captureAndValidate();

  // Allocate, capture and set kernel parameters
  int32_t AllocCaptureSetValidate(void** kernelParams, address kernArgs, size_t kernArgsSize);
};

class NativeFnCommand : public Command {
 private:
  void(CL_CALLBACK* nativeFn_)(void*);

  char* args_;
  size_t argsSize_;

  std::vector<Memory*> memObjects_;
  std::vector<size_t> memOffsets_;

 public:
  NativeFnCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                  void(CL_CALLBACK* nativeFn)(void*), const void* args, size_t argsSize,
                  size_t numMemObjs, const cl_mem* memObjs, const void** memLocs);

  ~NativeFnCommand() { delete[] args_; }

  void releaseResources() {
    for (const auto& memObject : memObjects_) {
      memObject->release();
    }
    Command::releaseResources();
  }

  virtual void submit(device::VirtualDevice& device) { device.submitNativeFn(*this); }

  int32_t invoke();
};


class ExternalSemaphoreCmd : public Command {
 public:
  enum ExternalSemaphoreCmdType { COMMAND_WAIT_EXTSEMAPHORE, COMMAND_SIGNAL_EXTSEMAPHORE };

 private:
  const void* sem_ptr_;                //!< Pointer to external semaphore
  uint64_t fence_;                     //!< semaphore value to be set
  ExternalSemaphoreCmdType cmd_type_;  //!< Signal or Wait semaphore command

 public:
  ExternalSemaphoreCmd(HostQueue& queue, const void* sem_ptr, uint64_t fence,
                       ExternalSemaphoreCmdType cmd_type)
      : Command::Command(queue, CL_COMMAND_USER),
        sem_ptr_(sem_ptr),
        fence_(fence),
        cmd_type_(cmd_type) {}

  virtual void submit(device::VirtualDevice& device) { device.submitExternalSemaphoreCmd(*this); }
  const void* sem_ptr() const { return sem_ptr_; }
  const uint64_t fence() { return fence_; }
  const ExternalSemaphoreCmdType semaphoreCmd() { return cmd_type_; }
};


class Marker : public Command {
 public:
  //! Create a new Marker
  Marker(HostQueue& queue, bool userVisible, const EventWaitList& eventWaitList = nullWaitList,
         const Event* waitingEvent = nullptr, bool cpu_wait = false)
      : Command(queue, userVisible ? CL_COMMAND_MARKER : 0, eventWaitList, 0, waitingEvent) {
    cpu_wait_ = cpu_wait;
  }

  //! The actual command implementation.
  virtual void submit(device::VirtualDevice& device) { device.submitMarker(*this); }
};

class AccumulateCommand : public Command {
 private:
  //! Kernel names and timestamps list for activity profiling
  std::vector<std::string> kernelNames_;
  std::vector<std::pair<uint64_t, uint64_t>> tsList_;

 public:
  //! Create a new Marker
  AccumulateCommand(HostQueue& queue, const EventWaitList& eventWaitList = nullWaitList,
                    const Event* waitingEvent = nullptr)
      : Command(queue, CL_COMMAND_TASK, eventWaitList, 0, waitingEvent) {}

  //! Add kernel name to the list if available
  void addKernelName(const std::string& kernelName) { kernelNames_.push_back(kernelName); }

  //! Add multiple kernel names in bulk
  void addKernelNames(const std::vector<std::string>& kernelNames) {
    kernelNames_.insert(kernelNames_.end(), kernelNames.begin(), kernelNames.end());
  }

  //! Add kernel timestamp to the list if available
  void addTimestamps(uint64_t startTs, uint64_t endTs) {
    tsList_.push_back(std::make_pair(startTs, endTs));
  }

  //! Return the kernel names
  const std::vector<std::string>& getKernelNames() const { return kernelNames_; }

  //! Return the kernel timestamps
  const std::vector<std::pair<uint64_t, uint64_t>>& getTimestamps() const { return tsList_; }

  //! The command implementation
  virtual void submit(device::VirtualDevice& device) { device.submitAccumulate(*this); }
};

/*! \brief  Maps CL objects created from external ones and syncs the contents (blocking).
 *
 */

class ExtObjectsCommand : public Command {
 private:
  std::vector<amd::Memory*> memObjects_;  //!< The list of Memory based classes

 public:
  //! Construct a new AcquireExtObjectsCommand
  ExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList, uint32_t num_objects,
                    const std::vector<amd::Memory*>& memoryObjects, cl_command_type type)
      : Command(queue, type, eventWaitList) {
    for (const auto& it : memoryObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  //! Release all resources associated with this command
  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  //! Get number of GL objects
  uint32_t getNumObjects() { return (uint32_t)memObjects_.size(); }
  //! Get pointer to GL object list
  const std::vector<amd::Memory*>& getMemList() const { return memObjects_; }
  bool validateMemory();
  virtual bool processGLResource(device::Memory* mem) = 0;
};

class AcquireExtObjectsCommand : public ExtObjectsCommand {
 public:
  //! Construct a new AcquireExtObjectsCommand
  AcquireExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                           uint32_t num_objects, const std::vector<amd::Memory*>& memoryObjects,
                           cl_command_type type)
      : ExtObjectsCommand(queue, eventWaitList, num_objects, memoryObjects, type) {}

  virtual void submit(device::VirtualDevice& device) { device.submitAcquireExtObjects(*this); }

  virtual bool processGLResource(device::Memory* mem);
};

class ReleaseExtObjectsCommand : public ExtObjectsCommand {
 public:
  //! Construct a new ReleaseExtObjectsCommand
  ReleaseExtObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                           uint32_t num_objects, const std::vector<amd::Memory*>& memoryObjects,
                           cl_command_type type)
      : ExtObjectsCommand(queue, eventWaitList, num_objects, memoryObjects, type) {}

  virtual void submit(device::VirtualDevice& device) { device.submitReleaseExtObjects(*this); }

  virtual bool processGLResource(device::Memory* mem);
};

class PerfCounterCommand : public Command {
 public:
  typedef std::vector<PerfCounter*> PerfCounterList;

  enum State {
    Begin = 0,  //!< Issue a begin command
    End = 1     //!< Issue an end command
  };

  //! Construct a new PerfCounterCommand
  PerfCounterCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                     const PerfCounterList& counterList, State state)
      : Command(queue, 1, eventWaitList), counterList_(counterList), state_(state) {
    for (uint i = 0; i < counterList_.size(); ++i) {
      counterList_[i]->retain();
    }
  }

  void releaseResources() {
    for (uint i = 0; i < counterList_.size(); ++i) {
      counterList_[i]->release();
    }
    Command::releaseResources();
  }

  //! Gets the number of PerfCounter objects
  size_t getNumCounters() const { return counterList_.size(); }

  //! Gets the list of all counters
  const PerfCounterList& getCounters() const { return counterList_; }

  //! Gets the performance counter state
  State getState() const { return state_; }

  //! Process the command on the device queue
  virtual void submit(device::VirtualDevice& device) { device.submitPerfCounter(*this); }

 private:
  PerfCounterList counterList_;  //!< The list of performance counters
  State state_;                  //!< State of the issued command
};

/*! \brief      Thread Trace memory objects command.
 *
 *  \details    Used for bindig memory objects to therad trace mechanism.
 */
class ThreadTraceMemObjectsCommand : public Command {
 public:
  //! Construct a new ThreadTraceMemObjectsCommand
  ThreadTraceMemObjectsCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                               size_t numMemoryObjects, const cl_mem* memoryObjects,
                               size_t sizeMemoryObject, ThreadTrace& threadTrace,
                               cl_command_type type)
      : Command(queue, type, eventWaitList),
        sizeMemObjects_(sizeMemoryObject),
        threadTrace_(threadTrace) {
    memObjects_.resize(numMemoryObjects);
    for (size_t i = 0; i < numMemoryObjects; ++i) {
      Memory* obj = as_amd(memoryObjects[i]);
      obj->retain();
      memObjects_[i] = obj;
    }
    threadTrace_.retain();
  }
  //! Release all resources associated with this command
  void releaseResources() {
    threadTrace_.release();
    for (const auto& itr : memObjects_) {
      itr->release();
    }
    Command::releaseResources();
  }

  //! Get number of CL memory objects
  uint32_t getNumObjects() { return (uint32_t)memObjects_.size(); }

  //! Get pointer to CL memory object list
  const std::vector<amd::Memory*>& getMemList() const { return memObjects_; }

  //! Submit command to bind memory object to the Thread Trace mechanism
  virtual void submit(device::VirtualDevice& device) { device.submitThreadTraceMemObjects(*this); }

  //! Return the thread trace object.
  ThreadTrace& getThreadTrace() const { return threadTrace_; }

  //! Get memory object size
  const size_t getMemoryObjectSize() const { return sizeMemObjects_; }

  //! Validate memory bound to the thread thrace
  bool validateMemory();

 private:
  std::vector<amd::Memory*> memObjects_;  //!< The list of memory objects,bound to the thread trace
  size_t sizeMemObjects_;     //!< The size of each memory object from memObjects_ list (all memory
                              //! objects have the smae size)
  ThreadTrace& threadTrace_;  //!< The Thread Trace object
};

/*! \brief      Thread Trace command.
 *
 *  \details    Used for issue begin/end/pause/resume for therad trace object.
 */
class ThreadTraceCommand : public Command {
 private:
  void* threadTraceConfig_;

 public:
  enum State {
    Begin = 0,  //!< Issue a begin command
    End = 1,    //!< Issue an end command
    Pause = 2,  //!< Issue a pause command
    Resume = 3  //!< Issue a resume command
  };

  //! Construct a new ThreadTraceCommand
  ThreadTraceCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                     const void* threadTraceConfig, ThreadTrace& threadTrace, State state,
                     cl_command_type type)
      : Command(queue, type, eventWaitList), threadTrace_(threadTrace), state_(state) {
    const unsigned int size = *static_cast<const unsigned int*>(threadTraceConfig);
    threadTraceConfig_ = static_cast<void*>(new char[size]);
    if (threadTraceConfig_) {
      memcpy(threadTraceConfig_, threadTraceConfig, size);
    }
    threadTrace_.retain();
  }

  //! Release all resources associated with this command
  void releaseResources() {
    threadTrace_.release();
    Command::releaseResources();
  }

  //! Get the thread trace object
  ThreadTrace& getThreadTrace() const { return threadTrace_; }

  //! Get the thread trace command state
  State getState() const { return state_; }

  //! Process the command on the device queue
  virtual void submit(device::VirtualDevice& device) { device.submitThreadTrace(*this); }
  // Accessor methods
  void* threadTraceConfig() const { return threadTraceConfig_; }

 private:
  ThreadTrace& threadTrace_;  //!< The list of performance counters
  State state_;               //!< State of the issued command
};

class SignalCommand : public OneMemoryArgCommand {
 private:
  uint32_t markerValue_;
  uint64_t markerOffset_;

 public:
  SignalCommand(HostQueue& queue, cl_command_type cmdType, const EventWaitList& eventWaitList,
                Memory& memory, uint32_t value, uint64_t offset = 0)
      : OneMemoryArgCommand(queue, cmdType, eventWaitList, memory),
        markerValue_(value),
        markerOffset_(offset) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSignal(*this); }

  const uint32_t markerValue() { return markerValue_; }
  Memory& memory() { return *memory_; }
  const uint64_t markerOffset() { return markerOffset_; }
};

class MakeBuffersResidentCommand : public Command {
 private:
  std::vector<amd::Memory*> memObjects_;
  cl_bus_address_amd* busAddresses_;

 public:
  MakeBuffersResidentCommand(HostQueue& queue, cl_command_type type,
                             const EventWaitList& eventWaitList,
                             const std::vector<amd::Memory*>& memObjects,
                             cl_bus_address_amd* busAddr)
      : Command(queue, type, eventWaitList), busAddresses_(busAddr) {
    for (const auto& it : memObjects) {
      it->retain();
      memObjects_.push_back(it);
    }
  }

  virtual void submit(device::VirtualDevice& device) { device.submitMakeBuffersResident(*this); }

  void releaseResources() {
    for (const auto& it : memObjects_) {
      it->release();
    }
    Command::releaseResources();
  }

  bool validateMemory();
  const std::vector<amd::Memory*>& memObjects() const { return memObjects_; }
  cl_bus_address_amd* busAddress() const { return busAddresses_; }
};

//! A deallocation command used to free SVM or system pointers.
class SvmFreeMemoryCommand : public Command {
 public:
  typedef void(CL_CALLBACK* freeCallBack)(cl_command_queue, uint32_t, void**, void*);

 private:
  std::vector<void*> svmPointers_;  //!< List of pointers to deallocate
  freeCallBack pfnFreeFunc_;        //!< User-defined deallocation callback
  void* userData_;                  //!< Data passed to user-defined callback

 public:
  SvmFreeMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList,
                       uint32_t numSvmPointers, void** svmPointers, freeCallBack pfnFreeFunc,
                       void* userData)
      : Command(queue, CL_COMMAND_SVM_FREE, eventWaitList),
        //! We copy svmPointers since it can be reused/deallocated after
        //  command creation
        svmPointers_(svmPointers, svmPointers + numSvmPointers),
        pfnFreeFunc_(pfnFreeFunc),
        userData_(userData) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmFreeMemory(*this); }

  std::vector<void*>& svmPointers() { return svmPointers_; }

  freeCallBack pfnFreeFunc() const { return pfnFreeFunc_; }

  void* userData() const { return userData_; }
};

//! A copy command where the origin and destination memory locations are SVM
// pointers.
class SvmCopyMemoryCommand : public Command {
 private:
  void* dst_;        //!< Destination pointer
  const void* src_;  //!< Source pointer
  size_t srcSize_;   //!< Size (in bytes) of the source buffer

 public:
  SvmCopyMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, void* dst,
                       const void* src, size_t srcSize)
      : Command(queue, CL_COMMAND_SVM_MEMCPY, eventWaitList),
        dst_(dst),
        src_(src),
        srcSize_(srcSize) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmCopyMemory(*this); }

  void* dst() const { return dst_; }

  const void* src() const { return src_; }

  size_t srcSize() const { return srcSize_; }
};

//! A fill command where the pattern and destination memory locations are SVM
// pointers.
class SvmFillMemoryCommand : public Command {
 private:
  void* dst_;                                           //!< Destination pointer
  char pattern_[FillMemoryCommand::MaxFillPatterSize];  //!< The fill pattern
  size_t patternSize_;                                  //!< Pattern size
  size_t times_;                                        //!< Number of times to fill the
  //   destination buffer with the source buffer

 public:
  SvmFillMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, void* dst,
                       const void* pattern, size_t patternSize, size_t size)
      : Command(queue, CL_COMMAND_SVM_MEMFILL, eventWaitList),
        dst_(dst),
        patternSize_(patternSize),
        times_(size / patternSize) {
    assert(amd::isMultipleOf(size, patternSize));
    //! We copy the pattern buffer since it can be reused/deallocated after
    //  command creation
    memcpy(pattern_, pattern, patternSize);
  }

  virtual void submit(device::VirtualDevice& device) { device.submitSvmFillMemory(*this); }

  void* dst() const { return dst_; }

  const char* pattern() const { return pattern_; }

  size_t patternSize() const { return patternSize_; }

  size_t times() const { return times_; }
};

/*! \brief A map memory command where the pointer to be mapped is a SVM shared
 * buffer
 */
class SvmMapMemoryCommand : public Command {
 private:
  Memory* svmMem_;  //!< the pointer to the amd::Memory object corresponding the svm pointer mapped
  Coord3D size_;    //!< the map size
  Coord3D origin_;  //!< the origin of the mapped svm pointer shift from the beginning of svm space
                    //! allocated
  cl_map_flags flags_;  //!< map flags
  void* svmPtr_;

 public:
  SvmMapMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, Memory* svmMem,
                      const size_t size, const size_t offset, cl_map_flags flags, void* svmPtr)
      : Command(queue, CL_COMMAND_SVM_MAP, eventWaitList),
        svmMem_(svmMem),
        size_(size),
        origin_(offset),
        flags_(flags),
        svmPtr_(svmPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmMapMemory(*this); }

  Memory* getSvmMem() const { return svmMem_; }

  Coord3D size() const { return size_; }

  cl_map_flags mapFlags() const { return flags_; }

  Coord3D origin() const { return origin_; }

  void* svmPtr() const { return svmPtr_; }

  bool isEntireMemory() const;
};

/*! \brief An unmap memory command where the unmapped pointer is a SVM shared
 * buffer
 */
class SvmUnmapMemoryCommand : public Command {
 private:
  Memory* svmMem_;  //!< the pointer to the amd::Memory object corresponding the svm pointer mapped
  void* svmPtr_;    //!< SVM pointer

 public:
  SvmUnmapMemoryCommand(HostQueue& queue, const EventWaitList& eventWaitList, Memory* svmMem,
                        void* svmPtr)
      : Command(queue, CL_COMMAND_SVM_UNMAP, eventWaitList), svmMem_(svmMem), svmPtr_(svmPtr) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmUnmapMemory(*this); }

  Memory* getSvmMem() const { return svmMem_; }

  void* svmPtr() const { return svmPtr_; }
};

/*! \brief      A P2P copy memory command
 *
 *  \details    Used for buffers only. Backends are expected
 *              to handle any required translation. Buffers are treated
 *              as 1D structures so origin_[0] and size_[0] are
 *              equivalent to offset_ and count_ respectively.
 */
class CopyMemoryP2PCommand : public CopyMemoryCommand {
 public:
  CopyMemoryP2PCommand(HostQueue& queue, cl_command_type cmdType,
                       const EventWaitList& eventWaitList, Memory& srcMemory, Memory& dstMemory,
                       Coord3D srcOrigin, Coord3D dstOrigin, Coord3D size)
      : CopyMemoryCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory, srcOrigin, dstOrigin,
                          size) {}

  CopyMemoryP2PCommand(HostQueue& queue, cl_command_type cmdType,
                       const EventWaitList& eventWaitList, Memory& srcMemory, Memory& dstMemory,
                       Coord3D srcOrigin, Coord3D dstOrigin, Coord3D size,
                       const BufferRect& srcRect, const BufferRect& dstRect,
                       amd::CopyMetadata copyMetadata = amd::CopyMetadata())
      : CopyMemoryCommand(queue, cmdType, eventWaitList, srcMemory, dstMemory, srcOrigin, dstOrigin,
                          size, srcRect, dstRect) {}

  virtual void submit(device::VirtualDevice& device) { device.submitCopyMemoryP2P(*this); }

  bool validateMemory();
};

/*! \brief      Prefetch command for SVM memory
 *
 *  \details    Prefetches SVM memory into the destination device or CPU
 */
class SvmPrefetchAsyncCommand : public Command {
  const void* dev_ptr_;  //!< Device pointer to memory for prefetch
  size_t count_;         //!< the size for prefetch
  bool cpu_access_;      //!< Prefetch data into CPU location
  int numa_id_;          //!< Host NUMA node id
  amd::Device* dev_;     //!< Destination device to prefetch to

 public:
  SvmPrefetchAsyncCommand(HostQueue& queue, const EventWaitList& eventWaitList, const void* dev_ptr,
                          size_t count, amd::Device* dev, bool cpu_access, int numa_id)
      : Command(queue, 1, eventWaitList),
        dev_ptr_(dev_ptr),
        count_(count),
        cpu_access_(cpu_access),
        dev_(dev),
        numa_id_(numa_id) {}

  virtual void submit(device::VirtualDevice& device) { device.submitSvmPrefetchAsync(*this); }

  bool validateMemory();

  const void* dev_ptr() const { return dev_ptr_; }
  size_t count() const { return count_; }
  amd::Device* device() const { return dev_; }
  size_t cpu_access() const { return cpu_access_; }
  int numa_id() const { return numa_id_; }
};

/*! \brief  A virtual map memory command.
 *
 */

class VirtualMapCommand : public Command {
 private:
  const void* ptr_;  //!< Virtual address to map to the memory

 protected:
  Memory* memory_;  //!< Memory to map, nullptr means unmap
  size_t size_;     //!< Size of the mapping in bytes

 public:
  //! Construct a new VirtualMapCommand
  VirtualMapCommand(HostQueue& queue, const EventWaitList& eventWaitList, void* ptr, size_t size,
                    Memory* memory)
      : Command(queue, 1, eventWaitList), ptr_(ptr), size_(size), memory_(memory) {
    // Sanity checks
    assert(size > 0 && "invalid");
    if (memory_) memory_->retain();
  }

  virtual void releaseResources() {
    if (memory_) memory_->release();
    DEBUG_ONLY(memory_ = nullptr);
    Command::releaseResources();
  }

  virtual void submit(device::VirtualDevice& device) { device.submitVirtualMap(*this); }

  //! Read the memory object
  Memory* memory() const { return memory_; }
  //! Read the size
  size_t size() const { return size_; }
  //! Read the pointer
  const void* ptr() const { return ptr_; }
};

//! Union used in memory suballocator, must be updated with the new commands
union ComputeCommand {
  ReadMemoryCommand cmd0;
  WriteMemoryCommand cmd1;
  FillMemoryCommand cmd2;
  CopyMemoryCommand cmd3;
  MapMemoryCommand cmd4;
  UnmapMemoryCommand cmd5;
  MigrateMemObjectsCommand cmd6;
  NDRangeKernelCommand cmd7;
  NativeFnCommand cmd8;
  ExternalSemaphoreCmd cmd9;
  Marker cmd10;
  AccumulateCommand cmd11;
  AcquireExtObjectsCommand cmd13;
  ReleaseExtObjectsCommand cmd14;
  PerfCounterCommand cmd15;
  ThreadTraceMemObjectsCommand cmd16;
  ThreadTraceCommand cmd17;
  SignalCommand cmd18;
  MakeBuffersResidentCommand cmd19;
  SvmFreeMemoryCommand cmd20;
  SvmCopyMemoryCommand cmd21;
  SvmFillMemoryCommand cmd22;
  SvmMapMemoryCommand cmd23;
  SvmUnmapMemoryCommand cmd24;
  CopyMemoryP2PCommand cmd25;
  SvmPrefetchAsyncCommand cmd26;
  VirtualMapCommand cmd27;
  BatchMemoryOperationCommand cmd28;
  ComputeCommand() {}
  ~ComputeCommand() {}
};

/*! @}
 *  @}
 */

}  // namespace amd

#endif /*COMMAND_HPP_*/
