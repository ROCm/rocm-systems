/* Copyright (c) 2012 - 2021 Advanced Micro Devices, Inc.

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

/*! \file commandqueue.hpp
 *  \brief  Declarations CommandQueue object.
 *
 * \author Laurent Morichetti
 * \date   October 2008
 */

#ifndef COMMAND_QUEUE_HPP_
#define COMMAND_QUEUE_HPP_

#include "thread/thread.hpp"
#include "platform/object.hpp"
#include "platform/command.hpp"
/*! \brief Holds commands that will be executed on a specific device.
 *
 *  \details A command queue is created on a specific device in
 *  a Context. A new virtual device will be instantiated from the given
 *  device and an execution environment (a thread) will be created to run
 *  the CommandQueue::loop() function.
 */

namespace amd {

class HostQueue;
class DeviceQueue;

class CommandQueue : public RuntimeObject {
 public:
  static constexpr uint RealTimeDisabled = 0xffffffff;
  enum class Priority : uint { Low = 0, Normal, Medium, High };

  struct Properties {
    typedef cl_command_queue_properties value_type;
    const value_type mask_;
    value_type value_;

    Properties(value_type mask, value_type value) : mask_(mask), value_(value & mask) {}

    bool set(value_type bits) {
      if ((mask_ & bits) != bits) {
        return false;
      }
      value_ |= bits;
      return true;
    }

    bool clear(value_type bits) {
      if ((mask_ & bits) != bits) {
        return false;
      }
      value_ &= ~bits;
      return true;
    }

    bool test(value_type bits) const { return (value_ & bits) != 0; }
  };

  //! Return the context this command queue is part of.
  Context& context() const { return context_(); }

  //! Return the device for this command queue.
  Device& device() const { return device_; }

  //! Return the command queue properties.
  Properties properties() const { return properties_; }
  Properties& properties() { return properties_; }

  //! Returns the base class object
  CommandQueue* asCommandQueue() { return this; }

  virtual ~CommandQueue() {}

  //! Returns TRUE if the object was successfully created
  virtual bool create() = 0;

  //! RTTI internal implementation
  virtual ObjectType objectType() const { return ObjectTypeQueue; }

  //! Rturns HostQueue object
  virtual HostQueue* asHostQueue() { return NULL; }

  //! Returns DeviceQueue object
  virtual DeviceQueue* asDeviceQueue() { return NULL; }

  //! Returns the number or requested real time CUs
  uint rtCUs() const { return rtCUs_; }

  //! Returns the queue priority
  Priority priority() const { return priority_; }

  //! Returns the CU mask array
  const std::vector<uint32_t>& cuMask() const { return cuMask_; }

  //! Returns the queue lock
  Monitor& lock() { return queueLock_; }

 protected:
  //! CommandQueue constructor is protected
  //! to keep the CommandQueue class as a virtual interface
  CommandQueue(Context& context,                         //!< Context object
               Device& device,                           //!< Device object
               cl_command_queue_properties properties,   //!< Queue properties
               cl_command_queue_properties propMask,     //!< Queue properties mask
               uint rtCUs = RealTimeDisabled,            //!< Avaialble real time compute units
               Priority priority = Priority::Normal,     //!< Queue priority
               const std::vector<uint32_t>& cuMask = {}  //!< CU mask
               )
      : properties_(propMask, properties),
        rtCUs_(rtCUs),
        priority_(priority),
        queueLock_(),
        lastCmdLock_(),
        device_(device),
        context_(context),
        cuMask_(cuMask) {}

  Properties properties_;               //!< Queue properties
  uint rtCUs_;                          //!< The number of used RT compute units
  Priority priority_;                   //!< Queue priority
  Monitor queueLock_;                   //!< Lock protecting the queue
  Monitor lastCmdLock_;                 //!< Lock protecting the last queued command
  Device& device_;                      //!< The device
  SharedReference<Context> context_;    //!< The context of this command queue
  const std::vector<uint32_t> cuMask_;  //!< The CU mask

 private:
  //! Disable copy constructor
  CommandQueue(const CommandQueue&);

  //! Disable assignment
  CommandQueue& operator=(const CommandQueue&);
};


class HostQueue : public CommandQueue {
  class Thread : public amd::Thread {
   public:
    //! True if this command queue thread is accepting commands.
    volatile bool acceptingCommands_;

    //! Create a new thread
    Thread()
        : amd::Thread("Command Queue Thread", CQ_THREAD_STACK_SIZE, !AMD_DIRECT_DISPATCH),
          acceptingCommands_(false),
          virtualDevice_(nullptr) {}

    //! The command queue thread entry point.
    void run(void* data) {
      HostQueue* queue = static_cast<HostQueue*>(data);
      virtualDevice_ = queue->device().createVirtualDevice(queue);
      if (virtualDevice_ != nullptr) {
        queue->loop(virtualDevice_);
        Release();
      } else {
        acceptingCommands_ = false;
        queue->flush();
      }
    }

    void Init(HostQueue* queue) {
      virtualDevice_ = queue->device().createVirtualDevice(queue);
      if (virtualDevice_ != nullptr) {
        acceptingCommands_ = true;
      }
    }

    void Release() const { delete virtualDevice_; }

    //! Get virtual device for the current thread
    device::VirtualDevice* vdev() const { return virtualDevice_; }

   private:
    device::VirtualDevice* virtualDevice_;  //!< Virtual device for this thread

  } thread_;  //!< The command queue thread instance.

 private:
  ConcurrentLinkedQueue<Command*> queue_;  //!< The queue.

  Command* lastEnqueueCommand_;  //!< The last submitted command

  //! Await commands and execute them as they become ready.
  void loop(device::VirtualDevice* virtualDevice);

 protected:
  virtual bool terminate();

 public:
  /*! \brief Construct a new host queue.
   *
   * \note A new virtual device instance will be created from the
   * given device.
   */
  HostQueue(Context& context, Device& device, cl_command_queue_properties properties,
            uint queueRTCUs = 0, Priority priority = Priority::Normal,
            const std::vector<uint32_t>& cuMask = {});

  //! Returns TRUE if this command queue can accept commands.
  virtual bool create() { return thread_.acceptingCommands_; }

  //! Append the given command to the queue.
  void append(Command& command);

  //! Return the thread object running the command loop.
  const Thread& thread() const { return thread_; }

  //! Signal to start processing the commands in the queue.
  void flush() {
    ScopedLock sl(queueLock_);
    queueLock_.notify();
  }

  //! Finish all queued commands
  void finish(bool cpu_wait = false);

  //! Wait until finish of one command
  void finishCommand(Command* command);

  //! Check if hostQueue empty snapshot
  bool isEmpty();

  //! Get virtual device for the current command queue
  device::VirtualDevice* vdev() const { return thread_.vdev(); }

  //! Return the current queue as the HostQueue
  virtual HostQueue* asHostQueue() { return this; }

  //! Get last enqueued command
  Command* getLastQueuedCommand(bool retain);

  //! Get the submitted batch
  Command* GetSubmissionBatch() const { return head_; }

  //! Get the current batch size
  size_t GetSubmissionBatchSize() const { return size_; }

  //! Insert a command into the linked list of submitted commands
  void FormSubmissionBatch(Command* command) {
    // Insert the command to the linked list.
    if (nullptr == head_) {  // if the list is empty
      head_ = tail_ = command;
    } else {
      tail_->setNext(command);
      tail_ = command;
    }
    size_++;
    command->setStatus(CL_SUBMITTED);
    command->retain();
    // @note: runtime needs double retain in order to maintain the batch,
    // because setStatus(COMPLETE) releases command and batch update may have
    // an invalid access
    command->retain();

    // Release the last command in the batch
    if (lastEnqueueCommand_ != nullptr) {
      lastEnqueueCommand_->release();
    } else {
      // The queue becomes active. Add it to the set of activeQueues.
      device_.addToActiveQueues(this);
    }

    // Extra retain for the last command
    command->retain();

    lastEnqueueCommand_ = command;
  }

  //! Flushes submitted commands if the batch size significantly grew
  void FlushSubmissionBatch(Command* command) {
    if (size_ > DEBUG_CLR_MAX_BATCH_SIZE) {
      command->notifyCmdQueue();
    }
  }
  //! Reset the command batch list
  void ResetSubmissionBatch() {
    head_ = nullptr;
    size_ = 0;
  }

  //! Set queue status
  void SetQueueStatus() { isActive_ = true; }

  //! Get queue status
  bool GetQueueStatus() { return isActive_; }

  //! Set the force destory to terminate queue without checking last command
  void SetForceDestroy(bool forceDestroy) { forceDestroy_ = forceDestroy; }

  uint64_t getQueueID() { return thread_.vdev()->getQueueID(); }

  //! Returns Synchronization Policy for the current stream
  amd::SyncPolicy GetSyncPolicy() const { return sync_policy_; }
  //! Set Synchronization Policy used by Queue
  void SetSyncPolicy(amd::SyncPolicy value) { sync_policy_ = value; }

 private:
  Command* head_;    //!< Head of the batch list
  Command* tail_;    //!< Tail of the batch list
  size_t size_ = 0;  //!< The current batch size

  //! True if this command queue is active
  bool isActive_;
  bool forceDestroy_ = false;  //!< Destroy the queue in the current state

  amd::SyncPolicy sync_policy_;  //!< Used for controlling stream synchronization
};

class DeviceQueue : public CommandQueue {
 public:
  DeviceQueue(Context& context,                        //!< Context object
              Device& device,                          //!< Device object
              cl_command_queue_properties properties,  //!< Queue properties
              uint size                                //!< Device queue size
              )
      : CommandQueue(context, device, properties,
                     device.info().queueOnDeviceProperties_ | CL_QUEUE_ON_DEVICE |
                         CL_QUEUE_ON_DEVICE_DEFAULT),
        size_(size),
        virtualDevice_(NULL) {}

  virtual ~DeviceQueue();

  //! Returns TRUE if device queue was successfully created
  virtual bool create();

  //! Return the current queue as the DeviceQueue
  virtual DeviceQueue* asDeviceQueue() { return this; }

  //! Returns the size of device queue
  uint size() const { return size_; }

  //! Returns virtual device for this device queue
  device::VirtualDevice* vDev() const { return virtualDevice_; }

 private:
  uint size_;                             //!< Device queue size
  device::VirtualDevice* virtualDevice_;  //!< Virtual device for this queue
};
}  // namespace amd

#endif  // COMMAND_QUEUE_HPP_
