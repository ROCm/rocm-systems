/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "utils/debug.hpp"
#include "top.hpp"
#include "utils/flags.hpp"

#include "device/devhcmessages.hpp"
#include "device/devhostcall.hpp"
#include "device/devrpc.hpp"
#include "device/devsignal.hpp"

#include "os/os.hpp"
#include "thread/monitor.hpp"
#include "utils/util.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"

#include <assert.h>
#include <string.h>
#include <algorithm>
#include <new>
#include <set>

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/devsanitizer.hpp"
#endif
#endif

namespace amd {

#ifndef USE_NEW_HOSTCALL_IMPL
PacketHeader* HostcallBuffer::getHeader(uint64_t ptr) const {
  return headers_ + (ptr & index_mask_);
}

Payload* HostcallBuffer::getPayload(uint64_t ptr) const { return payloads_ + (ptr & index_mask_); }

static uint32_t setControlField(uint32_t control, uint8_t offset, uint8_t width, uint32_t value) {
  uint32_t mask = ~(((1 << width) - 1) << offset);
  control &= mask;
  return control | (value << offset);
}

static uint32_t resetReadyFlag(uint32_t control) {
  return setControlField(control, CONTROL_OFFSET_READY_FLAG, CONTROL_WIDTH_READY_FLAG, 0);
}
#endif  // !USE_NEW_HOSTCALL_IMPL

/** \brief Signature for pointer accepted by the function call service.
 *  \param output Pointer to output arguments.
 *  \param input Pointer to input arguments.
 *
 *  The function can accept up to seven 64-bit arguments via the
 *  #input pointer, and can produce up to two 64-bit arguments via the
 *  #output pointer. The contents of these arguments are defined by
 *  the function being invoked.
 */
typedef void (*HostcallFunctionCall)(uint64_t* output, const uint64_t* input);

static void handlePayload(MessageHandler& messages, uint32_t service, uint64_t* payload,
                          const amd::Device& dev) {
  switch (service) {
    case SERVICE_FUNCTION_CALL: {
      uint64_t output[2];
      auto fptr = reinterpret_cast<HostcallFunctionCall>(payload[0]);
      fptr(output, payload + 1);
      memcpy(payload, output, sizeof(output));
      return;
    }
    case SERVICE_PRINTF:
      if (!messages.handlePayload(service, payload)) {
        ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: invalid request for service \"%d\".",
                service);
        guarantee(false, "Hostcall: invalid service request %d \n", service);
      }
      return;
    case SERVICE_DEVMEM: {
      guarantee(payload[0] != 0 || payload[1] != 0, "Both payloads cannot be 0 \n");
      if (payload[0]) {
        amd::Memory* mem = amd::MemObjMap::FindMemObj(reinterpret_cast<void*>(payload[0]));
        if (mem) {
          const_cast<amd::Device*>(&dev)->RemoveHostcallMemory(mem);
          amd::MemObjMap::RemoveMemObj(reinterpret_cast<void*>(payload[0]));
          mem->release();
        } else {
          ClPrint(amd::LOG_ERROR, amd::LOG_ALWAYS, "Hostcall: Unknown pointer %p in devmem service",
                  payload[0]);
        }
      } else {
        amd::Context& ctx = dev.context();
        amd::Buffer* buf = new (ctx) amd::Buffer(ctx, CL_MEM_READ_WRITE, payload[1], NULL,
                                                 (payload[1] == 2 * Mi) ? 2 * Mi : 0);
        uint64_t va = 0;
        if (buf) {
          if (buf->create()) {
            device::Memory* dm = buf->getDeviceMemory(dev);
#ifdef USE_NEW_HOSTCALL_IMPL
            if (dm != nullptr) {
              va = dm->virtualAddress();
            }
            if (va != 0) {
              amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
              const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
            } else {
              buf->release();
            }
#else  // !USE_NEW_HOSTCALL_IMPL
            va = dm->virtualAddress();
            amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
            const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
#endif  // USE_NEW_HOSTCALL_IMPL
          } else {
            buf->release();
          }
        }
        payload[0] = va;
      }
      return;
    }
    default:
      guarantee(false, "Hostcall: no handler found for service ID %d \n", service);
      return;
  }
}

uint32_t getHostcallBufferAlignment() { return alignof(Payload); }

#ifdef USE_NEW_HOSTCALL_IMPL
static uintptr_t getDevicePhaseOffset() {
  return amd::alignUp(sizeof(HostcallBuffer), alignof(uint32_t));
}

static uintptr_t getHostPhaseOffset(uint32_t num_packets) {
  return amd::alignUp(getDevicePhaseOffset() + num_packets * sizeof(uint32_t), alignof(uint32_t));
}

static uintptr_t getHeaderOffset(uint32_t num_packets) {
  return amd::alignUp(getHostPhaseOffset(num_packets) + num_packets * sizeof(uint32_t),
                      alignof(PacketHeader));
}

static uintptr_t getPayloadOffset(uint32_t num_packets) {
  return amd::alignUp(getHeaderOffset(num_packets) + num_packets * sizeof(PacketHeader),
                      alignof(Payload));
}

size_t getHostcallBufferSize(uint32_t num_packets) {
  return getPayloadOffset(num_packets) + num_packets * sizeof(Payload);
}

bool HostcallBuffer::initialize(uint32_t num_packets, amd::Memory* occupied_mem) {
  if (occupied_mem == nullptr || device_ == nullptr) {
    return false;
  }

  device::Memory* dm = occupied_mem->getDeviceMemory(*device_);
  if (dm == nullptr || dm->virtualAddress() == 0) {
    return false;
  }

  auto base = reinterpret_cast<uint8_t*>(this);
  device_phase_ = reinterpret_cast<std::atomic<uint32_t>*>(base + getDevicePhaseOffset());
  host_phase_ = reinterpret_cast<std::atomic<uint32_t>*>(base + getHostPhaseOffset(num_packets));
  occupied_ = static_cast<uintptr_t>(dm->virtualAddress());
  headers_ = reinterpret_cast<PacketHeader*>(base + getHeaderOffset(num_packets));
  payloads_ = reinterpret_cast<Payload*>(base + getPayloadOffset(num_packets));
  doorbell_ = nullptr;
  num_packets_ = num_packets;
  occupied_mem_ = occupied_mem;
  scan_limit_ = num_packets;

  for (uint32_t i = 0; i < num_packets; ++i) {
    new (&device_phase_[i]) std::atomic<uint32_t>(0);
    new (&host_phase_[i]) std::atomic<uint32_t>(0);
  }
  return true;
}

ProcessResult HostcallBuffer::processPackets(MessageHandler& messages) {
  uint32_t new_limit = 0;
  for (uint32_t i = 0; i < scan_limit_; ++i) {
    uint32_t dp = device_phase_[i].load(std::memory_order_relaxed);
    uint32_t hp = host_phase_[i].load(std::memory_order_relaxed);

    if (dp == hp) {
      continue;
    }

    new_limit = i + 1;
    std::atomic_thread_fence(std::memory_order_acquire);

    auto* header = &headers_[i];
    auto* payload = &payloads_[i];
    auto service = header->service_;
    auto activemask = header->activemask_;

#if defined(__clang__)
#if __has_feature(address_sanitizer)
    if (service == SERVICE_SANITIZER) {
      handleSanitizerService(payload, activemask, device_, uri_locator);
      // activemask zeroed to avoid subsequent handling for each work-item.
      activemask = 0;
    }
#endif
#endif
    while (activemask) {
      auto wi = amd::leastBitSet(activemask);
      activemask ^= static_cast<decltype(activemask)>(1) << wi;
      auto slot = payload->slots[wi];
      handlePayload(messages, service, slot, *device_);
    }

    std::atomic_thread_fence(std::memory_order_release);
    host_phase_[i].store(hp ^ 1, std::memory_order_relaxed);
  }

  if (new_limit > 0) {
    scan_limit_ = new_limit;
    return ProcessResult::kProcessed;
  }
  return scan_limit_ < num_packets_ ? ProcessResult::kIdleNarrow : ProcessResult::kIdleFull;
}
#else  // !USE_NEW_HOSTCALL_IMPL
void HostcallBuffer::processPackets(MessageHandler& messages) {
  // Grab the entire ready stack and set the top to 0. New requests from the
  // device will continue pushing on the stack while we process the packets that
  // we have grabbed.

  uint64_t ready_stack = std::atomic_exchange_explicit(&ready_stack_, static_cast<uint64_t>(0),
                                                       std::memory_order_acquire);
  if (!ready_stack) {
    return;
  }

  // Each wave can submit at most one packet at a time. The ready stack cannot
  // contain multiple packets from the same wave, so consuming ready packets in
  // a latest-first order does not affect ordering of hostcall within a wave.
  for (decltype(ready_stack) iter = ready_stack, next = 0; iter; iter = next) {
    auto header = getHeader(iter);
    // Remember the next packet pointer, because we will no longer own the
    // current packet at the end of this loop.
    next = header->next_;

    auto service = header->service_;
    auto payload = getPayload(iter);
    auto activemask = header->activemask_;

#if defined(__clang__)
#if __has_feature(address_sanitizer)
    if (service == SERVICE_SANITIZER) {
      handleSanitizerService(payload, activemask, device_, uri_locator);
      // activemask zeroed to avoid subsequent handling for each work-item.
      activemask = 0;
    }
#endif
#endif
    while (activemask) {
      auto wi = amd::leastBitSet(activemask);
      activemask ^= static_cast<decltype(activemask)>(1) << wi;
      auto slot = payload->slots[wi];
      handlePayload(messages, service, slot, *device_);
    }

    header->control_.store(resetReadyFlag(header->control_), std::memory_order_release);
  }
}

static uintptr_t getHeaderStart() {
  return amd::alignUp(sizeof(HostcallBuffer), alignof(PacketHeader));
}

static uintptr_t getPayloadStart(uint32_t num_packets) {
  auto header_start = getHeaderStart();
  auto header_end = header_start + sizeof(PacketHeader) * num_packets;
  return amd::alignUp(header_end, alignof(Payload));
}

size_t getHostcallBufferSize(uint32_t num_packets) {
  size_t buffer_size = getPayloadStart(num_packets);
  buffer_size += num_packets * sizeof(Payload);
  return buffer_size;
}

static uint64_t getIndexMask(uint32_t num_packets) {
  // The number of packets is at least equal to the maximum number of waves
  // supported by the device. That means we do not need to account for the
  // border cases where num_packets is zero or one.
  assert(num_packets > 1);
  if (!amd::isPowerOfTwo(num_packets)) {
    num_packets = amd::nextPowerOfTwo(num_packets);
  }
  return num_packets - 1;
}

void HostcallBuffer::initialize(uint32_t num_packets) {
  auto base = reinterpret_cast<uint8_t*>(this);
  headers_ = reinterpret_cast<PacketHeader*>((base + getHeaderStart()));
  payloads_ = reinterpret_cast<Payload*>((base + getPayloadStart(num_packets)));
  index_mask_ = getIndexMask(num_packets);

  // The null pointer is identical to (uint64_t)0. When using tagged pointers,
  // the tag and the index part of the array must never be zero at the same
  // time. In the initialized free stack, headers[1].next points to headers[0],
  // which has index 0. We initialize this pointer to have a tag of 1.
  uint64_t next = index_mask_ + 1;

  // Initialize the free stack.
  headers_[0].next_ = 0;
  for (uint32_t ii = 1; ii != num_packets; ++ii) {
    headers_[ii].next_ = next;
    next = ii;
  }
  free_stack_ = next;
  ready_stack_ = 0;
}
#endif  // USE_NEW_HOSTCALL_IMPL

/** \brief Manage a unique listener thread and its associated buffers.
 */
class HostcallListener {
  std::set<HostcallBuffer*> buffers_;
  std::set<RpcBufferInfo*> rpc_buffers_;
  device::Signal* doorbell_ = nullptr;
  device::Signal* rpc_doorbell_ = nullptr;
  MessageHandler messages_;
  // Keep track of devices for which signal creation have already been done
  std::set<const amd::Device*> devices_;
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  device::UriLocator* urilocator = nullptr;
#endif
#endif
  class Thread : public amd::Thread {
   public:
    // Take the system default stack: glibc allocates a thread's static TLS block inside the stack
    // requested through pthread_attr_setstacksize() and refuses the request when the two do not
    // fit, which is easy to hit from a process that links libraries with large PT_TLS segments.
    Thread() : amd::Thread("Hostcall Listener Thread") {}

    //! The hostcall listener thread entry point.
    void run(void* data) {
      auto listener = reinterpret_cast<HostcallListener*>(data);
      listener->consumePackets();
    }
  } thread_;  //!< The hostcall listener thread.

  void consumePackets();

 public:
  /** \brief Add a buffer to the listener.
   *
   *  Behaviour is undefined if:
   *  - hostcall_initialize_buffer() was not invoked successfully on
   *    the buffer prior to registration.
   *  - The same buffer is registered with multiple listeners.
   *  - The same buffer is associated with more than one hardware queue.
   */
  void addBuffer(HostcallBuffer* buffer);

  /** \brief Remove a buffer that is no longer in use.
   *
   *  The buffer can be reused after removal.  Behaviour is undefined if the
   *  buffer is freed without first removing it.
   */
  void removeBuffer(HostcallBuffer* buffer);

  /* \brief Return true if no buffers are registered.
   */
  void addRpcBuffer(RpcBufferInfo* info);
  void removeRpcBufferByPtr(void* raw);
  void flushRpcBufferByPtr(void* raw);
  bool idle() const { return buffers_.empty() && rpc_buffers_.empty(); }

  void terminate();
  bool initSignal(const amd::Device& dev);
  bool initDevice(const amd::Device& dev);
};

HostcallListener* hostcallListener = nullptr;
extern amd::Monitor listenerLock;

static bool listenerTerminating = false;
#ifdef USE_NEW_HOSTCALL_IMPL
constexpr static uint32_t kYieldSpins = 128;
constexpr static uint64_t kSignalTimeout = K * K;
#else  // !USE_NEW_HOSTCALL_IMPL
constexpr static uint64_t kTimeoutFloor = K * K * 4;
constexpr static uint64_t kTimeoutCeil = K * K * 16;
#endif  // USE_NEW_HOSTCALL_IMPL
static struct Init {
  enum class State { kDefault = 0, kInit, kDestroy, kExit };
  volatile State state = State::kDefault;
  ~Init() {
    if (state == State::kInit) {
      state = State::kDestroy;
      // @note: Under Linux thread destruction can be delayed and
      // ROCR may crash in a wait for event occasionally. Hence, runtime needs
      // an early exit. The logic isn't required for Windows.
      while (IS_LINUX && (state == State::kDestroy)) {
      }
    }
  }
} kHostThreadActive;
#ifdef USE_NEW_HOSTCALL_IMPL
void HostcallListener::consumePackets() {
  uint64_t signal_value = SIGNAL_INIT;
  kHostThreadActive.state = Init::State::kInit;
  uint32_t yield_spins = 0;

  while (true) {
    if (kHostThreadActive.state == Init::State::kDestroy) {
      kHostThreadActive.state = Init::State::kExit;
      return;
    }

    ProcessResult result = ProcessResult::kIdleFull;
    bool rpc_busy = false;
    {
      amd::ScopedLock lock{listenerLock};
      for (auto buf : buffers_) {
        result = std::min(result, buf->processPackets(messages_));
      }
      if (result == ProcessResult::kIdleNarrow) {
        for (auto buf : buffers_) {
          buf->resetScanLimit();
        }
      }
      for (auto info : rpc_buffers_) {
        rpc_busy |= processRpcBuffer(info);
      }
    }

    if (result != ProcessResult::kIdleFull || rpc_busy) {
      if (++yield_spins >= kYieldSpins) {
        yield_spins = 0;
        amd::Os::yield();
      }
      continue;
    }

    yield_spins = 0;
    // RPC and hostcall use separate doorbells.
    if (!rpc_buffers_.empty()) {
      rpc_doorbell_->Wait(0, device::Signal::Condition::Ne, kSignalTimeout);
      continue;
    }
    uint64_t new_value =
        doorbell_->Wait(signal_value, device::Signal::Condition::Ne, kSignalTimeout);
    if (new_value != signal_value) {
      signal_value = new_value;
    }
    if (signal_value == SIGNAL_DONE) {
      return;
    }
  }
}
#else  // !USE_NEW_HOSTCALL_IMPL
void HostcallListener::consumePackets() {
  uint64_t timeout = kTimeoutFloor;
  uint64_t signal_value = SIGNAL_INIT;
  kHostThreadActive.state = Init::State::kInit;
  while (true) {
    while (true) {
      if (kHostThreadActive.state == Init::State::kDestroy) {
        kHostThreadActive.state = Init::State::kExit;
        return;
      }

      bool have_hostcall = !buffers_.empty();
      bool have_rpc = !rpc_buffers_.empty();

      if (have_hostcall && !have_rpc) {
        uint64_t new_value = doorbell_->Wait(signal_value, device::Signal::Condition::Ne, timeout);
        if (new_value != signal_value) {
          signal_value = new_value;
          timeout = std::max(kTimeoutFloor, timeout >> 1);
          break;
        }
        timeout = std::min(kTimeoutCeil, timeout << 1);
      } else if (!have_hostcall && have_rpc) {
        // RPC only — block until the device increments the doorbell.
        rpc_doorbell_->Wait(0, device::Signal::Condition::Ne, timeout);
        timeout = std::min(kTimeoutCeil, timeout << 1);
        break;
      } else {
        uint64_t new_value =
            doorbell_->Wait(signal_value, device::Signal::Condition::Ne, kTimeoutFloor);
        if (new_value != signal_value) {
          signal_value = new_value;
        }
        break;
      }
    }

    if (signal_value == SIGNAL_DONE) {
      return;
    }

    if (!idle()) {
      amd::ScopedLock lock{listenerLock};

      for (auto ii : buffers_) {
        ii->processPackets(messages_);
      }
      for (auto ii : rpc_buffers_) {
        processRpcBuffer(ii);
      }
    }
  }

  return;
}
#endif  // USE_NEW_HOSTCALL_IMPL

void HostcallListener::terminate() {
  if (thread_.state() >= Thread::FINISHED || amd::Os::isThreadAlive(thread_)) {
    kHostThreadActive.state = Init::State::kDestroy;
    doorbell_->Reset(SIGNAL_DONE);
    if (rpc_doorbell_) rpc_doorbell_->Reset(1);

    // FIXME_lmoriche: fix termination handshake
    while (thread_.state() < Thread::FINISHED) {
      amd::Os::yield();
    }
  }

#if defined(__clang__)
#if __has_feature(address_sanitizer)
  delete urilocator;
#endif
#endif
  delete rpc_doorbell_;
  delete doorbell_;
  devices_.clear();
}

void HostcallListener::addBuffer(HostcallBuffer* buffer) {
  assert(buffers_.count(buffer) == 0 && "buffer already present");
  buffer->setDoorbell(doorbell_->getHandle());
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  buffer->setUriLocator(urilocator);
#endif
#endif
  buffers_.insert(buffer);
}

void HostcallListener::removeBuffer(HostcallBuffer* buffer) {
  assert(buffers_.count(buffer) != 0 && "unknown buffer");
  buffers_.erase(buffer);
}

void HostcallListener::addRpcBuffer(RpcBufferInfo* info) {
  info->messages = &messages_;
  if (!rpc_doorbell_) {
    rpc_doorbell_ = info->device->createSignal();
#if defined(WITH_PAL_DEVICE) && !defined(_WIN32)
    auto ws = device::Signal::WaitState::Active;
#else
    auto ws = device::Signal::WaitState::Blocked;
#endif
    rpc_doorbell_->Init(*info->device, 0, ws);
  }
  initRpcDoorbell(info->buffer, rpc_doorbell_->getHandle());
  rpc_buffers_.insert(info);
}

void HostcallListener::removeRpcBufferByPtr(void* raw) {
  for (auto* b : rpc_buffers_) {
    if (b->buffer == raw) {
      rpc_buffers_.erase(b);
      delete b;
      return;
    }
  }
}

void HostcallListener::flushRpcBufferByPtr(void* raw) {
  for (auto* b : rpc_buffers_) {
    if (b->buffer == raw) {
      flushRpcBuffer(b);
      return;
    }
  }
}

bool HostcallListener::initSignal(const amd::Device& dev) {
  doorbell_ = dev.createSignal();
#ifdef USE_NEW_HOSTCALL_IMPL
  if (doorbell_ == nullptr || !initDevice(dev)) {
    delete doorbell_;
    doorbell_ = nullptr;
    devices_.clear();
    return false;
  }
#else  // !USE_NEW_HOSTCALL_IMPL
  initDevice(dev);
#endif  // USE_NEW_HOSTCALL_IMPL
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  urilocator = dev.createUriLocator();
#endif
#endif
  // If the listener thread was not successfully initialized, clean
  // everything up and bail out.
  if (thread_.state() < Thread::INITIALIZED) {
    delete doorbell_;
#ifdef USE_NEW_HOSTCALL_IMPL
    doorbell_ = nullptr;
#endif  // USE_NEW_HOSTCALL_IMPL
    devices_.clear();
#if defined(__clang__)
#if __has_feature(address_sanitizer)
    delete urilocator;
#ifdef USE_NEW_HOSTCALL_IMPL
    urilocator = nullptr;
#endif  // USE_NEW_HOSTCALL_IMPL
#endif
#endif
    return false;
  }
  thread_.start(this);
  return true;
}

bool HostcallListener::initDevice(const amd::Device& dev) {
  // Create only one signal per device
  // This is to avoid conflicts when n signals are created for n HIP streams per device
  if (devices_.count(&dev) == 0) {
#if defined(WITH_PAL_DEVICE) && !defined(_WIN32)
    auto ws = device::Signal::WaitState::Active;
#else
    auto ws = device::Signal::WaitState::Blocked;
#endif
    if ((doorbell_ == nullptr) || !doorbell_->Init(dev, SIGNAL_INIT, ws)) {
      return false;
    }
    devices_.insert(&dev);
  }
  return true;
}

#ifdef USE_NEW_HOSTCALL_IMPL
bool enableHostcalls(const amd::Device& dev, void* bfr, uint32_t numPackets,
                     amd::Memory* occupiedMem) {
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  buffer->setDevice(&dev);
  if (!buffer->initialize(numPackets, occupiedMem)) {
    ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "Failed to initialize hostcall buffer");
    return false;
  }
#else  // !USE_NEW_HOSTCALL_IMPL
bool enableHostcalls(const amd::Device& dev, void* bfr, uint32_t numPackets) {
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  buffer->initialize(numPackets);
  buffer->setDevice(&dev);
#endif  // USE_NEW_HOSTCALL_IMPL

  amd::ScopedLock lock(listenerLock);
  
  // Don't start a new listener until any previous one is fully terminated.
  while (listenerTerminating) {
    listenerLock.wait();
  }
  if (!hostcallListener) {
    hostcallListener = new HostcallListener();
    if (!hostcallListener->initSignal(dev)) {
      ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
              "Failed to launch hostcall listener");
      delete hostcallListener;
      hostcallListener = nullptr;
      return false;
    }
    ClPrint(amd::LOG_INFO, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "Launched hostcall listener at %p", hostcallListener);
  }
// For PAL, create one signal per device (inside hostcallListener->initDevice(dev)) whose pointer is
// stored in this hostcall buffer For ROCr, create only one signal across all devices (inside
// hostcallListener->initSignal(dev)) whose pointer is stored in every hostcall buffer
#if defined(WITH_PAL_DEVICE)
  else if (!hostcallListener->initDevice(dev)) {
    ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "failed to initialize device for hostcall");
    return false;
  }
#endif  // defined(WITH_PAL_DEVICE)
  hostcallListener->addBuffer(buffer);
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Registered hostcall buffer %p with listener %p", buffer,
          hostcallListener);
  return true;
}

#ifdef USE_NEW_HOSTCALL_IMPL
void disableHostcalls(void* bfr) {
  assert(bfr && "expected a hostcall buffer");
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  HostcallListener* listenerToTerminate = nullptr;
  {
    amd::ScopedLock lock(listenerLock);
    if (!hostcallListener) {
      buffer->getOccupiedMem()->release();
      return;
    }
    hostcallListener->removeBuffer(buffer);
    if (hostcallListener->idle()) {
      listenerToTerminate = hostcallListener;
      hostcallListener = nullptr;
      listenerTerminating = true;
    }
  }
  buffer->getOccupiedMem()->release();
  if (listenerToTerminate) {
    listenerToTerminate->terminate();
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
    delete listenerToTerminate;
    {
      amd::ScopedLock lock(listenerLock);
      listenerTerminating = false;
      listenerLock.notifyAll();
    }
  }
}
#else  // !USE_NEW_HOSTCALL_IMPL
void disableHostcalls(void* bfr) {
  HostcallListener* listenerToTerminate = nullptr;
  {
    amd::ScopedLock lock(listenerLock);
    if (!hostcallListener) return;
    assert(bfr && "expected a hostcall buffer");
    auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
    hostcallListener->removeBuffer(buffer);
    if (hostcallListener->idle()) {
      listenerToTerminate = hostcallListener;
      hostcallListener = nullptr;
      listenerTerminating = true;
    }
  }
  if (listenerToTerminate) {
    listenerToTerminate->terminate();
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
    delete listenerToTerminate;
    {
      amd::ScopedLock lock(listenerLock);
      listenerTerminating = false;
      listenerLock.notifyAll();
    }
  }
}
#endif  // USE_NEW_HOSTCALL_IMPL

bool enableRpc(const amd::Device& dev, void* bfr, uint32_t num_ports) {
  auto* info = new RpcBufferInfo{bfr, dev.info().wavefrontWidth_, num_ports, &dev};

  amd::ScopedLock lock(listenerLock);
  if (!hostcallListener) {
    hostcallListener = new HostcallListener();
    if (!hostcallListener->initSignal(dev)) {
      ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
              "Failed to launch hostcall listener for RPC");
      delete hostcallListener;
      hostcallListener = nullptr;
      delete info;
      return false;
    }
    ClPrint(amd::LOG_INFO, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "Launched hostcall listener at %p (for RPC)", hostcallListener);
  }
#if defined(WITH_PAL_DEVICE)
  else if (!hostcallListener->initDevice(dev)) {
    ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "failed to initialize device for RPC");
    delete info;
    return false;
  }
#endif
  hostcallListener->addRpcBuffer(info);
  ClPrint(amd::LOG_INFO, amd::LOG_QUEUE, "Registered RPC buffer %p with listener %p", bfr,
          hostcallListener);
  return true;
}

void flushRpc(void* bfr) {
  amd::ScopedLock lock(listenerLock);
  if (!hostcallListener) {
    return;
  }
  assert(bfr && "expected an RPC buffer");
  hostcallListener->flushRpcBufferByPtr(bfr);
}

void disableRpc(void* bfr) {
  {
    amd::ScopedLock lock(listenerLock);
    if (!hostcallListener) {
      return;
    }
    assert(bfr && "expected an RPC buffer");
    hostcallListener->removeRpcBufferByPtr(bfr);
  }
  if (hostcallListener->idle()) {
    hostcallListener->terminate();
    delete hostcallListener;
    hostcallListener = nullptr;
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
  }
}
}  // namespace amd
