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
#include "device/devsignal.hpp"

#include "os/os.hpp"
#include "thread/monitor.hpp"
#include "utils/util.hpp"
#include "utils/debug.hpp"
#include "utils/flags.hpp"

#include <assert.h>
#include <string.h>
#include <new>
#include <set>

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/devsanitizer.hpp"
#endif
#endif

namespace amd {

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
            if (dm != nullptr) {
              va = dm->virtualAddress();
            }
            if (va != 0) {
              amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
              const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
            } else {
              buf->release();
            }
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

uint32_t getHostcallBufferAlignment() { return alignof(Payload); }

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
  occupied_ = reinterpret_cast<uint32_t*>(dm->virtualAddress());
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

bool HostcallBuffer::hasWorkPending() const {
  for (uint32_t i = 0; i < scan_limit_; ++i) {
    uint32_t dp = device_phase_[i].load(std::memory_order_relaxed);
    uint32_t hp = host_phase_[i].load(std::memory_order_relaxed);
    if (dp != hp) {
      return true;
    }
  }
  return false;
}

void HostcallBuffer::processPackets(MessageHandler& messages) {
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
  }
}

/** \brief Manage a unique listener thread and its associated buffers.
 */
class HostcallListener {
  std::set<HostcallBuffer*> buffers_;
  device::Signal* doorbell_;
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
    Thread() : amd::Thread("Hostcall Listener Thread", CQ_THREAD_STACK_SIZE) {}

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
  bool idle() const { return buffers_.empty(); }

  void terminate();
  bool initSignal(const amd::Device& dev);
  bool initDevice(const amd::Device& dev);
};

HostcallListener* hostcallListener = nullptr;
extern amd::Monitor listenerLock;
constexpr static uint32_t kSpinIterations = 1024;
constexpr static uint64_t kSignalTimeout = K * K;
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
void HostcallListener::consumePackets() {
  uint64_t signal_value = SIGNAL_INIT;
  kHostThreadActive.state = Init::State::kInit;
  while (true) {
    if (kHostThreadActive.state == Init::State::kDestroy) {
      kHostThreadActive.state = Init::State::kExit;
      return;
    }

    for (uint32_t spin = 0; spin < kSpinIterations; ++spin) {
      if (kHostThreadActive.state == Init::State::kDestroy) {
        kHostThreadActive.state = Init::State::kExit;
        return;
      }

      amd::ScopedLock lock{listenerLock};
      for (auto buf : buffers_) {
        buf->processPackets(messages_);
      }
    }

    {
      amd::ScopedLock lock{listenerLock};
      for (auto buf : buffers_) {
        buf->resetScanLimit();
      }
      for (auto buf : buffers_) {
        buf->processPackets(messages_);
      }
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

void HostcallListener::terminate() {
  if (thread_.state() >= Thread::FINISHED || amd::Os::isThreadAlive(thread_)) {
    kHostThreadActive.state = Init::State::kExit;
    doorbell_->Reset(SIGNAL_DONE);

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

bool HostcallListener::initSignal(const amd::Device& dev) {
  doorbell_ = dev.createSignal();
  if (doorbell_ == nullptr || !initDevice(dev)) {
    delete doorbell_;
    doorbell_ = nullptr;
    devices_.clear();
    return false;
  }
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  urilocator = dev.createUriLocator();
#endif
#endif
  // If the listener thread was not successfully initialized, clean
  // everything up and bail out.
  if (thread_.state() < Thread::INITIALIZED) {
    delete doorbell_;
    doorbell_ = nullptr;
    devices_.clear();
#if defined(__clang__)
#if __has_feature(address_sanitizer)
    delete urilocator;
    urilocator = nullptr;
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

bool enableHostcalls(const amd::Device& dev, void* bfr, uint32_t numPackets,
                     amd::Memory* occupiedMem) {
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  buffer->setDevice(&dev);
  if (!buffer->initialize(numPackets, occupiedMem)) {
    ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
            "Failed to initialize hostcall buffer");
    return false;
  }

  amd::ScopedLock lock(listenerLock);
  if (!hostcallListener) {
    hostcallListener = new HostcallListener();
    if (!hostcallListener) {
      ClPrint(amd::LOG_ERROR, (amd::LOG_INIT | amd::LOG_QUEUE | amd::LOG_RESOURCE),
              "Failed to allocate hostcall listener");
      return false;
    }
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

void disableHostcalls(void* bfr) {
  assert(bfr && "expected a hostcall buffer");
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  {
    amd::ScopedLock lock(listenerLock);
    if (!hostcallListener) {
      buffer->getOccupiedMem()->release();
      return;
    }
    hostcallListener->removeBuffer(buffer);
  }
  buffer->getOccupiedMem()->release();

  if (hostcallListener->idle()) {
    hostcallListener->terminate();
    delete hostcallListener;
    hostcallListener = nullptr;
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
  }
}
}  // namespace amd
