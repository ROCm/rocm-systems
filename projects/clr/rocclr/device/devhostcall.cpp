/* Copyright (c) 2019 - 2021 Advanced Micro Devices, Inc.

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
#include <set>

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "device/devsanitizer.hpp"
#endif
#endif

namespace amd {

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
            va = dm->virtualAddress();
            amd::MemObjMap::AddMemObj(reinterpret_cast<void*>(va), buf);
            const_cast<amd::Device*>(&dev)->TrackHostcallMemory(buf);
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

uint32_t getHostcallBufferAlignment() { return alignof(Payload); }

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
constexpr static uint64_t kTimeoutFloor = K * K * 4;
constexpr static uint64_t kTimeoutCeil = K * K * 16;
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
  uint64_t timeout = kTimeoutFloor;
  uint64_t signal_value = SIGNAL_INIT;
  kHostThreadActive.state = Init::State::kInit;
  while (true) {
    while (true) {
      if (kHostThreadActive.state == Init::State::kDestroy) {
        kHostThreadActive.state = Init::State::kExit;
        return;
      }
      uint64_t new_value = doorbell_->Wait(signal_value, device::Signal::Condition::Ne, timeout);
      if (new_value != signal_value) {
        signal_value = new_value;
        // Reduce the timeout for quicker processing
        timeout = timeout >> 0x1;
        timeout = std::max(kTimeoutFloor, timeout);
        break;
      }
      // Increase the timeout since we dont need to check as frequently
      timeout = timeout << 0x1;
      timeout = std::min(kTimeoutCeil, timeout);
    }

    if (signal_value == SIGNAL_DONE) {
      return;
    }

    if (!idle()) {
      amd::ScopedLock lock{listenerLock};

      for (auto ii : buffers_) {
        ii->processPackets(messages_);
      }
    }
  }

  return;
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
  initDevice(dev);
#if defined(__clang__)
#if __has_feature(address_sanitizer)
  urilocator = dev.createUriLocator();
#endif
#endif
  // If the listener thread was not successfully initialized, clean
  // everything up and bail out.
  if (thread_.state() < Thread::INITIALIZED) {
    delete doorbell_;
    devices_.clear();
#if defined(__clang__)
#if __has_feature(address_sanitizer)
    delete urilocator;
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

bool enableHostcalls(const amd::Device& dev, void* bfr, uint32_t numPackets) {
  auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
  buffer->initialize(numPackets);
  buffer->setDevice(&dev);

  amd::ScopedLock lock(listenerLock);
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

void disableHostcalls(void* bfr) {
  {
    amd::ScopedLock lock(listenerLock);
    if (!hostcallListener) {
      return;
    }
    assert(bfr && "expected a hostcall buffer");
    auto buffer = reinterpret_cast<HostcallBuffer*>(bfr);
    hostcallListener->removeBuffer(buffer);
  }
  if (hostcallListener->idle()) {
    hostcallListener->terminate();
    delete hostcallListener;
    hostcallListener = nullptr;
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Terminated hostcall listener");
  }
}
}  // namespace amd
