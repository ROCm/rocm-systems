/* Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc.

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

#include "device/device.hpp"
#include "palsignal.hpp"
#include "paldevice.hpp"
#include "os/os.hpp"

#include <functional>

namespace amd::pal {

Signal::~Signal() {
  dev_->GlbCtx().svmFree(amdSignal_);

  if (ws_ == device::Signal::WaitState::Blocked) {
#if defined(_WIN32)
    Pal::Result result = Pal::Result::Success;

    Pal::UnregisterEventInfo eventInfo = {};
    eventInfo.pEvent = &event_;
    eventInfo.trackingType = Pal::EventTrackingType::ShaderInterrupt;
    result = dev_->iDev()->UnregisterEvent(eventInfo);
    if (result != Pal::Result::Success) {
      ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
              "Failed to unregister SQ event needed for hostcall buffer");
    }
#endif
  }
}

bool Signal::Init(const amd::Device& dev, uint64_t init, device::Signal::WaitState ws) {
  dev_ = static_cast<const pal::Device*>(&dev);
  ws_ = ws;

  void* buffer = dev_->GlbCtx().svmAlloc(sizeof(amd_signal_t), alignof(amd_signal_t),
                                         CL_MEM_SVM_FINE_GRAIN_BUFFER | CL_MEM_SVM_ATOMICS);
  if (!buffer) {
    ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE, "Failed to create amd_signal_t buffer");
    return false;
  }
  std::memset(buffer, 0, sizeof(amd_signal_t));

  amdSignal_ = new (buffer) amd_signal_t();
  amdSignal_->value = init;

  if (ws_ == device::Signal::WaitState::Blocked) {
#if defined(_WIN32)
    Pal::Result result = Pal::Result::Success;

    Util::EventCreateFlags flags = {};
    flags.manualReset = false;
    flags.initiallySignaled = false;
    result = event_.Init(flags);
    if (result != Pal::Result::Success) {
      ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
              "Failed to create Pal::Util::Event needed for hostcall buffer");
      return false;
    }

    result = event_.Set();
    if (result != Pal::Result::Success) {
      ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
              "Failed to set Pal::Util::Event needed for hostcall buffer");
      return false;
    }

    Pal::RegisterEventInfo eventInputInfo = {};
    eventInputInfo.pEvent = &event_;
    eventInputInfo.trackingType = Pal::EventTrackingType::ShaderInterrupt;
    Pal::RegisterEventOutputInfo eventOutputInfo = {};
    result = dev_->iDev()->RegisterEvent(eventInputInfo, &eventOutputInfo);
    if (result != Pal::Result::Success) {
      ClPrint(amd::LOG_ERROR, amd::LOG_QUEUE,
              "Failed to register SQ event needed for hostcall buffer");
      return false;
    }
    amdSignal_->event_id = eventOutputInfo.shaderInterrupt.eventId;
    amdSignal_->event_mailbox_ptr = eventOutputInfo.shaderInterrupt.eventMailboxGpuVa;
    ClPrint(amd::LOG_INFO, amd::LOG_INIT, "Registered SQ event %d with mailbox slot %p",
            amdSignal_->event_id, amdSignal_->event_mailbox_ptr);
#endif
  }

  return true;
}

uint64_t Signal::Wait(uint64_t value, device::Signal::Condition c, uint64_t timeout) {
  auto cmp = [](device::Signal::Condition c) -> std::function<bool(uint64_t, uint64_t)> {
    switch (c) {
      case device::Signal::Condition::Eq:
        return [](auto ls, auto rs) { return ls == rs; };
      case device::Signal::Condition::Ne:
        return [](auto ls, auto rs) { return ls != rs; };
      case device::Signal::Condition::Lt:
        return [](auto ls, auto rs) { return ls < rs; };
      case device::Signal::Condition::Gte:
        return [](auto ls, auto rs) { return ls >= rs; };
    };
    ShouldNotReachHere();
    return [](auto ls, auto rs) { return false; };
  }(c);

  if (ws_ == device::Signal::WaitState::Blocked) {
#if defined(_WIN32)
    Pal::Result result = Pal::Result::Success;

    float timeoutInSec = timeout / (1000 * 1000);
    result = event_.Wait(Util::fseconds{timeoutInSec});

    if ((result != Pal::Result::Success) && (result != Pal::Result::Timeout)) {
      return -1;
    }

    std::atomic_thread_fence(std::memory_order_acquire);
    return amdSignal_->value;
#endif
  } else if (ws_ == device::Signal::WaitState::Active) {
    auto start = amd::Os::timeNanos();
    while (true) {
      auto end = amd::Os::timeNanos();
      auto duration = 1000 * (end - start);  // convert to us
      if (duration >= timeout) {
        return amdSignal_->value;
      }

      if (!cmp(amdSignal_->value, value)) {
        amd::Os::yield();
        continue;
      }

      std::atomic_thread_fence(std::memory_order_acquire);
      return amdSignal_->value;
    }
  }

  return -1;
}

void Signal::Reset(uint64_t value) { amdSignal_->value = value; }

};  // namespace amd::pal
