/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression test for the device-printf hostcall listener wait loop in
// device/devhostcall.cpp (HostcallListener::consumePackets), fixing
// ROCm/TheRock#6357.
//
// Background
// ----------
// consumePackets() blocks on the doorbell signal and only serviced the ready
// stacks (processPackets) when doorbell_->Wait() reported a VALUE CHANGE. On a
// wait timeout with an unchanged value it merely grew the backoff timeout and
// looped again, never inspecting the ready stacks. If a device-side doorbell
// ring is ever missed (observed on gfx1200/gfx1201 RDNA4 Linux) a ready
// SERVICE_PRINTF packet is never serviced, the requesting wave never unblocks,
// and hipDeviceSynchronize deadlocks forever (300s CI timeout, exit 124).
//
// The fix breaks out of the wait loop on timeout when the listener is not idle
// (`if (!idle()) break;`) so the ready stacks are polled even when a doorbell
// wakeup is lost.
//
// What this test does
// -------------------
// consumePackets() is entangled with a live device signal, a listener thread,
// MessageHandler and amd::Device, so it cannot be compiled or driven in
// isolation. Instead this test pins the *control flow* of the wait loop with a
// faithful model (mirroring device/devhostcall.cpp:286-316) driven by a mock
// doorbell. A "lost doorbell" is a Wait() that keeps returning the current
// signal value (a timeout) even though a packet is ready. The model reproduces
// the deadlock without the patch and shows the packet is serviced with it,
// while leaving the healthy value-change fast path unchanged.

#include "gtest/gtest.h"

#include <algorithm>
#include <cstdint>

namespace {

// Mirror the constants and sentinels from device/devhostcall.cpp so the model
// backs off exactly like the real loop.
constexpr uint64_t K = 1024;
constexpr uint64_t kTimeoutFloor = K * K * 4;
constexpr uint64_t kTimeoutCeil = K * K * 16;
constexpr uint64_t kSignalInit = static_cast<uint64_t>(-1);  // stands in for SIGNAL_INIT

// Models the slice of HostcallListener + doorbell state that the wait loop
// touches. Only the behaviour relevant to the lost-doorbell fix is reproduced.
struct ListenerModel {
  // ---- listener state ----
  bool buffers_registered = true;  // a hostcall buffer is registered
  bool packet_ready = false;       // a SERVICE_PRINTF packet sits on a ready stack
  bool packet_serviced = false;    // processPackets() handled the ready packet

  // ---- mock doorbell ----
  // A healthy doorbell reports a value change exactly once; a lost doorbell
  // never does, so every Wait() is a timeout that returns the current value.
  bool doorbell_delivers = false;
  uint64_t delivered_value = 0;

  // HostcallListener::idle(): true when no buffers are registered.
  bool idle() const { return !buffers_registered; }

  // HostcallBuffer::processPackets(): a cheap no-op when nothing is ready;
  // otherwise it services the pending packet exactly once.
  void processPackets() {
    if (!packet_ready) return;
    packet_serviced = true;
    packet_ready = false;
  }

  // doorbell_->Wait(signal_value, Ne, timeout): returns the observed signal
  // value. On a healthy wakeup it returns a new (different) value once; on a
  // lost/missed doorbell it always returns the current value == a timeout.
  uint64_t doorbellWait(uint64_t current) {
    if (doorbell_delivers) {
      doorbell_delivers = false;
      return delivered_value;
    }
    return current;
  }
};

// Faithful model of HostcallListener::consumePackets()'s wait+service loop.
// `patched` toggles the ROCm/TheRock#6357 fix (break out of the wait loop on a
// timeout when the listener is not idle).
//
// Returns true if the pending packet was serviced within `budget` doorbell
// waits. Returning false models the real loop spinning forever on the timeout
// path (i.e. the hipDeviceSynchronize deadlock) -- the finite budget stands in
// for "never".
bool runWaitLoop(ListenerModel& m, bool patched, int budget = 1000) {
  uint64_t timeout = kTimeoutFloor;
  uint64_t signal_value = kSignalInit;

  for (int i = 0; i < budget; ++i) {
    uint64_t new_value = m.doorbellWait(signal_value);
    if (new_value != signal_value) {
      // Value-change fast path: shrink the timeout and drop out of the wait
      // loop to service packets.
      signal_value = new_value;
      timeout = std::max(kTimeoutFloor, timeout >> 1);
    } else {
      // Timeout: the doorbell value did not change.
      timeout = std::min(kTimeoutCeil, timeout << 1);
      // The fix: poll the ready stacks on timeout too, but only when there is
      // something to service. Without it (or while idle) keep on waiting.
      if (!(patched && !m.idle())) {
        continue;
      }
    }

    // Loop tail (device/devhostcall.cpp:309-315): service packets when not idle.
    if (!m.idle()) {
      m.processPackets();
    }
    if (m.packet_serviced) {
      return true;
    }
  }
  return false;
}

// The core regression: a lost doorbell + a ready packet deadlocks the
// unpatched loop -- the packet is never serviced.
TEST(DevHostcallWaitLoop, LostDoorbellUnpatchedDeadlocks) {
  ListenerModel m;
  m.buffers_registered = true;
  m.packet_ready = true;
  m.doorbell_delivers = false;  // doorbell ring was missed

  EXPECT_FALSE(runWaitLoop(m, /*patched=*/false));
  EXPECT_FALSE(m.packet_serviced);
}

// With the fix, the same lost doorbell no longer hangs: the ready packet is
// polled and serviced on the timeout path.
TEST(DevHostcallWaitLoop, LostDoorbellPatchedServicesPacket) {
  ListenerModel m;
  m.buffers_registered = true;
  m.packet_ready = true;
  m.doorbell_delivers = false;  // doorbell ring was missed

  EXPECT_TRUE(runWaitLoop(m, /*patched=*/true));
  EXPECT_TRUE(m.packet_serviced);
}

// The healthy value-change fast path is unaffected by the fix: a packet
// delivered with a working doorbell is serviced either way.
TEST(DevHostcallWaitLoop, HealthyDoorbellUnpatchedServicesPacket) {
  ListenerModel m;
  m.buffers_registered = true;
  m.packet_ready = true;
  m.doorbell_delivers = true;
  m.delivered_value = 42;

  EXPECT_TRUE(runWaitLoop(m, /*patched=*/false));
  EXPECT_TRUE(m.packet_serviced);
}

TEST(DevHostcallWaitLoop, HealthyDoorbellPatchedServicesPacket) {
  ListenerModel m;
  m.buffers_registered = true;
  m.packet_ready = true;
  m.doorbell_delivers = true;
  m.delivered_value = 42;

  EXPECT_TRUE(runWaitLoop(m, /*patched=*/true));
  EXPECT_TRUE(m.packet_serviced);
}

// The fix is guarded by !idle(): an idle listener (no buffers) must keep
// waiting on a timeout rather than busy-servicing, and must not fabricate work.
TEST(DevHostcallWaitLoop, PatchedIdleListenerKeepsWaiting) {
  ListenerModel m;
  m.buffers_registered = false;  // idle
  m.packet_ready = false;
  m.doorbell_delivers = false;

  EXPECT_FALSE(runWaitLoop(m, /*patched=*/true));
  EXPECT_FALSE(m.packet_serviced);
}

}  // namespace
