/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <condition_variable>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>
#include "device/rocm/rocrctx.hpp"
#include "utils/flags.hpp"

namespace amd::roc {

class Device;
class ProfilingSignal;

//! Dumps one JSON Lines record per AQL kernel dispatch when GPU_DUMP_AQL_DISPATCH names a directory
class AqlDispatchDumper {
 public:
  //! kGraph packets are replayed from a captured batch, so signals attached at replay are not
  //! recorded; cooperative and GraphExecClassic nodes re-dispatch per launch and report as kEager.
  enum class Origin { kEager, kGraph };

  //! Hot-path gate; true when GPU_DUMP_AQL_DISPATCH is set
  static bool Enabled() {
    static const bool enabled = !flagIsDefault(GPU_DUMP_AQL_DISPATCH);
    return enabled;
  }

  //! Appends one record. |packet| is the host-side 64B packet; |header| and |rest| carry the
  //! out-of-band header/setup dword halves. Safe to call with an unknown kernel_object.
  //! A non-null |signal| is retained and polled for a matching completion record.
  static void Record(const Device& dev, const hsa_queue_t* queue, uint64_t index,
                     const void* packet, bool ext_packet, uint16_t header, uint16_t rest,
                     Origin origin, ProfilingSignal* signal);

  //! Stops the poller and drops held signals, quiescing any still armed so their destructor
  //! cannot block. Idempotent; must run before Hsa::shut_down().
  static void Shutdown();

 private:
  //! One retained dispatch signal awaiting completion
  struct InFlight {
    ProfilingSignal* signal;
    hsa_agent_t agent;
    uint64_t queue_id;
    uint64_t packet_index;
    Origin origin;
    std::string kernel;
  };

  AqlDispatchDumper();
  static AqlDispatchDumper& Instance();
  void PollLoop();
  //! Emits a completion record for every held signal that reached zero. Caller holds mutex_.
  void Drain();

  std::mutex mutex_;
  std::ofstream out_;
  std::condition_variable cv_;
  std::vector<InFlight> inflight_;
  std::thread poller_;
  bool stop_ = false;
};

}  // namespace amd::roc
