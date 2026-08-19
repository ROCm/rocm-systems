// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_server.h"

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/amdgpu/pci/scratch_pci_device.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "util/log.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <format>
#include <stop_token>
#include <thread>

namespace rocjitsu {
namespace {

/// @brief How often the waiting thread rechecks for a shutdown signal.
constexpr long kSignalPollNanoseconds = 100'000'000;

} // namespace

int run_vfio_server(const std::string &socket_path) {
  RegisterSymbols symbols;
  add_pre_discovery_symbols(symbols, ScratchPciDevice::kBarIndex);
  BarAccessTrace trace(symbols);

  // Vendor 0x1002 is AMD; the device ID is a placeholder until the emulated GPU
  // supplies its own identity. Class 0x12/0x00 is a processing accelerator,
  // which is what a compute GPU without a display presents as.
  const simdojo::PciId id = {.vendor = 0x1002,
                             .device = 0x0000,
                             .subsys_vendor = 0x1002,
                             .subsys = 0x0000,
                             .cls = 0x12,
                             .subcls = 0x00,
                             .prog_if = 0x00,
                             .revision = 0x00};
  ScratchPciDevice device("rocjitsu-scratch", id, &trace);

  // Shutdown signals are blocked and then consumed synchronously, the way the
  // daemon does it. Almost nothing is safe to touch from a signal handler, least
  // of all the synchronization a stop request runs through.
  sigset_t shutdown_signals;
  sigemptyset(&shutdown_signals);
  sigaddset(&shutdown_signals, SIGINT);
  sigaddset(&shutdown_signals, SIGTERM);
  sigset_t previous_signals;
  if (pthread_sigmask(SIG_BLOCK, &shutdown_signals, &previous_signals) != 0) {
    util::Logger::warn("vfu: cannot block shutdown signals");
    return 1;
  }

  int status = 0;
  {
    VfioDeviceHost host(socket_path, device);
    if (!host.build()) {
      pthread_sigmask(SIG_SETMASK, &previous_signals, nullptr);
      return 1;
    }

    util::Logger::warn(std::format("vfu: serving {} on {}", device.name(), socket_path));

    // The serving thread inherits the blocked mask, so a shutdown signal is
    // delivered to the wait below and interrupts nothing mid-protocol.
    std::atomic<bool> serving_failed = false;
    std::atomic<bool> serving_finished = false;
    std::jthread serving_thread([&](std::stop_token stop_token) {
      serving_failed = host.run(stop_token) == VfioDeviceHost::ServeResult::Failed;
      serving_finished = true;
    });

    // Waiting only for a signal would leave the process alive but serving
    // nothing if the transport failed on its own: a supervisor would see a
    // healthy process in front of a dead socket.
    while (!serving_finished.load()) {
      const timespec timeout{.tv_sec = 0, .tv_nsec = kSignalPollNanoseconds};
      const int signal = sigtimedwait(&shutdown_signals, nullptr, &timeout);
      if (signal == SIGINT || signal == SIGTERM) {
        break;
      }
      if (signal < 0 && errno != EAGAIN && errno != EINTR) {
        util::Logger::warn("vfu: failed while waiting for a shutdown signal");
        status = 1;
        break;
      }
    }

    // Stop and join before the host is destroyed, so no callback can run against
    // a context that is being torn down.
    serving_thread.request_stop();
    serving_thread.join();
    host.detach();
    if (serving_failed.load()) {
      status = 1;
    }
  }

  // Drain anything that arrived while shutting down. Restoring the previous mask
  // with a shutdown signal still pending would kill the process on the way out
  // of an otherwise orderly stop.
  while (true) {
    const timespec no_wait{.tv_sec = 0, .tv_nsec = 0};
    if (sigtimedwait(&shutdown_signals, nullptr, &no_wait) < 0) {
      break;
    }
  }
  pthread_sigmask(SIG_SETMASK, &previous_signals, nullptr);

  const std::string report = trace.unmodeled_report();
  if (!report.empty()) {
    util::Logger::warn(report);
  }
  return status;
}

} // namespace rocjitsu
