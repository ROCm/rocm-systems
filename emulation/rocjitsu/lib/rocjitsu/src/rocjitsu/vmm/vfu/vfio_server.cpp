// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_server.h"

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "util/log.h"

#include "embedded_schema.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <exception>
#include <format>
#include <stop_token>
#include <thread>

namespace rocjitsu {
namespace {

/// @brief How often the waiting thread rechecks for a shutdown signal.
constexpr long kSignalPollNanoseconds = 100'000'000;

} // namespace

int run_vfio_server(const std::string &config_path, const std::string &socket_path) {
  config::DeviceIdentityConfig identity;
  try {
    identity = config::load_device_identity(config_path, kEmbeddedSchema);
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: cannot read {}: {}", config_path, error.what()));
    return 1;
  }
  RegisterSymbols symbols;
  add_pre_discovery_symbols(symbols, GpuPciDevice::kRegisterBar);
  BarAccessTrace trace(symbols);

  const std::string device_name =
      identity.device.marketing_name.empty() ? "gpu" : identity.device.marketing_name;
  GpuPciDevice device(device_name, gpu_pci_spec_from_config(identity.device, identity.pci), &trace);
  if (!device.usable()) {
    return 1;
  }

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

  // What the driver said about its interrupt ring. Reported next to the
  // unmodelled registers because it answers the same question -- how far the
  // driver got before it stopped telling us anything -- and because the ring is
  // the one thing the device now knows that it cannot yet act on.
  const GpuPciDevice::InterruptRing ring = device.interrupt_ring();
  // Only when there is something to say. A reset or a disconnect clears these
  // registers, so a guest that detached cleanly has already taken the ring with
  // it, and warning about its absence would make an ordinary shutdown look like
  // a fault. What the driver said while attached is reported when it says it.
  if (ring.programmed()) {
    util::Logger::warn(std::format("vfu: the driver left {}", GpuPciDevice::describe(ring)));
  }

  const std::string report = trace.unmodeled_report();
  if (!report.empty()) {
    util::Logger::warn(report);
  }
  return status;
}

} // namespace rocjitsu
