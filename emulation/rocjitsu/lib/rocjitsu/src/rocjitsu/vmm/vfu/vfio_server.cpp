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

/// @brief Consume any pending handled signal, then put the old mask back.
///
/// @details Restoring the mask with one of these still queued kills the
/// process: terminating is the default disposition for the request signal as
/// well as the shutdown ones. Every path that unmasks has to drain first, which
/// is why this is a function rather than a sequence written out at each of
/// them.
/// @param[in] handled The signals this server blocked.
/// @param[in] previous The mask to restore.
void drain_and_restore(const sigset_t &handled, const sigset_t &previous) {
  while (true) {
    const timespec no_wait{.tv_sec = 0, .tv_nsec = 0};
    if (sigtimedwait(&handled, nullptr, &no_wait) < 0) {
      break;
    }
  }
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
}

/// @brief How often the waiting thread rechecks for a signal.
constexpr long kSignalPollNanoseconds = 100'000'000;

/// @brief What a requested interrupt reports itself as.
///
/// @details This device runs no work, so it has nothing of its own to report
/// and no block it could honestly attribute an interrupt to. What is wanted is
/// an identifier the driver accepts as well formed and then finds no handler
/// for, so the entry exercises its dispatch rather than its malformed-entry
/// path while naming no hardware that did anything. Anything at or above the
/// client count is rejected as invalid, and of the values below it this is one
/// that no block registers a source for -- not because the architecture
/// reserves it on this generation, but because nothing here claims it. It will
/// stop being inert the day something does.
constexpr uint8_t kRequestedInterruptClient = 0x1a;
constexpr uint8_t kRequestedInterruptSource = 0x00;

} // namespace

ServerSignalAction action_for_signal(int signal) {
  if (signal == SIGINT || signal == SIGTERM) {
    return ServerSignalAction::Stop;
  }
  if (signal == SIGUSR1) {
    return ServerSignalAction::DeliverInterrupt;
  }
  return ServerSignalAction::KeepServing;
}

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
  //
  // POSIX rather than std, deliberately and not for want of looking: C++ offers
  // no per-thread signal mask and no synchronous signal wait. std::signal gives
  // only an async handler, which is the thing being avoided here, and sigprocmask
  // is unspecified in a process with threads -- which this one has, since serving
  // runs on its own. pthread_sigmask is the correct call, and <csignal> declares
  // it; it does not need <pthread.h>.
  sigset_t handled_signals;
  sigemptyset(&handled_signals);
  sigaddset(&handled_signals, SIGINT);
  sigaddset(&handled_signals, SIGTERM);
  // Delivering an interrupt on request is a bring-up affordance, not a model of
  // anything: the device has no event source yet, so the only way to show that
  // the interrupt path works end to end is for something outside to ask.
  sigaddset(&handled_signals, SIGUSR1);
  sigset_t previous_signals;
  if (pthread_sigmask(SIG_BLOCK, &handled_signals, &previous_signals) != 0) {
    util::Logger::warn("vfu: cannot block the signals this server waits on");
    return 1;
  }

  int status = 0;
  {
    VfioDeviceHost host(socket_path, device);
    if (!host.build()) {
      // Drained on this path too: a request signal queued while the mask was on
      // is still pending, and unmasking with one outstanding kills the process
      // on the way out of a failure it has already reported.
      drain_and_restore(handled_signals, previous_signals);
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
      const int signal = sigtimedwait(&handled_signals, nullptr, &timeout);
      const ServerSignalAction action = action_for_signal(signal);
      if (action == ServerSignalAction::Stop) {
        break;
      }
      if (action == ServerSignalAction::DeliverInterrupt) {
        // Handed to the serving thread rather than done here: that thread is
        // otherwise the only one that touches the device, and waiting on its
        // lock would let a stalled client make this loop miss a shutdown.
        // Refused when one is still outstanding: the serving thread has not run
        // the previous request yet, so a second would only queue behind work that
        // is itself waiting on a client. Say so rather than appearing to comply.
        const bool accepted = host.ask_serving_thread([&device] {
          if (device.deliver_interrupt({.client_id = kRequestedInterruptClient,
                                        .source_id = kRequestedInterruptSource})) {
            // The entry is in the ring either way; whether a message went with
            // it is the driver's choice, and saying otherwise would misreport a
            // silent ring as a delivered interrupt.
            util::Logger::warn(device.interrupt_ring().raises_messages
                                   ? "vfu: delivered an interrupt on request"
                                   : "vfu: put an entry in the ring on request, with messages "
                                     "switched off by the driver");
          } else {
            util::Logger::warn(std::format("vfu: cannot deliver an interrupt: {}",
                                           GpuPciDevice::describe(device.interrupt_ring())));
          }
        });
        if (!accepted) {
          util::Logger::warn("vfu: an interrupt request is already pending; ignoring this one");
        }
        continue;
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

  drain_and_restore(handled_signals, previous_signals);

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
