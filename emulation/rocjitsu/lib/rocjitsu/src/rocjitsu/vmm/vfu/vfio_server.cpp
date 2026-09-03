// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_server.h"

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/virtual_machine.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "simdojo/sim/simulation.h"
#include "util/log.h"

#include "embedded_schema.h"

#include <atomic>
#include <cerrno>
#include <csignal>
#include <ctime>
#include <exception>
#include <format>
#include <memory>
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
  config::LoadedConfig loaded;
  try {
    loaded = config::load_config(config_path, kEmbeddedSchema);
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: cannot read {}: {}", config_path, error.what()));
    return 1;
  }
  SoC *soc = loaded.soc();
  if (soc == nullptr) {
    util::Logger::warn(std::format("vfu: {} describes no GPU to present", config_path));
    return 1;
  }
  // One socket serves one function. A config describing several GPUs builds them
  // all and would have every one but the first silently dropped, which reads to
  // whoever wrote the config as a device that lost most of itself.
  if (loaded.num_gpus > 1) {
    util::Logger::warn(std::format("vfu: {} describes {} GPUs; serving presents one function",
                                   config_path, loaded.num_gpus));
    return 1;
  }
  RegisterSymbols symbols;
  add_pre_discovery_symbols(symbols, GpuPciDevice::kRegisterBar);
  BarAccessTrace trace(symbols);

  // Named for what it is in the machine, not for what it is sold as. This string
  // becomes the component's path, which link specs and per-block register models
  // address it by, and a marketing name varies per config and per product.
  constexpr const char *kPciFunctionName = "pci";

  // The PCI function is a child of the machine it speaks for, alongside the SoC
  // rather than in place of it. Until now this server had no engine at all,
  // which was survivable only because the device answers register reads out of
  // its own state and never asks the simulation for anything. A device that
  // drives a command processor will, and a component outside the topology has no
  // engine pointer and no partition: scheduling from it asserts in a debug build
  // and is silently dropped in a release one.
  //
  // The whole machine is built here, not a placeholder for one. It costs what
  // load_device_identity exists to avoid -- a large part's register files are
  // gigabytes -- and that is the right trade: the hardware behind the bus face
  // is what the next stages drive, and a function parented to a stand-in would
  // have to be re-parented to reach any of it.
  //
  // Built before create(), because there is no hot-attach: the engine sizes its
  // partitions and hands out engine pointers there, and a component added later
  // gets neither.
  simdojo::SimulationEngine::Config engine_config = loaded.engine_config;
  engine_config.max_ticks = 0;
  engine_config.await_primaries = true;
  if (engine_config.num_threads != 1) {
    // Partitioning assigns threads per XCD, and a PCI function is not one, so it
    // would be left without a partition -- the exact defect this is fixing. It
    // also wants to share a partition with whatever it eventually drives, so the
    // placement is a decision to make with that, not ahead of it.
    util::Logger::warn(std::format("vfu: serving single-threaded; {} asked for {} threads",
                                   config_path, engine_config.num_threads));
    engine_config.num_threads = 1;
  }
  simdojo::SimulationEngine engine(engine_config);

  // The build result owns the SoC as its root; ownership moves to the machine.
  auto built_root = loaded.take_root();
  built_root.release();
  auto machine = std::make_unique<VirtualMachine>(std::unique_ptr<SoC>(soc), /*daemon_mode=*/false);
  VirtualMachine *machine_ptr = machine.get();
  engine.topology().set_root(std::move(machine));
  loaded.wire_links(engine.topology());
  soc->wire_backing(engine.topology());

  auto *device_ptr =
      static_cast<GpuPciDevice *>(machine_ptr->add_child(std::make_unique<GpuPciDevice>(
          kPciFunctionName, gpu_pci_spec_from_config(loaded.device, loaded.pci), &trace, soc)));
  GpuPciDevice &device = *device_ptr;
  struct FrontendShutdown {
    GpuPciDevice &device;
    ~FrontendShutdown() { (void)device.shutdown_frontend(); }
  } frontend_shutdown{device};
  if (!device.usable()) {
    return 1;
  }

  engine.create();
  // Without a primary the engine treats an idle machine as a finished one and
  // returns from run() immediately. This machine is idle by construction between
  // guest accesses, so it needs to be told that quiescence is not completion.
  //
  // The KFD driver the machine carries is deliberately never opened: under
  // vfio-user the guest's own amdgpu is the driver, and opening ours would put
  // a second one in front of the same hardware.
  engine.register_as_primary();

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
  // Started after the mask is in place so it inherits it: a shutdown signal must
  // reach the thread waiting for one below, not interrupt the engine.
  //
  // run() blocks until request_exit, and latches readiness on the way out of the
  // thread rather than only inside run(), so a failure before run() is entered
  // cannot strand the wait below forever.
  //
  // The request to exit is tied to a destructor rather than written out at each
  // return. run() finishes only when asked, and a jthread's stop token does not
  // ask it -- the lambda has none to observe -- so a return that forgot would
  // join a thread that never ends. A server that reports a failure and then
  // hangs is worse than one that crashes, because a supervisor sees a live
  // process in front of a dead socket, and it is exactly what happens when this
  // is left to six separate exit paths to remember.
  struct EngineRun {
    simdojo::SimulationEngine &engine;
    std::jthread thread;
    ~EngineRun() {
      engine.request_exit("serving ended", 0);
      if (thread.joinable()) {
        thread.join();
      }
    }
  };
  EngineRun engine_run{engine, std::jthread([&engine] {
                         engine.run();
                         engine.latch_startup_if_unlatched(/*failed=*/true);
                       })};

  if (!engine.wait_until_started()) {
    util::Logger::warn("vfu: the simulation engine did not start");
    drain_and_restore(handled_signals, previous_signals);
    return 1;
  }

  {
    VfioDeviceHost host(socket_path, device);
    if (!host.build()) {
      // Drained on this path too: a request signal queued while the mask was on
      // is still pending, and unmasking with one outstanding kills the process
      // on the way out of a failure it has already reported.
      drain_and_restore(handled_signals, previous_signals);
      return 1;
    }

    util::Logger::warn(std::format(
        "vfu: serving {} on {}",
        loaded.device.marketing_name.empty() ? device.name() : loaded.device.marketing_name,
        socket_path));

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
                                           describe(device.interrupt_ring())));
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

  // Every external source of work is quiesced before the engine: the serving
  // thread is stopped and joined above and the host is gone with the scope, so
  // nothing can call into the device while its engine is torn down by
  // ~EngineRun below. The device outlives both, because the topology owns it and
  // the engine owns that.
  drain_and_restore(handled_signals, previous_signals);

  // What the driver said about its interrupt ring. Reported next to the
  // unmodelled registers because it answers the same question -- how far the
  // driver got before it stopped telling us anything -- and because the ring is
  // the one thing the device now knows that it cannot yet act on.
  const InterruptRing ring = device.interrupt_ring();
  // Only when there is something to say. A reset or a disconnect clears these
  // registers, so a guest that detached cleanly has already taken the ring with
  // it, and warning about its absence would make an ordinary shutdown look like
  // a fault. What the driver said while attached is reported when it says it.
  if (ring.programmed()) {
    util::Logger::warn(std::format("vfu: the driver left {}", describe(ring)));
  }

  const std::string report = trace.unmodeled_report();
  if (!report.empty()) {
    util::Logger::warn(report);
  }
  return status;
}

} // namespace rocjitsu
