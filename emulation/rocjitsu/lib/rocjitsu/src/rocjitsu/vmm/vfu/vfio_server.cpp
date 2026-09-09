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
#include <cstdint>
#include <ctime>
#include <exception>
#include <format>
#include <memory>
#include <optional>
#include <stop_token>
#include <thread>
#include <unistd.h>

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

/// @brief Block every signal the server handles before any worker can exist.
class ServerSignalMask {
public:
  ServerSignalMask() {
    sigemptyset(&handled_);
    sigaddset(&handled_, SIGINT);
    sigaddset(&handled_, SIGTERM);
    sigaddset(&handled_, SIGUSR1);
    active_ = pthread_sigmask(SIG_BLOCK, &handled_, &previous_) == 0;
  }

  ~ServerSignalMask() { restore(); }

  ServerSignalMask(const ServerSignalMask &) = delete;
  ServerSignalMask &operator=(const ServerSignalMask &) = delete;

  [[nodiscard]] bool active() const { return active_; }
  [[nodiscard]] const sigset_t &handled() const { return handled_; }

  void restore() {
    if (!active_)
      return;
    drain_and_restore(handled_, previous_);
    active_ = false;
  }

private:
  sigset_t handled_{};
  sigset_t previous_{};
  bool active_ = false;
};

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

/// @brief Own the launcher's readiness descriptor until startup succeeds.
class ReadinessPipe {
public:
  explicit ReadinessPipe(int fd) : fd_(fd) {}
  ~ReadinessPipe() {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  ReadinessPipe(const ReadinessPipe &) = delete;
  ReadinessPipe &operator=(const ReadinessPipe &) = delete;

  [[nodiscard]] bool signal() {
    if (fd_ < 0) {
      return true;
    }

    constexpr uint8_t ready = 1;
    ssize_t written = 0;
    do {
      written = write(fd_, &ready, sizeof(ready));
    } while (written < 0 && errno == EINTR);
    close(fd_);
    fd_ = -1;
    return written == static_cast<ssize_t>(sizeof(ready));
  }

private:
  int fd_;
};

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

namespace {

int run_vfio_server_impl(const std::string &config_path, const std::string &socket_path,
                         int ready_fd, std::optional<int> engine_exit_code_for_test) {
  ReadinessPipe readiness(ready_fd);
  // Config loading and machine construction may create helper threads. Block
  // before either one so every thread in the server inherits the mask and a
  // process-directed shutdown signal can only be consumed by sigtimedwait().
  ServerSignalMask signal_mask;
  if (!signal_mask.active()) {
    util::Logger::warn("vfu: cannot block the signals this server waits on");
    return 1;
  }
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
  std::unique_ptr<simdojo::CompositeComponent> built_root = loaded.take_root();
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
  class FrontendShutdown {
  public:
    explicit FrontendShutdown(GpuPciDevice &device) : device_(device) {}
    ~FrontendShutdown() { (void)device_.shutdown_frontend(); }

  private:
    GpuPciDevice &device_;
  } frontend_shutdown(device);
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
  class EngineRun {
  public:
    EngineRun(simdojo::SimulationEngine &engine, std::jthread thread)
        : engine_(engine), thread_(std::move(thread)) {}

    ~EngineRun() { stop(); }

    void stop() {
      engine_.request_exit("serving ended", 0);
      if (thread_.joinable()) {
        thread_.join();
      }
    }

  private:
    simdojo::SimulationEngine &engine_;
    std::jthread thread_;
  };
  simdojo::ExitStatus engine_status;
  std::atomic<bool> engine_finished = false;
  EngineRun engine_run(engine, std::jthread([&engine, &engine_status, &engine_finished] {
                         engine_status = engine.run();
                         engine.latch_startup_if_unlatched(/*failed=*/true);
                         engine_finished.store(true, std::memory_order_release);
                       }));

  if (!engine.wait_until_started()) {
    util::Logger::warn("vfu: the simulation engine did not start");
    return 1;
  }
  if (engine_finished.load(std::memory_order_acquire)) {
    engine_run.stop();
    util::Logger::warn(std::format("vfu: simulation engine stopped before serving: {} (code {})",
                                   engine_status.message, engine_status.code));
    return engine_status.code == 0 ? 1 : engine_status.code;
  }

  {
    VfioDeviceHost host(socket_path, device);
    if (!host.build()) {
      return 1;
    }

    // build() has bound and started listening on the AF_UNIX socket. Publish
    // that state directly instead of making the launcher infer it by polling a
    // path against a guessed deadline. Closing the pipe without this byte is
    // the corresponding startup-failure notification.
    if (!readiness.signal()) {
      util::Logger::warn("vfu: cannot report server readiness to the launcher");
      return 1;
    }
    if (engine_exit_code_for_test.has_value()) {
      engine.request_exit("test-requested simulation failure", *engine_exit_code_for_test);
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

    // A clean client disconnect does not own the server process lifetime. The
    // launcher observes QEMU itself and sends the shutdown signal after QEMU
    // exits, avoiding a timing-dependent grace period between disconnect and
    // waitpid. A transport failure still ends the process immediately.
    bool engine_ended_while_serving = false;
    while ((!serving_finished.load() || !serving_failed.load()) &&
           !engine_finished.load(std::memory_order_acquire)) {
      const timespec timeout{.tv_sec = 0, .tv_nsec = kSignalPollNanoseconds};
      const int signal = sigtimedwait(&signal_mask.handled(), nullptr, &timeout);
      const ServerSignalAction action = action_for_signal(signal);
      if (action == ServerSignalAction::Stop) {
        break;
      }
      if (action == ServerSignalAction::DeliverInterrupt) {
        // Queue onto the simulation owner with CP and SDMA completions. Ring
        // publication spans guest memory, register state, and message delivery,
        // so no second thread may execute it concurrently.
        if (!device.request_interrupt(
                {.client_id = kRequestedInterruptClient, .source_id = kRequestedInterruptSource}))
          util::Logger::warn("vfu: cannot queue an interrupt request");
        continue;
      }
      if (signal < 0 && errno != EAGAIN && errno != EINTR) {
        util::Logger::warn("vfu: failed while waiting for a shutdown signal");
        status = 1;
        break;
      }
    }
    engine_ended_while_serving = engine_finished.load(std::memory_order_acquire);

    // Stop and join before the host is destroyed, so no callback can run against
    // a context that is being torn down.
    serving_thread.request_stop();
    serving_thread.join();
    host.detach();
    if (serving_failed.load()) {
      status = 1;
    }

    if (engine_ended_while_serving) {
      status = engine_status.code == 0 ? 1 : engine_status.code;
      util::Logger::warn(std::format("vfu: simulation engine stopped while serving: {} (code {})",
                                     engine_status.message, engine_status.code));
    }
  }

  engine_run.stop();
  if (engine_status.code != 0)
    status = engine_status.code;

  // Every external source of work is quiesced before the engine: the serving
  // thread is stopped and joined above and the host is gone with the scope, so
  // nothing can call into the device while the engine is stopped and joined.
  // The device outlives both, because the topology owns it and the engine owns
  // that.
  signal_mask.restore();

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

} // namespace

int run_vfio_server(const std::string &config_path, const std::string &socket_path, int ready_fd) {
  return run_vfio_server_impl(config_path, socket_path, ready_fd, std::nullopt);
}

int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::string &socket_path, int ready_fd,
                                              int exit_code) {
  return run_vfio_server_impl(config_path, socket_path, ready_fd, exit_code);
}

} // namespace rocjitsu
