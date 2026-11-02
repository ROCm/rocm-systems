// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/vfio_server.h"

#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"
#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device_spec.h"
#include "rocjitsu/vm/amdgpu/pci/register_symbols.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vm/topology_assembly.h"
#include "rocjitsu/vm/virtual_machine.h"
#include "rocjitsu/vmm/vfu/vfio_device_host.h"
#include "simdojo/sim/simulation.h"
#include "util/diagnostic.h"
#include "util/log.h"

#include "embedded_schema.h"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <filesystem>
#include <format>
#include <memory>
#include <optional>
#include <stop_token>
#include <sys/un.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

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
/// @brief Time for the VMM to close every function after the first leaves.
constexpr auto kGroupDisconnectGrace = std::chrono::seconds(5);

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

/// @brief Reject path aliases and paths libvfio-user cannot bind as AF_UNIX sockets.
bool validate_socket_paths(const std::vector<std::string> &socket_paths, std::size_t gpu_count) {
  if (socket_paths.empty() || socket_paths.size() != gpu_count) {
    util::Logger::warn(std::format("vfu: {} local GPU(s) require exactly {} socket path(s); got {}",
                                   gpu_count, gpu_count, socket_paths.size()));
    return false;
  }

  std::unordered_set<std::string> normalized_paths;
  for (const std::string &socket_path : socket_paths) {
    if (socket_path.empty() || socket_path.find('\0') != std::string::npos) {
      util::Logger::warn("vfu: socket paths must be nonempty filesystem paths without NUL bytes");
      return false;
    }
    if (socket_path.size() >= sizeof(sockaddr_un::sun_path)) {
      util::Logger::warn(std::format("vfu: socket path is too long for AF_UNIX: {}", socket_path));
      return false;
    }

    try {
      const std::string normalized =
          std::filesystem::absolute(socket_path).lexically_normal().string();
      if (!normalized_paths.insert(normalized).second) {
        util::Logger::warn(
            std::format("vfu: socket paths resolve to the same lexical path: {}", socket_path));
        return false;
      }
    } catch (const std::exception &error) {
      util::Logger::warn(
          std::format("vfu: cannot normalize socket path {}: {}", socket_path, error.what()));
      return false;
    }
  }
  return true;
}

enum class ServingOutcome { Running, RequestedStop, Stopped, Failed };

int run_vfio_server_impl(const std::string &config_path,
                         const std::vector<std::string> &socket_paths, int ready_fd,
                         std::optional<int> engine_exit_code_for_test) {
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
    util::StringDiagnostic diagnostic;
    auto maybe_loaded = config::load_config(config_path, kEmbeddedSchema, diagnostic.emitter());
    if (maybe_loaded.failed()) {
      util::Logger::warn(std::format("vfu: cannot read {}: {}", config_path, diagnostic.message()));
      return 1;
    }
    loaded = std::move(maybe_loaded.value());
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: cannot read {}: {}", config_path, error.what()));
    return 1;
  }
  SoC *soc = loaded.soc();
  if (soc == nullptr) {
    util::Logger::warn(std::format("vfu: {} describes no GPU to present", config_path));
    return 1;
  }
  if (!validate_socket_paths(socket_paths, loaded.num_gpus)) {
    return 1;
  }
  if (loaded.num_gpus > 1 && loaded.devices.size() != loaded.num_gpus) {
    util::Logger::warn(std::format("vfu: {} local GPUs have only {} device identities",
                                   loaded.num_gpus, loaded.devices.size()));
    return 1;
  }
  RegisterSymbols symbols;
  add_pre_discovery_symbols(symbols, GpuPciDevice::kRegisterBar);
  // A PCI function retains its trace until the engine-owned topology destroys
  // it, so traces must be declared before the engine.
  std::vector<std::unique_ptr<BarAccessTrace>> traces;

  // The whole local node is built before create(): the engine sizes its one
  // partition and gives every function an engine pointer during creation. The
  // guest's driver owns these GPU models; the host KFD driver stays unopened.
  simdojo::SimulationEngine::Config engine_config = loaded.engine_config;
  engine_config.max_ticks = 0;
  engine_config.await_primaries = true;
  if (engine_config.num_threads != 1) {
    // All local GPUs and their PCI functions must live in the same partition.
    util::Logger::warn(std::format("vfu: serving single-threaded; {} asked for {} threads",
                                   config_path, engine_config.num_threads));
    engine_config.num_threads = 1;
  }
  simdojo::SimulationEngine engine(engine_config);

  util::StringDiagnostic diagnostic;
  VirtualMachine *machine_ptr = nullptr;
  try {
    auto machine = assemble_virtual_machine_topology(loaded, engine.topology(),
                                                     /*daemon_mode=*/false, diagnostic.emitter());
    if (machine.failed()) {
      util::Logger::warn("vfu: cannot assemble the local node: ", diagnostic.message());
      return 1;
    }
    machine_ptr = machine.value();
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: cannot assemble the local node: {}", error.what()));
    return 1;
  }
  if (machine_ptr->num_socs() != socket_paths.size()) {
    util::Logger::warn(std::format("vfu: assembled node has {} local GPU(s), expected {}",
                                   machine_ptr->num_socs(), socket_paths.size()));
    return 1;
  }

  // Each trace is separately allocated because its mutex makes it nonmovable.
  std::vector<GpuPciDevice *> pci_devices;
  class FrontendShutdown {
  public:
    explicit FrontendShutdown(const std::vector<GpuPciDevice *> &devices) : devices_(devices) {}
    ~FrontendShutdown() { (void)shutdown(); }

    [[nodiscard]] bool shutdown() {
      if (shut_down_)
        return complete_;
      shut_down_ = true;
      for (GpuPciDevice *device : devices_)
        if (!device->shutdown_frontend())
          complete_ = false;
      return complete_;
    }

  private:
    const std::vector<GpuPciDevice *> &devices_;
    bool shut_down_ = false;
    bool complete_ = true;
  } frontend_shutdown(pci_devices);

  try {
    traces.reserve(socket_paths.size());
    pci_devices.reserve(socket_paths.size());
    for (std::size_t gpu_index = 0; gpu_index < socket_paths.size(); ++gpu_index) {
      SoC *local_soc = machine_ptr->soc(static_cast<uint32_t>(gpu_index));
      if (local_soc == nullptr) {
        util::Logger::warn(std::format("vfu: local GPU {} has no SoC", gpu_index));
        return 1;
      }
      traces.push_back(std::make_unique<BarAccessTrace>(symbols));
      const config::KfdDeviceConfig &identity =
          loaded.devices.empty() ? loaded.device : loaded.devices[gpu_index];
      const std::string name = socket_paths.size() == 1 ? "pci" : std::format("pci{}", gpu_index);
      auto *device = static_cast<GpuPciDevice *>(machine_ptr->add_child(
          std::make_unique<GpuPciDevice>(name, gpu_pci_spec_from_config(identity, loaded.pci),
                                         traces.back().get(), local_soc)));
      pci_devices.push_back(device);
      if (!device->usable())
        return 1;
    }
    if (engine.create(diagnostic.emitter()).failed()) {
      util::Logger::warn("vfu: cannot create the local GPU node: ", diagnostic.message());
      return 1;
    }
  } catch (const std::exception &error) {
    util::Logger::warn(std::format("vfu: cannot create the local GPU node: {}", error.what()));
    return 1;
  }

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
      if (stopped_)
        return;
      engine_.request_exit("serving ended", 0);
      if (thread_.joinable())
        thread_.join();
      stopped_ = true;
    }

  private:
    simdojo::SimulationEngine &engine_;
    std::jthread thread_;
    bool stopped_ = false;
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
    status = 1;
  } else if (engine_finished.load(std::memory_order_acquire)) {
    util::Logger::warn(std::format("vfu: simulation engine stopped before serving: {} (code {})",
                                   engine_status.message, engine_status.code));
    status = engine_status.code == 0 ? 1 : engine_status.code;
  }

  // Declare hosts before threads so exceptional unwinding also joins every
  // serving thread before destroying a host or stopping the engine.
  std::vector<std::unique_ptr<VfioDeviceHost>> hosts;
  std::vector<std::unique_ptr<std::atomic<ServingOutcome>>> serving_outcomes;
  std::vector<std::jthread> serving_threads;
  if (status == 0) {
    try {
      hosts.reserve(socket_paths.size());
      for (std::size_t gpu_index = 0; gpu_index < socket_paths.size(); ++gpu_index) {
        hosts.push_back(
            std::make_unique<VfioDeviceHost>(socket_paths[gpu_index], *pci_devices[gpu_index]));
        if (!hosts.back()->build()) {
          status = 1;
          break;
        }
      }
    } catch (const std::exception &error) {
      util::Logger::warn(std::format("vfu: cannot build a PCI transport: {}", error.what()));
      status = 1;
    }
  }

  if (status == 0) {
    try {
      serving_outcomes.reserve(hosts.size());
      serving_threads.reserve(hosts.size());
      for (const std::unique_ptr<VfioDeviceHost> &host : hosts) {
        serving_outcomes.push_back(
            std::make_unique<std::atomic<ServingOutcome>>(ServingOutcome::Running));
        VfioDeviceHost *host_ptr = host.get();
        std::atomic<ServingOutcome> *outcome = serving_outcomes.back().get();
        // Each host has its own protocol loop; all inherit the blocked mask.
        serving_threads.emplace_back([host_ptr, outcome](std::stop_token stop_token) {
          try {
            const VfioDeviceHost::ServeResult result = host_ptr->run(stop_token);
            const ServingOutcome reason =
                result == VfioDeviceHost::ServeResult::Failed
                    ? ServingOutcome::Failed
                    : (stop_token.stop_requested() ? ServingOutcome::RequestedStop
                                                   : ServingOutcome::Stopped);
            outcome->store(reason, std::memory_order_release);
          } catch (const std::exception &error) {
            util::Logger::warn(std::format("vfu: PCI serving failed: {}", error.what()));
            outcome->store(ServingOutcome::Failed, std::memory_order_release);
          } catch (...) {
            util::Logger::warn("vfu: PCI serving failed with a non-standard exception");
            outcome->store(ServingOutcome::Failed, std::memory_order_release);
          }
        });
      }
    } catch (const std::exception &error) {
      util::Logger::warn(std::format("vfu: cannot start PCI serving threads: {}", error.what()));
      status = 1;
    }
  }

  // Every host has bound and every serving thread exists before readiness is
  // published. Closing the pipe without its byte reports any setup failure.
  if (status == 0 && engine_finished.load(std::memory_order_acquire)) {
    util::Logger::warn("vfu: simulation engine stopped before all sockets were ready");
    status = 1;
  }
  if (status == 0) {
    for (std::size_t gpu_index = 0; gpu_index < serving_outcomes.size(); ++gpu_index) {
      const ServingOutcome reason = serving_outcomes[gpu_index]->load(std::memory_order_acquire);
      if (reason == ServingOutcome::Failed ||
          (socket_paths.size() > 1 && reason != ServingOutcome::Running)) {
        util::Logger::warn(std::format("vfu: PCI function {} stopped before readiness", gpu_index));
        status = 1;
        break;
      }
    }
  }
  if (status == 0 && !readiness.signal()) {
    util::Logger::warn("vfu: cannot report server readiness to the launcher");
    status = 1;
  }
  if (status == 0) {
    if (engine_exit_code_for_test.has_value())
      engine.request_exit("test-requested simulation failure", *engine_exit_code_for_test);

    for (std::size_t gpu_index = 0; gpu_index < hosts.size(); ++gpu_index) {
      const config::KfdDeviceConfig &identity =
          loaded.devices.empty() ? loaded.device : loaded.devices[gpu_index];
      util::Logger::warn(std::format("vfu: serving {} on {}",
                                     identity.marketing_name.empty()
                                         ? pci_devices[gpu_index]->name()
                                         : identity.marketing_name,
                                     socket_paths[gpu_index]));
    }

    // With one function, an orderly permanent disconnect leaves process
    // lifetime to the launcher. A multi-function VMM closes its sockets one
    // after another, so the first observed closure starts a bounded wait for
    // every function to leave before judging the group incomplete. A closed
    // function may still be draining a transport session and resetting.
    std::optional<std::chrono::steady_clock::time_point> first_closure;
    sigset_t shutdown_signals{};
    sigemptyset(&shutdown_signals);
    sigaddset(&shutdown_signals, SIGINT);
    sigaddset(&shutdown_signals, SIGTERM);
    while (true) {
      if (engine_finished.load(std::memory_order_acquire)) {
        status = engine_status.code == 0 ? 1 : engine_status.code;
        util::Logger::warn(std::format("vfu: simulation engine stopped while serving: {} (code {})",
                                       engine_status.message, engine_status.code));
        break;
      }
      bool session_ended = false;
      bool all_sockets_closed = socket_paths.size() > 1;
      for (std::size_t gpu_index = 0; gpu_index < serving_outcomes.size(); ++gpu_index) {
        const ServingOutcome outcome = serving_outcomes[gpu_index]->load(std::memory_order_acquire);
        if (outcome == ServingOutcome::Failed) {
          util::Logger::warn(
              std::format("vfu: PCI function {} stopped serving the node", gpu_index));
          status = 1;
          session_ended = true;
          break;
        }
        if (socket_paths.size() == 1)
          continue;

        const VfioDeviceHost::ClientSnapshot snapshot = hosts[gpu_index]->client_snapshot();
        if (outcome == ServingOutcome::Stopped && snapshot.completed_disconnects == 0) {
          util::Logger::warn(std::format(
              "vfu: PCI function {} stopped without a completed disconnect", gpu_index));
          status = 1;
          session_ended = true;
          break;
        }
        const bool socket_closed = !snapshot.attached && (snapshot.completed_disconnects != 0 ||
                                                          snapshot.disconnect_in_progress);
        if (socket_closed && !first_closure.has_value())
          first_closure = std::chrono::steady_clock::now();
        all_sockets_closed &= socket_closed;
      }
      if (session_ended)
        break;

      if (first_closure.has_value() && !all_sockets_closed &&
          std::chrono::steady_clock::now() - *first_closure >= kGroupDisconnectGrace) {
        // The launcher can already be shutting down the node. Give a pending
        // SIGTERM priority over the incomplete-group deadline before failing.
        const timespec no_wait{.tv_sec = 0, .tv_nsec = 0};
        const int pending = sigtimedwait(&shutdown_signals, nullptr, &no_wait);
        if (action_for_signal(pending) == ServerSignalAction::Stop)
          break;
        // A final function can close after its first snapshot but before the
        // deadline decision. Recheck at the decision point so that closure is
        // not reported as a missing peer merely because of poll ordering.
        bool peer_still_present = false;
        for (const std::unique_ptr<VfioDeviceHost> &host : hosts) {
          const VfioDeviceHost::ClientSnapshot snapshot = host->client_snapshot();
          if (snapshot.attached ||
              (snapshot.completed_disconnects == 0 && !snapshot.disconnect_in_progress)) {
            peer_still_present = true;
            break;
          }
        }
        if (peer_still_present) {
          util::Logger::warn("vfu: PCI function group did not disconnect together");
          status = 1;
          break;
        }
      }

      const timespec timeout{.tv_sec = 0, .tv_nsec = kSignalPollNanoseconds};
      const int signal = sigtimedwait(&signal_mask.handled(), nullptr, &timeout);
      const ServerSignalAction action = action_for_signal(signal);
      if (action == ServerSignalAction::Stop)
        break;
      if (action == ServerSignalAction::DeliverInterrupt) {
        // Queue onto the simulation owner with CP and SDMA completions. Ring
        // publication spans guest memory, register state, and message delivery,
        // so no second thread may execute it concurrently. A signal addresses
        // the node, not one particular function, and requests one per GPU.
        for (std::size_t gpu_index = 0; gpu_index < pci_devices.size(); ++gpu_index)
          if (!pci_devices[gpu_index]->request_interrupt(
                  {.client_id = kRequestedInterruptClient, .source_id = kRequestedInterruptSource}))
            util::Logger::warn(
                socket_paths.size() == 1
                    ? "vfu: cannot queue an interrupt request"
                    : std::format("vfu: cannot queue an interrupt request for GPU {}", gpu_index));
        continue;
      }
      if (signal < 0 && errno != EAGAIN && errno != EINTR) {
        util::Logger::warn("vfu: failed while waiting for a shutdown signal");
        status = 1;
        break;
      }
    }
  }

  // No callback may reach a host during detach or a frontend during teardown.
  for (std::jthread &thread : serving_threads)
    thread.request_stop();
  for (std::jthread &thread : serving_threads)
    thread.join();
  for (const std::unique_ptr<std::atomic<ServingOutcome>> &outcome : serving_outcomes) {
    const ServingOutcome reason = outcome->load(std::memory_order_acquire);
    if (reason == ServingOutcome::Failed)
      status = 1;
  }
  for (const std::unique_ptr<VfioDeviceHost> &host : hosts)
    host->detach();
  hosts.clear();
  // Stop event execution before dismantling queues, while the engine still
  // owns the complete topology. Frontends release their borrowed SoC state
  // before the engine object later destroys that topology.
  engine_run.stop();
  std::vector<InterruptRing> interrupt_rings;
  interrupt_rings.reserve(pci_devices.size());
  for (GpuPciDevice *device : pci_devices)
    interrupt_rings.push_back(device->interrupt_ring());
  if (!frontend_shutdown.shutdown())
    status = 1;
  if (engine_status.code != 0)
    status = engine_status.code;

  // Every external source of work is quiesced before the engine: the serving
  // threads are joined and the hosts are gone. The SoCs and PCI frontends
  // remain owned by the topology until after engine destruction.
  signal_mask.restore();

  for (std::size_t gpu_index = 0; gpu_index < pci_devices.size(); ++gpu_index) {
    // A clean disconnect clears the interrupt ring, so report only a ring the
    // guest left programmed. Keep per-function reports separate for diagnosis.
    const InterruptRing &ring = interrupt_rings[gpu_index];
    if (ring.programmed()) {
      util::Logger::warn(
          socket_paths.size() == 1
              ? std::format("vfu: the driver left {}", describe(ring))
              : std::format("vfu: GPU {}: the driver left {}", gpu_index, describe(ring)));
    }
    const std::string report = traces[gpu_index]->unmodeled_report();
    if (!report.empty()) {
      util::Logger::warn(
          socket_paths.size() == 1 ? report : std::format("vfu: GPU {}: {}", gpu_index, report));
    }
  }
  return status;
}

} // namespace

int run_vfio_server(const std::string &config_path, const std::string &socket_path, int ready_fd) {
  return run_vfio_server(config_path, std::vector<std::string>{socket_path}, ready_fd);
}

int run_vfio_server(const std::string &config_path, const std::vector<std::string> &socket_paths,
                    int ready_fd) {
  return run_vfio_server_impl(config_path, socket_paths, ready_fd, std::nullopt);
}

int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::string &socket_path, int ready_fd,
                                              int exit_code) {
  return run_vfio_server_with_engine_exit_for_test(
      config_path, std::vector<std::string>{socket_path}, ready_fd, exit_code);
}

int run_vfio_server_with_engine_exit_for_test(const std::string &config_path,
                                              const std::vector<std::string> &socket_paths,
                                              int ready_fd, int exit_code) {
  return run_vfio_server_impl(config_path, socket_paths, ready_fd, exit_code);
}

} // namespace rocjitsu
