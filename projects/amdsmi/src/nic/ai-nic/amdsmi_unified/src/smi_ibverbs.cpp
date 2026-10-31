/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "smi_ibverbs.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>

#ifdef LIBIBVERBS_INSTALLED

#include <dlfcn.h>

#include <cstring>

namespace {

uint16_t mtu_enum_to_bytes(enum ibv_mtu mtu) {
  switch (mtu) {
    case IBV_MTU_256:
      return 256;
    case IBV_MTU_512:
      return 512;
    case IBV_MTU_1024:
      return 1024;
    case IBV_MTU_2048:
      return 2048;
    case IBV_MTU_4096:
      return 4096;
  }
  return 0;
}

}  // namespace

namespace smi_ibverbs {

IbverbsLoader& IbverbsLoader::instance() {
  static IbverbsLoader inst;
  return inst;
}

IbverbsLoader::IbverbsLoader() {
  /*
   * dlopen() the libibverbs SONAME, resolved through the usual dynamic
   * linker search rules (ld.so.cache, LD_LIBRARY_PATH, default dirs).
   * The SONAME is hardcoded rather than detected at build time.
   *
   * If this fails, the loader stays unavailable and queries return nullopt.
   */
  handle_ = ::dlopen("libibverbs.so.1", RTLD_LAZY | RTLD_LOCAL);
  if (!handle_) {
    return;
  }

  get_device_list =
      reinterpret_cast<decltype(get_device_list)>(::dlsym(handle_, "ibv_get_device_list"));
  free_device_list =
      reinterpret_cast<decltype(free_device_list)>(::dlsym(handle_, "ibv_free_device_list"));
  get_device_name =
      reinterpret_cast<decltype(get_device_name)>(::dlsym(handle_, "ibv_get_device_name"));
  open_device = reinterpret_cast<decltype(open_device)>(::dlsym(handle_, "ibv_open_device"));
  close_device = reinterpret_cast<decltype(close_device)>(::dlsym(handle_, "ibv_close_device"));
  query_port = reinterpret_cast<decltype(query_port)>(::dlsym(handle_, "ibv_query_port"));

  available_ = get_device_list && free_device_list && get_device_name && open_device &&
               close_device && query_port;
}

std::optional<PortMtu> query_port_mtu_in_process(const std::string& rdma_dev, uint8_t port_num) {
  auto& ibv = IbverbsLoader::instance();
  if (!ibv.is_available()) {
    return std::nullopt;
  }

  int num_devices = 0;
  struct ibv_device** dev_list = ibv.get_device_list(&num_devices);
  if (!dev_list) {
    return std::nullopt;
  }

  std::optional<PortMtu> result;
  for (int i = 0; i < num_devices; i++) {
    const char* name = ibv.get_device_name(dev_list[i]);
    if (!name || rdma_dev != name) {
      continue;
    }

    struct ibv_context* ctx = ibv.open_device(dev_list[i]);
    if (!ctx) {
      break;
    }

    struct ibv_port_attr attr;
    std::memset(&attr, 0, sizeof(attr));
    if (ibv.query_port(ctx, port_num, &attr) == 0) {
      uint16_t active = mtu_enum_to_bytes(attr.active_mtu);
      uint16_t max = mtu_enum_to_bytes(attr.max_mtu);
      if (active != 0 && max != 0) {
        result = PortMtu{active, max};
      }
    }

    ibv.close_device(ctx);
    break;
  }

  ibv.free_device_list(dev_list);
  return result;
}

}  // namespace smi_ibverbs

#endif  // LIBIBVERBS_INSTALLED

namespace smi_ibverbs {

namespace {

constexpr int kFatalSignals[] = {SIGSEGV, SIGBUS, SIGABRT, SIGILL, SIGFPE};

// A handler stops the kernel and systemd-coredump from logging a crash on every init.
void exit_on_fatal_signal(int /*signal_number*/) { ::_exit(1); }

void handle_fatal_signals() {
  struct sigaction action = {};
  action.sa_handler = exit_on_fatal_signal;
  ::sigemptyset(&action.sa_mask);
  for (const int signal_number : kFatalSignals) {
    ::sigaction(signal_number, &action, nullptr);
  }
}

}  // namespace

std::optional<PortMtu> run_isolated(const MtuProbe_t& probe, std::chrono::milliseconds timeout) {
  int fds[2];
  if (::pipe2(fds, O_CLOEXEC) != 0) {
    return std::nullopt;
  }

  const pid_t child = ::fork();
  if (child < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    return std::nullopt;
  }

  if (child == 0) {
    // Only _exit() leaves the child: the parent's buffers and atexit handlers must not run twice.
    const struct rlimit no_core = {0, 0};
    ::setrlimit(RLIMIT_CORE, &no_core);
    handle_fatal_signals();
    ::close(fds[0]);
    const auto mtu = probe();
    if (mtu.has_value()) {
      const auto written = ::write(fds[1], &mtu.value(), sizeof(PortMtu));
      ::_exit((written == static_cast<ssize_t>(sizeof(PortMtu))) ? 0 : 1);
    }
    ::_exit(0);
  }

  ::close(fds[1]);
  auto answer = PortMtu{};
  auto received = size_t{0};
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (received < sizeof(PortMtu)) {
    const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (left.count() <= 0) {
      break;
    }
    struct pollfd waiting = {fds[0], POLLIN, 0};
    const int ready = ::poll(&waiting, 1, static_cast<int>(left.count()));
    if ((ready < 0) && (errno == EINTR)) {
      continue;
    }
    if (ready <= 0) {
      break;
    }
    const auto got =
        ::read(fds[0], reinterpret_cast<char*>(&answer) + received, sizeof(PortMtu) - received);
    if (got <= 0) {
      break;
    }
    received += static_cast<size_t>(got);
  }
  ::close(fds[0]);

  // A child that answered exits by itself; anything still running is hung and gets killed.
  ::kill(child, SIGKILL);
  auto status = int{0};
  while ((::waitpid(child, &status, 0) < 0) && (errno == EINTR)) {
  }

  if (received != sizeof(PortMtu)) {
    return std::nullopt;
  }
  return answer;
}

std::optional<PortMtu> query_port_mtu(const std::string& rdma_dev, uint8_t port_num) {
#ifdef LIBIBVERBS_INSTALLED
  // Load the library before forking so an absent one costs no child and the child reuses the map.
  if (!IbverbsLoader::instance().is_available()) {
    return std::nullopt;
  }
  return run_isolated([&]() { return query_port_mtu_in_process(rdma_dev, port_num); },
                      kMtuQueryTimeout);
#else
  static_cast<void>(rdma_dev);
  static_cast<void>(port_num);
  return std::nullopt;
#endif
}

}  // namespace smi_ibverbs
