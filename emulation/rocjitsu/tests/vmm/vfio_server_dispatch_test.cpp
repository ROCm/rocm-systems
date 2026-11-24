// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "scoped_temp.h"

#include "embedded_schema.h"
#include "rocjitsu/config/config_loader.h"
#include "rocjitsu/vm/amdgpu/command_processor.h"
#include "rocjitsu/vm/soc.h"
#include "rocjitsu/vmm/vfu/vfio_server.h"
#include "util/diagnostic.h"

#include <gtest/gtest.h>

#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <iterator>
#include <regex>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {

using rocjitsu::test::initialize_runtime_directory;

/// @brief How long a server is given to start, report and stop.
/// @details Far longer than a healthy server needs, so that only a server that
/// is stuck runs into it; a sanitizer build on a loaded machine is not stuck.
constexpr std::chrono::seconds kServerDeadline{30};

/// @brief The one-GPU config the other server tests serve.
std::filesystem::path single_function_config() {
  return std::filesystem::path(RJ_VFU_TEST_CONFIG_PATH);
}

/// @brief The same machine four times over, which a server presents as a
/// group of four functions.
std::filesystem::path grouped_function_config() {
  return single_function_config().parent_path() / "gfx1250_mi455x_kmd_4gpu.json";
}

/// @brief Read a shipped config and make it ask for @p threads dispatch threads.
/// @details The shipped configs either leave the width to the allocation table,
/// which resolves differently on every host, or pin it to one. A test of the
/// width has to state it, and the server reads a config only from a file, so
/// the request is written into the text. A top-level member of a shipped config
/// starts its own line; the entries of the allocation table, which carry the
/// same key, do not.
/// @param[in] config_path Shipped config to start from.
/// @param[in] threads The width the returned text asks for.
/// @param[out] config_text The config, asking for @p threads.
[[nodiscard]] testing::AssertionResult config_asking_for(const std::filesystem::path &config_path,
                                                         uint32_t threads,
                                                         std::string &config_text) {
  std::ifstream input(config_path, std::ios::binary);
  if (!input) {
    return testing::AssertionFailure() << "cannot read " << config_path;
  }
  const std::string shipped{std::istreambuf_iterator<char>(input),
                            std::istreambuf_iterator<char>()};
  const std::regex top_level_request(R"(\n[ \t]*"cpu_dispatch_threads"[ \t]*:[ \t]*[0-9]+[ \t]*,)");
  config_text = std::regex_replace(shipped, top_level_request, "");
  const std::size_t opening_brace = config_text.find('{');
  if (opening_brace == std::string::npos) {
    return testing::AssertionFailure() << config_path << " is not a JSON object";
  }
  config_text.insert(opening_brace + 1, std::format(R"("cpu_dispatch_threads":{},)", threads));
  return testing::AssertionSuccess();
}

/// @brief What the loader makes of a config, read without serving it.
struct ResolvedConfig {
  /// @brief The dispatch width resolved for each local GPU, in ordinal order.
  std::vector<uint32_t> dispatch;
  /// @brief The widest the first GPU can dispatch: one submitting thread, and
  /// one worker for each compute unit beyond the first in every XCD.
  uint32_t dispatch_capacity = 0;
};

/// @brief Load @p config_text the way a server would and report what it resolves to.
/// @details Every test here states what its config asks for before it looks at
/// what a server did with it. Without that, a config the edit above failed to
/// change would be served at whatever width it happened to resolve to, and a
/// server that ignored the request could not be told from one that obeyed it.
[[nodiscard]] testing::AssertionResult resolve(const std::string &config_text,
                                               ResolvedConfig &resolved) {
  util::StringDiagnostic diagnostic;
  util::FailureOr<rocjitsu::config::LoadedConfig> loaded =
      rocjitsu::config::load_config_from_string(config_text, rocjitsu::kEmbeddedSchema,
                                                diagnostic.emitter());
  if (loaded.failed()) {
    return testing::AssertionFailure() << diagnostic.message();
  }
  rocjitsu::SoC *soc = loaded.value().soc();
  if (soc == nullptr) {
    return testing::AssertionFailure() << "the config describes no GPU";
  }
  resolved.dispatch = loaded.value().execution_threads.dispatch;
  std::size_t capacity = 1;
  soc->for_each_cp([&capacity](const rocjitsu::amdgpu::CommandProcessor *command_processor) {
    if (!command_processor->compute_units().empty()) {
      capacity += command_processor->compute_units().size() - 1;
    }
  });
  resolved.dispatch_capacity = static_cast<uint32_t>(capacity);
  return testing::AssertionSuccess();
}

/// @brief Read exactly @p bytes from @p fd, giving up at @p deadline.
/// @returns Whether every byte arrived. End of stream before then is a failure.
[[nodiscard]] bool read_exactly(int fd, std::span<std::byte> bytes,
                                std::chrono::steady_clock::time_point deadline) {
  std::size_t received = 0;
  while (received < bytes.size()) {
    const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
        deadline - std::chrono::steady_clock::now());
    if (remaining.count() <= 0) {
      return false;
    }
    pollfd wait{.fd = fd, .events = POLLIN | POLLHUP, .revents = 0};
    const int ready = ::poll(&wait, 1, static_cast<int>(remaining.count()));
    if (ready < 0 && errno == EINTR) {
      continue;
    }
    if (ready <= 0) {
      return false;
    }
    const ssize_t count = ::read(fd, bytes.data() + received, bytes.size() - received);
    if (count < 0 && errno == EINTR) {
      continue;
    }
    if (count <= 0) {
      return false;
    }
    received += static_cast<std::size_t>(count);
  }
  return true;
}

/// @brief A vfio-user server in a child process that reports its dispatch widths.
///
/// @details The server has to be a process of its own: it stops on a signal
/// sent to the process, which a server on a thread of the test would share with
/// every other test. The widths therefore cross a pipe. The child is killed on
/// destruction if the test did not stop it.
class ReportingServer {
public:
  ReportingServer() = default;
  ~ReportingServer() {
    if (child_ > 0) {
      (void)::kill(child_, SIGKILL);
      int status = 0;
      while (::waitpid(child_, &status, 0) == -1 && errno == EINTR) {
      }
    }
    close_descriptor(dispatch_read_);
  }

  ReportingServer(const ReportingServer &) = delete;
  ReportingServer &operator=(const ReportingServer &) = delete;

  /// @brief Start a server on @p config_text and collect what it reports.
  /// @param[in] config_text Simulation config to serve.
  /// @param[in] function_count How many functions to present, which must be how
  ///            many local GPUs the config describes.
  /// @details Returns once the server has reported a width for every function
  /// and then that all of its sockets are listening.
  [[nodiscard]] testing::AssertionResult start(const std::string &config_text,
                                               uint32_t function_count) {
    if (const testing::AssertionResult created = config_.initialize("rj-vfu-dispatch-"); !created) {
      return created;
    }
    if (const testing::AssertionResult written = config_.write(config_text); !written) {
      return written;
    }
    if (const testing::AssertionResult created = initialize_runtime_directory(runtime_, "rj-v-");
        !created) {
      return created;
    }
    std::vector<std::string> socket_paths;
    for (uint32_t function_index = 0; function_index < function_count; ++function_index) {
      socket_paths.push_back(std::format("{}/gpu{}.sock", runtime_.path(), function_index));
    }

    std::array<int, 2> ready_pipe{-1, -1};
    std::array<int, 2> dispatch_pipe{-1, -1};
    if (::pipe(ready_pipe.data()) != 0) {
      return testing::AssertionFailure() << "cannot create a pipe: " << std::strerror(errno);
    }
    if (::pipe(dispatch_pipe.data()) != 0) {
      const int pipe_error = errno;
      close_descriptor(ready_pipe[0]);
      close_descriptor(ready_pipe[1]);
      return testing::AssertionFailure() << "cannot create a pipe: " << std::strerror(pipe_error);
    }
    child_ = ::fork();
    const int fork_error = errno;
    if (child_ == 0) {
      ::close(ready_pipe[0]);
      ::close(dispatch_pipe[0]);
      ::_exit(rocjitsu::run_vfio_server_reporting_dispatch_for_test(
          config_.path(), socket_paths, ready_pipe[1], dispatch_pipe[1]));
    }
    // The child holds the only other write ends, so its exit is an end of
    // stream here rather than a wait that runs to the deadline.
    close_descriptor(ready_pipe[1]);
    close_descriptor(dispatch_pipe[1]);
    dispatch_read_ = dispatch_pipe[0];
    if (child_ < 0) {
      close_descriptor(ready_pipe[0]);
      return testing::AssertionFailure() << "cannot fork a server: " << std::strerror(fork_error);
    }

    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + kServerDeadline;
    widths_.assign(function_count, 0);
    const bool reported =
        read_exactly(dispatch_read_, std::as_writable_bytes(std::span(widths_)), deadline);
    std::array<std::byte, 1> ready{};
    const bool listening = reported && read_exactly(ready_pipe[0], ready, deadline) &&
                           std::to_integer<uint8_t>(ready[0]) == 1;
    close_descriptor(ready_pipe[0]);
    if (!reported) {
      return testing::AssertionFailure()
             << "the server did not report a dispatch width for each of " << function_count
             << " function(s)";
    }
    if (!listening) {
      return testing::AssertionFailure() << "the server did not report that it was ready";
    }
    return testing::AssertionSuccess();
  }

  /// @brief The dispatch width the server reported for each function, in order.
  [[nodiscard]] const std::vector<uint32_t> &widths() const { return widths_; }

  /// @brief Ask the server to stop, and expect it to leave in good order
  /// having reported nothing beyond one width for each function.
  [[nodiscard]] testing::AssertionResult stop() {
    if (child_ <= 0) {
      return testing::AssertionFailure() << "no server is running";
    }
    if (::kill(child_, SIGTERM) != 0) {
      return testing::AssertionFailure() << "cannot signal the server: " << std::strerror(errno);
    }
    const std::chrono::steady_clock::time_point deadline =
        std::chrono::steady_clock::now() + kServerDeadline;
    int status = 0;
    while (true) {
      const pid_t waited = ::waitpid(child_, &status, WNOHANG);
      if (waited == child_) {
        break;
      }
      if (waited < 0 && errno != EINTR) {
        return testing::AssertionFailure() << "cannot reap the server: " << std::strerror(errno);
      }
      if (std::chrono::steady_clock::now() >= deadline) {
        return testing::AssertionFailure() << "the server did not stop when asked to";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    child_ = -1;
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
      return testing::AssertionFailure()
             << "the server did not exit cleanly, wait status " << status;
    }
    std::array<std::byte, 1> surplus{};
    if (read_exactly(dispatch_read_, surplus, deadline)) {
      return testing::AssertionFailure() << "the server reported more widths than functions";
    }
    return testing::AssertionSuccess();
  }

private:
  static void close_descriptor(int &fd) {
    if (fd >= 0) {
      ::close(fd);
    }
    fd = -1;
  }

  rocjitsu::test::ScopedTempFile config_;
  rocjitsu::test::ScopedTempDirectory runtime_;
  pid_t child_ = -1;
  int dispatch_read_ = -1;
  std::vector<uint32_t> widths_;
};

constexpr uint32_t kGroupedFunctions = 4;

} // namespace

// A lone function is the configuration the dispatch workers were measured in,
// with guest memory copied through mapped windows rather than through the
// transport, so it gets the width launch mode would give the same config.
TEST(VfioServerDispatch, ALoneFunctionDispatchesAsWideAsItsConfigAsks) {
  constexpr uint32_t kRequested = 4;
  std::string config;
  ASSERT_TRUE(config_asking_for(single_function_config(), kRequested, config));
  ResolvedConfig resolved;
  ASSERT_TRUE(resolve(config, resolved));
  ASSERT_EQ(resolved.dispatch, std::vector<uint32_t>{kRequested});
  ASSERT_GT(resolved.dispatch_capacity, kRequested) << "the request must not be capped here";

  ReportingServer server;
  ASSERT_TRUE(server.start(config, 1));
  EXPECT_EQ(server.widths(), std::vector<uint32_t>{kRequested});
  EXPECT_TRUE(server.stop());
}

// A SoC cannot use more dispatch threads than one submitter and a worker for
// each further compute unit of every XCD, so a request beyond that is cut down
// to it rather than honoured with workers that could never be given work.
TEST(VfioServerDispatch, ALoneFunctionIsCappedByItsComputeUnits) {
  constexpr uint32_t kRequested = 100'000;
  std::string config;
  ASSERT_TRUE(config_asking_for(single_function_config(), kRequested, config));
  ResolvedConfig resolved;
  ASSERT_TRUE(resolve(config, resolved));
  const uint32_t capacity = resolved.dispatch_capacity;
  ASSERT_GT(capacity, 1u) << "a machine with nothing to parallelise cannot show a cap";
  ASSERT_LT(capacity, kRequested) << "the request must exceed what the machine can use";

  ReportingServer server;
  ASSERT_TRUE(server.start(config, 1));
  EXPECT_EQ(server.widths(), std::vector<uint32_t>{capacity})
      << "asked for " << kRequested << " dispatch threads";
  EXPECT_TRUE(server.stop());
}

TEST(VfioServerDispatch, ALoneFunctionAskingForOneThreadStaysSerial) {
  std::string config;
  ASSERT_TRUE(config_asking_for(single_function_config(), 1, config));
  ResolvedConfig resolved;
  ASSERT_TRUE(resolve(config, resolved));
  ASSERT_EQ(resolved.dispatch, std::vector<uint32_t>{1});

  ReportingServer server;
  ASSERT_TRUE(server.start(config, 1));
  EXPECT_EQ(server.widths(), std::vector<uint32_t>{1});
  EXPECT_TRUE(server.stop());
}

// Functions served as a group copy guest memory through the transport, one
// copy at a time, because an address there may belong to a peer and only the
// group's grant table can tell. Dispatch workers queueing for serialized memory
// access were measured several times slower than serial dispatch, so a group
// stays serial even when its config resolves to a width launch mode would
// honour.
TEST(VfioServerDispatch, GroupedFunctionsStaySerialWhateverTheirConfigAsks) {
  constexpr uint32_t kRequested = 4;
  std::string config;
  ASSERT_TRUE(config_asking_for(grouped_function_config(), kRequested, config));
  ResolvedConfig resolved;
  ASSERT_TRUE(resolve(config, resolved));
  ASSERT_EQ(resolved.dispatch, std::vector<uint32_t>(kGroupedFunctions, kRequested))
      << "the config must ask every GPU for more than one thread, or serial proves nothing";

  ReportingServer server;
  ASSERT_TRUE(server.start(config, kGroupedFunctions));
  EXPECT_EQ(server.widths(), std::vector<uint32_t>(kGroupedFunctions, 1));
  EXPECT_TRUE(server.stop());
}
