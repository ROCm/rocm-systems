// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file daemon_test.cpp
/// @brief Runs HIP kernel tests against a shared rocjitsu daemon via the CLI.
///
/// Each fixture starts one session with `rocjitsu session start --daemon`,
/// which brings the emulator daemon up and holds it there, then runs each
/// workload into that same session with `rocjitsu exec --session <id>` and
/// stops it again. Paths are injected via CMake compile definitions.
///
/// The daemon is what these cases are about: several client processes
/// sharing one emulated machine's GPU memory. Everything the fixture used
/// to do by hand — forking a launcher, waiting for a socket to appear
/// under a runtime directory it had chosen, and LD_PRELOADing the
/// interposer onto each client — is now the CLI's job, and going through
/// it means these tests exercise the same path a user takes.

#include <gtest/gtest.h>

#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <vector>

namespace {

struct ProcessResult {
  std::string output;
  int exit_code = -1;
};

struct TestPaths {
  std::string cli_bin = RJ_CLI_BIN;
  std::string daemon_config = RJ_DAEMON_CONFIG;
  std::string daemon_config_2gpu = RJ_DAEMON_CONFIG_2GPU;
  std::string preload_lib = RJ_PRELOAD_LIB;
  std::string hip_vector_add_bin = RJ_HIP_VECTOR_ADD_BIN;
  std::string hip_memcpy_bin = RJ_HIP_MEMCPY_BIN;
  std::string hip_rccl_bin = RJ_HIP_RCCL_BIN;
  std::string daemon_logging_config = RJ_DAEMON_LOGGING_CONFIG;
  std::string interposer_dup_bin = RJ_INTERPOSER_DUP_BIN;
};

std::filesystem::path resolve_relative_to_exe(const std::filesystem::path &exe_dir,
                                              const char *path) {
  std::filesystem::path candidate(path);
  if (!candidate.is_absolute())
    candidate = exe_dir / candidate;
  return candidate.lexically_normal();
}

std::filesystem::path current_exe_dir() {
  std::error_code ec;
  std::filesystem::path exe = std::filesystem::read_symlink("/proc/self/exe", ec);
  if (ec)
    return {};
  return exe.parent_path();
}

bool installed_paths_exist(const TestPaths &paths) {
  // Only the UNCONDITIONALLY-built artifacts are required here. hip_rccl_bin is
  // deliberately excluded: the RCCL test binary (and the RcclDaemonTest cases
  // that use it) are built/registered only when RCCL_LIB is found at configure
  // time, so requiring it would make this check fail on non-RCCL installs and
  // wrongly fall back to build-tree paths that a pure install does not have.
  return std::filesystem::exists(paths.cli_bin) && std::filesystem::exists(paths.daemon_config) &&
         std::filesystem::exists(paths.daemon_config_2gpu) &&
         std::filesystem::exists(paths.preload_lib) &&
         std::filesystem::exists(paths.hip_vector_add_bin) &&
         std::filesystem::exists(paths.hip_memcpy_bin) &&
         std::filesystem::exists(paths.daemon_logging_config) &&
         std::filesystem::exists(paths.interposer_dup_bin);
}

TestPaths installed_paths(const std::filesystem::path &exe_dir) {
  return {
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_CLI_BIN).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_DAEMON_CONFIG).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_DAEMON_CONFIG_2GPU).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_PRELOAD_LIB).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_HIP_VECTOR_ADD_BIN).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_HIP_MEMCPY_BIN).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_HIP_RCCL_BIN).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_DAEMON_LOGGING_CONFIG).string(),
      resolve_relative_to_exe(exe_dir, RJ_INSTALLED_INTERPOSER_DUP_BIN).string(),
  };
}

TestPaths &test_paths() {
  static TestPaths paths = [] {
    std::filesystem::path exe_dir = current_exe_dir();
    if (!exe_dir.empty()) {
      TestPaths installed = installed_paths(exe_dir);
      if (installed_paths_exist(installed))
        return installed;
    }
    return TestPaths{};
  }();
  return paths;
}

const char *cli_bin() { return test_paths().cli_bin.c_str(); }

const char *daemon_config() { return test_paths().daemon_config.c_str(); }

const char *daemon_config_2gpu() { return test_paths().daemon_config_2gpu.c_str(); }

const char *preload_lib() { return test_paths().preload_lib.c_str(); }

const char *hip_vector_add_bin() { return test_paths().hip_vector_add_bin.c_str(); }

const char *hip_memcpy_bin() { return test_paths().hip_memcpy_bin.c_str(); }

const char *hip_rccl_bin() { return test_paths().hip_rccl_bin.c_str(); }

const char *daemon_logging_config() { return test_paths().daemon_logging_config.c_str(); }
const char *interposer_dup_bin() { return test_paths().interposer_dup_bin.c_str(); }

/// A path as a single shell word.
std::string quoted(const std::string &path) { return "'" + path + "'"; }

/// Run a shell command and collect what it wrote.
///
/// `merge_stderr` is what a failing case wants — the CLI's diagnostics
/// are on stderr, and a bare exit code says nothing — but it is wrong
/// wherever the output is read rather than reported: `session list`
/// writes ids to stdout and "no sessions are up" to stderr, and merging
/// the two turns "no sessions" into a session named `no`.
ProcessResult capture(const std::string &command, bool merge_stderr = true) {
  ProcessResult result;
  std::array<char, 4096> buf;
  FILE *pipe = popen((command + (merge_stderr ? " 2>&1" : " 2>/dev/null")).c_str(), "r");
  if (!pipe) {
    result.exit_code = -1;
    return result;
  }
  while (fgets(buf.data(), buf.size(), pipe) != nullptr)
    result.output += buf.data();
  int status = pclose(pipe);
  result.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return result;
}

/// One test's private view of the CLI's state.
///
/// Each case gets its own runtime and config roots, so `session list`
/// sees exactly the session this test started even when ctest runs the
/// suite with -j. ROCJITSU_LIB pins the interposer to the library this
/// build produced; without it the CLI would search the ROCm install
/// locations and could find an older one.
struct CliEnv {
  std::string tmp_dir;

  std::string prefix() const {
    std::string e = "env";
    e += " XDG_RUNTIME_DIR=" + quoted(tmp_dir);
    e += " ROCJITSU_CLI_RUNTIME_DIR=" + quoted(tmp_dir + "/runtime");
    e += " ROCJITSU_CLI_CONFIG_DIR=" + quoted(tmp_dir + "/config");
    e += " ROCJITSU_LIB=" + quoted(preload_lib());
    return e;
  }

  /// Apply the same variables to this process, for a forked child that
  /// goes on to exec the CLI directly.
  void apply() const {
    setenv("XDG_RUNTIME_DIR", tmp_dir.c_str(), 1);
    setenv("ROCJITSU_CLI_RUNTIME_DIR", (tmp_dir + "/runtime").c_str(), 1);
    setenv("ROCJITSU_CLI_CONFIG_DIR", (tmp_dir + "/config").c_str(), 1);
    setenv("ROCJITSU_LIB", preload_lib(), 1);
  }
};

/// A private temporary directory, removed with the fixture.
std::string make_temp_dir(const char *stem) {
  const char *xdg = std::getenv("XDG_RUNTIME_DIR");
  std::string tmpl = std::string(xdg ? xdg : "/tmp") + "/" + stem + "-XXXXXX";
  if (mkdtemp(tmpl.data()) == nullptr)
    return {};
  return tmpl;
}

/// The ids `rocjitsu session list` reports, one per line.
std::vector<std::string> live_sessions(const CliEnv &env) {
  ProcessResult r =
      capture(env.prefix() + " " + quoted(cli_bin()) + " session list", /*merge_stderr=*/false);
  std::vector<std::string> ids;
  if (r.exit_code != 0)
    return ids;
  std::istringstream lines(r.output);
  for (std::string line; std::getline(lines, line);) {
    while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
      line.pop_back();
    if (!line.empty())
      ids.push_back(line);
  }
  return ids;
}

TEST(RocjitsuCliDaemon, LaunchesApplicationAfterDaemonIsReady) {
  if (!std::filesystem::exists(cli_bin()))
    GTEST_SKIP() << "no rocjitsu CLI at " << cli_bin();

  CliEnv env{make_temp_dir("rocjitsu-launch")};
  ASSERT_FALSE(env.tmp_dir.empty()) << "mkdtemp failed: " << strerror(errno);
  struct TempCleanup {
    std::string path;
    ~TempCleanup() {
      std::error_code error;
      std::filesystem::remove_all(path, error);
    }
  } cleanup{env.tmp_dir};

  // The one-shot shape: bring the daemon up, run the workload against it,
  // and take the whole session away again on the way out.
  ProcessResult r = capture(env.prefix() + " " + quoted(cli_bin()) + " --daemon --config " +
                            quoted(daemon_config()) + " -- " + quoted(hip_vector_add_bin()) +
                            " --gtest_filter=HipVectorAddTest.CorrectResult");
  EXPECT_EQ(r.exit_code, 0) << r.output;

  // Nothing survives the command that owned it. This is the assertion the
  // socket-file check used to stand in for, asked of the CLI rather than
  // of a path the test had to know.
  EXPECT_TRUE(live_sessions(env).empty());
}

/// One daemon-backed session, held open for the lifetime of one test.
class DaemonTest : public ::testing::Test {
protected:
  // Config the session is started with. The base fixture uses the plain KMD
  // config; subclasses can synthesize a config (e.g. to enable plugins) into
  // tmp_dir_ and return its path.
  virtual std::string daemon_config_for_test() { return daemon_config(); }

  /// How many emulated GPUs the session has. Overridden by the RCCL
  /// fixture, which needs two.
  virtual std::string session_config() { return daemon_config_for_test(); }

  void SetUp() override {
    if (!std::filesystem::exists(cli_bin()))
      GTEST_SKIP() << "no rocjitsu CLI at " << cli_bin();

    env_.tmp_dir = make_temp_dir("rocjitsu-test");
    ASSERT_FALSE(env_.tmp_dir.empty()) << "mkdtemp failed: " << strerror(errno);
    tmp_dir_ = env_.tmp_dir;

    ASSERT_NO_FATAL_FAILURE(config_path_ = session_config());
    ASSERT_FALSE(config_path_.empty()) << "session config unavailable";

    // `session start` owns the session for as long as it runs, so it is a
    // child of this process rather than a command that returns. It exits
    // when `session stop` asks it to, in TearDown.
    owner_pid_ = fork();
    ASSERT_GE(owner_pid_, 0) << "fork failed: " << strerror(errno);
    if (owner_pid_ == 0) {
      env_.apply();
      execl(cli_bin(), cli_bin(), "session", "start", "--daemon", "--config", config_path_.c_str(),
            nullptr);
      _exit(127);
    }

    // A run answers its socket from before bring-up finishes — that is
    // deliberate, so `rocjitsu exec` can say "not ready (pulling)" rather
    // than time out — which means `session list` naming the session is
    // not yet permission to use it. What settles it is an exec that the
    // session accepts, so that is what is waited for: the session is
    // ready exactly when it can run something, asked through the same
    // interface the cases themselves use.
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
    std::string last;
    while (std::chrono::steady_clock::now() < deadline) {
      std::vector<std::string> ids = live_sessions(env_);
      if (!ids.empty()) {
        session_ = ids.front();
        ProcessResult probe = run_in_session("/bin/true", nullptr);
        if (probe.exit_code == 0)
          return;
        last = probe.output;
      }
      int status = 0;
      if (waitpid(owner_pid_, &status, WNOHANG) > 0) {
        owner_pid_ = -1;
        FAIL() << "`session start` exited before the session came up";
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    FAIL() << "no session was ready within 60s. Last attempt:\n" << last;
  }

  void TearDown() override {
    if (owner_pid_ > 0) {
      ProcessResult stopped =
          capture(env_.prefix() + " " + quoted(cli_bin()) + " session stop " + session_);
      EXPECT_EQ(stopped.exit_code, 0) << stopped.output;

      int status = 0;
      EXPECT_EQ(waitpid(owner_pid_, &status, 0), owner_pid_);
      EXPECT_TRUE(WIFEXITED(status));
      EXPECT_EQ(WEXITSTATUS(status), 0);
      owner_pid_ = -1;

      // Teardown removed it: the daemon is stopped and its scratch
      // directory is gone, which is what an empty list means here.
      EXPECT_TRUE(live_sessions(env_).empty());
    }
    if (!tmp_dir_.empty())
      std::filesystem::remove_all(tmp_dir_);
  }

  /// Run one workload in this session.
  ///
  /// The environment the emulated workload needs — SDMA copies, no
  /// scratch reclaim, RCCL kept off transports the simulated topology
  /// does not model — is injected by the CLI's rocjitsu backend, so only
  /// what is specific to a case is passed here.
  ProcessResult run_in_session(const char *binary, const char *gtest_filter,
                               const std::vector<std::string> &envs = {},
                               const std::vector<std::string> &args = {}) {
    std::string cmd = env_.prefix() + " " + quoted(cli_bin()) + " exec --session " + session_;
    for (const std::string &kv : envs)
      cmd += " --env " + quoted(kv);
    cmd += " -- " + quoted(binary);
    for (const std::string &arg : args)
      cmd += " " + quoted(arg);
    if (gtest_filter && gtest_filter[0])
      cmd += " --gtest_filter=" + std::string(gtest_filter);
    return capture(cmd);
  }

  ProcessResult run_hip_test(const char *binary, const char *gtest_filter) {
    return run_in_session(binary, gtest_filter);
  }

  ProcessResult run_rccl_rank(int rank, int world_size, const std::string &shared_dir,
                              const char *gtest_filter) {
    return run_in_session(
        hip_rccl_bin(), gtest_filter, {"HIP_VISIBLE_DEVICES=" + std::to_string(rank)},
        {"--rank=" + std::to_string(rank), "--world-size=" + std::to_string(world_size),
         "--shared-dir=" + shared_dir});
  }

  void run_collective(const char *filter, int world_size = 2) {
    std::string shared_tmpl = tmp_dir_ + "/coll-XXXXXX";
    ASSERT_NE(mkdtemp(shared_tmpl.data()), nullptr) << strerror(errno);
    std::string shared_dir = shared_tmpl;

    std::vector<std::thread> threads(world_size);
    std::vector<ProcessResult> results(world_size);

    for (int r = 0; r < world_size; ++r)
      threads[r] =
          std::thread([&, r] { results[r] = run_rccl_rank(r, world_size, shared_dir, filter); });
    for (auto &t : threads)
      t.join();

    for (int r = 0; r < world_size; ++r)
      EXPECT_EQ(results[r].exit_code, 0) << "Rank " << r << " failed:\n" << results[r].output;

    std::filesystem::remove_all(shared_dir);
  }

  CliEnv env_;
  pid_t owner_pid_ = -1;
  std::string tmp_dir_;
  std::string session_;
  std::string config_path_;
};

// --- hip_vector_add_test ---

TEST_F(DaemonTest, HipVectorAdd) {
  auto r = run_hip_test(hip_vector_add_bin(), "HipVectorAddTest.CorrectResult");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, ExecFailureLeavesTheSessionUsable) {
  // A workload that cannot be started must not take the daemon with it.
  // The old form of this asserted that the daemon's socket file was still
  // there; running a second workload afterwards asserts the stronger and
  // more useful thing, that the daemon still serves.
  auto failed = run_in_session("/does/not/exist", nullptr);
  EXPECT_NE(failed.exit_code, 0) << failed.output;

  auto ok = run_hip_test(hip_vector_add_bin(), "HipVectorAddTest.CorrectResult");
  EXPECT_EQ(ok.exit_code, 0) << ok.output;
}

// --- hip_memcpy_test ---

TEST_F(DaemonTest, HipMemcpyRoundTripFloat) {
  auto r = run_hip_test(hip_memcpy_bin(), "HipMemcpyTest.RoundTripFloat");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, HipMemcpyRoundTripInt) {
  auto r = run_hip_test(hip_memcpy_bin(), "HipMemcpyTest.RoundTripInt");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, HipMemcpyRoundTripPageableAbovePinThreshold) {
  auto r = run_hip_test(hip_memcpy_bin(), "HipMemcpyTest.RoundTripPageableAbovePinThreshold");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, HipMemcpyDeviceToDevice) {
  auto r = run_hip_test(hip_memcpy_bin(), "HipMemcpyTest.DeviceToDevice");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

// --- Multi-client tests ---

TEST_F(DaemonTest, TwoIndependentClients) {
  std::thread t1, t2;
  ProcessResult r1, r2;

  t1 = std::thread(
      [&] { r1 = run_hip_test(hip_vector_add_bin(), "HipVectorAddTest.CorrectResult"); });
  t2 = std::thread([&] { r2 = run_hip_test(hip_memcpy_bin(), "HipMemcpyTest.RoundTripFloat"); });

  t1.join();
  t2.join();

  EXPECT_EQ(r1.exit_code, 0) << "Client 1 (vector_add):\n" << r1.output;
  EXPECT_EQ(r2.exit_code, 0) << "Client 2 (memcpy):\n" << r2.output;
}

// --- Daemon-mode plugin coverage ---
//
// Plugins declared in the config load and run inside the forked daemon, not the
// interposer. The daemon resolves librocjitsu_plugin_logging.so by explicit
// path from the CLI's own directory, so this exercises the same plugin
// discovery + load contract as the interposer path.
class DaemonPluginTest : public DaemonTest {
protected:
  // Rewrite the logging-plugin config template's placeholder sink dir to a
  // private, writable directory for this test, then run the session on it.
  std::string daemon_config_for_test() override {
    sink_dir_ = tmp_dir_ + "/plugin_sink";
    std::error_code ec;
    std::filesystem::create_directories(sink_dir_, ec);

    std::ifstream in(daemon_logging_config());
    if (!in.good()) {
      ADD_FAILURE() << "cannot open daemon logging config: " << daemon_logging_config();
      return {};
    }
    std::string cfg{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};

    const std::string token = RJ_DAEMON_LOGGING_SINK_TOKEN;
    for (size_t pos = cfg.find(token); pos != std::string::npos; pos = cfg.find(token, pos)) {
      cfg.replace(pos, token.size(), sink_dir_);
      pos += sink_dir_.size();
    }

    std::string out_path = tmp_dir_ + "/daemon_logging_config.json";
    std::ofstream out(out_path);
    out << cfg;
    if (!out.good()) {
      ADD_FAILURE() << "cannot write daemon logging config: " << out_path;
      return {};
    }
    return out_path;
  }

  std::string sink_dir_;
};

TEST_F(DaemonPluginTest, LoggingPluginDispatchLogged) {
  auto r = run_hip_test(hip_vector_add_bin(), "HipVectorAddTest.CorrectResult");
  ASSERT_EQ(r.exit_code, 0) << r.output;

  // The logging plugin runs in the daemon and writes <sink_dir>/logging.log as
  // it executes the client's kernel dispatch. Its presence and contents prove
  // the plugin loaded and ran in daemon mode.
  std::string log = sink_dir_ + "/logging.log";
  ASSERT_TRUE(std::filesystem::exists(log))
      << "daemon did not produce plugin log at " << log << "\n"
      << r.output;
  std::ifstream f(log);
  std::string contents{std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
  EXPECT_NE(contents.find("dispatch"), std::string::npos) << "log:\n" << contents;
  EXPECT_NE(contents.find("vgprs="), std::string::npos) << "log:\n" << contents;
}

// --- Remote-backend dup/backend bookkeeping ---
//
// Runs the interposer dup/dup2/dup3 regression binary against a live daemon, so
// open("/dev/kfd") is serviced by the RemoteDriver (Remote backend) rather than
// the in-process SimulatedKfd. This gives the fd/backend state machine coverage
// on the remote path — the primary-fd re-mint (reissue_synthetic_kfd_fd) and the
// invalidation-vs-open serialization — which the CLI-launched (local) variant of
// the same tests cannot reach.

TEST_F(DaemonTest, InterposerDupReopenAfterPrimaryOverwriteRemote) {
  auto r = run_hip_test(interposer_dup_bin(),
                        "InterposerDupTest.ReopenAfterPrimaryOverwriteKeepsBackend");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, InterposerDupSerializedReopenUnderContentionRemote) {
  auto r = run_hip_test(interposer_dup_bin(),
                        "InterposerDupTest.SerializedReopenUnderContentionStaysRoutable");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, InterposerDupKeepsRoutingAfterPrimaryCloseRemote) {
  auto r =
      run_hip_test(interposer_dup_bin(), "InterposerDupTest.DupKeepsKfdRoutingAfterPrimaryClose");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, InterposerDup2OverPrimaryInvalidatesRemote) {
  auto r =
      run_hip_test(interposer_dup_bin(), "InterposerDupTest.Dup2OverPrimaryInvalidatesKfdIdentity");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

TEST_F(DaemonTest, InterposerProcMapsNamesRemoteKfdMarker) {
  auto r = run_hip_test(interposer_dup_bin(), "InterposerDupTest.ProcMapsNamesRemoteKfdMarker");
  EXPECT_EQ(r.exit_code, 0) << r.output;
}

// --- RCCL collective tests (2-GPU daemon) ---
//
// The same fixture on the two-GPU config: each rank is its own
// `rocjitsu exec` into the one session, which is how several processes
// come to share the daemon's emulated machine.

class RcclDaemonTest : public DaemonTest {
protected:
  std::string session_config() override { return daemon_config_2gpu(); }
};

TEST_F(RcclDaemonTest, AllReduce) { run_collective("RcclTest.AllReduce"); }
TEST_F(RcclDaemonTest, Broadcast) { run_collective("RcclTest.Broadcast"); }
TEST_F(RcclDaemonTest, AllGather) { run_collective("RcclTest.AllGather"); }
TEST_F(RcclDaemonTest, ReduceScatter) { run_collective("RcclTest.ReduceScatter"); }
TEST_F(RcclDaemonTest, SendRecv) { run_collective("RcclTest.SendRecv"); }

} // namespace
