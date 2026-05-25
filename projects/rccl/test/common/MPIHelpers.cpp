/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "MPIHelpers.hpp"

#ifdef MPI_TESTS_ENABLED

    #include "MPITestCore.hpp"
    #include "MPIEnvironment.hpp"
    #include <cctype>
    #include <cerrno>
    #include <cstdlib>
    #include <cstring>
    #include <dlfcn.h>
    #include <fcntl.h>
    #include <fstream>
    #include <hip/hip_runtime.h>
    #include <iostream>
    #include <mpi.h>
    #include <sstream>
    #include <sys/stat.h>
    #include <unistd.h>
    // RCCL_MPIHelpers_HAS_GTEST is set by the build system (CMakeLists.txt) for
    // targets that actually link against GTest, so this guard is guaranteed to
    // match the link graph rather than relying on header presence alone.
    #if defined(RCCL_MPIHelpers_HAS_GTEST)
        #include <gtest/gtest.h>
    #endif

    // resetNcclDebugFile() calls ncclResetDebugInit() via dlsym so that we
    // never reference the deprecated symbol by name at compile or link time.
    //
    // Why dlsym instead of a direct call:
    //   - ncclResetDebugInit is marked __attribute__((deprecated)) in nccl.h,
    //     so a direct call triggers -Wdeprecated-declarations and requires a
    //     #pragma GCC diagnostic suppression.
    //   - pncclResetDebugInit (the non-deprecated profiling wrapper) is NOT
    //     exported from librccl.so, so it cannot be linked against directly.
    //   - dlsym(RTLD_DEFAULT, ...) resolves the symbol from the already-loaded
    //     librccl.so at runtime — no pragma, no link-time dependency.
    //   - If RCCL removes the API in a future release the call becomes a no-op
    //     (the static fn pointer stays null) rather than a link or build error.
    static void resetNcclDebugFile()
    {
        using NcclResetFn = void (*)();
        static NcclResetFn fn = reinterpret_cast<NcclResetFn>(
            ::dlsym(RTLD_DEFAULT, "ncclResetDebugInit"));
        if(fn)
            fn();
    }

namespace MPIHelpers
{
namespace
{
    std::string sliceFromOffset(const std::string& full, std::uintmax_t off)
    {
        if(off >= full.size())
        {
            return {};
        }
        return full.substr(static_cast<std::size_t>(off));
    }

    // Determine the base directory for per-rank and NCCL debug log files.
    //
    // Priority:
    //   1. RCCL_TEST_LOG_DIR env var — explicit override, useful in CI.
    //   2. Directory of the test binary (/proc/self/exe) — ties logs to the
    //      specific build that produced them; different builds never share logs.
    //      Skipped if not writable (e.g. system-installed or read-only prefix).
    //   3. /tmp — last-resort fallback.
    std::string getLogBaseDir()
    {
        const char* env = std::getenv("RCCL_TEST_LOG_DIR");
        if(env && env[0] != '\0')
            return env;

        char buf[4096];
        const ssize_t len = ::readlink("/proc/self/exe", buf, sizeof(buf) - 1);
        if(len > 0)
        {
            buf[len]    = '\0';
            char* slash = ::strrchr(buf, '/');
            if(slash)
            {
                *slash = '\0';
                if(::access(buf, W_OK) == 0)
                    return std::string{buf};
                // Binary dir is not writable; fall through to /tmp.
            }
        }

        return "/tmp";
    }

    // Replace every character that is not alphanumeric, '.', or '-' with '_'
    // so that test-suite and test-case names are safe to embed in a file path.
    std::string sanitizeForFilename(const std::string& s)
    {
        std::string out;
        out.reserve(s.size());
        for(unsigned char c : s)
        {
            out += (std::isalnum(c) || c == '.' || c == '-') ? static_cast<char>(c) : '_';
        }
        return out;
    }

    // Build a per-test, per-rank NCCL debug log path.
    //
    // Pattern: <logdir>/rccl_<TestSuite>.<TestName>_rank<R>_pid<P>.log
    // PID suffix ensures each process invocation gets a unique file so that
    // stale files from a different user or a previous run never block fopen("w").
    std::string makeTestNameLogPath(int rank)
    {
        const auto        pid     = std::to_string(::getpid());
        const std::string logdir  = getLogBaseDir();
#if defined(RCCL_MPIHelpers_HAS_GTEST)
        std::string suite;
        std::string test;

        const ::testing::TestInfo* info
            = ::testing::UnitTest::GetInstance()->current_test_info();
        if(info)
        {
            if(info->test_suite_name())
                suite = info->test_suite_name();
            if(info->name())
                test = info->name();
        }

        if(!suite.empty() && !test.empty())
        {
            return logdir + "/rccl_" + sanitizeForFilename(suite) + "."
                   + sanitizeForFilename(test) + "_rank" + std::to_string(rank)
                   + "_pid" + pid + ".log";
        }
#endif
        return logdir + "/rccl_rank" + std::to_string(rank)
               + "_pid" + pid + ".log";
    }
} // namespace


// ============================================================================
// FileDescriptor Implementation
// ============================================================================

FileDescriptor::FileDescriptor(int fd) noexcept : fd_(fd) {}

FileDescriptor::~FileDescriptor()
{
    if(fd_ >= 0)
    {
        ::close(fd_);
    }
}

FileDescriptor::FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.fd_)
{
    other.fd_ = -1;
}

FileDescriptor& FileDescriptor::operator=(FileDescriptor&& other) noexcept
{
    if(this != &other)
    {
        if(fd_ >= 0)
        {
            ::close(fd_);
        }
        fd_       = other.fd_;
        other.fd_ = -1;
    }
    return *this;
}

int FileDescriptor::get() const noexcept
{
    return fd_;
}

bool FileDescriptor::is_valid() const noexcept
{
    return fd_ >= 0;
}

int FileDescriptor::release() noexcept
{
    const auto fd = fd_;
    fd_           = -1;
    return fd;
}

// ============================================================================
// TeeThread Implementation
// ============================================================================

TeeThread::TeeThread(int read_fd, int console_fd, int log_fd)
    : read_fd_(read_fd), console_fd_(console_fd), log_fd_(log_fd), running_(true)
{
    thread_ = std::thread([this]() { this->tee_loop(); });
}

TeeThread::~TeeThread()
{
    running_ = false;
    if(thread_.joinable())
    {
        thread_.join();
    }
}

void TeeThread::tee_loop()
{
    std::array<char, 4096> buffer;
    while(running_)
    {
        const auto bytes_read = ::read(read_fd_, buffer.data(), buffer.size());
        if(bytes_read <= 0)
        {
            if(bytes_read == 0 || errno != EINTR)
            {
                break; // EOF or error
            }
            continue;
        }

        // Write to console
        [[maybe_unused]] auto console_written = ::write(console_fd_, buffer.data(), bytes_read);

        // Write to log file
        [[maybe_unused]] auto log_written = ::write(log_fd_, buffer.data(), bytes_read);
    }
}

// ============================================================================
// MPI Initialization
// ============================================================================

MPIContext initializeMPI(int* argc, char*** argv)
{
    MPIContext ctx;

    // Initialize MPI with thread support
    MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &ctx.thread_support);
    MPI_Comm_rank(MPI_COMM_WORLD, &ctx.world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ctx.world_size);

    // Update global environment
    MPIEnvironment::world_rank      = ctx.world_rank;
    MPIEnvironment::world_size      = ctx.world_size;
    MPIEnvironment::mpi_initialized = true;

    return ctx;
}

// ============================================================================
// GPU Setup
// ============================================================================

void setupGPU(int world_rank)
{
    int device_count = 0;
    (void)hipGetDeviceCount(&device_count);

    if(device_count > 0)
    {
        // Use MPI_COMM_TYPE_SHARED to detect local ranks on same node
        MPI_Comm node_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);

        int local_rank, local_size;
        MPI_Comm_rank(node_comm, &local_rank);
        MPI_Comm_size(node_comm, &local_size);

        // Cache multi-node detection result for isMultiNodeTest()
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPIEnvironment::cached_multi_node_result = (local_size < world_size) ? 1 : 0;

        // Assign GPU in round-robin fashion
        int device_id = local_rank % device_count;
        (void)hipSetDevice(device_id);

        MPI_Comm_free(&node_comm);
    }
}

// ============================================================================
// Per-Rank Logging
// ============================================================================

std::optional<RankLogConfig> setupRankLogging(int rank)
{
    const auto* env_value                = std::getenv("RCCL_MPI_LOG_ALL_RANKS");
    const bool  per_rank_logging_enabled = (env_value && std::string(env_value) == "1");

    RankLogConfig config;
    config.logging_enabled = per_rank_logging_enabled;
    config.is_rank_zero    = (rank == 0);

    // Non-zero ranks: Always redirect output (either to log file or /dev/null)
    if(rank != 0)
    {
        // Save original stdout/stderr
        config.saved_stdout = FileDescriptor{::dup(STDOUT_FILENO)};
        config.saved_stderr = FileDescriptor{::dup(STDERR_FILENO)};

        if(!config.saved_stdout->is_valid() || !config.saved_stderr->is_valid())
        {
            TEST_WARN("Rank %d: Failed to duplicate stdout/stderr", rank);
            return std::nullopt;
        }

        if(per_rank_logging_enabled)
        {
            const auto log_filename = getRankLogFilePath(rank);

            const auto log_fd = ::open(log_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

            if(log_fd < 0)
            {
                TEST_WARN("Rank %d: Failed to create log file: %s", rank, log_filename.c_str());
                return std::nullopt;
            }

            config.log_fd = FileDescriptor{log_fd};

            // Redirect stdout/stderr to log file
            if(::dup2(log_fd, STDOUT_FILENO) < 0 || ::dup2(log_fd, STDERR_FILENO) < 0)
            {
                TEST_WARN("Rank %d: Failed to redirect to log file", rank);
                return std::nullopt;
            }

            // Debug: Write initial marker to log file (AFTER redirection)
            TEST_INFO("===== LOG FILE FOR RANK %d =====", rank);
        }
        else
        {
            // Default: Suppress all output by redirecting to /dev/null
            const auto null_fd = ::open("/dev/null", O_WRONLY);
            if(null_fd < 0)
            {
                TEST_WARN("Rank %d: Failed to open /dev/null", rank);
                return std::nullopt;
            }

            // Redirect stdout/stderr to /dev/null
            if(::dup2(null_fd, STDOUT_FILENO) < 0 || ::dup2(null_fd, STDERR_FILENO) < 0)
            {
                TEST_WARN("Rank %d: Failed to redirect to /dev/null", rank);
                ::close(null_fd);
                return std::nullopt;
            }

            ::close(null_fd);
        }

        // Disable buffering for immediate output
        std::setvbuf(stdout, nullptr, _IONBF, 0);
        std::setvbuf(stderr, nullptr, _IONBF, 0);

        return config;
    }

    // Rank 0: Only redirect if per-rank logging is enabled (for tee functionality)
    if(!per_rank_logging_enabled)
    {
        return std::nullopt; // Rank 0 outputs to console normally
    }

    // Create log file for rank 0
    const auto log_filename = getRankLogFilePath(rank);

    TEST_TRACE("Rank %d (rank 0 tee mode) opening log file: %s", rank, log_filename.c_str());

    const auto log_fd = ::open(log_filename.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);

    if(log_fd < 0)
    {
        TEST_WARN("Rank %d: Failed to create log file: %s", rank, log_filename.c_str());
        return std::nullopt;
    }

    config.log_fd = FileDescriptor{log_fd};

    // Debug: Write initial marker directly to log file (BEFORE redirection)
    const char*           marker  = "===== LOG FILE FOR RANK 0 (TEE MODE) =====\n";
    [[maybe_unused]] auto written = ::write(log_fd, marker, std::strlen(marker));

    // Rank 0 with per-rank logging: Output to BOTH console AND log file (tee behavior)
    // Print banner before redirection
    TEST_INFO("Per-Rank Logging ENABLED (RCCL_MPI_LOG_ALL_RANKS=1)");
    TEST_INFO("Rank 0     : Output to BOTH console AND %s", log_filename.c_str());
    TEST_INFO("Ranks 1-N  : Output redirected to per-rank log files");
    TEST_INFO("Log dir    : %s  (override: RCCL_TEST_LOG_DIR)", getLogBaseDir().c_str());

    // Save original stdout/stderr for tee thread
    config.saved_stdout = FileDescriptor{::dup(STDOUT_FILENO)};
    config.saved_stderr = FileDescriptor{::dup(STDERR_FILENO)};

    if(!config.saved_stdout->is_valid() || !config.saved_stderr->is_valid())
    {
        TEST_WARN("Rank %d: Failed to duplicate stdout/stderr", rank);
        return std::nullopt;
    }

    // Create pipes for tee functionality
    int pipe_fds[2];
    if(::pipe(pipe_fds) < 0)
    {
        TEST_WARN("Rank %d: Failed to create pipe", rank);
        return std::nullopt;
    }

    config.pipe_read_fd  = FileDescriptor{pipe_fds[0]};
    config.pipe_write_fd = FileDescriptor{pipe_fds[1]};

    // Start tee thread to duplicate output to both console and log file
    try
    {
        config.tee_thread = std::make_unique<TeeThread>(config.pipe_read_fd->get(),
                                                        config.saved_stdout->get(),
                                                        log_fd);
    }
    catch(const std::exception& e)
    {
        TEST_WARN("Rank %d: Failed to start tee thread: %s", rank, e.what());
        return std::nullopt;
    }

    // Redirect stdout/stderr to the pipe write end
    if(::dup2(config.pipe_write_fd->get(), STDOUT_FILENO) < 0
       || ::dup2(config.pipe_write_fd->get(), STDERR_FILENO) < 0)
    {
        TEST_WARN("Rank %d: Failed to redirect to pipe", rank);
        return std::nullopt;
    }

    // Disable buffering for immediate output
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::setvbuf(stderr, nullptr, _IONBF, 0);

    return config;
}

std::string getRankLogFilePath(int rank)
{
    return getLogBaseDir() + "/rccl_test_rank_" + std::to_string(rank)
           + "_pid" + std::to_string(::getpid()) + ".log";
}

std::uintmax_t getFileSizeBytes(const std::string& path)
{
    struct stat st
    {
    };
    if(::stat(path.c_str(), &st) != 0)
    {
        return 0;
    }
    return static_cast<std::uintmax_t>(st.st_size);
}

std::string readTextFile(const std::string& path)
{
    std::ifstream file(path);
    if(!file.is_open())
    {
        TEST_WARN("Could not open file: %s", path.c_str());
        return "";
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

std::string readRankLogFile(int rank)
{
    std::fflush(stdout);
    std::fflush(stderr);
    return readTextFile(getRankLogFilePath(rank));
}

TestLogAssertionOptions makeNcclDebugFileAssertionOptions(int mpi_rank)
{
    TestLogAssertionOptions o;
    o.mpi_rank                        = mpi_rank;
    o.capture_nccl_debug_file         = true;
    o.read_per_rank_stderr_log        = false;
    // Test-name-based paths are unique per test, so there is no truncation risk,
    // auto-unlink log files on pass, keep on failure is implemented in the destructor.
    // Set RCCL_KEEP_TEST_LOGS=1 to retain all logs unconditionally.
    o.unlink_auto_generated_nccl_path = true;
    return o;
}

TestLogAssertionOptions makePerRankStderrAssertionOptions(int mpi_rank)
{
    TestLogAssertionOptions o;
    o.mpi_rank                  = mpi_rank;
    o.capture_nccl_debug_file   = false;
    o.read_per_rank_stderr_log = true;
    return o;
}

TestLogAssertionOptions makeCombinedAssertionLogOptions(int mpi_rank)
{
    TestLogAssertionOptions o;
    o.mpi_rank                  = mpi_rank;
    o.capture_nccl_debug_file   = true;
    o.read_per_rank_stderr_log = true;
    return o;
}

TestLogAssertionContext::TestLogAssertionContext(const TestLogAssertionOptions& opts)
    : opts_(opts)
    , capture_nccl_(opts.capture_nccl_debug_file)
    , read_per_rank_(opts.read_per_rank_stderr_log)
{
    const int rank = opts.mpi_rank;

    // Capture the test label for use in the log-file header (written in the
    // destructor).  GTest tests get "Suite/TestName"; standalone binaries
    // that don't link against GTest get "pid_<PID>" as a fallback so the
    // log can still be correlated to the process that produced it.
#if defined(RCCL_MPIHelpers_HAS_GTEST)
    {
        const ::testing::TestInfo* info
            = ::testing::UnitTest::GetInstance()->current_test_info();
        if(info && info->test_suite_name() && info->name())
            test_label_ = std::string{info->test_suite_name()} + "/" + info->name();
    }
#endif
    if(test_label_.empty())
        test_label_ = std::string{"pid_"} + std::to_string(::getpid());

    if(capture_nccl_)
    {
        if(opts_.nccl_debug_file_path.empty())
        {
            nccl_path_      = makeTestNameLogPath(rank);
            auto_nccl_path_ = true;
        }
        else
        {
            nccl_path_      = opts_.nccl_debug_file_path;
            auto_nccl_path_ = false;
        }

        const char* prev = std::getenv("NCCL_DEBUG_FILE");
        if(prev != nullptr)
        {
            saved_nccl_debug_env_     = prev;
            saved_nccl_debug_present_ = true;
        }
        if(::setenv("NCCL_DEBUG_FILE", nccl_path_.c_str(), 1) != 0)
        {
            TEST_WARN("Rank %d: setenv NCCL_DEBUG_FILE failed", rank);
        }
        else
        {
            env_modified_ = true;
            // Tell NCCL to close its current log file and re-read NCCL_DEBUG_FILE
            // on the next log output.  Without this the global ncclDebugFile remains
            // pointed at the previous test's (possibly already-unlinked) file handle.
            resetNcclDebugFile();
        }

        // Remove any log file left over from a previous test run (e.g. a
        // skipped run that kept its log).  NCCL always opens the file with
        // fopen("w") which truncates, so if we recorded the OLD file's size
        // as nccl_start_offset_, readNcclDebugLog() would slice past all of
        // the new content and return an empty string on every subsequent run.
        // Only unlink auto-generated paths; do not clobber a user-supplied file.
        if(auto_nccl_path_)
        {
            (void)::unlink(nccl_path_.c_str());
        }

        if(opts_.isolate_new_output)
        {
            // File was just unlinked; getFileSizeBytes always returns 0 here.
            // The assignment is kept for symmetry and clarity.
            nccl_start_offset_ = getFileSizeBytes(nccl_path_);
        }
    }

    if(read_per_rank_)
    {
        if(opts_.isolate_new_output)
        {
            std::fflush(stdout);
            std::fflush(stderr);
            per_rank_start_offset_ = getFileSizeBytes(getRankLogFilePath(rank));
        }
    }
}

TestLogAssertionContext::~TestLogAssertionContext()
{
    if(env_modified_)
    {
        if(saved_nccl_debug_present_)
        {
            (void)::setenv("NCCL_DEBUG_FILE", saved_nccl_debug_env_.c_str(), 1);
        }
        else
        {
            (void)::unsetenv("NCCL_DEBUG_FILE");
        }
        // Flush all stdio streams (including NCCL's internal FILE* buffer) before
        // reading the log file.  resetNcclDebugFile() arms the lazy close but does
        // not flush or close the handle immediately; fflush(nullptr) ensures every
        // pending byte has been written to disk before we read and rewrite the file.
        std::fflush(nullptr);
        resetNcclDebugFile();
    }

    if(capture_nccl_ && !nccl_path_.empty())
    {
        // ---------------------------------------------------------------
        // Decide whether this log will be kept or deleted BEFORE doing
        // any file I/O: large NCCL_DEBUG=INFO logs on passing tests would
        // otherwise add avoidable teardown I/O and rank skew if we read
        // and rewrite a file we are about to delete.
        // ---------------------------------------------------------------

        // Treat RCCL_KEEP_TEST_LOGS=0 (or empty) as "don't keep"; any other
        // non-empty value (typically "1") activates unconditional retention.
        const char* keep_env = std::getenv("RCCL_KEEP_TEST_LOGS");
        const bool  keep_all = keep_env && keep_env[0] != '\0' && keep_env[0] != '0';

        bool test_passed = false; // conservative default: keep when uncertain
#if defined(RCCL_MPIHelpers_HAS_GTEST)
        {
            const ::testing::TestInfo* info
                = ::testing::UnitTest::GetInstance()->current_test_info();
            if(info && info->result())
                test_passed = info->result()->Passed();
        }
#endif

        const bool unlink_it
            = !keep_all
              && test_passed
              && ((auto_nccl_path_ && opts_.unlink_auto_generated_nccl_path)
                  || (!auto_nccl_path_ && opts_.unlink_explicit_nccl_path));

        if(unlink_it)
        {
            (void)::unlink(nccl_path_.c_str());
        }
        else
        {
            // ---------------------------------------------------------------
            // Prepend a one-line header to the log file so it is
            // self-identifying regardless of whether the name is human-readable
            // (GTest path) or opaque (pid-based standalone path).
            //
            // The header is written here (destructor) rather than in the
            // constructor because NCCL opens the file with fopen("w") on its
            // first lazy-init log call, which would truncate anything written
            // earlier.  By writing in the destructor — after the communicator
            // has already been torn down and fflush(nullptr) has drained NCCL's
            // internal buffer — the file is complete and safe to rewrite.
            // ---------------------------------------------------------------
            const std::string content = readTextFile(nccl_path_);
            if(!content.empty())
            {
                std::ostringstream header;
                header << "=== RCCL Test Log"
                       << " | Test: "  << test_label_
                       << " | Rank: "  << opts_.mpi_rank
                       << " | PID: "   << ::getpid()
                       << " ===\n";
                const std::string hdr = header.str();

                if(FILE* f = std::fopen(nccl_path_.c_str(), "w"))
                {
                    std::fwrite(hdr.c_str(),     1, hdr.size(),     f);
                    std::fwrite(content.c_str(), 1, content.size(), f);
                    std::fclose(f);
                }
            }
        }
    }
}

std::string TestLogAssertionContext::readNcclDebugLog() const
{
    if(!capture_nccl_)
    {
        return {};
    }
    // NCCL writes to a stdio FILE* with block-buffering.  Flush all open
    // streams so that any bytes still sitting in the libc buffer (e.g.
    // "Init CE" / "CE: rank" lines written during the most recent collective)
    // are committed to disk before we read the file back.
    std::fflush(nullptr);
    const std::string full = readTextFile(nccl_path_);
    return opts_.isolate_new_output ? sliceFromOffset(full, nccl_start_offset_) : full;
}

std::string TestLogAssertionContext::readPerRankStderrLog() const
{
    if(!read_per_rank_)
    {
        return {};
    }
    std::fflush(stdout);
    std::fflush(stderr);
    const std::string full = readTextFile(getRankLogFilePath(opts_.mpi_rank));
    return opts_.isolate_new_output ? sliceFromOffset(full, per_rank_start_offset_) : full;
}

void restoreRankLogging(RankLogConfig& config)
{
    // Only restore if we actually redirected (have saved stdout/stderr)
    if(!config.saved_stdout || !config.saved_stdout->is_valid())
    {
        return;
    }

    // Flush any pending output
    std::fflush(stdout);
    std::fflush(stderr);

    // CRITICAL: Restore stdout/stderr BEFORE closing pipe
    // The tee thread won't get EOF until ALL write ends are closed
    if(config.saved_stdout && config.saved_stdout->is_valid())
    {
        ::dup2(config.saved_stdout->get(), STDOUT_FILENO);
    }

    if(config.saved_stderr && config.saved_stderr->is_valid())
    {
        ::dup2(config.saved_stderr->get(), STDERR_FILENO);
    }

    if(config.is_rank_zero && config.tee_thread)
    {
        // For rank 0 with per-rank logging: Stop the tee thread
        // Close the pipe write end to signal EOF to the tee thread
        config.pipe_write_fd.reset();

        // Wait for tee thread to finish processing
        config.tee_thread.reset();

        // Close pipe read end
        config.pipe_read_fd.reset();
    }
}

} // namespace MPIHelpers

#endif // MPI_TESTS_ENABLED
