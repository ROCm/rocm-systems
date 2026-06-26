/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file MPIHelpers.hpp
 * @brief Shared MPI utility functions for both GTest and standalone tests
 *
 * Provides common functionality for MPI test initialization, GPU setup,
 * and per-rank logging that can be used by both GTest-based tests and
 * standalone tests (performance benchmarks, etc.).
 */

#ifndef MPI_HELPERS_HPP
#define MPI_HELPERS_HPP

#ifdef MPI_TESTS_ENABLED

#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <type_traits>

/**
 * @namespace MPIHelpers
 * @brief Shared MPI utilities for test infrastructure
 */
namespace MPIHelpers
{

// Returns the MPI world rank as a string for diagnostic messages.
// Probes OpenMPI, MPICH/PMIx, and SLURM env vars in order; returns "?" if none set.
inline const char* getMpiRankStr()
{
    for(const char* var : {"OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "SLURM_PROCID"})
    {
        const char* val = std::getenv(var);
        if(val)
            return val;
    }
    return "?";
}

/**
 * @struct MPIContext
 * @brief MPI environment context information
 */
struct MPIContext
{
    int world_rank; ///< MPI rank in MPI_COMM_WORLD
    int world_size; ///< Total number of MPI processes
    int thread_support; ///< MPI thread support level provided
};

/**
 * @brief Initialize MPI with thread support
 *
 * Initializes MPI with MPI_THREAD_MULTIPLE support and returns context info.
 *
 * @param argc Pointer to argc from main()
 * @param argv Pointer to argv from main()
 * @return MPIContext with rank, size, and thread support info
 *
 * @note Must be called before any other MPI operations
 * @note Automatically sets MPIEnvironment static variables
 */
MPIContext initializeMPI(int* argc, char*** argv);

/**
 * @brief Setup GPU device for this MPI rank
 *
 * Assigns GPU device based on local rank (ranks on same node).
 * Uses MPI_COMM_TYPE_SHARED to detect node topology and assigns
 * GPUs in round-robin fashion.
 *
 * @param world_rank MPI rank in MPI_COMM_WORLD
 *
 * @note Handles multiple ranks per node automatically
 * @note Uses hipSetDevice() to assign GPU
 */
void setupGPU(int world_rank);

/**
 * @class FileDescriptor
 * @brief RAII wrapper for POSIX file descriptors
 *
 * Automatically closes file descriptor on destruction.
 * Move-only semantics prevent accidental duplication.
 */
class FileDescriptor
{
public:
    explicit FileDescriptor(int fd = -1) noexcept;
    ~FileDescriptor();

    // Move-only semantics
    FileDescriptor(FileDescriptor&& other) noexcept;
    FileDescriptor& operator=(FileDescriptor&& other) noexcept;

    // Delete copy operations
    FileDescriptor(const FileDescriptor&)            = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;

    [[nodiscard]] int  get() const noexcept;
    [[nodiscard]] bool is_valid() const noexcept;
    int                release() noexcept;

private:
    int fd_;
};

/**
 * @class TeeThread
 * @brief Thread for duplicating output to console and log file
 *
 * Used by rank 0 when per-rank logging is enabled to send output
 * to both console and log file simultaneously.
 */
class TeeThread
{
public:
    TeeThread(int read_fd, int console_fd, int log_fd);
    ~TeeThread();

    // Delete copy/move operations
    TeeThread(const TeeThread&)            = delete;
    TeeThread& operator=(const TeeThread&) = delete;
    TeeThread(TeeThread&&)                 = delete;
    TeeThread& operator=(TeeThread&&)      = delete;

private:
    void tee_loop();

    int               read_fd_;
    int               console_fd_;
    int               log_fd_;
    std::atomic<bool> running_;
    std::thread       thread_;
};

/**
 * @struct RankLogConfig
 * @brief Per-rank logging configuration and state
 *
 * Manages file descriptors and threads for per-rank logging when
 * RCCL_MPI_LOG_ALL_RANKS=1 environment variable is set.
 */
struct RankLogConfig
{
    std::optional<FileDescriptor> log_fd; ///< Log file descriptor
    std::optional<FileDescriptor> saved_stdout; ///< Saved stdout for restoration
    std::optional<FileDescriptor> saved_stderr; ///< Saved stderr for restoration
    std::optional<FileDescriptor> pipe_read_fd; ///< Pipe read end (rank 0 only)
    std::optional<FileDescriptor> pipe_write_fd; ///< Pipe write end (rank 0 only)
    std::unique_ptr<TeeThread>    tee_thread; ///< Tee thread (rank 0 only)
    bool                          logging_enabled{false}; ///< Is per-rank logging enabled?
    bool                          is_rank_zero{false}; ///< Is this rank 0?
};

/**
 * @brief Setup per-rank logging if RCCL_MPI_LOG_ALL_RANKS=1
 *
 * Configures output redirection for MPI ranks:
 * - Rank 0: Output to BOTH console AND log file (tee behavior)
 * - Rank 1-N: Output redirected to rccl_test_rank_<N>.log
 *
 * If RCCL_MPI_LOG_ALL_RANKS is not set:
 * - Rank 0: Normal console output
 * - Rank 1-N: Output suppressed (redirected to /dev/null)
 *
 * @param rank MPI rank in MPI_COMM_WORLD
 * @return Optional RankLogConfig if logging was configured, std::nullopt otherwise
 *
 * @note Call before any test output
 * @note Must call restoreRankLogging() at end to cleanup
 */
std::optional<RankLogConfig> setupRankLogging(int rank);

/**
 * @brief Restore original stdout/stderr after per-rank logging
 *
 * Cleans up per-rank logging configuration and restores original
 * stdout/stderr file descriptors.
 *
 * @param config RankLogConfig to cleanup
 *
 * @note Safe to call multiple times
 * @note Flushes pending output before restoration
 */
void restoreRankLogging(RankLogConfig& config);

/**
 * @brief True when RCCL_MPI_LOG_ALL_RANKS=1 (per-rank log files are active)
 */
inline bool isPerRankLoggingEnabled()
{
    const char* env = std::getenv("RCCL_MPI_LOG_ALL_RANKS");
    return env != nullptr && std::string(env) == "1";
}

/**
 * @brief Path to the per-rank log file used when RCCL_MPI_LOG_ALL_RANKS=1
 *
 * Matches the filename created by setupRankLogging(). The name embeds the
 * --gtest_filter test label when it identifies a single test
 * (rccl_test_<Label>_rank_<R>_pid<P>.log) so every rank's file can be matched
 * to the test that produced it; otherwise it falls back to
 * rccl_test_rank_<R>_pid<P>.log. The file is shared by all tests in the process
 * unless callers use TestLogAssertionContext::isolate_new_output.
 */
std::string getRankLogFilePath(int rank);

/**
 * @brief Read an entire file into a string (e.g. NCCL_DEBUG_FILE output)
 *
 * @return File contents, or empty string if the file could not be read
 */
std::string readTextFile(const std::string& path);

/**
 * @brief Read the full contents of a rank log file
 *
 * Flushes stdout/stderr first so output redirected via per-rank logging (tee/pipe)
 * reaches the file before reading.
 *
 * @return File contents, or empty string if the file could not be read
 */
std::string readRankLogFile(int rank);

/**
 * @brief Byte length of an existing file, or 0 if missing or on error
 */
std::uintmax_t getFileSizeBytes(const std::string& path);

/**
 * @struct TestLogAssertionOptions
 * @brief Configure scoped log capture for MPI test assertions
 *
 * Two independent mechanisms can be enabled at once:
 * - capture_nccl_debug_file: RAII set/restore NCCL_DEBUG_FILE so NCCL writes debug
 *   lines to a dedicated file (set before ncclComm* init). Works alongside global
 *   per-rank logging from main_mpi.cpp.
 * - read_per_rank_stderr_log: after the action under test, read rccl_test_rank_<r>.log
 *   (requires RCCL_MPI_LOG_ALL_RANKS=1 for non-empty stderr capture on that rank).
 *
 * Use readNcclDebugLog() / readPerRankStderrLog() after the code that should emit logs.
 * With isolate_new_output, only bytes appended after this object is constructed are
 * returned (reduces bleed from earlier tests in the same process).
 */
struct TestLogAssertionOptions
{
    int mpi_rank{0};

    bool capture_nccl_debug_file{false};
    /** If capture_nccl_debug_file and empty, auto-path is built from getLogBaseDir()
     *  (priority: RCCL_TEST_LOG_DIR env var → binary directory → /tmp):
     *    GTest binary : <logdir>/rccl_<Suite>.<Test>_rank<R>_pid<P>.log
     *    Standalone   : <logdir>/rccl_rank<R>_pid<P>.log
     *  Path is derived from GTest current_test_info() at construction time. */
    std::string nccl_debug_file_path;
    /** When true, the log file is unlinked after a PASSING test.
     *  Failing tests always keep their log so it is available for post-mortem
     *  inspection.  Set RCCL_KEEP_TEST_LOGS=1 to keep all logs regardless. */
    bool unlink_auto_generated_nccl_path{true};
    bool unlink_explicit_nccl_path{false};

    bool read_per_rank_stderr_log{false};
    bool isolate_new_output{true};
};

/**
 * @class TestLogAssertionContext
 * @brief RAII scope for NCCL_DEBUG_FILE and/or per-rank log reads
 *
 * Construct at the start of a TEST_F (before communicator init when using
 * capture_nccl_debug_file). Destructor restores NCCL_DEBUG_FILE and optionally
 * unlinks the NCCL temp file.
 */
class TestLogAssertionContext
{
public:
    explicit TestLogAssertionContext(const TestLogAssertionOptions& opts);
    ~TestLogAssertionContext();

    TestLogAssertionContext(const TestLogAssertionContext&)            = delete;
    TestLogAssertionContext& operator=(const TestLogAssertionContext&) = delete;

    [[nodiscard]] bool capturesNcclDebugFile() const noexcept { return capture_nccl_; }
    [[nodiscard]] bool readsPerRankStderrLog() const noexcept { return read_per_rank_; }

    /** Path passed to NCCL via NCCL_DEBUG_FILE when capture_nccl_debug_file is set */
    [[nodiscard]] const std::string& ncclDebugFilePath() const noexcept { return nccl_path_; }

    /** Content from the NCCL debug file (slice if isolate_new_output was set) */
    [[nodiscard]] std::string readNcclDebugLog() const;

    /** Content from rccl_test_rank_<rank>.log after flush (slice if isolated) */
    [[nodiscard]] std::string readPerRankStderrLog() const;

private:
    TestLogAssertionOptions opts_;
    std::string             nccl_path_;
    std::string             test_label_;           ///< "Suite/Test" or "pid_<P>" for standalone
    std::string             saved_nccl_debug_env_;
    bool                    saved_nccl_debug_present_{false};
    bool                    env_modified_{false};
    bool                    auto_nccl_path_{false};
    bool                    capture_nccl_{false};
    bool                    read_per_rank_{false};
    std::uintmax_t          nccl_start_offset_{0};
    std::uintmax_t          per_rank_start_offset_{0};
};

/** Scoped NCCL_DEBUG_FILE only (NCCL_DEBUG=INFO typical) */
TestLogAssertionOptions makeNcclDebugFileAssertionOptions(int mpi_rank);

/** Read rccl_test_rank_<rank>.log only (RCCL_MPI_LOG_ALL_RANKS=1) */
TestLogAssertionOptions makePerRankStderrAssertionOptions(int mpi_rank);

/** Both: NCCL debug file + per-rank stderr log (either read may contain the line) */
TestLogAssertionOptions makeCombinedAssertionLogOptions(int mpi_rank);

/**
 * @brief Return the effective value of a numeric NCCL env var (mirrors NCCL_PARAM semantics).
 * @param name        Env var name (e.g. "NCCL_CTA_POLICY").
 * @param ncclDefault NCCL compiled-in default (returned when the var is unset or empty).
 *
 * Mirrors ncclLoadParam behaviour: uses base 0 so that "0x10" parses as 16, "010" as 8, etc.
 * An empty string is treated as unset (returns ncclDefault), consistent with RCCL's param code.
 */
template<typename T>
inline T getEnvParam(const char* name, T ncclDefault)
{
    static_assert(std::is_integral_v<T>, "getEnvParam: T must be an integer type");
    const char* v = std::getenv(name);
    if(!v || v[0] == '\0') return ncclDefault;

    char* endptr = nullptr;
    errno        = 0;
    if constexpr(std::is_unsigned_v<T>)
    {
        const unsigned long long val = std::strtoull(v, &endptr, 0);
        if(errno == ERANGE || endptr == v) return ncclDefault;
        return static_cast<T>(val);
    }
    else
    {
        const long long val = std::strtoll(v, &endptr, 0);
        if(errno == ERANGE || endptr == v) return ncclDefault;
        return static_cast<T>(val);
    }
}

/** RAII scoped set/restore of a single env var; restores (or unsets) on destruction.
 *
 *  Warns on stderr (via getMpiRankStr()) when overriding an existing value so the
 *  caller knows the in-test override is active.  The original value is always restored
 *  on destruction regardless of test outcome.
 */
struct MpiEnvGuard
{
    const char* name;
    std::string saved;
    bool        had{false};

    MpiEnvGuard(const char* n, const char* v) : name(n)
    {
        const char* prev = std::getenv(n);
        had              = (prev != nullptr);
        if(had)
        {
            saved = prev;
            if(saved != v)
            {
#ifdef RCCL_TEST_CHECKS_HPP
                TEST_INFO("MpiEnvGuard overriding %s: '%s' -> '%s' (will restore on scope exit)",
                          n, saved.c_str(), v);
#else
                fprintf(stderr,
                        "[MpiEnvGuard rank %s] overriding %s: '%s' -> '%s' (will restore on scope exit)\n",
                        getMpiRankStr(), n, saved.c_str(), v);
#endif
            }
        }
        ::setenv(n, v, /*overwrite=*/1);
    }
    ~MpiEnvGuard()
    {
        if(had)
            ::setenv(name, saved.c_str(), 1);
        else
            ::unsetenv(name);
    }

    MpiEnvGuard(const MpiEnvGuard&)            = delete;
    MpiEnvGuard& operator=(const MpiEnvGuard&) = delete;
};

} // namespace MPIHelpers

#endif // MPI_TESTS_ENABLED

#endif // MPI_HELPERS_HPP
