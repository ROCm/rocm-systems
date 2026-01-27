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
#include <memory>
#include <optional>
#include <string>
#include <thread>

/**
 * @namespace MPIHelpers
 * @brief Shared MPI utilities for test infrastructure
 */
namespace MPIHelpers
{

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

// ============================================================================
// Stderr Capture for Debug Log Parsing
// ============================================================================

/**
 * @class StderrCapture
 * @brief Captures stderr output to a string for analysis
 *
 * Useful for capturing NCCL debug output (e.g., NCCL_DEBUG=INFO)
 * and parsing it for specific patterns.
 *
 * Usage:
 *   StderrCapture capture;
 *   capture.start();
 *   // ... code that writes to stderr ...
 *   capture.stop();
 *   std::string output = capture.getOutput();
 *   if (output.find("some pattern") != std::string::npos) { ... }
 */
class StderrCapture
{
public:
    StderrCapture();
    ~StderrCapture();

    // Non-copyable, non-movable
    StderrCapture(const StderrCapture&)            = delete;
    StderrCapture& operator=(const StderrCapture&) = delete;
    StderrCapture(StderrCapture&&)                 = delete;
    StderrCapture& operator=(StderrCapture&&)      = delete;

    /**
     * @brief Start capturing stderr
     * @return true if capture started successfully
     */
    bool start();

    /**
     * @brief Stop capturing and read captured output
     */
    void stop();

    /**
     * @brief Get the captured stderr output
     * @return Captured output as string (empty if capture not started/stopped)
     */
    [[nodiscard]] const std::string& getOutput() const;

    /**
     * @brief Check if a pattern exists in captured output
     * @param pattern Substring to search for
     * @return true if pattern found
     */
    [[nodiscard]] bool hasPattern(const std::string& pattern) const;

    /**
     * @brief Reset capture state for reuse
     */
    void reset();

    /**
     * @brief Check if currently capturing
     */
    [[nodiscard]] bool isCapturing() const;

private:
    int         m_savedStderr;
    int         m_tempFd;
    std::string m_tempPath;
    std::string m_capturedOutput;
    bool        m_capturing;
};

/**
 * @class StderrCaptureScope
 * @brief RAII wrapper for StderrCapture - automatically starts/stops capture
 *
 * Usage:
 *   StderrCapture capture;
 *   {
 *       StderrCaptureScope scope(capture);
 *       // ... code that writes to stderr ...
 *   }  // capture stops here
 *   if (capture.hasPattern("some pattern")) { ... }
 */
class StderrCaptureScope
{
public:
    explicit StderrCaptureScope(StderrCapture& capture);
    ~StderrCaptureScope();

    // Non-copyable
    StderrCaptureScope(const StderrCaptureScope&)            = delete;
    StderrCaptureScope& operator=(const StderrCaptureScope&) = delete;

private:
    StderrCapture& m_capture;
};

// ============================================================================
// NCCL Debug Environment Helpers
// ============================================================================

/**
 * @brief Check if NCCL debug logging is enabled for a specific subsystem
 *
 * Checks NCCL_DEBUG and NCCL_DEBUG_SUBSYS environment variables.
 *
 * @param subsystem Subsystem to check (e.g., "REG", "NET", "INIT", "COLL")
 *                  Pass empty string to check if any debug is enabled
 * @param minLevel Minimum debug level required (default: "INFO")
 *                 Levels: "WARN", "INFO", "TRACE"
 * @return true if debug logging is enabled for the specified subsystem
 *
 * Examples:
 *   isNCCLDebugEnabled("REG")      // true if NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS contains "REG"
 *   isNCCLDebugEnabled("NET")      // true if NCCL_DEBUG=INFO and NCCL_DEBUG_SUBSYS contains "NET"
 *   isNCCLDebugEnabled("")         // true if any NCCL_DEBUG level is set
 */
bool isNCCLDebugEnabled(const std::string& subsystem = "", const std::string& minLevel = "INFO");

/**
 * @brief Get the current NCCL debug level
 * @return Debug level string ("WARN", "INFO", "TRACE") or empty if not set
 */
std::string getNCCLDebugLevel();

/**
 * @brief Get list of enabled NCCL debug subsystems
 * @return Comma-separated list of subsystems or "ALL" or empty
 */
std::string getNCCLDebugSubsystems();

} // namespace MPIHelpers

#endif // MPI_TESTS_ENABLED

#endif // MPI_HELPERS_HPP
