/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "MPIHelpers.hpp"

#ifdef MPI_TESTS_ENABLED

    #include "MPITestCore.hpp"
    #include "MPIEnvironment.hpp"
    #include <cerrno>
    #include <cstring>
    #include <fcntl.h>
    #include <hip/hip_runtime.h>
    #include <iostream>
    #include <mpi.h>
    #include <unistd.h>

namespace MPIHelpers
{

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
    hipGetDeviceCount(&device_count);

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
        hipSetDevice(device_id);

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
            // Per-rank logging enabled: Redirect to log file
            const auto log_filename
                = std::string{"rccl_test_rank_"} + std::to_string(rank) + ".log";

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
    const auto log_filename = std::string{"rccl_test_rank_"} + std::to_string(rank) + ".log";

    // Debug: Print to stderr BEFORE creating log file
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
    TEST_INFO("Ranks 1-N  : Output redirected to rccl_test_rank_<N>.log");
    TEST_INFO("Location   : Log files created in current working directory");

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
