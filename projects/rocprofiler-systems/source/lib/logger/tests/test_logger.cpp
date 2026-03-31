// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "logger/logger.hpp"

#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace
{

// Read all available data from a file descriptor into a string.
// Used to collect child stdout redirected through a pipe.
std::string
read_fd(int fd)
{
    std::string result;
    char        buf[4096];
    ssize_t     n;
    while((n = read(fd, buf, sizeof(buf))) > 0)
        result.append(buf, static_cast<size_t>(n));
    return result;
}

}  // namespace

class logger_test : public ::testing::Test
{
protected:
};

TEST_F(logger_test, include_process_id_in_filename_with_extension)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename("logfile.log");
    auto expected = "logfile_" + pid + ".log";

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_without_extension)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename("logfile");
    auto expected = "logfile_" + pid;

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_with_path)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename("/var/log/myapp.log");
    auto expected = "/var/log/myapp_" + pid + ".log";

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_with_path_no_extension)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename("/var/log/myapp");
    auto expected = "/var/log/myapp_" + pid;

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_empty)
{
    auto result = rocprofsys::include_process_id_in_filename("");
    EXPECT_TRUE(result.empty());
}

TEST_F(logger_test, include_process_id_in_filename_multiple_dots)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename("file.name.with.dots.txt");
    auto expected = "file.name.with.dots_" + pid + ".txt";

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_dot_in_directory)
{
    auto pid = std::to_string(getpid());
    auto result =
        rocprofsys::include_process_id_in_filename("/path.with.dots/logfile.log");
    auto expected = "/path.with.dots/logfile_" + pid + ".log";

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_hidden_file)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename(".hidden");
    auto expected = ".hidden_" + pid;

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, include_process_id_in_filename_hidden_file_with_extension)
{
    auto pid      = std::to_string(getpid());
    auto result   = rocprofsys::include_process_id_in_filename(".hidden.log");
    auto expected = ".hidden_" + pid + ".log";

    EXPECT_EQ(result, expected);
}

TEST_F(logger_test, parse_boolean_env_true_values)
{
    EXPECT_TRUE(rocprofsys::parse_boolean_env("1"));
    EXPECT_TRUE(rocprofsys::parse_boolean_env("on"));
    EXPECT_TRUE(rocprofsys::parse_boolean_env("true"));
    EXPECT_TRUE(rocprofsys::parse_boolean_env("yes"));
}

TEST_F(logger_test, parse_boolean_env_false_values)
{
    EXPECT_FALSE(rocprofsys::parse_boolean_env("0"));
    EXPECT_FALSE(rocprofsys::parse_boolean_env("off"));
    EXPECT_FALSE(rocprofsys::parse_boolean_env("false"));
    EXPECT_FALSE(rocprofsys::parse_boolean_env("no"));
    EXPECT_FALSE(rocprofsys::parse_boolean_env("random"));
    EXPECT_FALSE(rocprofsys::parse_boolean_env(nullptr));
}

TEST_F(logger_test, to_lower_conversion)
{
    EXPECT_EQ(rocprofsys::to_lower("HELLO"), "hello");
    EXPECT_EQ(rocprofsys::to_lower("Hello World"), "hello world");
    EXPECT_EQ(rocprofsys::to_lower("already_lower"), "already_lower");
    EXPECT_EQ(rocprofsys::to_lower("MiXeD123"), "mixed123");
    EXPECT_EQ(rocprofsys::to_lower(""), "");
}

TEST_F(logger_test, logger_settings_parse_level)
{
    rocprofsys::logger_settings_t settings;

    EXPECT_EQ(settings.parse_level("trace"), spdlog::level::trace);
    EXPECT_EQ(settings.parse_level("TRACE"), spdlog::level::trace);
    EXPECT_EQ(settings.parse_level("debug"), spdlog::level::debug);
    EXPECT_EQ(settings.parse_level("DEBUG"), spdlog::level::debug);
    EXPECT_EQ(settings.parse_level("info"), spdlog::level::info);
    EXPECT_EQ(settings.parse_level("INFO"), spdlog::level::info);
    EXPECT_EQ(settings.parse_level("warn"), spdlog::level::warn);
    EXPECT_EQ(settings.parse_level("warning"), spdlog::level::warn);
    EXPECT_EQ(settings.parse_level("WARNING"), spdlog::level::warn);
    EXPECT_EQ(settings.parse_level("error"), spdlog::level::err);
    EXPECT_EQ(settings.parse_level("err"), spdlog::level::err);
    EXPECT_EQ(settings.parse_level("ERROR"), spdlog::level::err);
    EXPECT_EQ(settings.parse_level("critical"), spdlog::level::critical);
    EXPECT_EQ(settings.parse_level("CRITICAL"), spdlog::level::critical);
    EXPECT_EQ(settings.parse_level("off"), spdlog::level::off);
    EXPECT_EQ(settings.parse_level("OFF"), spdlog::level::off);
    EXPECT_EQ(settings.parse_level("invalid"), spdlog::level::info);
}

TEST_F(logger_test, logger_instance_returns_valid_logger)
{
    auto& logger = rocprofsys::logger_t::instance();
    EXPECT_NE(logger.name(), "");
    EXPECT_EQ(logger.name(), "rocprofiler-systems");

    testing::internal::CaptureStdout();
    logger.info("stdout_capture_test_marker");
    logger.flush();
    auto captured = testing::internal::GetCapturedStdout();

    EXPECT_NE(captured.find("stdout_capture_test_marker"), std::string::npos)
        << "Log message not found in stdout. Captured: " << captured;
    EXPECT_NE(captured.find("[info]"), std::string::npos)
        << "Log level not found in stdout. Captured: " << captured;

    auto pid_marker = "P:" + std::to_string(getpid());
    EXPECT_NE(captured.find(pid_marker), std::string::npos)
        << "PID not found in stdout. Captured: " << captured;
}

TEST_F(logger_test, logger_instance_is_singleton)
{
    auto& logger1 = rocprofsys::logger_t::instance();
    auto& logger2 = rocprofsys::logger_t::instance();

    EXPECT_EQ(&logger1, &logger2);
}

TEST_F(logger_test, fork_child_gets_different_pid_in_filename)
{
    pid_t parent_pid = getpid();

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        pid_t current_pid = getpid();

        auto child_filename    = rocprofsys::include_process_id_in_filename("test.log");
        auto expected_filename = "test_" + std::to_string(current_pid) + ".log";

        bool pid_differs      = (current_pid != parent_pid);
        bool filename_correct = (child_filename == expected_filename);

        exit((pid_differs && filename_correct) ? 0 : 1);
    }
    else
    {
        int status;
        waitpid(child_pid, &status, 0);
        int child_exit_code = WEXITSTATUS(status);

        EXPECT_EQ(child_exit_code, 0) << "Child should have different PID in filename";
    }
}

TEST_F(logger_test, fork_resets_logger_in_child)
{
    auto& parent_logger     = rocprofsys::logger_t::instance();
    auto* parent_logger_ptr = &parent_logger;

    int pipefd[2];
    ASSERT_EQ(pipe(pipefd), 0) << "pipe() failed";

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        auto& child_logger = rocprofsys::logger_t::instance();

        bool logger_has_name = (child_logger.name() == "rocprofiler-systems");

        child_logger.info("fork_reset_child_marker");
        child_logger.flush();

        _exit(logger_has_name ? 0 : 1);
    }

    close(pipefd[1]);
    auto child_output = read_fd(pipefd[0]);
    close(pipefd[0]);

    int status;
    waitpid(child_pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Child logger should be valid after fork reset";

    EXPECT_NE(child_output.find("fork_reset_child_marker"), std::string::npos)
        << "Child log message not found in piped stdout. Got: " << child_output;
    EXPECT_NE(child_output.find("[info]"), std::string::npos)
        << "Log level not found in child output. Got: " << child_output;

    auto child_pid_marker = "P:" + std::to_string(child_pid);
    EXPECT_NE(child_output.find(child_pid_marker), std::string::npos)
        << "Child PID not found in log output. Expected P:" << child_pid
        << " in: " << child_output;

    auto& post_fork_parent_logger = rocprofsys::logger_t::instance();
    EXPECT_EQ(&post_fork_parent_logger, parent_logger_ptr)
        << "Parent logger should remain the same after fork";
}

TEST_F(logger_test, logger_settings_default_values)
{
    unsetenv("ROCPROFSYS_LOG_LEVEL");
    unsetenv("ROCPROFSYS_LOG_FILE");

    rocprofsys::logger_settings_t settings;

    EXPECT_EQ(settings.get_log_level(), spdlog::level::info);
    EXPECT_TRUE(settings.get_log_file().empty());
}

TEST_F(logger_test, logger_settings_with_env_vars)
{
    setenv("ROCPROFSYS_LOG_LEVEL", "debug", 1);
    setenv("ROCPROFSYS_LOG_FILE", "/tmp/test.log", 1);

    rocprofsys::logger_settings_t settings;

    EXPECT_EQ(settings.get_log_level(), spdlog::level::debug);
    EXPECT_EQ(settings.get_log_file(), "/tmp/test.log");

    unsetenv("ROCPROFSYS_LOG_LEVEL");
    unsetenv("ROCPROFSYS_LOG_FILE");
}

TEST_F(logger_test, logger_settings_monochrome)
{
    setenv("ROCPROFSYS_MONOCHROME", "1", 1);

    rocprofsys::logger_settings_t settings;
    const auto*                   pattern = settings.get_log_pattern();

    EXPECT_TRUE(std::string(pattern).find("%^") == std::string::npos);
    EXPECT_TRUE(std::string(pattern).find("%$") == std::string::npos);

    unsetenv("ROCPROFSYS_MONOCHROME");
}

TEST_F(logger_test, fork_child_creates_log_file_with_child_pid)
{
    std::string test_log_base = "/tmp/test_fork_logger";
    std::string test_log_ext  = ".log";
    std::string test_log_file = test_log_base + test_log_ext;

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        setenv("ROCPROFSYS_LOG_FILE", test_log_file.c_str(), 1);

        pid_t current_pid = getpid();

        auto& child_logger = rocprofsys::logger_t::instance();

        std::string expected_child_log_file =
            test_log_base + "_" + std::to_string(current_pid) + test_log_ext;

        child_logger.info("Child process log entry");
        child_logger.flush();

        std::ifstream child_log(expected_child_log_file);
        bool          child_log_exists = child_log.good();

        std::string content;
        bool        has_content = false;
        if(child_log_exists)
        {
            std::getline(child_log, content);
            has_content = content.find("Child process log entry") != std::string::npos;
        }

        std::remove(expected_child_log_file.c_str());

        exit(child_log_exists && has_content ? 0 : 1);
    }
    else
    {
        int status;
        waitpid(child_pid, &status, 0);
        int child_exit_code = WEXITSTATUS(status);

        EXPECT_EQ(child_exit_code, 0)
            << "Child process failed to create its own log file with child PID";
    }
}

TEST_F(logger_test, concurrent_logging_during_fork_no_deadlock)
{
    auto& logger = rocprofsys::logger_t::instance();

    constexpr int     num_threads    = 4;
    constexpr int     log_iterations = 200;
    std::atomic<bool> keep_logging{ true };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([&logger, &keep_logging, i] {
            int iter = 0;
            while(keep_logging.load(std::memory_order_relaxed))
            {
                logger.info("Thread {} iteration {}", i, iter++);
                if(iter >= log_iterations) break;
            }
        });
    }

    int pipefd[2];
    ASSERT_EQ(pipe(pipefd), 0) << "pipe() failed";

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        // In child: threads don't exist, logger was reset by atfork handler.
        // Verify we can acquire a fresh logger without deadlocking.
        auto& child_logger = rocprofsys::logger_t::instance();
        child_logger.info("concurrent_fork_child_marker");
        child_logger.flush();
        _exit(0);
    }

    close(pipefd[1]);
    auto child_output = read_fd(pipefd[0]);
    close(pipefd[0]);

    keep_logging.store(false, std::memory_order_relaxed);
    for(auto& t : threads)
        t.join();

    int   status;
    pid_t waited = waitpid(child_pid, &status, 0);
    ASSERT_NE(waited, -1) << "waitpid failed";
    ASSERT_TRUE(WIFEXITED(status)) << "Child did not exit normally (possible deadlock)";
    EXPECT_EQ(WEXITSTATUS(status), 0);

    EXPECT_NE(child_output.find("concurrent_fork_child_marker"), std::string::npos)
        << "Child log message not found after concurrent fork. Got: " << child_output;

    auto child_pid_marker = "P:" + std::to_string(child_pid);
    EXPECT_NE(child_output.find(child_pid_marker), std::string::npos)
        << "Child PID not in log output. Expected P:" << child_pid
        << " in: " << child_output;
}

TEST_F(logger_test, multiple_sequential_forks)
{
    constexpr int num_forks = 3;

    for(int i = 0; i < num_forks; ++i)
    {
        int pipefd[2];
        ASSERT_EQ(pipe(pipefd), 0) << "pipe() failed on fork " << i;

        pid_t child_pid = fork();
        if(child_pid == 0)
        {
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            close(pipefd[1]);

            auto& child_logger = rocprofsys::logger_t::instance();
            child_logger.info("sequential_fork_child_{}", i);
            child_logger.flush();

            bool valid = (child_logger.name() == "rocprofiler-systems");
            _exit(valid ? 0 : 1);
        }

        close(pipefd[1]);
        auto child_output = read_fd(pipefd[0]);
        close(pipefd[0]);

        int status;
        waitpid(child_pid, &status, 0);
        ASSERT_TRUE(WIFEXITED(status)) << "Fork " << i << " child did not exit normally";
        EXPECT_EQ(WEXITSTATUS(status), 0)
            << "Fork " << i << " child failed to reinitialize logger";

        auto expected_marker = "sequential_fork_child_" + std::to_string(i);
        EXPECT_NE(child_output.find(expected_marker), std::string::npos)
            << "Fork " << i << " child output missing marker. Got: " << child_output;

        auto child_pid_marker = "P:" + std::to_string(child_pid);
        EXPECT_NE(child_output.find(child_pid_marker), std::string::npos)
            << "Fork " << i << " child PID not in output. Got: " << child_output;
    }

    // Parent logger should still work after multiple forks
    testing::internal::CaptureStdout();
    auto& logger = rocprofsys::logger_t::instance();
    EXPECT_EQ(logger.name(), "rocprofiler-systems");
    logger.info("parent_after_{}_forks", num_forks);
    logger.flush();
    auto parent_output = testing::internal::GetCapturedStdout();

    EXPECT_NE(parent_output.find("parent_after_3_forks"), std::string::npos)
        << "Parent log not captured after sequential forks. Got: " << parent_output;
}

TEST_F(logger_test, parent_continues_logging_after_fork)
{
    std::string log_base = "/tmp/test_parent_post_fork";
    std::string log_file = log_base + ".log";

    setenv("ROCPROFSYS_LOG_FILE", log_file.c_str(), 1);

    // Force a fresh logger with the file sink
    // (the singleton was already created without a file sink,
    //  so we test via a child that gets a fresh logger)
    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        auto& child_logger = rocprofsys::logger_t::instance();
        child_logger.info("parent_post_fork_marker");
        child_logger.flush();

        pid_t       current_pid = getpid();
        std::string expected_log_file =
            log_base + "_" + std::to_string(current_pid) + ".log";

        std::ifstream log(expected_log_file);
        bool          found = false;
        std::string   line;
        while(std::getline(log, line))
        {
            if(line.find("parent_post_fork_marker") != std::string::npos)
            {
                found = true;
                break;
            }
        }

        std::remove(expected_log_file.c_str());
        _exit(found ? 0 : 1);
    }

    int status;
    waitpid(child_pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Child failed to log after fork";

    // Parent logger should remain functional
    auto& parent_logger = rocprofsys::logger_t::instance();
    EXPECT_NO_THROW(parent_logger.info("Parent logging after child completed"));

    unsetenv("ROCPROFSYS_LOG_FILE");
}

TEST_F(logger_test, fork_child_gets_new_logger_instance)
{
    auto& parent_logger = rocprofsys::logger_t::instance();
    EXPECT_EQ(parent_logger.name(), "rocprofiler-systems");
    EXPECT_FALSE(parent_logger.sinks().empty());

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        auto& child_logger = rocprofsys::logger_t::instance();

        // The child logger should be freshly created with default sinks
        // (stdout only, since ROCPROFSYS_LOG_FILE is not set)
        bool has_sinks = !child_logger.sinks().empty();
        bool has_name  = (child_logger.name() == "rocprofiler-systems");
        bool can_log   = true;

        try
        {
            child_logger.info("Verifying child logger works");
            child_logger.flush();
        } catch(...)
        {
            can_log = false;
        }

        _exit((has_sinks && has_name && can_log) ? 0 : 1);
    }

    int status;
    waitpid(child_pid, &status, 0);
    ASSERT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0) << "Child should get a fresh, functional logger";

    // Parent logger should remain valid after fork
    EXPECT_EQ(parent_logger.name(), "rocprofiler-systems");
    EXPECT_FALSE(parent_logger.sinks().empty());
    EXPECT_NO_THROW(parent_logger.info("Parent still works after child fork"));
}

TEST_F(logger_test, concurrent_logging_stress_with_fork)
{
    auto& logger = rocprofsys::logger_t::instance();

    constexpr int     num_threads      = 8;
    constexpr int     iterations       = 500;
    constexpr int     child_iterations = 10;
    std::atomic<bool> start{ false };
    std::atomic<int>  ready_count{ 0 };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(int i = 0; i < num_threads; ++i)
    {
        threads.emplace_back([&logger, &start, &ready_count, i] {
            ready_count.fetch_add(1, std::memory_order_release);
            while(!start.load(std::memory_order_acquire))
            {
            }
            for(int j = 0; j < iterations; ++j)
            {
                logger.info("Stress thread {} iter {}", i, j);
            }
        });
    }

    // Wait for all threads to be ready
    while(ready_count.load(std::memory_order_acquire) < num_threads)
    {
    }
    start.store(true, std::memory_order_release);

    int pipefd[2];
    ASSERT_EQ(pipe(pipefd), 0) << "pipe() failed";

    // Fork while threads are actively logging
    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);

        // Child: threads don't exist. Logger must be usable.
        auto& child_logger = rocprofsys::logger_t::instance();
        for(int j = 0; j < child_iterations; ++j)
        {
            child_logger.info("stress_child_iter_{}", j);
        }
        child_logger.flush();
        _exit(0);
    }

    close(pipefd[1]);
    auto child_output = read_fd(pipefd[0]);
    close(pipefd[0]);

    for(auto& t : threads)
        t.join();

    int   status;
    pid_t waited = waitpid(child_pid, &status, 0);
    ASSERT_NE(waited, -1) << "waitpid failed";
    ASSERT_TRUE(WIFEXITED(status))
        << "Child did not exit normally under stress (possible deadlock)";
    EXPECT_EQ(WEXITSTATUS(status), 0);

    // Validate child produced all expected log lines
    for(int j = 0; j < child_iterations; ++j)
    {
        auto marker = "stress_child_iter_" + std::to_string(j);
        EXPECT_NE(child_output.find(marker), std::string::npos)
            << "Missing child stress iteration " << j << " in output";
    }

    auto child_pid_marker = "P:" + std::to_string(child_pid);
    EXPECT_NE(child_output.find(child_pid_marker), std::string::npos)
        << "Child PID not in stress test output. Got: " << child_output;

    // Parent logger should still work
    EXPECT_NO_THROW(logger.info("Parent done after stress fork"));
}
