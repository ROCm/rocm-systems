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

#include <cstdlib>
#include <fstream>
#include <set>
#include <string>

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

    pid_t child_pid = fork();
    if(child_pid == 0)
    {
        auto& child_logger     = rocprofsys::logger_t::instance();
        auto* child_logger_ptr = &child_logger;

        bool logger_is_valid = (child_logger_ptr != nullptr);
        bool logger_has_name = (child_logger.name() == "rocprofiler-systems");

        exit((logger_is_valid && logger_has_name) ? 0 : 1);
    }
    else
    {
        int status;
        waitpid(child_pid, &status, 0);
        int child_exit_code = WEXITSTATUS(status);

        EXPECT_EQ(child_exit_code, 0) << "Child logger should be valid after fork reset";

        auto& post_fork_parent_logger = rocprofsys::logger_t::instance();
        EXPECT_EQ(&post_fork_parent_logger, parent_logger_ptr)
            << "Parent logger should remain the same after fork";
    }
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
