#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <fstream>
#include <filesystem>
#include <thread>
#include <chrono>
#include <sstream>
#include <cstdlib>
#include <unistd.h>

#include "../logger.h"

// Define static members for Logger class
namespace aql_profile {
Logger::mutex_t Logger::mutex_;
Logger* Logger::instance_ = nullptr;
}

namespace aql_profile {

class LoggerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Clean up any existing instance
        Logger::Destroy();
        
        // Remove any existing log file
        if (std::filesystem::exists(log_file_path_)) {
            std::filesystem::remove(log_file_path_);
        }
        
        // Clear environment variable
        unsetenv("HSA_VEN_AMD_AQLPROFILE_LOG");
    }

    void TearDown() override {
        // Clean up after each test
        Logger::Destroy();
        unsetenv("HSA_VEN_AMD_AQLPROFILE_LOG");
        
        // Remove test log file
        if (std::filesystem::exists(log_file_path_)) {
            std::filesystem::remove(log_file_path_);
        }
    }

    const std::string log_file_path_ = "/tmp/aql_profile_log.txt";
    
    // Helper function to read log file content
    std::string ReadLogFile() {
        std::ifstream file(log_file_path_);
        if (!file.is_open()) return "";
        
        std::stringstream buffer;
        buffer << file.rdbuf();
        return buffer.str();
    }
    
    // Helper function to enable file logging
    void EnableFileLogging() {
        setenv("HSA_VEN_AMD_AQLPROFILE_LOG", "1", 1);
    }
};

// Test singleton pattern
TEST_F(LoggerTest, SingletonPattern) {
    Logger& logger1 = Logger::Instance();
    Logger& logger2 = Logger::Instance();
    
    // Should be the same instance
    EXPECT_EQ(&logger1, &logger2);
}

// Test basic logging without file output
TEST_F(LoggerTest, BasicLoggingWithoutFile) {
    Logger& logger = Logger::Instance();
    
    // Should not crash when logging without file
    logger << "Test message";
    
    // Verify log file doesn't exist
    EXPECT_FALSE(std::filesystem::exists(log_file_path_));
}

// Test basic logging with file output
TEST_F(LoggerTest, BasicLoggingWithFile) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "Test message";
    
    // Give some time for file operations
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // Verify log file exists and contains content
    EXPECT_TRUE(std::filesystem::exists(log_file_path_));
    
    std::string content = ReadLogFile();
    EXPECT_FALSE(content.empty());
    EXPECT_THAT(content, testing::HasSubstr("Test message"));
    EXPECT_THAT(content, testing::HasSubstr("pid"));
    EXPECT_THAT(content, testing::HasSubstr("tid"));
}

// Test streaming operations
TEST_F(LoggerTest, StreamingOperations) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "Number: " << 42 << " String: " << "test" << " Float: " << 3.14;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    EXPECT_THAT(content, testing::HasSubstr("Number: 42"));
    EXPECT_THAT(content, testing::HasSubstr("String: test"));
    EXPECT_THAT(content, testing::HasSubstr("Float: 3.14"));
}

// Test endl manipulator
TEST_F(LoggerTest, EndlManipulator) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "First line" << Logger::endl << "Second line";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    EXPECT_THAT(content, testing::HasSubstr("First line"));
    EXPECT_THAT(content, testing::HasSubstr("Second line"));
    
    // Should have multiple log entries with timestamps
    size_t pid_count = 0;
    size_t pos = 0;
    while ((pos = content.find("pid", pos)) != std::string::npos) {
        pid_count++;
        pos += 3;
    }
    EXPECT_GE(pid_count, 2); // At least 2 log entries
}


// Test concurrent logging from multiple threads
TEST_F(LoggerTest, ConcurrentLogging) {
    EnableFileLogging();
    
    const int num_threads = 4;
    const int messages_per_thread = 10;
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back([i, messages_per_thread]() {
            Logger& logger = Logger::Instance();
            for (int j = 0; j < messages_per_thread; ++j) {
                logger << "Thread " << i << " Message " << j;
            }
        });
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    
    // Verify log file contains messages from all threads
    std::string content = ReadLogFile();
    EXPECT_FALSE(content.empty());
    
    // Count messages from each thread
    for (int i = 0; i < num_threads; ++i) {
        std::string thread_pattern = "Thread " + std::to_string(i);
        EXPECT_THAT(content, testing::HasSubstr(thread_pattern));
    }
}

// Test logging with special characters
TEST_F(LoggerTest, SpecialCharacters) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    std::string special_msg = "Special chars: !@#$%^&*()_+-=[]{}|;':\",./<>?";
    logger << special_msg;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    EXPECT_THAT(content, testing::HasSubstr(special_msg));
}

// Test large message logging
TEST_F(LoggerTest, LargeMessage) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    std::string large_msg(1000, 'A'); // 1000 character message
    logger << large_msg;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    EXPECT_THAT(content, testing::HasSubstr(large_msg));
}


// Test timestamp format in logs
TEST_F(LoggerTest, TimestampFormat) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "Timestamp test";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    
    // Check for timestamp pattern (YYYY-MM-DD HH:MM:SS)
    EXPECT_THAT(content, testing::MatchesRegex(".*[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}.*"));
}

// Test PID and TID in logs
TEST_F(LoggerTest, PidTidInLogs) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "PID/TID test";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    
    // Check for PID and TID patterns
    EXPECT_THAT(content, testing::HasSubstr("pid"));
    EXPECT_THAT(content, testing::HasSubstr("tid"));
    
    // Verify they contain numbers
    EXPECT_THAT(content, testing::MatchesRegex(".*pid[0-9]+.*"));
    EXPECT_THAT(content, testing::MatchesRegex(".*tid[0-9]+.*"));
}

// Test empty message handling
TEST_F(LoggerTest, EmptyMessage) {
    Logger& logger = Logger::Instance();
    
    Logger::begm();
    Logger::endl();
    
    const std::string& msg = Logger::LastMessage();
    EXPECT_EQ(msg, "");
}

// Test multiple consecutive endl calls
TEST_F(LoggerTest, MultipleEndl) {
    EnableFileLogging();
    
    Logger& logger = Logger::Instance();
    logger << "Test" << Logger::endl << Logger::endl << "After multiple endl";
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    std::string content = ReadLogFile();
    EXPECT_THAT(content, testing::HasSubstr("Test"));
    EXPECT_THAT(content, testing::HasSubstr("After multiple endl"));
}

} // namespace aql_profile
