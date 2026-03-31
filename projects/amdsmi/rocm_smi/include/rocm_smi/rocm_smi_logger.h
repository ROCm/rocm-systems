/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Detail Description:
 * Implemented complete logging mechanism, supporting multiple logging type
 * like as file based logging, console base logging etc. It also supported
 * for different log types.
 *
 * Thread Safe logging mechanism. Compatible with G++ (Linux platform)
 *
 * Supported Log Type: ERROR, ALARM, ALWAYS, INFO, BUFFER, TRACE, DEBUG
 * No control for ERROR, ALRAM and ALWAYS messages. These type of messages
 * should be always captured -- IF logging is enabled.
 *
 * WARNING: Logging is controlled by users environment variable - RSMI_LOGGING.
 * Enabling RSMI_LOGGING, by export RSMI_LOGGING=<any value>. No logs will
 * be printed, unless RSMI_LOGGING is enabled.
 *
 * BUFFER log type should be use while logging raw buffer or raw messages
 * Having direct interface as well as C++ Singleton inface. Can use
 * whatever interface fits your needs.
 */

#ifndef _ROCM_SMI_LOGGER_H_
#define _ROCM_SMI_LOGGER_H_

// C Header File(s)

// C++ Header File(s)
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

// Code Specific Header Files(s)

namespace ROCmLogging {
// Direct Interface for logging into log file or console using MACRO(s)
#define LOG_ERROR(x) (ROCmLogging::Logger::getInstance()->error(x))
#define LOG_ALARM(x) (ROCmLogging::Logger::getInstance()->alarm(x))
#define LOG_ALWAYS(x) (ROCmLogging::Logger::getInstance()->always(x))
#define LOG_INFO(x) (ROCmLogging::Logger::getInstance()->info(x))
#define LOG_WARN(x) (ROCmLogging::Logger::getInstance()->warn(x))
#define LOG_BUFFER(x) (ROCmLogging::Logger::getInstance()->buffer(x))
#define LOG_TRACE(x) (ROCmLogging::Logger::getInstance()->trace(x))
#define LOG_DEBUG(x) (ROCmLogging::Logger::getInstance()->debug(x))

// enum for LOG_LEVEL
typedef enum LOG_LEVEL {
  DISABLE_LOG = 1,
  LOG_LEVEL_INFO = 2,
  LOG_LEVEL_WARN = 3,
  LOG_LEVEL_BUFFER = 4,
  LOG_LEVEL_TRACE = 5,
  LOG_LEVEL_DEBUG = 6,
  ENABLE_LOG = 7,
} LogLevel;

// enum for LOG_TYPE
typedef enum LOG_TYPE { NO_LOG = 1, CONSOLE = 2, FILE_LOG = 3, BOTH_FILE_AND_CONSOLE = 4 } LogType;

class Logger {
 public:
  static Logger* getInstance() noexcept;

  Logger& operator<<(std::string& s) {
    switch (this->m_LogLevel) {
      case DISABLE_LOG:
        break;
      case LOG_LEVEL_INFO:
        info(s);
        break;
      case LOG_LEVEL_WARN:
        warn(s);
        break;
      case LOG_LEVEL_BUFFER:
        buffer(s);
        break;
      case LOG_LEVEL_TRACE:
        trace(s);
        break;
      case LOG_LEVEL_DEBUG:
        debug(s);
        break;
      case ENABLE_LOG:
        always(s);
        break;
      default:
        break;
    }
    return *getInstance();
  }

  Logger& operator<<(const char* s) { return operator<<(std::string(s)); }

  template <class T>
  Logger& operator<<(const T& v) {
    std::ostringstream s;
    s << v;
    std::string str = s.str();
    return operator<<(str);
  }

  // Interface for Error Log
  void error(const char* text) noexcept;
  void error(std::string& text) noexcept;  // NOLINT
  void error(std::ostringstream& stream) noexcept;

  // Interface for Alarm Log
  void alarm(const char* text) noexcept;
  void alarm(std::string& text) noexcept;  // NOLINT
  void alarm(std::ostringstream& stream) noexcept;

  // Interface for Always Log
  void always(const char* text) noexcept;
  void always(std::string& text) noexcept;  // NOLINT
  void always(std::ostringstream& stream) noexcept;

  // Interface for Buffer Log
  void buffer(const char* text) noexcept;
  void buffer(std::string& text) noexcept;  // NOLINT
  void buffer(std::ostringstream& stream) noexcept;

  // Interface for Info Log
  void info(const char* text) noexcept;
  void info(std::string& text) noexcept;  // NOLINT
  void info(std::ostringstream& stream) noexcept;

  // Interface for Warn Log
  void warn(const char* text) noexcept;
  void warn(std::string& text) noexcept;  // NOLINT
  void warn(std::ostringstream& stream) noexcept;

  // Interface for Trace log
  void trace(const char* text) noexcept;
  void trace(std::string& text) noexcept;  // NOLINT
  void trace(std::ostringstream& stream) noexcept;

  // Interface for Debug log
  void debug(const char* text) noexcept;
  void debug(std::string& text) noexcept;  // NOLINT
  void debug(std::ostringstream& stream) noexcept;

  // Error and Alarm log must be always enable
  // Hence, there is no interfce to control error and alarm logs

  // Interfaces to control log levels
  void updateLogLevel(LogLevel logLevel);
  void enableAllLogLevels();  // Enable all log levels
  void disableLog();          // Disable all log levels, except error and alarm

  // Interfaces to control log Types
  void updateLogType(LogType logType);
  void enableConsoleLogging();
  void enableFileLogging();
  std::string getLogSettings();
  bool isLoggerEnabled();

 protected:
  Logger();
  ~Logger();

  // Wrapper function for lock/unlock
  // For Extensible feature, lock and unlock should be in protected
  void lock();
  void unlock();

  std::string getCurrentTime();

 private:
  std::ofstream m_File;
  bool m_loggingIsOn = false;
  LogLevel m_LogLevel;
  LogType m_LogType;
  std::mutex m_Mutex;
  std::unique_lock<std::mutex> m_Lock{m_Mutex, std::defer_lock};

  void logIntoFile(std::string& data);   // NOLINT
  void logOnConsole(std::string& data);  // NOLINT
  void operator=(const Logger&) {}
  void initialize_resources();
  void destroy_resources();
};

}  // namespace ROCmLogging

#endif  // End of _ROCM_SMI_LOGGER_H_
