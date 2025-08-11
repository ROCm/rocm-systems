/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
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

// C++ Header File(s)
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <mutex>

// POSIX Socket Header File(s)
#include <errno.h>

// Code Specific Header Files(s)


namespace ROCmLogging {
// Direct Interface for logging into log file or console using MACRO(s)
#define LOG_ERROR(x) (ROCmLogging::Logger::getInstance()->error(x))
#define LOG_ALARM(x) (ROCmLogging::Logger::getInstance()->alarm(x))
#define LOG_ALWAYS(x) (ROCmLogging::Logger::getInstance()->always(x))
#define LOG_INFO(x) (ROCmLogging::Logger::getInstance()->info(x))
#define LOG_BUFFER(x) (ROCmLogging::Logger::getInstance()->buffer(x))
#define LOG_TRACE(x) (ROCmLogging::Logger::getInstance()->trace(x))
#define LOG_DEBUG(x) (ROCmLogging::Logger::getInstance()->debug(x))

// enum for LOG_LEVEL
typedef enum LOG_LEVEL {
  DISABLE_LOG = 1,
  LOG_LEVEL_INFO = 2,
  LOG_LEVEL_BUFFER = 3,
  LOG_LEVEL_TRACE = 4,
  LOG_LEVEL_DEBUG = 5,
  ENABLE_LOG = 6,
} LogLevel;

// enum for LOG_TYPE
typedef enum LOG_TYPE {
  NO_LOG = 1,
  CONSOLE = 2,
  FILE_LOG = 3,
  BOTH_FILE_AND_CONSOLE = 4
} LogType;

class Logger {
 public:
  static Logger* getInstance() throw();

  Logger& operator<<(std::string &s) {
    switch (this->m_LogLevel) {
      case DISABLE_LOG:
        break;
      case LOG_LEVEL_INFO:
        info(s);
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

  Logger &operator<<(const char* s) {
    return operator<<(std::string(s));
  }

  template <class T> Logger &operator<<(const T &v) {
    std::ostringstream s;
    s << v;
    std::string str = s.str();
    return operator<<(str);
  }

  // Interface for Error Log
  void error(const char* text) throw();
  void error(std::string& text) throw();
  void error(std::ostringstream& stream) throw();

  // Interface for Alarm Log
  void alarm(const char* text) throw();
  void alarm(std::string& text) throw();
  void alarm(std::ostringstream& stream) throw();

  // Interface for Always Log
  void always(const char* text) throw();
  void always(std::string& text) throw();
  void always(std::ostringstream& stream) throw();

  // Interface for Buffer Log
  void buffer(const char* text) throw();
  void buffer(std::string& text) throw();
  void buffer(std::ostringstream& stream) throw();

  // Interface for Info Log
  void info(const char* text) throw();
  void info(std::string& text) throw();
  void info(std::ostringstream& stream) throw();

  // Interface for Trace log
  void trace(const char* text) throw();
  void trace(std::string& text) throw();
  void trace(std::ostringstream& stream) throw();

  // Interface for Debug log
  void debug(const char* text) throw();
  void debug(std::string& text) throw();
  void debug(std::ostringstream& stream) throw();

  // Error and Alarm log must be always enable
  // Hence, there is no interfce to control error and alarm logs

  // Interfaces to control log levels
  void updateLogLevel(LogLevel logLevel);
  void enableAllLogLevels();    // Enable all log levels
  void disableLog();  // Disable all log levels, except error and alarm

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
  static Logger* m_Instance;
  std::ofstream m_File;
  bool m_loggingIsOn = false;
  LogLevel m_LogLevel;
  LogType m_LogType;
  std::mutex m_Mutex;
  std::unique_lock<std::mutex> m_Lock{m_Mutex, std::defer_lock};

  void logIntoFile(std::string& data);
  void logOnConsole(std::string& data);
  void operator=(const Logger&) {}
  void initialize_resources();
  void destroy_resources();
};

}  // namespace ROCmLogging

#endif  // End of _ROCM_SMI_LOGGER_H_
