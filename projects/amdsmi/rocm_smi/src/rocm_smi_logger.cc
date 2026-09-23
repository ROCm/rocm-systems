// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
 * Having direct interface as well as C++ Singleton iface. Can use
 * whatever interface fits your needs.
 */

// C++ Header File(s)
#include <cstdlib>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>

// Code Specific Header Files(s)
#include "rocm_smi/rocm_smi_logger.h"
#include "rocm_smi/rocm_smi_main.h"

// Log file name
// WARNING: File name should be changed here and
// pre/post install/remove/upgrade scripts. Changing
// in one place will cause a mismatch in these scripts,
// files may not have proper permissions, and logrotate
// would not function properly.
#define LOGPATH "/var/log/amd_smi_lib/"
#define LOGBASE_FNAME "AMD-SMI-lib"
#define LOGEXTENSION ".log"
const char* logFileName = LOGPATH LOGBASE_FNAME LOGEXTENSION;

ROCmLogging::Logger::Logger() { initialize_resources(); }

ROCmLogging::Logger::~Logger() {
  if (m_loggingIsOn) {
    destroy_resources();
  }
}

// Updated to Meyers Singleton pattern
ROCmLogging::Logger* ROCmLogging::Logger::getInstance() noexcept {
  static Logger instance;  // Thread-safe since C++11, no dynamic allocation
  return &instance;
}

void ROCmLogging::Logger::lock() { m_Lock.lock(); }

void ROCmLogging::Logger::unlock() { m_Lock.unlock(); }

void ROCmLogging::Logger::logIntoFile(std::string& data) {
  lock();
  if (!m_File.is_open()) {
    initialize_resources();
    if (!m_File.is_open()) {
      std::cout << "WARNING: re-initializing resources was unsuccessful."
                << " Unable to print the following message." << std::endl;
      logOnConsole(data);
      unlock();
      return;
    }
  }
  m_File << getCurrentTime() << "  " << data << std::endl;
  unlock();
}

void ROCmLogging::Logger::logOnConsole(std::string& data) {
  std::cout << getCurrentTime() << "  " << data << std::endl;
}

// Returns: In string format, YY-MM-DD HH:MM:SS.microseconds
std::string ROCmLogging::Logger::getCurrentTime(void) {
  std::string currentTime;

  // get current time
  auto now = std::chrono::system_clock::now();

  // get number of milliseconds for the current second
  // (remainder after division into seconds)
  auto ms = std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

  // convert to std::time_t in order to convert to std::tm (broken time)
  auto timer = std::chrono::system_clock::to_time_t(now);

  // convert to broken time
  std::tm bt = *std::localtime(&timer);

  std::ostringstream oss;

  // YY-MM-DD HH:MM:SS.microseconds
  oss << std::put_time(&bt, "%F %T");
  oss << '.' << std::setfill('0') << std::setw(4) << ms.count();
  currentTime = oss.str();
  return currentTime;
}

// Interface for Error Log
void ROCmLogging::Logger::error(const char* text) noexcept {
  // By default, logging is disabled
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[ERROR]: ");
  data.append(text);

  // ERROR must be capture
  if (m_LogType == FILE_LOG) {
    logIntoFile(data);
  } else if (m_LogType == CONSOLE) {
    logOnConsole(data);
  } else if (m_LogType == BOTH_FILE_AND_CONSOLE) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::error(std::string& text) noexcept { error(text.data()); }

void ROCmLogging::Logger::error(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  error(text.data());
  stream.str("");
}

// Interface for Alarm Log
void ROCmLogging::Logger::alarm(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[ALARM]: ");
  data.append(text);

  // ALARM must be capture
  if (m_LogType == FILE_LOG) {
    logIntoFile(data);
  } else if (m_LogType == CONSOLE) {
    logOnConsole(data);
  } else if (m_LogType == BOTH_FILE_AND_CONSOLE) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::alarm(std::string& text) noexcept { alarm(text.data()); }

void ROCmLogging::Logger::alarm(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  alarm(text.data());
  stream.str("");
}

// Interface for Always Log
void ROCmLogging::Logger::always(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[ALWAYS]: ");
  data.append(text);

  // No check for ALWAYS logs
  if (m_LogType == FILE_LOG) {
    logIntoFile(data);
  } else if (m_LogType == CONSOLE) {
    logOnConsole(data);
  } else if (m_LogType == BOTH_FILE_AND_CONSOLE) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::always(std::string& text) noexcept { always(text.data()); }

void ROCmLogging::Logger::always(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  always(text.data());
  stream.str("");
}

// Interface for Buffer Log
void ROCmLogging::Logger::buffer(const char* text) noexcept {
  // Buffer is the special case. So don't add log level
  // and timestamp in the buffer message. Just log the raw bytes.
  if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_BUFFER)) {
    lock();
    if (!m_File.is_open()) {
      initialize_resources();
      if (!m_File.is_open()) {
        std::cout << "WARNING: re-initializing resources was unsuccessful."
                  << " Unable to print the following message." << std::endl;
        std::string txtStr(text);
        std::cout << txtStr << std::endl;
        unlock();
        return;
      }
    }
    m_File << text << std::endl;
    unlock();
  } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_BUFFER)) {
    std::cout << text << std::endl;
  }
}

void ROCmLogging::Logger::buffer(std::string& text) noexcept { buffer(text.data()); }

void ROCmLogging::Logger::buffer(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  buffer(text.data());
  stream.str("");
}

// Interface for Info Log
void ROCmLogging::Logger::info(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[INFO]: ");
  data.append(text);

  if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_INFO)) {
    logIntoFile(data);
  } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_INFO)) {
    logOnConsole(data);
  } else if ((m_LogType == BOTH_FILE_AND_CONSOLE) && (m_LogLevel >= LOG_LEVEL_INFO)) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::info(std::string& text) noexcept { info(text.data()); }

void ROCmLogging::Logger::info(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  info(text.data());
  stream.str("");
}

// Interface for Warn Log
void ROCmLogging::Logger::warn(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[WARN]: ");
  data.append(text);

  if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_WARN)) {
    logIntoFile(data);
  } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_WARN)) {
    logOnConsole(data);
  } else if ((m_LogType == BOTH_FILE_AND_CONSOLE) && (m_LogLevel >= LOG_LEVEL_WARN)) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::warn(std::string& text) noexcept { warn(text.data()); }

void ROCmLogging::Logger::warn(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  warn(text.data());
  stream.str("");
}

// Interface for Trace Log
void ROCmLogging::Logger::trace(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[TRACE]: ");
  data.append(text);

  if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
    logIntoFile(data);
  } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
    logOnConsole(data);
  } else if ((m_LogType == BOTH_FILE_AND_CONSOLE) && (m_LogLevel >= LOG_LEVEL_TRACE)) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::trace(std::string& text) noexcept { trace(text.data()); }

void ROCmLogging::Logger::trace(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  trace(text.data());
  stream.str("");
}

// Interface for Debug Log
void ROCmLogging::Logger::debug(const char* text) noexcept {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  if (!m_loggingIsOn) {
    return;
  }

  std::string data;
  data.append("[DEBUG]: ");
  data.append(text);

  if ((m_LogType == FILE_LOG) && (m_LogLevel >= LOG_LEVEL_DEBUG)) {
    logIntoFile(data);
  } else if ((m_LogType == CONSOLE) && (m_LogLevel >= LOG_LEVEL_DEBUG)) {
    logOnConsole(data);
  } else if ((m_LogType == BOTH_FILE_AND_CONSOLE) && (m_LogLevel >= LOG_LEVEL_DEBUG)) {
    logOnConsole(data);
    logIntoFile(data);
  }
}

void ROCmLogging::Logger::debug(std::string& text) noexcept { debug(text.data()); }

void ROCmLogging::Logger::debug(std::ostringstream& stream) noexcept {
  std::string text = stream.str();
  debug(text.data());
  stream.str("");
}

// Interfaces to control log levels
void ROCmLogging::Logger::updateLogLevel(LogLevel logLevel) { m_LogLevel = logLevel; }

void ROCmLogging::Logger::enableAllLogLevels() { m_LogLevel = ENABLE_LOG; }

// Disable all log levels, except error and alarm
void ROCmLogging::Logger::disableLog() { m_LogLevel = DISABLE_LOG; }

// Interfaces to control log Types
void ROCmLogging::Logger::updateLogType(LogType logType) { m_LogType = logType; }

void ROCmLogging::Logger::enableConsoleLogging() { m_LogType = CONSOLE; }

void ROCmLogging::Logger::enableFileLogging() { m_LogType = FILE_LOG; }

// Returns a string of details on current log settings
std::string ROCmLogging::Logger::getLogSettings() {
  std::string logSettings;

  logSettings += ("Logging is " + std::string((m_loggingIsOn ? "ENABLED" : "DISABLED")) + ", ");

  if (m_File.is_open()) {
    logSettings += "OpenStatus = File (" + std::string(logFileName) + ") is open";
  } else {
    logSettings += "OpenStatus = File (" + std::string(logFileName) + ") is not open";
  }
  logSettings += ", ";

  switch (m_LogType) {
    case NO_LOG:
      logSettings += "LogType = NO_LOG";
      break;
    case FILE_LOG:
      logSettings += "LogType = FILE_LOG";
      break;
    case CONSOLE:
      logSettings += "LogType = CONSOLE";
      break;
    case BOTH_FILE_AND_CONSOLE:
      logSettings += "LogType = BOTH_FILE_AND_CONSOLE";
      break;
    default:
      logSettings += "LogType = <undefined>";
  }
  logSettings += ", ";

  switch (m_LogLevel) {
    case DISABLE_LOG:
      logSettings += "LogLevel = DISABLE_LOG";
      break;
    case LOG_LEVEL_INFO:
      logSettings += "LogLevel = LOG_LEVEL_INFO";
      break;
    case LOG_LEVEL_WARN:
      logSettings += "LogLevel = LOG_LEVEL_WARN";
      break;
    case LOG_LEVEL_BUFFER:
      logSettings += "LogLevel = LOG_LEVEL_BUFFER";
      break;
    case LOG_LEVEL_TRACE:
      logSettings += "LogLevel = LOG_LEVEL_TRACE";
      break;
    case LOG_LEVEL_DEBUG:
      logSettings += "LogLevel = LOG_LEVEL_DEBUG";
      break;
    case ENABLE_LOG:
      logSettings += "LogLevel = ENABLE_LOG";
      break;
    default:
      logSettings += "LogLevel = <undefined>";
  }

  return logSettings;
}

// Returns current reported enabled logging state. State is controlled by
// user's environment variable RSMI_LOGGING.
bool ROCmLogging::Logger::isLoggerEnabled() { return m_loggingIsOn; }

void ROCmLogging::Logger::initialize_resources() {
  // By default, logging is disabled (ie. no RSMI_LOGGING)
  // The check below allows us to toggle logging through RSMI_LOGGING
  // set or unset
  try {
    m_loggingIsOn = amd::smi::RocmSMI::getInstance().isLoggingOn();
  } catch (...) {
    m_loggingIsOn = false;  // Fail safe - disable logging
  }

  if (!m_loggingIsOn) {
    return;
  }
  m_File.open(logFileName, std::ios::out | std::ios::app);
  m_LogLevel = LOG_LEVEL_TRACE;
  // RSMI_LOGGING = 1, output to logs only
  // RSMI_LOGGING = 2, output to console only
  // RSMI_LOGGING = 3, output to logs and console
  switch (amd::smi::RocmSMI::getInstance().getLogSetting()) {
    case 0:
      m_LogType = NO_LOG;
      break;
    case 1:
      m_LogType = FILE_LOG;
      break;
    case 2:
      m_LogType = CONSOLE;
      break;
    case 3:
      m_LogType = BOTH_FILE_AND_CONSOLE;
      break;
    default:
      m_LogType = NO_LOG;
      break;
  }
  if (!m_File.is_open()) {
    std::cout << "WARNING: Issue opening log file (" << logFileName << ") to write." << std::endl;
  }
  if (m_File.fail()) {
    std::cout << "WARNING: Failed opening log file." << std::endl;
  }
  // 0664 to match the mode the package postinst sets and avoid churn on the
  // first root-run. The log is root-owned, so the group-write bit has no
  // practical effect; the security-relevant part is withholding world-write
  // (S_IWOTH), which would let any local user tamper with a root-owned log
  // (CWE-732).
  chmod(logFileName, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP | S_IROTH);
}

void ROCmLogging::Logger::destroy_resources() { m_File.close(); }
