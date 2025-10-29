#ifndef CUID_UTIL_H
#define CUID_UTIL_H

#include "cuid.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>

enum LogLevel { DEBUG, INFO, WARN, ERROR };

class Logger {
public:
    static Logger& instance() {
        static Logger logger_;
        return logger_;
    }

    void set_level(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    const char* LogLevelName(LogLevel level) const;

    void log(LogLevel level, const std::string& msg) const;

private:
    Logger() : level_(INFO) {}
    LogLevel level_;
};

#define LOG(level, msg) \
    do { \
        std::ostringstream _log_stream_; \
        _log_stream_ << msg; \
        Logger::instance().log(level, _log_stream_.str()); \
    } while (0)

namespace AmdCuidUtilities {
    std::string read_sysfs_file(const std::string &path);
    std::string readlink_bdf(const std::string &device_path);
    amdcuid_status_t generate_secondary_cuid(const amdcuid* primary_id, amdcuid* secondary_id);
    amdcuid_status_t generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t component_type, amdcuid* id);
    char* get_cuid_as_string(const amdcuid *id);
}

#endif
