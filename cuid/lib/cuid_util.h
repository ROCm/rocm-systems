#ifndef CUID_UTIL_H
#define CUID_UTIL_H

#include "cuid.h"
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <sstream>
#include <map>

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

using SectionMap = std::map<std::string, std::map<std::string, std::string>>;
namespace CuidUtilities {
    std::string read_sysfs_file(const std::string &path);
    std::string readlink_bdf(const std::string &device_path);
    amdcuid get_secondary_cuid(amdcuid_salt_t salt, const amdcuid* primary_id);
    amdcuid generate_primary_cuid(uint64_t serial_number, uint8_t unit_id_part1, uint8_t unit_id_part2,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t component_type);
    char* get_cuid_as_string(const amdcuid *id);
    const char *cuid_status_to_string(amdcuid_status_t status);
    SectionMap parse_cuid_file(const std::string &filename);
    void write_cuid_file(const std::string &filename, const SectionMap &sections);

}

#endif