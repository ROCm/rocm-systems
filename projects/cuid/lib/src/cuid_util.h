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

#ifndef CUID_UTIL_H
#define CUID_UTIL_H

#include "include/amd_cuid.h"
#include "src/cuid_internal.h"
#include "hmac.h"
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

namespace CuidUtilities {
    std::string read_sysfs_file(const std::string &path);
    std::string readlink_bdf(const std::string &device_path);
    std::string bdf_to_device_path(const std::string &bdf, amdcuid_device_type_t device_type);
    std::string real_dev_path_from_fd(int fd);
    std::string get_real_path(const std::string path);
    amdcuid_status_t generate_derived_cuid(const amdcuid_primary_id* primary_id, amdcuid_derived_id* derived_id, cuid_hmac* hmac);
    amdcuid_status_t generate_primary_cuid(uint64_t serial_number, uint16_t unit_id,
                                 uint8_t revision_id, uint16_t device_id, uint16_t vendor_id,
                                 uint8_t device_type, amdcuid_primary_id* primary_id);
    void remove_UUIDv8_bits(amdcuid_id_t* id, uint8_t out_raw_bits[16]);
    std::string get_cuid_as_string(const amdcuid_id_t *id);
    amdcuid_status_t uuid_string_to_uint8(const std::string& uuid_str, uint8_t* uuid);
    std::string device_type_to_string(amdcuid_device_type_t type);

    // Use inline functions to avoid static initialization order fiasco
    inline const std::string& cuid_file() {
        static const std::string path = "/tmp/cuid";
        return path;
    }
    inline const std::string& priv_cuid_file() {
        static const std::string path = "/tmp/priv_cuid";
        return path;
    }
}

#endif
