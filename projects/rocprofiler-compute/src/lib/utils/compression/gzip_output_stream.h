// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <string>
#include <string_view>

// Forward-declare gzFile so this header need not include zlib.h.
struct gzFile_s;

namespace rocprofiler_compute_tool::compression
{

inline constexpr std::string_view kGzipSuffix = ".gz";

// Level 1 for write-once counter CSVs; matches GZIP_LEVEL in csv_compression.py.
inline constexpr int kCompressionLevel = 1;

/// Thin gzip adapter over zlib gzFile. Caller supplies path including kGzipSuffix.
class GzipFileOutputStream
{
public:
    explicit GzipFileOutputStream(const std::string& path, int level = kCompressionLevel);
    ~GzipFileOutputStream();

    GzipFileOutputStream(const GzipFileOutputStream&)            = delete;
    GzipFileOutputStream& operator=(const GzipFileOutputStream&) = delete;

    bool write(std::string_view data);
    bool close();
    bool ok() const;
    bool is_open() const;

private:
    std::string m_path;
    gzFile_s*   m_file   = nullptr;
    bool        m_failed = false;
};

}  // namespace rocprofiler_compute_tool::compression
