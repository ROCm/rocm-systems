// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "gzip_output_stream.h"

#include <zlib.h>

#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

namespace rocprofiler_compute_tool::compression
{
namespace
{
// gzwrite length is unsigned but capped at INT_MAX; chunk larger writes.
constexpr std::size_t kMaxChunk = std::numeric_limits<int>::max();
}  // namespace

GzipFileOutputStream::GzipFileOutputStream(const std::string& path, int level)
    : m_path{path}
{
    const auto mode = "wb" + std::to_string(std::clamp(level, 0, 9));
    m_file          = gzopen(path.c_str(), mode.c_str());
    m_failed        = m_file == nullptr;
}

GzipFileOutputStream::~GzipFileOutputStream()
{
    if (m_file && !close())
        std::cerr << "Failed to close gzip stream: " << m_path << std::endl;
}

bool GzipFileOutputStream::write(std::string_view data)
{
    if (m_failed || !m_file)
    {
        m_failed = true;
        return false;
    }
    if (data.empty())
        return true;

    while (!data.empty())
    {
        const auto chunk = std::min(data.size(), kMaxChunk);

        if (gzwrite(m_file, data.data(), static_cast<unsigned>(chunk)) == 0)
        {
            m_failed = true;
            return false;
        }

        data.remove_prefix(chunk);
    }

    return true;
}

bool GzipFileOutputStream::close()
{
    if (!m_file)
        return !m_failed;

    m_failed |= gzclose(m_file) != Z_OK;
    m_file = nullptr;
    return !m_failed;
}

bool GzipFileOutputStream::ok() const
{
    return !m_failed;
}

bool GzipFileOutputStream::is_open() const
{
    return m_file != nullptr;
}

}  // namespace rocprofiler_compute_tool::compression
