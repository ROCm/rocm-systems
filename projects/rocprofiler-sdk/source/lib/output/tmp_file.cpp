// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "tmp_file.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <cerrno>
#include <cstring>

namespace fs = ::rocprofiler::common::filesystem;

namespace
{
bool
ensure_parent_directory(const std::string& filename)
{
    auto fpath = fs::path{filename}.parent_path();
    if(fpath.empty()) return true;

    auto ec = std::error_code{};
    if(fs::exists(fpath, ec)) return true;

    if(ec)
    {
        ROCP_ERROR << "failed to stat temporary file directory '" << fpath.string() << "' for '"
                   << filename << "' :: " << ec.message();
        return false;
    }

    fs::create_directories(fpath, ec);
    if(ec)
    {
        ROCP_ERROR << "failed to create temporary file directory '" << fpath.string() << "' for '"
                   << filename << "' :: " << ec.message();
        return false;
    }

    return true;
}

bool
ensure_file_exists(const std::string& filename, std::ios::openmode mode = std::ofstream::out)
{
    auto ec = std::error_code{};
    if(fs::exists(filename, ec)) return true;

    if(ec)
    {
        ROCP_ERROR << "failed to stat temporary file '" << filename << "' :: " << ec.message();
        return false;
    }

    auto ofs = std::ofstream{};
    ofs.open(filename, mode);
    if(!ofs)
    {
        ROCP_ERROR << "failed to create temporary file: '" << filename << "'";
        return false;
    }

    return true;
}
}  // namespace

bool
tmp_file::fopen(const char* _mode)
{
    if(!ensure_parent_directory(filename)) return false;

    // if the filepath does not exist, open in out mode to create it
    if(!ensure_file_exists(filename)) return false;

    ROCP_INFO << "opening (via fopen) temporary file: '" << filename << "'...";
    file = std::fopen(filename.c_str(), _mode);
    if(file)
        fd = ::fileno(file);
    else
        ROCP_ERROR << "failed to open temporary file via fopen: '" << filename
                   << "' :: " << std::strerror(errno);

    return (file != nullptr && fd > 0);
}

tmp_file::tmp_file(std::string _filename)
: filename(std::move(_filename))
{}

tmp_file::~tmp_file()
{
    close();
    remove();
}

bool
tmp_file::flush()
{
    if(stream.is_open())
    {
        ROCP_INFO << "flushing temporary file: '" << filename << "'...";
        stream.flush();
    }
    else if(file != nullptr)
    {
        ROCP_INFO << "flushing temporary file: '" << filename << "'...";
        int _ret = fflush(file);
        int _cnt = 0;
        while(_ret == EAGAIN || _ret == EINTR)
        {
            // std::this_thread::sleep_for(std::chrono::milliseconds{ 100 });
            _ret = fflush(file);
            if(++_cnt > 10) break;
        }
        return (_ret == 0);
    }

    return true;
}

bool
tmp_file::close()
{
    flush();

    if(stream.is_open())
    {
        ROCP_INFO << "closing temporary file: '" << filename << "'...";
        stream.close();
        return !stream.is_open();
    }
    else if(file != nullptr)
    {
        ROCP_INFO << "closing temporary file: '" << filename << "'...";
        auto _ret = fclose(file);
        if(_ret == 0)
        {
            file = nullptr;
            fd   = -1;
        }
        return (_ret == 0);
    }

    return true;
}

bool
tmp_file::open(std::ios::openmode _mode)
{
    if(!ensure_parent_directory(filename)) return false;

    // if the filepath does not exist, open in out mode to create it
    if(!ensure_file_exists(filename, std::ofstream::binary | std::ofstream::out)) return false;

    if(stream.is_open() && stream.good())
    {
        ROCP_TRACE << "temporary file: '" << filename << "' is already open...";
        return true;
    }

    ROCP_INFO << "opening temporary file: '" << filename << "'...";
    stream.open(filename, _mode);
    if(!stream.is_open() || !stream.good())
        ROCP_ERROR << "failed to open temporary file: '" << filename << "'";
    return (stream.is_open() && stream.good());
}

bool
tmp_file::remove()
{
    close();
    auto ec = std::error_code{};
    if(fs::exists(filename, ec))
    {
        ROCP_INFO << "removing temporary file: '" << filename << "'...";
        auto _ret = fs::remove(filename, ec);

        if(ec)
            ROCP_WARNING << fmt::format(
                "Error removing temporary file '{}' :: {}", filename, ec.message());
        else if(!_ret)
            ROCP_WARNING << fmt::format("Error removing temporary file '{}' :: Unknown error",
                                        filename);

        return _ret;
    }
    else if(ec)
    {
        ROCP_WARNING << fmt::format(
            "Error checking temporary file '{}' for removal :: {}", filename, ec.message());
        return false;
    }

    return true;
}

bool
tmp_file::exists() const
{
    auto ec  = std::error_code{};
    auto ret = fs::exists(filename, ec);
    if(ec)
    {
        ROCP_WARNING << fmt::format(
            "Error checking temporary file '{}' :: {}", filename, ec.message());
        return false;
    }
    return ret;
}

tmp_file::operator bool() const
{
    return (stream.is_open() && stream.good()) || (file != nullptr && fd > 0);
}
