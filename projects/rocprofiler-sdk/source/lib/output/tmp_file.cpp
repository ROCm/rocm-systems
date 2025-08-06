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

namespace fs = ::rocprofiler::common::filesystem;

bool
tmp_file::fopen(const char* _mode)
{
    auto fpath = fs::path{filename}.parent_path();
    if(!fs::exists(fpath)) fs::create_directories(fpath);

    if(!fs::exists(filename))
    {
        // if the filepath does not exist, open in out mode to create it
        std::ofstream _ofs{filename};
    }

    ROCP_INFO << "opening (via fopen) temporary file: '" << filename << "'...";
    file = std::fopen(filename.c_str(), _mode);
    if(file) fd = ::fileno(file);

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
    auto fpath = fs::path{filename}.parent_path();
    if(!fs::exists(fpath)) fs::create_directories(fpath);

    if(!fs::exists(filename))
    {
        // if the filepath does not exist, open in out mode to create it
        std::ofstream _ofs{};
        _ofs.open(filename, std::ofstream::binary | std::ofstream::out);
    }

    ROCP_INFO << "opening temporary file: '" << filename << "'...";
    stream.open(filename, _mode);
    return (stream.is_open() && stream.good());
}

bool
tmp_file::remove()
{
    close();
    if(fs::exists(filename))
    {
        ROCP_INFO << "removing temporary file: '" << filename << "'...";
        auto _ec  = std::error_code{};
        auto _ret = fs::remove(filename, _ec);

        if(_ec)
            ROCP_WARNING << fmt::format(
                "Error removing temporary file '{}' :: {}", filename, _ec.message());
        else if(!_ret)
            ROCP_WARNING << fmt::format("Error removing temporary file '{}' :: Unknown error",
                                        filename);

        return _ret;
    }

    return true;
}

bool
tmp_file::exists() const
{
    return fs::exists(filename);
}

tmp_file::operator bool() const
{
    return (stream.is_open() && stream.good()) || (file != nullptr && fd > 0);
}
