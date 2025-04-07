// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "core/defines.hpp"

#include <array>
#include <iostream>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>

namespace rocprofsys
{
inline namespace config
{
bool
get_debug() ROCPROFSYS_HOT;

int
get_verbose() ROCPROFSYS_HOT;
}  // namespace config

struct redirect
{
    redirect(std::ostream& _os, std::string _expected)
    : m_os{ _os }
    , m_expected{ std::move(_expected) }
    {
        if(!get_debug())
        {
            // save stream buffer
            m_strm_buffer = m_os.rdbuf();
            // redirect to stringstream
            _os.rdbuf(m_buffer.rdbuf());
        }
    }

    ~redirect()
    {
        if(!m_strm_buffer) return;
        // restore stream buffer
        m_os.rdbuf(m_strm_buffer);
        auto _v      = m_buffer.str();
        _v           = replace<3>(m_buffer.str(), { '\n', '\t', ' ' });
        auto _expect = replace<3>(m_expected, { '\n', '\t', ' ' });
        if(_v != _expect)
        {
            if(get_verbose() > 0)
                std::cerr
                    << "[rocprofsys::redirect] Expected:\n[rocprofsys::redirect]    "
                    << _expect
                    << "\n[rocprofsys::redirect] Found:\n[rocprofsys::redirect]    " << _v
                    << "\n";
            if(get_verbose() <= 0 || (&m_os != &std::cerr && &m_os != &std::cout))
                m_os << m_buffer.str() << std::flush;
        }
    }

private:
    template <size_t N>
    static std::string replace(std::string _v, const std::array<char, N>& _c,
                               const std::string& _s = "")
    {
        for(const auto& itr : _c)
        {
            while(true)
            {
                auto _pos = _v.find(itr);
                if(_pos == std::string::npos) break;
                _v = _v.replace(_pos, 1, _s);
            }
        }
        return _v;
    }

    std::ostream&     m_os;
    std::string       m_expected = {};
    std::stringstream m_buffer{};
    std::streambuf*   m_strm_buffer = nullptr;
};
}  // namespace rocprofsys
