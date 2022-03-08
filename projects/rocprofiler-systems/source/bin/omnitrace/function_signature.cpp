// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "function_signature.hpp"

function_signature::function_signature(string_t _ret, const string_t& _name,
                                       string_t _file, location_t _row, location_t _col,
                                       bool _loop, bool _info_beg, bool _info_end)
: m_loop(_loop)
, m_info_beg(_info_beg)
, m_info_end(_info_end)
, m_row(std::move(_row))
, m_col(std::move(_col))
, m_return(std::move(_ret))
, m_name(tim::demangle(_name))
, m_file(std::move(_file))
{
    if(m_file.find('/') != string_t::npos)
        m_file = m_file.substr(m_file.find_last_of('/') + 1);
}

function_signature::function_signature(const string_t& _ret, const string_t& _name,
                                       const string_t&              _file,
                                       const std::vector<string_t>& _params,
                                       location_t _row, location_t _col, bool _loop,
                                       bool _info_beg, bool _info_end)
: function_signature(_ret, _name, _file, _row, _col, _loop, _info_beg, _info_end)
{
    m_params = "(";
    for(const auto& itr : _params)
        m_params.append(itr + ", ");
    if(!_params.empty()) m_params = m_params.substr(0, m_params.length() - 2);
    m_params += ")";
}

string_t
function_signature::get(function_signature& sig)
{
    return sig.get();
}

string_t
function_signature::get() const
{
    std::stringstream ss;
    if(use_return_info && !m_return.empty()) ss << m_return << " ";
    ss << m_name;
    if(use_args_info) ss << m_params;
    if(m_loop && m_info_beg)
    {
        auto _row_col_str = [](unsigned long _row, unsigned long _col) {
            std::stringstream _ss{};
            if(_row == 0 && _col == 0) return std::string{};
            if(_col > 0)
                _ss << "{" << _row << "," << _col << "}";
            else
                _ss << "{" << _row << "}";
            return _ss.str();
        };

        auto _rc1 = _row_col_str(m_row.first, m_col.first);
        auto _rc2 = _row_col_str(m_row.second, m_col.second);
        if(m_info_end && !_rc1.empty() && !_rc2.empty() && _rc1 != _rc2)
            ss << " [" << _rc1 << "-" << _rc2 << "]";
        else if(m_info_end && !_rc1.empty() && !_rc2.empty() && _rc1 == _rc2)
            ss << " [" << _rc1 << "]";
        else if(m_info_end && !_rc1.empty() && _rc2.empty())
            ss << " [" << _rc1 << "]";
        else if(!m_info_end && !_rc1.empty())
            ss << " [" << _rc1 << "]";
        else
            errprintf(1, "loop line info is empty!");
    }
    if(use_file_info && m_file.length() > 0) ss << " [" << m_file;
    if(use_line_info && m_row.first > 0) ss << ":" << m_row.first;
    if(use_file_info && m_file.length() > 0) ss << "]";

    m_signature = ss.str();
    return m_signature;
}
