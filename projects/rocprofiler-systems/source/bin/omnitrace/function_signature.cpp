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

function_signature::function_signature(std::string_view _ret, std::string_view _name,
                                       std::string_view _file, location_t _row,
                                       location_t _col, bool _loop, bool _info_beg,
                                       bool _info_end)
: m_loop(_loop)
, m_info_beg(_info_beg)
, m_info_end(_info_end)
, m_row(std::move(_row))
, m_col(std::move(_col))
, m_return(_ret)
, m_name(tim::demangle(_name.data()))
, m_file(_file)
{
    if(m_file.find('/') != std::string_view::npos)
        m_file = m_file.substr(m_file.find_last_of('/') + 1);
}

function_signature::function_signature(std::string_view _ret, std::string_view _name,
                                       std::string_view                _file,
                                       const std::vector<std::string>& _params,
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

std::string
function_signature::get(function_signature& sig)
{
    return sig.get();
}

std::string
function_signature::get(bool _all, bool _save) const
{
    if(!_all && _save && !m_signature.empty()) return m_signature;

    std::stringstream ss;
    if((_all || use_return_info) && !m_return.empty()) ss << m_return << " ";
    ss << m_name;
    if(_all || use_args_info) ss << m_params;
    if(m_loop)
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
        else if(m_loop_num < std::numeric_limits<uint32_t>::max())
            ss << " [loop#" << m_loop_num << "]";
        else
            errprintf(3, "line info for %s is empty! [{%s}] [{%s}]\n", m_name.c_str(),
                      _rc1.c_str(), _rc2.c_str());
    }
    if((_all || use_file_info) && m_file.length() > 0) ss << " [" << m_file;
    if((_all || use_line_info) && m_row.first > 0) ss << ":" << m_row.first;
    if((_all || use_file_info) && m_file.length() > 0) ss << "]";

    if(_save) m_signature = ss.str();
    return ss.str();
}

std::string
function_signature::get_coverage(bool _basic_block) const
{
    std::stringstream ss;
    if(!m_return.empty()) ss << m_return << " ";
    ss << m_name << m_params;
    if(_basic_block && m_loop && m_info_beg)
    {
        if(m_file.length() > 0) ss << " [" << m_file << "]";
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
            errprintf(3, "line info for %s is empty!\n", m_name.c_str());
    }
    else
    {
        if(m_file.length() > 0) ss << " [" << m_file;
        if(m_row.first > 0) ss << ":" << m_row.first;
        if(m_file.length() > 0) ss << "]";
    }

    return ss.str();
}
