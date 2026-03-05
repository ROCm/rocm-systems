// MIT License
//
// Copyright (c) 2024-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "util.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>
#include <filesystem>

namespace rocprofiler
{
namespace att_wrapper
{
using Fspath = std::filesystem::path;

struct CodeobjLoadInfo
{
    std::string name{};
    size_t      id{0};
    size_t      addr{0};
    size_t      size{0};
};

class ATTDecoder
{
public:
    ATTDecoder();

    using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

    /**
     * Parse a list of att files
     * @param[in] input directory where att_files and codeobj_files are relative to
     * @param[in] output_dir location where ui_ files are generated
     * @param[in] att_files list of ATT files, ideally from the same kernel launch
     * @param[in] codeobj_files list of code object information loaded at the time of the trace
     * @param[in] output_formats List of comma-separated output formats, e.g. "json,csv"
     */
    void parse(const Fspath&                       input_dir,
               const Fspath&                       output_dir,
               const std::vector<std::string>&     att_files,
               std::shared_ptr<AddressTable>&      codeobj_files,
               const std::vector<std::string>&     counters_names,
               const std::string&                  output_formats);
};

class ATTFileMgr
{
    using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

public:
    ATTFileMgr(Fspath _dir, std::vector<std::string> _counters, std::shared_ptr<AddressTable>& codeobj_files);
    ~ATTFileMgr();

    void parseShader(int se_id, const std::vector<char>& data);

    Fspath dir{};

    std::shared_ptr<class CodeFile>    codefile{nullptr};
    std::shared_ptr<class FilenameMgr> filenames{nullptr};
    std::shared_ptr<AddressTable>      table{nullptr};

    std::map<size_t, std::vector<occupancy_t>> occupancy;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
