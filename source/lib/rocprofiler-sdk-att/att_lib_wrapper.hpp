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

#include "lib/common/filesystem.hpp"
#include "lib/rocprofiler-sdk-att/util.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <vector>

namespace rocprofiler
{
namespace att_wrapper
{
using Fspath = rocprofiler::common::filesystem::path;

struct CodeobjLoadInfo
{
    std::string name{};
    size_t      id{0};
    size_t      addr{0};
    size_t      size{0};
};

enum tool_att_capability_t
{
    ATT_CAPABILITIES_TESTING = 0,  // used for code coverage testing
    ATT_CAPABILITIES_SUMMARY,      // used for CSV output only
    ATT_CAPABILITIES_TRACE,        // used for all outputs
    ATT_CAPABILITIES_DEBUG,
    ATT_CAPABILITIES_LAST = ATT_CAPABILITIES_DEBUG,
};

/**
 * Query decoder library capability. Returns list of supported capabilities.
 */
std::vector<tool_att_capability_t>
query_att_decode_capability();

class ATTDecoder
{
public:
    ATTDecoder(tool_att_capability_t cap);

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
               const std::vector<CodeobjLoadInfo>& codeobj_files,
               const std::string&                  output_formats);

    bool valid() const;

protected:
    std::shared_ptr<class DL> dl{nullptr};
};

class ATTFileMgr
{
    using AddressTable = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

public:
    ATTFileMgr(Fspath _dir, std::shared_ptr<DL> _dl);
    ~ATTFileMgr();

    void parseShader(int se_id, const std::vector<char>& data);

    Fspath dir{};

    std::shared_ptr<class DL>          dl{nullptr};
    std::shared_ptr<class CodeFile>    codefile{nullptr};
    std::shared_ptr<class FilenameMgr> filenames{nullptr};
    std::shared_ptr<AddressTable>      table{nullptr};

    std::map<size_t, std::vector<att_occupancy_info_v2_t>>              occupancy;
    std::array<std::shared_ptr<class WstatesFile>, ATT_WAVE_STATE_LAST> wstates;
};

}  // namespace att_wrapper
}  // namespace rocprofiler
