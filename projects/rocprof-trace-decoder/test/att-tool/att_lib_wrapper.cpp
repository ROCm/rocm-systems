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

#include "att_lib_wrapper.hpp"
#include "filenames.hpp"
#include "occupancy.hpp"
#include "profile_interface.hpp"
#include "wave.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>
#include <vector>

namespace rocprofiler
{
namespace att_wrapper
{
ATTFileMgr::ATTFileMgr(Fspath _dir, std::vector<std::string> _counters, std::shared_ptr<AddressTable>& codeobj_files)
: dir(std::move(_dir))
, table(codeobj_files)
{
    std::filesystem::create_directories(dir);
    codefile  = std::make_shared<CodeFile>(dir, table);
    filenames = std::make_shared<FilenameMgr>(dir);

    filenames->perfcounters = std::move(_counters);
}

ATTFileMgr::~ATTFileMgr() { OccupancyFile::OccupancyFile(dir, table, occupancy); }

void
ATTFileMgr::parseShader(int se_id, const std::vector<char>& data)
{
    WaveConfig config(se_id, filenames, codefile);
    ToolData   tooldata(data, config);

    if(!config.occupancy.empty()) occupancy.emplace(se_id, std::move(config.occupancy));

    for(auto& [pc, kernel] : config.kernel_names)
        codefile->kernel_names.emplace(pc, std::move(kernel));
}

int
get_shader_id(const std::string& name)
{
    auto run_pos = name.rfind('_');
    if(run_pos == std::string::npos) throw std::runtime_error("Invalid name");

    std::string stripped      = name.substr(0, run_pos);
    auto        se_number_pos = stripped.rfind('_');
    if(se_number_pos == std::string::npos || se_number_pos + 1 >= stripped.size())
        throw std::runtime_error("Invalid name");

    return std::stoi(std::string(stripped.substr(se_number_pos + 1)));
}

ATTDecoder::ATTDecoder() {}

void
ATTDecoder::parse(const Fspath&                       input_dir,
                  const Fspath&                       output_dir,
                  const std::vector<std::string>&     att_files,
                  std::shared_ptr<AddressTable>&      codeobj_files,
                  const std::vector<std::string>&     counters_names,
                  const std::string&                  /* output_formats */)
{
    ATTFileMgr mgr(output_dir, counters_names, codeobj_files);

    for(const auto& shader : att_files)
    {
        int shader_id = get_shader_id(shader);

        std::vector<char> shader_data;

        {
            std::ifstream file(input_dir / shader, std::ios::binary);
            if(!file.is_open())
            {
                WARNING("could not open " << shader);
                continue;
            }

            file.seekg(0, std::ios::end);
            shader_data.resize(file.tellg());
            file.seekg(0);
            file.read(shader_data.data(), shader_data.size());
        }

        mgr.parseShader(shader_id, shader_data);
    }
}

}  // namespace att_wrapper
}  // namespace rocprofiler
