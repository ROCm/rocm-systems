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
#include "wstates.hpp"

#include "lib/rocprofiler-sdk/registration.hpp"

#include <dlfcn.h>
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
ATTFileMgr::ATTFileMgr(Fspath                                _dir,
                       std::vector<std::string>              _counters,
                       rocprofiler_thread_trace_decoder_id_t _decoder)
: dir(std::move(_dir))
, decoder(_decoder)
{
    rocprofiler::common::filesystem::create_directories(dir);
    table     = std::make_shared<AddressTable>();
    codefile  = std::make_shared<CodeFile>(dir, table);
    filenames = std::make_shared<FilenameMgr>(dir);

    for(size_t i = 0; i < ROCPROFILER_THREAD_TRACE_DECODER_WSTATE_LAST; i++)
        wstates.at(i) = std::make_shared<WstatesFile>(i, dir);
    filenames->perfcounters = std::move(_counters);
}

ATTFileMgr::~ATTFileMgr()
{
    for(auto id : codeobjs_to_delete)
    {
        auto status = rocprofiler_thread_trace_decoder_codeobj_unload(decoder, id);
        ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS) << "unable to delete codeobj " << id;
    }

    OccupancyFile::OccupancyFile(dir, table, occupancy);
}

void
ATTFileMgr::addDecoder(const char* filepath, uint64_t id, uint64_t load_addr, uint64_t memsize)
{
    if(filepath == nullptr) return;

    std::vector<char> buffer{};

    {
        std::ifstream file(filepath, std::ios::in | std::ios::binary);

        if(!file.is_open()) throw std::runtime_error("Invalid file " + std::string(filepath));

        file.seekg(0, file.end);
        buffer.resize(file.tellg());
        file.seekg(0, file.beg);
        file.read(buffer.data(), buffer.size());
    }

    auto status = rocprofiler_thread_trace_decoder_codeobj_load(
        decoder, id, load_addr, memsize, buffer.data(), buffer.size());
    ROCP_ERROR_IF(status != ROCPROFILER_STATUS_SUCCESS) << "Unable to load codeobj: " << filepath;

    codeobjs_to_delete.push_back(id);
    table->addDecoder(buffer.data(), buffer.size(), id, load_addr, memsize);
}

void
ATTFileMgr::parseShader(int se_id, std::vector<char>& data)
{
    WaveConfig config(se_id, filenames, codefile, wstates);
    ToolData   tooldata(data, config, decoder);

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

ATTDecoder::ATTDecoder(const std::string& path)
{
    auto status = rocprofiler_thread_trace_decoder_create(&decoder, path.c_str());
    ROCP_FATAL_IF(status != ROCPROFILER_STATUS_SUCCESS) << "Error loading decoder: " << status;
};

ATTDecoder::~ATTDecoder() { rocprofiler_thread_trace_decoder_destroy(decoder); }

void
ATTDecoder::parse(const Fspath&                       input_dir,
                  const Fspath&                       output_dir,
                  const std::vector<std::string>&     att_files,
                  const std::vector<CodeobjLoadInfo>& codeobj_files,
                  const std::vector<std::string>&     counters_names,
                  const std::string&                  output_formats)
{
    auto& formats = GlobalDefs::get().output_formats;
    formats       = output_formats;
    std::transform(formats.begin(), formats.end(), formats.begin(), [](unsigned char c) {
        return std::tolower(c);
    });

    ATTFileMgr mgr(output_dir, counters_names, decoder);

    for(const auto& file : codeobj_files)
    {
        if(file.id == 0 && file.name.empty()) continue;

        if(file.name.find("memory://") == 0)
        {
            ROCP_WARNING << file.name << " was not loaded";
            continue;
        }

        try
        {
            mgr.addDecoder((input_dir / file.name).c_str(), file.id, file.addr, file.size);
        } catch(std::exception& e)
        {
            ROCP_ERROR << file.id << ':' << file.name << " - " << e.what();
        }
    }

    for(const auto& shader : att_files)
    {
        int shader_id = 0;
        try
        {
            shader_id = get_shader_id(shader);
        } catch(std::exception& e)
        {
            ROCP_WARNING << "Could not retrieve shader_id: " << e.what();
            continue;
        }

        std::vector<char> shader_data;

        {
            std::ifstream file(input_dir / shader, std::ios::binary);
            if(!file.is_open())
            {
                ROCP_WARNING << "could not open " << shader;
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

bool
ATTDecoder::valid() const
{
    return decoder.handle != 0;
}

}  // namespace att_wrapper
}  // namespace rocprofiler
