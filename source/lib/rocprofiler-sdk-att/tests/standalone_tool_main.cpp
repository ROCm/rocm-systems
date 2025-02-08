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

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk-att/att_lib_wrapper.hpp"
#include "lib/rocprofiler-sdk-att/outputfile.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <iostream>
#include <mutex>
#include <nlohmann/json.hpp>

using rocprofiler::att_wrapper::ATTDecoder;
using Fspath = rocprofiler::att_wrapper::Fspath;

int
main(int argc, char** argv)
{
    if(argc < 2 || std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h")
    {
        std::cout
            << "Usage: att-parser-tool json_filepath [output_dir] [output_formats]\nParameters:\n"
            << "\tjson_filepath: Path of rocprofv3's generated results.json\n"
            << "\toutput_dir: Optional output directory. Default: json_filepath parent dir\n"
            << "\toutput_formats: json, perfetto, csv. Default: all\n"
            << std::endl;
        exit(0);
    }

    static auto flag = std::once_flag{};
    std::call_once(flag, []() {
        auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
        rocprofiler::common::init_logging("ROCPROF", logging_cfg);
        FLAGS_colorlogtostderr = true;
    });

    auto cap = rocprofiler::att_wrapper::tool_att_capability_t::ATT_CAPABILITIES_SUMMARY;
    {
        auto query = rocprofiler::att_wrapper::query_att_decode_capability();
        ROCP_FATAL_IF(query.empty()) << "No decoder capability available!";

        for(auto& avail_cap : query)
            cap = std::max(cap, avail_cap);
    }

    ATTDecoder decoder(cap);
    ROCP_FATAL_IF(!decoder.valid()) << "Failed to initialize decoder library!";

    auto get_run_number = [](const std::string& path) -> int {
        auto get_filename = [](const std::string& _path) -> std::string {
            return Fspath(_path).filename().c_str();
        };

        auto name          = get_filename(path);
        auto run_pos       = name.rfind('_');
        auto extension_pos = name.rfind(".att");

        if(extension_pos == std::string::npos || extension_pos <= run_pos) throw std::exception();

        return std::stoi(name.substr(run_pos + 1, extension_pos - run_pos));
    };

    Fspath input_path = rocprofiler::common::filesystem::absolute(argv[1]);

    std::string formats = "json,csv";
    if(argc >= 4) formats = argv[3];

    Fspath output_path = input_path.parent_path();
    if(argc >= 3) output_path = argv[2];

    nlohmann::json sdk_json;
    {
        nlohmann::json full_json;
        std::ifstream  ifile(input_path);
        ROCP_FATAL_IF(!ifile.is_open()) << "Failed to open json file!";
        ifile >> full_json;
        sdk_json = full_json["rocprofiler-sdk-tool"][0];
    }

    std::unordered_map<int, std::vector<std::string>> all_runs;

    for(auto& file : sdk_json["strings"]["att_files"])
    {
        try
        {
            int n = get_run_number(file);
            if(all_runs.find(n) == all_runs.end()) all_runs[n] = {};

            all_runs[n].push_back(file);

        } catch(std::exception&)
        {
            ROCP_WARNING << "Invalid ATT filename " << file;
        }
    }

    for(auto& [run_number, att_filenames] : all_runs)
    {
        std::vector<Fspath>                                    att_files{};
        std::vector<rocprofiler::att_wrapper::CodeobjLoadInfo> codeobj_files{};

        std::map<size_t, std::string> snapshot_files{};
        for(auto elem : sdk_json["strings"]["code_object_snapshot_files"])
            snapshot_files[elem["key"]] = elem["value"];

        for(auto& codeobj : sdk_json["code_objects"])
            if(!std::string{codeobj["uri"]}.empty())
            {
                std::string filename = codeobj["uri"];
                size_t      id       = size_t(codeobj["code_object_id"]);
                if(filename.empty()) continue;

                try
                {
                    filename = snapshot_files.at(id);
                } catch(...)
                {
                    ROCP_WARNING << "codeobject id " << id << " not found " << filename;
                }

                try
                {
                    codeobj_files.push_back({filename,
                                             id,
                                             size_t(codeobj["load_delta"]),
                                             size_t(codeobj["load_size"])});
                } catch(std::exception& e)
                {
                    ROCP_WARNING << "Could not load " << filename << ": " << e.what();
                } catch(std::string& r)
                {
                    ROCP_WARNING << "Could not load " << filename << ": " << r;
                } catch(...)
                {
                    ROCP_WARNING << "Could not load " << filename;
                }
            }

        std::string run_name = input_path.filename().c_str();
        std::string ui_name  = run_name.substr(0, run_name.find(".json"));
        auto output_dir      = output_path / ("ui_output_" + ui_name + std::to_string(run_number));
        decoder.parse(input_path.parent_path(), output_dir, att_filenames, codeobj_files, formats);
    }

    ROCP_INFO << "Finalizing ATT Tool";

    return 0;
}
