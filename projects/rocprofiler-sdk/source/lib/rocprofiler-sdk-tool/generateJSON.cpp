// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "generateJSON.hpp"
#include "config.hpp"
#include "helper.hpp"
#include "output_file.hpp"
#include "statistics.hpp"

#include "lib/common/string_entry.hpp"
#include "lib/common/utility.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/marker/api_id.h>

#include <utility>

namespace rocprofiler
{
namespace tool
{
void
write_json(tool_table*                                                      tool_functions,
           uint64_t                                                         pid,
           const domain_stats_vec_t&                                        domain_stats,
           std::vector<rocprofiler_agent_v0_t>                              agent_data,
           std::vector<rocprofiler_tool_counter_info_t>                     counter_data,
           std::deque<rocprofiler_buffer_tracing_hip_api_record_t>*         hip_api_deque,
           std::deque<rocprofiler_buffer_tracing_hsa_api_record_t>*         hsa_api_deque,
           std::deque<rocprofiler_buffer_tracing_kernel_dispatch_record_t>* kernel_dispatch_deque,
           std::deque<rocprofiler_buffer_tracing_memory_copy_record_t>*     memory_copy_deque,
           std::deque<rocprofiler_tool_counter_collection_record_t>*       counter_collection_deque,
           std::deque<rocprofiler_buffer_tracing_marker_api_record_t>*     marker_api_deque,
           std::deque<rocprofiler_buffer_tracing_scratch_memory_record_t>* scratch_memory_deque,
           std::deque<rocprofiler_buffer_tracing_rccl_api_record_t>*       rccl_api_deque)

{
    using JSONOutputArchive = cereal::MinimalJSONOutputArchive;

    constexpr auto json_prec   = 32;
    constexpr auto json_indent = JSONOutputArchive::Options::IndentChar::space;
    auto           json_opts   = JSONOutputArchive::Options{json_prec, json_indent, 1};
    auto           filename    = std::string_view{"results"};
    auto           ofs         = get_output_stream(filename, ".json");

    {
        auto json_ar = JSONOutputArchive{*ofs.stream, json_opts};
        json_ar.setNextName("rocprofiler-sdk-tool");
        json_ar.startNode();

        json_ar.makeArray();
        json_ar.startNode();

        // metadata
        {
            json_ar.setNextName("metadata");
            json_ar.startNode();
            auto* timestamps = tool_functions->tool_get_app_timestamps_fn();
            json_ar(cereal::make_nvp("pid", pid));
            json_ar(cereal::make_nvp("init_time", timestamps->app_start_time));
            json_ar(cereal::make_nvp("fini_time", timestamps->app_end_time));
            json_ar(cereal::make_nvp("config", get_config()));
            json_ar(cereal::make_nvp("command", common::read_command_line(getpid())));
            json_ar.finishNode();
        }

        // summary
        {
            json_ar.setNextName("summary");
            json_ar.startNode();
            json_ar.makeArray();

            for(const auto& itr : domain_stats)
            {
                auto _name = get_domain_column_name(itr.first);
                json_ar.startNode();

                json_ar(cereal::make_nvp("domain", std::string{_name}));
                json_ar(cereal::make_nvp("stats", itr.second));
                // itr.second.serialize(json_ar, 0);

                json_ar.finishNode();
            }

            json_ar.finishNode();
        }

        json_ar(cereal::make_nvp("agents", agent_data));
        json_ar(cereal::make_nvp("counters", counter_data));

        {
            auto callback_name_info = get_callback_id_names();
            auto buffer_name_info   = get_buffer_id_names();
            auto counter_dims       = get_tool_counter_dimension_info();
            auto marker_msg_data    = get_callback_roctx_msg();

            json_ar.setNextName("strings");
            json_ar.startNode();
            json_ar(cereal::make_nvp("callback_records", callback_name_info));
            json_ar(cereal::make_nvp("buffer_records", buffer_name_info));
            json_ar(cereal::make_nvp("marker_api", marker_msg_data));

            {
                auto _extern_corr_id_strings = std::map<size_t, std::string>{};
                if(tool::get_config().kernel_rename)
                {
                    for(auto itr : *kernel_dispatch_deque)
                    {
                        auto _value = itr.correlation_id.external.value;
                        if(_value > 0)
                        {
                            const auto* _str = common::get_string_entry(_value);
                            if(_str) _extern_corr_id_strings.emplace(_value, *_str);
                        }
                    }
                }

                json_ar.setNextName("correlation_id");
                json_ar.startNode();
                json_ar(cereal::make_nvp("external", _extern_corr_id_strings));
                json_ar.finishNode();
            }

            {
                json_ar.setNextName("counters");
                json_ar.startNode();
                json_ar(cereal::make_nvp("dimension_ids", counter_dims));
                json_ar.finishNode();
            }

            json_ar.finishNode();
        }

        {
            auto kern_sym_data = get_kernel_symbol_data();
            auto code_obj_data = get_code_object_data();

            json_ar(cereal::make_nvp("code_objects", code_obj_data));
            json_ar(cereal::make_nvp("kernel_symbols", kern_sym_data));
        }

        {
            json_ar.setNextName("callback_records");
            json_ar.startNode();
            json_ar(cereal::make_nvp("counter_collection", *counter_collection_deque));
            json_ar.finishNode();
        }

        {
            json_ar.setNextName("buffer_records");
            json_ar.startNode();
            json_ar(cereal::make_nvp("kernel_dispatch", *kernel_dispatch_deque));
            json_ar(cereal::make_nvp("hip_api", *hip_api_deque));
            json_ar(cereal::make_nvp("hsa_api", *hsa_api_deque));
            json_ar(cereal::make_nvp("marker_api", *marker_api_deque));
            json_ar(cereal::make_nvp("rccl_api", *rccl_api_deque));
            json_ar(cereal::make_nvp("memory_copy", *memory_copy_deque));
            json_ar(cereal::make_nvp("scratch_memory", *scratch_memory_deque));
            json_ar.finishNode();
        }

        json_ar.finishNode();  // end array
        json_ar.finishNode();
    }

    ofs.close();
}

}  // namespace tool
}  // namespace rocprofiler
