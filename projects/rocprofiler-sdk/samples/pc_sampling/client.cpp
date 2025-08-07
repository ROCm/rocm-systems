// MIT License
//
// Copyright (c) 2023-2025 ROCm Developer Tools
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

// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

/**
 * @file samples/pc_sampling_library/client.cpp
 *
 * @brief Example rocprofiler client (tool)
 */

#include "pcs.hpp"
#include "utils.hpp"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include "common/defines.hpp"
#include "common/filesystem.hpp"
#include "common/secure_filename.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

namespace client
{
namespace
{
rocprofiler_client_id_t*      client_id        = nullptr;
rocprofiler_client_finalize_t client_fini_func = nullptr;
rocprofiler_context_id_t      client_ctx{0};

int
tool_init(rocprofiler_client_finalize_t fini_func, void* /*tool_data*/)
{
    client_fini_func = fini_func;

    // Initialize necessary data structures
    pcs::init();

    client::pcs::find_all_gpu_agents_supporting_pc_sampling();

    if(client::pcs::gpu_agents.empty())
    {
        *utils::get_output_stream() << "No availabe gpu agents supporting PC sampling" << std::endl;
        // Emit the message to explicitly skip the sample.
        std::cerr << "PC sampling unavailable" << std::endl;
        // Exit with no error if none of the GPUs support PC sampling.
        exit(0);
    }

    // The relations assumed:
    // - One context for all gpu agents
    // - a buffer per agent
    // - a callback thread per buffer
    // - a pc sampling service per agent/buffer

    ROCPROFILER_CHECK(rocprofiler_create_context(&client_ctx));

    auto* buff_ids_vec = pcs::get_pc_sampling_buffer_ids();

    for(auto& gpu_agent : pcs::gpu_agents)
    {
        // creating a buffer that will hold pc sampling information
        rocprofiler_buffer_policy_t drop_buffer_action = ROCPROFILER_BUFFER_POLICY_LOSSLESS;
        auto                        buffer_id          = rocprofiler_buffer_id_t{};
        ROCPROFILER_CHECK(rocprofiler_create_buffer(client_ctx,
                                                    client::pcs::BUFFER_SIZE_BYTES,
                                                    client::pcs::WATERMARK,
                                                    drop_buffer_action,
                                                    client::pcs::rocprofiler_pc_sampling_callback,
                                                    nullptr,
                                                    &buffer_id));

        client::pcs::configure_pc_sampling_prefer_stochastic(
            gpu_agent.get(), client_ctx, buffer_id);

        // One helper thread per GPU agent's buffer.
        auto client_agent_thread = rocprofiler_callback_thread_t{};
        ROCPROFILER_CHECK(rocprofiler_create_callback_thread(&client_agent_thread));

        ROCPROFILER_CHECK(rocprofiler_assign_callback_thread(buffer_id, client_agent_thread));

        buff_ids_vec->emplace_back(buffer_id);
    }

    int valid_ctx = 0;
    ROCPROFILER_CHECK(rocprofiler_context_is_valid(client_ctx, &valid_ctx));
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    // Start PC sampling
    ROCPROFILER_CHECK(rocprofiler_start_context(client_ctx));

    return 0;
}

void
tool_fini(void* /*tool_data*/)
{
    if(client_id)
    {
        // Assert the context is inactive.
        int state = -1;
        ROCPROFILER_CHECK(rocprofiler_context_is_active(client_ctx, &state))
        assert(state == 0);

        // No need to stop the context, since it has been stopped implicitly by the rocprofiler-SDK.
        for(auto buff_id : *pcs::get_pc_sampling_buffer_ids())
        {
            // Flush the buffer explicitly
            ROCPROFILER_CHECK(rocprofiler_flush_buffer(buff_id));
            // Destroying the buffer
            rocprofiler_status_t status = rocprofiler_destroy_buffer(buff_id);
            if(status == ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
            {
                *utils::get_output_stream()
                    << "The buffer is busy, so we cannot destroy it at the moment." << std::endl;
            }
            else
            {
                ROCPROFILER_CHECK(status);
            }
        }
    }

    // deallocation
    pcs::fini();
}

}  // namespace

// forward declaration
void
setup();

void
setup()
{
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        ROCPROFILER_CHECK(rocprofiler_force_configure(&rocprofiler_configure));
    }
}

void
shutdown()
{}

}  // namespace client

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "PCSamplingExampleTool";

    // store client info
    client::client_id = id;

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " is using rocprofiler v" << major << "." << minor << "." << patch << " ("
         << runtime_version << ")";

    std::clog << info.str() << std::endl;

    std::ostream* output_stream = nullptr;
    std::string   filename      = common::secure::get_safe_output_filename("pc_sampling.log");
    if(filename == "stdout")
        output_stream = &std::cout;
    else if(filename == "stderr")
        output_stream = &std::cerr;
    else
        output_stream = new std::ofstream{filename};

    client::utils::get_output_stream() = output_stream;

    // create configure data
    static auto cfg =
        rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                            &client::tool_init,
                                            &client::tool_fini,
                                            static_cast<void*>(output_stream)};

    // return pointer to configure data
    return &cfg;
}
