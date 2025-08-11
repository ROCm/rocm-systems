// MIT License
//
// Copyright (c) 2024-2025 ROCm Developer Tools
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

#include "client.hpp"

#include "address_translation.hpp"
#include "cid_retirement.hpp"
#include "codeobj.hpp"
#include "external_cid.hpp"
#include "kernel_tracing.hpp"
#include "pcs.hpp"
#include "utils.hpp"

#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/internal_threading.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

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

    address_translation::init();
    external_cid::init();
    pcs::init();

    ROCPROFILER_CALL(rocprofiler_create_context(&client_ctx), "Cannot create context\n");

    pcs::configure_pc_sampling_on_all_agents(client_ctx);

    // Enable code object tracing service, to match PC samples to corresponding code object
    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(client_ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       client::codeobj::codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

    cid_retirement::configure_cid_retirement_tracing(client_ctx);
    // Kernel tracing service need for external correlation service.
    kernel_tracing::configure_kernel_tracing_service(client_ctx);
    external_cid::configure_external_correlation_service(client_ctx);

    int valid_ctx = 0;
    ROCPROFILER_CALL(rocprofiler_context_is_valid(client_ctx, &valid_ctx),
                     "failure checking context validity");
    if(valid_ctx == 0)
    {
        // notify rocprofiler that initialization failed
        // and all the contexts, buffers, etc. created
        // should be ignored
        return -1;
    }

    ROCPROFILER_CALL(rocprofiler_start_context(client_ctx), "rocprofiler context start failed");

    return 0;
}

void
tool_fini(void* /*tool_data*/)
{
    // Drain all retired correlation IDs
    client::sync();

    if(client_id)
    {
        // Assert the context is inactive.
        int state = -1;
        ROCPROFILER_CALL(rocprofiler_context_is_active(client_ctx, &state),
                         "Cannot inspect the stat of the context.")
        assert(state == 0);

        // No need to stop the context, since it has been stopped implicitly by the rocprofiler-SDK.

        // Flush remaining PC samples
        pcs::flush_and_destroy_buffers();
    }

    address_translation::dump_flat_profile();
    // deallocation
    address_translation::fini();
    external_cid::fini();
    pcs::fini();
}

}  // namespace

// forward declaration
void
setup();

void
setup()
{
    // Do not force configuration
    if(int status = 0;
       rocprofiler_is_initialized(&status) == ROCPROFILER_STATUS_SUCCESS && status == 0)
    {
        *utils::get_output_stream() << "Client forces rocprofiler configuration.\n" << std::endl;
        ROCPROFILER_CALL(rocprofiler_force_configure(&rocprofiler_configure),
                         "failed to force configuration");
    }
}

void
shutdown()
{}

void
sync()
{
    // Flush rocprofiler-SDK's buffers containing PC samples.
    pcs::flush_buffers();

    // Flush retired correlation IDs.
    cid_retirement::flush_retired_cids();
}

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
    std::string   filename      = "pc_sampling_integration_test.log";
    if(auto* outfile = getenv("ROCPROFILER_SAMPLE_OUTPUT_FILE"); outfile) filename = outfile;
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
