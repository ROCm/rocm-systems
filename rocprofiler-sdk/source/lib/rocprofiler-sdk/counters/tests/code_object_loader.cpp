// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/counters/tests/code_object_loader.hpp"
#include <fcntl.h>

#include "lib/common/logging.hpp"

namespace rocprofiler
{
namespace counters
{
namespace testing
{
hsa_status_t
load_code_object(const std::string& filename, hsa_agent_t agent, CodeObject& code_object)
{
    hsa_status_t err;
    code_object.file = open(filename.c_str(), O_RDONLY);
    if(code_object.file == -1)
    {
        ROCP_FATAL << "Could not load code object " << filename;
    }

    err = hsa_code_object_reader_create_from_file(code_object.file, &code_object.code_obj_rdr);
    if(err != HSA_STATUS_SUCCESS) return err;

    err = hsa_executable_create_alt(HSA_PROFILE_FULL,
                                    HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    nullptr,
                                    &code_object.executable);
    if(err != HSA_STATUS_SUCCESS) return err;
    err = hsa_executable_load_agent_code_object(
        code_object.executable, agent, code_object.code_obj_rdr, nullptr, nullptr);
    if(err != HSA_STATUS_SUCCESS) return err;

    err = hsa_executable_freeze(code_object.executable, nullptr);

    return err;
}

hsa_status_t
get_kernel(const CodeObject&  code_object,
           const std::string& kernel,
           hsa_agent_t        agent,
           Kernel&            kern)
{
    hsa_executable_symbol_t symbol;
    hsa_status_t            err =
        hsa_executable_get_symbol_by_name(code_object.executable, kernel.c_str(), &agent, &symbol);
    if(err != HSA_STATUS_SUCCESS)
    {
        err = hsa_executable_get_symbol_by_name(
            code_object.executable, (kernel + ".kd").c_str(), &agent, &symbol);
        if(err != HSA_STATUS_SUCCESS)
        {
            return err;
        }
    }
    ROCP_INFO << "kernel-name: " << kernel.c_str() << "\n";
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kern.handle);

    return err;
}

void
search_hasco(const common::filesystem::path& directory, std::string& filename)
{
    for(const auto& entry : common::filesystem::directory_iterator(directory))
    {
        if(common::filesystem::is_regular_file(entry))
        {
            if(entry.path().filename() == filename)
            {
                filename = entry.path();
            }
        }
        else if(common::filesystem::is_directory(entry))
        {
            search_hasco(entry, filename);  // Recursive call for subdirectories
        }
    }
}
}  // namespace testing
}  // namespace counters
}  // namespace rocprofiler
