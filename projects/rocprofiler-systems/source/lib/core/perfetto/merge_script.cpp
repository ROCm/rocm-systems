// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/merge_script.hpp"

#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cstdlib>
#include <string>

namespace rocprofsys
{
namespace core
{
namespace perfetto
{
namespace
{
constexpr const char* MERGE_SCRIPT_BASENAME = "rocprof-sys-merge-output.sh";
}  // namespace

void
run_merge_script(const std::string& output_folder)
{
    if(dmp::rank() != 0) return;

    auto script_path = std::string{ MERGE_SCRIPT_BASENAME };
    auto script_dir  = get_env("ROCPROFSYS_SCRIPT_PATH", std::string{}, false);

    if(!script_dir.empty())
        script_path = fmt::format("{}/{}", script_dir, MERGE_SCRIPT_BASENAME);

    if(!filepath::exists(script_path))
    {
        LOG_WARNING("Merge script not found: {}", script_path);
        return;
    }

    // system() rather than fork()+execv() because raw fork after MPI init
    // trips libfabric EFA's fork-safety abort on AWS-class CI runners
    // (RDMAV_FORK_SAFE is not enabled by default). system() routes through
    // /bin/sh which the EFA hook handles without aborting. The script path
    // is project-controlled (ROCPROFSYS_SCRIPT_PATH points at our libexec
    // dir) and output_folder comes from our own config, so the shell
    // interpolation surface is internal-only.
    auto command = fmt::format("{} '{}'", script_path, output_folder);
    int  result  = std::system(command.c_str());
    if(result != 0)
        LOG_ERROR("Failed to execute: {} (status={})", command, result);
    else
        LOG_INFO("Successfully executed: {}", command);
}
}  // namespace perfetto
}  // namespace core
}  // namespace rocprofsys
