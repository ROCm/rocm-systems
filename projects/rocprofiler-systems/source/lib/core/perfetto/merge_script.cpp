// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/perfetto/merge_script.hpp"

#include "core/timemory.hpp"
#include "core/utility.hpp"
#include "logger/debug.hpp"

#include <cerrno>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace rocprofsys
{
namespace core
{
namespace perfetto
{
namespace
{
// POSIX convention: 127 is "command not found / exec failure". Used by the
// shell when execve returns ENOENT and mirrored here so callers observing
// the merge-script child's exit status see the standard signal.
constexpr int EXEC_FAILURE_EXIT_CODE = 127;

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

    // Refuse to resolve through $PATH: a hostile PATH entry would win the
    // execve race. The script must arrive as an absolute (or at least
    // explicitly-rooted) path so we know which binary will run.
    if(script_path.find('/') == std::string::npos)
    {
        LOG_WARNING("Merge script path lacks '/': refusing to resolve through "
                    "PATH (got '{}')",
                    script_path);
        return;
    }

    if(!filepath::exists(script_path))
    {
        LOG_WARNING("Merge script not found: {}", script_path);
        return;
    }

    pid_t     pid        = ::fork();
    const int fork_errno = errno;
    if(pid < 0)
    {
        LOG_ERROR("fork failed for merge script {}: errno={}", script_path, fork_errno);
        return;
    }
    if(pid == 0)
    {
        // execv (not execlp) — script_path already includes the directory,
        // so PATH search is neither needed nor desired.
        char* const argv[] = { const_cast<char*>(script_path.c_str()),
                               const_cast<char*>(output_folder.c_str()), nullptr };
        ::execv(script_path.c_str(), argv);
        // execv only returns on failure.
        ::_exit(EXEC_FAILURE_EXIT_CODE);
    }

    int status = 0;
    while(::waitpid(pid, &status, 0) < 0)
    {
        if(errno == EINTR) continue;
        LOG_ERROR("waitpid failed for merge script {}: errno={}", script_path, errno);
        return;
    }
    if(WIFEXITED(status) && WEXITSTATUS(status) == 0)
        LOG_INFO("Successfully executed: {} {}", script_path, output_folder);
    else
        LOG_ERROR("Failed to execute: {} {} (status={})", script_path, output_folder,
                  status);
}
}  // namespace perfetto
}  // namespace core
}  // namespace rocprofsys
