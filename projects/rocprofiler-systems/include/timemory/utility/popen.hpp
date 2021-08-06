// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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
//

#pragma once

#include "timemory/macros/os.hpp"
#include "timemory/utility/macros.hpp"
#include "timemory/utility/utility.hpp"

#if !defined(TIMEMORY_WINDOWS)

#    include <sstream>
#    include <string>
#    include <vector>

#    include <cerrno>
#    include <cstdio>
#    include <cstdlib>
#    include <cstring>
#    include <ctime>
#    include <fcntl.h>
#    include <grp.h>
#    include <paths.h>
#    include <sys/param.h>
#    include <sys/stat.h>
#    include <sys/types.h>
#    include <sys/wait.h>
#    include <unistd.h>

namespace tim
{
namespace popen
{
//
using string_t = std::string;
using strvec_t = std::vector<string_t>;
//
struct TIMEMORY_PIPE
{
    FILE* read_fd  = nullptr;
    FILE* write_fd = nullptr;
    pid_t child_pid;
    int   child_status = std::numeric_limits<int>::max();
};
//
TIMEMORY_UTILITY_LINKAGE(TIMEMORY_PIPE*)
popen(const char* path, char** argv = nullptr, char** envp = nullptr);
//
TIMEMORY_UTILITY_LINKAGE(int)
pclose(TIMEMORY_PIPE* p);
//
TIMEMORY_UTILITY_LINKAGE(pid_t)
fork();
//
TIMEMORY_UTILITY_LINKAGE(void)  // NOLINT
sanitize_files();
//
TIMEMORY_UTILITY_LINKAGE(int)
open_devnull(int fd);
//
TIMEMORY_UTILITY_LINKAGE(void)
drop_privileges(int permanent);
//
TIMEMORY_UTILITY_LINKAGE(void)  // NOLINT
restore_privileges();
//
inline strvec_t
read_fork(TIMEMORY_PIPE* proc, int max_counter = 50)
{
    int      counter = 0;
    strvec_t linked_libraries;

    while(proc)
    {
        char  buffer[4096];
        auto* ret = fgets(buffer, 4096, proc->read_fd);
        if(ret == nullptr || strlen(buffer) == 0)
        {
            if(max_counter == 0)
            {
                pid_t cpid = waitpid(proc->child_pid, &proc->child_status, WNOHANG);
                if(cpid == 0)
                    continue;
                else
                    break;
            }
            if(counter++ > max_counter)
                break;
            continue;
        }
        auto line = string_t(buffer);
        auto loc  = string_t::npos;
        while((loc = line.find_first_of("\n\t")) != string_t::npos)
            line.erase(loc, 1);
        auto delim = delimit(line, " \n\t=>");
        for(const auto& itr : delim)
        {
            if(itr.find('/') == 0)
                linked_libraries.push_back(itr);
        }
    }

    return linked_libraries;
}
//
inline std::ostream&
flush_output(std::ostream& os, TIMEMORY_PIPE* proc, int max_counter = 0)
{
    int counter = 0;
    while(proc)
    {
        char  buffer[4096];
        auto* ret = fgets(buffer, 4096, proc->read_fd);
        if(ret == nullptr || strlen(buffer) == 0)
        {
            if(max_counter == 0)
            {
                pid_t cpid = waitpid(proc->child_pid, &proc->child_status, WNOHANG);
                if(cpid == 0)
                    continue;
                else
                    break;
            }
            if(counter++ > max_counter)
                break;
            continue;
        }
        os << string_t{ buffer } << std::flush;
    }

    return os;
}
//
//--------------------------------------------------------------------------------------//
//
}  // namespace popen
}  // namespace tim

#    if !defined(TIMEMORY_UTILITY_SOURCE) && !defined(TIMEMORY_USE_UTILITY_EXTERN)
#        include "timemory/utility/popen.cpp"
#    endif

#endif

namespace tim
{
//
//--------------------------------------------------------------------------------------//
//
inline bool
launch_process(const char* cmd, const std::string& extra, std::ostream* os)
{
#if !defined(TIMEMORY_WINDOWS)
    auto                       delim = tim::delimit(cmd, " \t");
    tim::popen::TIMEMORY_PIPE* fp    = nullptr;
    if(delim.size() < 2)
    {
        fp = tim::popen::popen(cmd, nullptr, nullptr);
    }
    else
    {
        static std::string   _c = "-c";
        std::array<char*, 4> _args;
        _args.fill(nullptr);
        char*       _cshell = getenv("SHELL");
        char*       _ushell = getusershell();
        std::string _shell = (_cshell) ? _cshell : (_ushell) ? getusershell() : "/bin/sh";
        _args.at(0)        = (char*) _shell.c_str();
        _args.at(1)        = (char*) _c.c_str();
        _args.at(2)        = (char*) cmd;
        fp                 = tim::popen::popen(_args.at(0), _args.data());
    }

    if(fp == nullptr)
    {
        std::stringstream ss;
        ss << "[timemory]> Error launching command: '" << cmd << "'... " << extra;
        perror(ss.str().c_str());
        return false;
    }
    if(os)
    {
        popen::flush_output(*os, fp);
    }

    auto ec = tim::popen::pclose(fp);
    if(ec != 0)
    {
        std::stringstream ss;
        ss << "[timemory]> Command: '" << cmd << "' returned a non-zero exit code: " << ec
           << "... " << extra;
        perror(ss.str().c_str());
        return false;
    }
#else
    if(std::system(nullptr) != 0)
    {
        int ec = std::system(cmd);

        if(ec != 0)
        {
            fprintf(stderr,
                    "[timemory]> Command: '%s' returned a non-zero exit code: %i... %s\n",
                    cmd, ec, extra.c_str());
            return false;
        }
    }
    else
    {
        fprintf(stderr, "std::system unavailable for command: '%s'... %s\n", cmd,
                extra.c_str());
        return false;
    }
    (void) os;
#endif

    return true;
}
}  // namespace tim
