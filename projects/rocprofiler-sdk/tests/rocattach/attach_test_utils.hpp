// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// Helpers shared by the rocattach test drivers.

#pragma once

#include <sys/types.h>
#include <sys/wait.h>

#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <system_error>
#include <thread>

// Waits until the target process has started its "rocp-bg-attach" thread, which rocprofiler-sdk
// creates once it has initialized with attachment support and into which rocattach injects. Gives
// up after 30 seconds, or as soon as a target that is a child of this process has exited. An
// exited child is left for the caller to wait for.
inline bool
wait_for_attach_ready(pid_t pid)
{
    constexpr int max_wait_sec = 30;
    auto          task_dir     = "/proc/" + std::to_string(pid) + "/task";

    std::cout << "Waiting for rocp-bg-attach thread in PID " << pid << "...\n";
    for(int elapsed = 0; elapsed < max_wait_sec; ++elapsed)
    {
        auto info = siginfo_t{};
        if(waitid(P_PID, pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == pid)
        {
            std::cout << "Target process " << pid
                      << " exited before it became ready for attachment\n";
            return false;
        }

        std::error_code ec;
        for(const auto& entry : std::filesystem::directory_iterator(task_dir, ec))
        {
            if(!entry.is_directory()) continue;
            std::ifstream comm(entry.path() / "comm");
            std::string   name;
            if(std::getline(comm, name) && name == "rocp-bg-attach")
            {
                std::cout << "Attachment ready (" << elapsed << "s elapsed)\n";
                return true;
            }
        }
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Timed out after " << max_wait_sec << "s waiting for rocp-bg-attach thread\n";
    return false;
}
