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

/**
 * @file samples/process_attachment/client.cpp
 *
 * @brief Process attachment client utility - demonstrates how to attach/detach to running processes
 */

#include <dlfcn.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

class ProcessAttachmentClient
{
private:
    void* attach_lib_handle = nullptr;
    int (*attach_func)(uint32_t) = nullptr;
    int (*detach_func)() = nullptr;
    uint32_t target_pid = 0;
    bool attached = false;

public:
    ProcessAttachmentClient() = default;
    
    ~ProcessAttachmentClient()
    {
        if(attached)
        {
            detach();
        }
        if(attach_lib_handle)
        {
            dlclose(attach_lib_handle);
        }
    }

    bool initialize()
    {
        // Try to load the rocprofv3-attach library
        const char* lib_paths[] = {
            "librocprofv3-attach.so",
            "/opt/rocm/lib/rocprofiler-sdk/librocprofv3-attach.so"
        };
        
        for(const char* lib_path : lib_paths)
        {
            attach_lib_handle = dlopen(lib_path, RTLD_NOW);
            if(attach_lib_handle) 
            {
                std::cout << "Loaded attachment library: " << lib_path << std::endl;
                break;
            }
        }
        
        if(!attach_lib_handle)
        {
            std::cerr << "Failed to load rocprofv3-attach library: " << dlerror() << std::endl;
            std::cerr << "Make sure rocprofiler-sdk is built and installed" << std::endl;
            return false;
        }

        // Get function pointers
        attach_func = (int(*)(uint32_t))dlsym(attach_lib_handle, "attach");
        detach_func = (int(*)())dlsym(attach_lib_handle, "detach");

        if(!attach_func || !detach_func)
        {
            std::cerr << "Failed to find attachment functions: " << dlerror() << std::endl;
            return false;
        }

        return true;
    }

    bool attach_to_process(uint32_t pid, const std::string& tool_lib_path)
    {
        if(attached)
        {
            std::cerr << "Already attached to a process" << std::endl;
            return false;
        }

        // Validate the target process
        if(kill(pid, 0) != 0)
        {
            std::cerr << "Target process " << pid << " is not accessible or doesn't exist" << std::endl;
            return false;
        }

        std::cout << "Attaching to process " << pid << "..." << std::endl;
        std::cout << "Tool library: " << tool_lib_path << std::endl;

        // Set environment variable for our tool library
        setenv("ROCPROF_ATTACH_TOOL_LIBRARY", tool_lib_path.c_str(), 1);

        int result = attach_func(pid);
        if(result != 0)
        {
            std::cerr << "Attachment failed with code: " << result << std::endl;
            return false;
        }

        target_pid = pid;
        attached = true;
        std::cout << "Successfully attached to process " << pid << std::endl;
        return true;
    }

    bool detach()
    {
        if(!attached)
        {
            std::cerr << "Not currently attached to any process" << std::endl;
            return false;
        }

        std::cout << "Detaching from process " << target_pid << "..." << std::endl;

        int result = detach_func();
        if(result != 0)
        {
            std::cerr << "Detachment failed with code: " << result << std::endl;
            return false;
        }

        attached = false;
        target_pid = 0;
        std::cout << "Successfully detached from process" << std::endl;
        return true;
    }

    bool is_attached() const { return attached; }
    uint32_t get_target_pid() const { return target_pid; }
};

void print_usage(const char* prog_name)
{
    std::cout << "Usage: " << prog_name << " [options] <pid>" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --tool-lib <path>        Path to tool library (required)" << std::endl;
    std::cout << "  -d, --duration <seconds> Attach for specified duration (default: 10)" << std::endl;
    std::cout << "  --interactive            Interactive mode - press Enter to detach" << std::endl;
    std::cout << "  -h, --help              Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << prog_name << " --tool-lib ./tool.so --pid 1234" << std::endl;
    std::cout << "  " << prog_name << " --tool-lib ./tool.so -d 30 --pid 1234" << std::endl;
    std::cout << std::endl;
    std::cout << "Interactive mode commands:" << std::endl;
    std::cout << "  Press Enter to detach and exit" << std::endl;
}

bool process_exists(uint32_t pid)
{
    return kill(pid, 0) == 0;
}

int main(int argc, char* argv[])
{
    uint32_t target_pid = 0;
    int duration_seconds = 10;  // default 10 seconds, -1 means interactive mode
    std::string tool_lib_path;
    bool interactive_mode = false;

    // Parse command line arguments
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        
        if(arg == "-h" || arg == "--help")
        {
            print_usage(argv[0]);
            return 0;
        }
        else if(arg == "--tool-lib")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: --tool-lib option requires a path" << std::endl;
                return 1;
            }
            tool_lib_path = argv[++i];
        }
        else if(arg == "--pid")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: --pid option requires a value" << std::endl;
                return 1;
            }
            target_pid = std::atoi(argv[++i]);
            if(target_pid <= 0)
            {
                std::cerr << "Error: Invalid PID: " << argv[i] << std::endl;
                return 1;
            }
        }
        else if(arg == "--interactive")
        {
            interactive_mode = true;
            duration_seconds = -1;
        }
        else if(arg == "-d" || arg == "--duration")
        {
            if(i + 1 >= argc)
            {
                std::cerr << "Error: Duration option requires a value" << std::endl;
                return 1;
            }
            duration_seconds = std::atoi(argv[++i]);
            if(duration_seconds <= 0)
            {
                std::cerr << "Error: Duration must be a positive integer" << std::endl;
                return 1;
            }
        }
        else if(arg[0] != '-')
        {
            // This should be the PID
            target_pid = std::atoi(arg.c_str());
            if(target_pid <= 0)
            {
                std::cerr << "Error: Invalid PID: " << arg << std::endl;
                return 1;
            }
        }
        else
        {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }

    if(target_pid == 0)
    {
        std::cerr << "Error: No target PID specified" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    if(tool_lib_path.empty())
    {
        std::cerr << "Error: Tool library path not specified (use --tool-lib)" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "Process Attachment Client" << std::endl;
    std::cout << "=========================" << std::endl;
    std::cout << "Target PID: " << target_pid << std::endl;
    std::cout << "Tool Library: " << tool_lib_path << std::endl;
    if(interactive_mode)
        std::cout << "Attachment Mode: interactive" << std::endl;
    else
        std::cout << "Attachment Mode: timed (" << duration_seconds << " seconds)" << std::endl;

    ProcessAttachmentClient client;
    if(!client.initialize())
    {
        std::cerr << "Failed to initialize attachment client" << std::endl;
        return 1;
    }

    if(!client.attach_to_process(target_pid, tool_lib_path))
    {
        std::cerr << "Failed to attach to target process" << std::endl;
        return 1;
    }

    // Handle attachment duration
    if(duration_seconds > 0)
    {
        std::cout << "Profiling for " << duration_seconds << " seconds..." << std::endl;
        
        // Monitor target process while profiling
        for(int i = 0; i < duration_seconds; ++i)
        {
            if(!process_exists(target_pid))
            {
                std::cout << "Target process " << target_pid << " has exited" << std::endl;
                break;
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            // Show progress every 10 seconds
            if((i + 1) % 10 == 0)
            {
                std::cout << "Profiling... " << (i + 1) << "/" << duration_seconds << " seconds" << std::endl;
            }
        }
    }
    else
    {
        // Interactive mode
        std::cout << std::endl;
        std::cout << "=== Interactive Mode ===" << std::endl;
        std::cout << "Profiling process " << target_pid << std::endl;
        std::cout << "Press Enter to detach and exit..." << std::endl;
        
        std::string input;
        std::getline(std::cin, input);
    }

    // Detach from the process
    if(!client.detach())
    {
        std::cerr << "Warning: Failed to cleanly detach from process" << std::endl;
    }

    std::cout << "Process attachment session completed" << std::endl;
    return 0;
}
