// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "ptrace_session.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"

#include <rocprofiler-sdk/defines.h>

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>

extern char** environ;

namespace common = ::rocprofiler::common;

namespace
{
std::unique_ptr<rocprofiler::attach::PTraceSession> ptrace_session;
std::thread                                         ptrace_thread;
std::atomic<bool>                                   finished_setup(false);
}  // namespace

ROCPROFILER_EXTERN_C_INIT
int
attach(uint32_t pid) ROCPROFILER_EXPORT;

int
detach() ROCPROFILER_EXPORT;
ROCPROFILER_EXTERN_C_FINI

void
initialize_logging()
{
    auto logging_cfg = rocprofiler::common::logging_config{.install_failure_handler = true};
    common::init_logging("ROCPROF", logging_cfg);
    FLAGS_colorlogtostderr = true;
}

namespace
{
// Helper function to allocate memory in target process and write data
bool
write_data_to_target(const std::string&          description,
                     const std::vector<uint8_t>& data,
                     void*&                      allocated_addr)
{
    // Allocate memory in target process
    if(!ptrace_session->simple_mmap(allocated_addr, data.size()))
    {
        ROCP_ERROR << "Failed to allocate memory for " << description << " in target process";
        return false;
    }
    ROCP_TRACE << "Allocated memory for " << description << " at " << allocated_addr;

    // Stop target process for writing
    if(!ptrace_session->stop())
    {
        ROCP_ERROR << "Failed to stop target process for " << description << " writing";
        return false;
    }

    // Write data to target process memory
    if(!ptrace_session->write(reinterpret_cast<size_t>(allocated_addr), data, data.size()))
    {
        ROCP_ERROR << "Failed to write " << description << " to target process";
        return false;
    }

    // Continue target process
    if(!ptrace_session->cont())
    {
        ROCP_ERROR << "Failed to continue target process after " << description << " writing";
        return false;
    }

    ROCP_TRACE << "Wrote " << description << " to target process";
    return true;
}

// Helper function to build environment buffer
std::vector<uint8_t>
build_environment_buffer()
{
    std::vector<uint8_t> environment_buffer(4);
    uint32_t             var_count = 0;

    char** invars = environ;
    for(; *invars; invars++)
    {
        const char* var = *invars;
        if(strncmp("ROCP", var, 4) != 0)
        {
            continue;
        }

        var_count++;
        ROCP_TRACE << "Adding to environment buffer: " << var;

        // Add variable name
        while(*var != '=')
        {
            environment_buffer.emplace_back(*var++);
        }
        environment_buffer.emplace_back(0);

        // Add variable value
        var++;
        while(*var != 0)
        {
            environment_buffer.emplace_back(*var++);
        }
        environment_buffer.emplace_back(0);
    }

    // Store count in first 4 bytes
    const uint8_t* var_count_bytes = reinterpret_cast<uint8_t*>(&var_count);
    std::copy(var_count_bytes, var_count_bytes + 4, environment_buffer.data());

    return environment_buffer;
}
// Helper function to connect to Unix socket with retries
int
connect_to_socket(const char* socket_path, int max_retries, int retry_delay_ms)
{
    int                socket_fd;
    struct sockaddr_un server_addr;

    for(int retry = 0; retry < max_retries; ++retry)
    {
        // Create socket
        socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if(socket_fd == -1)
        {
            ROCP_ERROR << "Failed to create socket: " << strerror(errno);
            return -1;
        }

        // Set up server address
        memset(&server_addr, 0, sizeof(server_addr));
        server_addr.sun_family = AF_UNIX;
        strncpy(server_addr.sun_path, socket_path, sizeof(server_addr.sun_path) - 1);

        // Try to connect
        if(connect(socket_fd, (struct sockaddr*) &server_addr, sizeof(server_addr)) == 0)
        {
            ROCP_TRACE << "Connected to socket " << socket_path << " on attempt " << (retry + 1);
            return socket_fd;
        }

        // Connection failed, close socket and retry
        close(socket_fd);

        if(retry < max_retries - 1)
        {
            ROCP_TRACE << "Connection attempt " << (retry + 1) << " failed, retrying in "
                       << retry_delay_ms << "ms...";
            std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
        }
    }

    ROCP_ERROR << "Failed to connect to socket " << socket_path << " after " << max_retries
               << " attempts";
    return -1;
}

// Helper function to send all bytes
bool
send_all(int fd, const void* buffer, size_t size)
{
    const uint8_t* buf        = static_cast<const uint8_t*>(buffer);
    size_t         total_sent = 0;
    while(total_sent < size)
    {
        ssize_t bytes_sent = send(fd, buf + total_sent, size - total_sent, 0);
        if(bytes_sent <= 0)
        {
            if(bytes_sent == 0)
            {
                ROCP_ERROR << "Connection closed by server";
            }
            else if(errno != EINTR)
            {
                ROCP_ERROR << "Failed to send data: " << strerror(errno);
            }
            return false;
        }
        total_sent += bytes_sent;
    }
    return true;
}

// Helper function to receive all bytes
bool
recv_all(int fd, void* buffer, size_t size)
{
    uint8_t* buf        = static_cast<uint8_t*>(buffer);
    size_t   total_read = 0;
    while(total_read < size)
    {
        ssize_t bytes_read = recv(fd, buf + total_read, size - total_read, 0);
        if(bytes_read <= 0)
        {
            if(bytes_read == 0)
            {
                ROCP_ERROR << "Connection closed by server";
            }
            else if(errno != EINTR)
            {
                ROCP_ERROR << "Failed to receive data: " << strerror(errno);
            }
            return false;
        }
        total_read += bytes_read;
    }
    return true;
}
}  // anonymous namespace

ROCPROFILER_EXTERN_C_INIT

void
handle_attach_operations(uint32_t pid)
{
    // Setup attachment for rocprofiler via socket communication
    ROCP_TRACE << "Socket-based attachment library called for pid " << pid;
    ptrace_session = std::make_unique<rocprofiler::attach::PTraceSession>(pid);
    ROCP_TRACE << "Attempting attachment to pid " << pid;
    if(!ptrace_session->attach())
    {
        ROCP_ERROR << "Attachment failed to pid " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Attachment success to pid " << pid;

    // Build environment buffer
    auto environment_buffer = build_environment_buffer();
    ROCP_TRACE << "Built environment buffer with " << environment_buffer.size() << " bytes";

    // Get tool library path
    auto tool_lib_path =
        rocprofiler::common::get_env("ROCPROF_ATTACH_TOOL_LIBRARY", "librocprofiler-sdk-tool.so");
    ROCP_TRACE << "Tool library path: " << tool_lib_path;

    // Execute rocprofiler_register_thread_init in target process to create socket_monitor_thread
    if(!ptrace_session->call_function("librocprofiler-register.so",
                                      "rocprofiler_register_thread_init"))
    {
        ROCP_ERROR << "Failed to call rocprofiler_register_thread_init in target process " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Successfully created socket_monitor_thread in target process";

    // Give the socket thread time to start up
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    // Connect to the Unix socket with retries
    constexpr int max_retries    = 10;
    constexpr int retry_delay_ms = 200;
    // Generate socket path with target process PID
    std::string socket_path      = common::get_socket_path_for_pid(pid);

    int socket_fd = connect_to_socket(socket_path.c_str(), max_retries, retry_delay_ms);
    if(socket_fd == -1)
    {
        ROCP_ERROR << "Failed to connect to socket " << socket_path;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Connected to socket " << socket_path;

    // Serialize and send ATTACH message
    auto message = common::serialize_attach_message(common::ATTACH_OP_ATTACH,
                                                    tool_lib_path,
                                                    environment_buffer);

    if(!send_all(socket_fd, message.data(), message.size()))
    {
        ROCP_ERROR << "Failed to send ATTACH message to target process";
        close(socket_fd);
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Sent ATTACH message (" << message.size() << " bytes)";

    // Receive response
    common::attach_response_message response;
    if(!recv_all(socket_fd, &response, sizeof(response)))
    {
        ROCP_ERROR << "Failed to receive response from target process";
        close(socket_fd);
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Check response
    if(response.op != common::ATTACH_OP_RESPONSE)
    {
        ROCP_ERROR << "Unexpected response op code: " << response.op;
        close(socket_fd);
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    if(response.status != common::ATTACH_STATUS_SUCCESS)
    {
        ROCP_ERROR << "ATTACH operation failed with rocprofiler_register error code: " << response.status;
        // Map rocprofiler_register error codes to more specific rocprofiler status
        rocprofiler_status_t mapped_status = ROCPROFILER_STATUS_ERROR;
        switch(response.status)
        {
            case ROCP_REG_NO_TOOLS:
                ROCP_ERROR << "  -> No profiling tools found";
                break;
            case ROCP_REG_DEADLOCK:
                ROCP_ERROR << "  -> Deadlock detected in rocprofiler-register";
                break;
            case ROCP_REG_INVALID_ARGUMENT:
                ROCP_ERROR << "  -> Invalid argument provided";
                mapped_status = ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT;
                break;
            case ROCP_REG_ATTACHMENT_NOT_AVAILABLE:
                ROCP_ERROR << "  -> Attachment not available (rocprofiler-attach library was never loaded)";
                ROCP_ERROR << "  -> Start the app with ROCP_TOOL_ATTACH=1 or build rocprofiler-register with ROCP_REG_DEFAULT_ATTACHMENT=ON";
                break;
            default:
                ROCP_ERROR << "  -> Unknown error";
                break;
        }
        close(socket_fd);
        ptrace_session->m_setup_status.store(mapped_status);
        finished_setup.store(true);
        return;
    }

    ROCP_TRACE << "ATTACH operation completed successfully";
    close(socket_fd);

    // Allow main thread to continue
    finished_setup.store(true);
    if(!ptrace_session->handle_signals())
    {
        ROCP_ERROR << "Signal handling loop terminated unexpectedly for pid " << pid;
        // don't return, try to detach anyways
    }

    // Detach rocprofiler via socket
    ROCP_TRACE << "Detaching rocprofiler from pid " << pid << " via socket";

    // Connect to socket for detach operation
    int detach_socket_fd = connect_to_socket(socket_path.c_str(), max_retries, retry_delay_ms);
    if(detach_socket_fd == -1)
    {
        ROCP_ERROR << "Failed to connect to socket for detach operation";
        // Continue with ptrace detach anyway
    }
    else
    {
        // Send DETACH message (empty environment buffer and tool_lib_path)
        std::vector<uint8_t> empty_env_buffer;
        auto                 detach_message = common::serialize_attach_message(common::ATTACH_OP_DETACH,
                                                                               "",
                                                                               empty_env_buffer);

        if(!send_all(detach_socket_fd, detach_message.data(), detach_message.size()))
        {
            ROCP_ERROR << "Failed to send DETACH message";
        }
        else
        {
            ROCP_TRACE << "Sent DETACH message";

            // Receive response
            common::attach_response_message detach_response;
            if(recv_all(detach_socket_fd, &detach_response, sizeof(detach_response)))
            {
                if(detach_response.op == common::ATTACH_OP_RESPONSE &&
                   detach_response.status == common::ATTACH_STATUS_SUCCESS)
                {
                    ROCP_TRACE << "DETACH operation completed successfully";
                }
                else
                {
                    ROCP_ERROR << "DETACH operation failed with rocprofiler_register error code: " << detach_response.status;
                    // Log more specific error information based on error code
                    switch(detach_response.status)
                    {
                        case ROCP_REG_NO_TOOLS:  // ROCP_REG_NO_TOOLS
                            ROCP_ERROR << "  -> No profiling tools found (detach entry point is NULL)";
                            break;
                        case ROCP_REG_ATTACHMENT_NOT_AVAILABLE:  // ROCP_REG_ATTACHMENT_NOT_AVAILABLE
                            ROCP_ERROR << "  -> Attachment not available (rocprofiler-attach library was never loaded)";
                            break;
                        default:
                            ROCP_ERROR << "  -> Unknown error during detach";
                            break;
                    }
                }
            }
            else
            {
                ROCP_ERROR << "Failed to receive DETACH response";
            }
        }
        close(detach_socket_fd);
    }

    ptrace_session->stop();
    ptrace_session->detach();
    ptrace_session.reset();
}

void
handle_ptrace_operations(uint32_t pid)
{
    // Setup attachement for rocprofiler
    ROCP_TRACE << "Attachment library called for pid " << pid;
    ptrace_session = std::make_unique<rocprofiler::attach::PTraceSession>(pid);
    ROCP_TRACE << "Attempting attachment to pid " << pid;
    if(!ptrace_session->attach())
    {
        ROCP_ERROR << "Attachment failed to pid " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT);
        finished_setup.store(true);
        return;
    }
    ROCP_TRACE << "Attachment success to pid " << pid;

    // Build and write environment buffer to target process
    auto  environment_buffer      = build_environment_buffer();
    void* environment_buffer_addr = nullptr;
    if(!write_data_to_target("environment buffer", environment_buffer, environment_buffer_addr))
    {
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Build and write tool library path to target process
    auto tool_lib_path_env =
        rocprofiler::common::get_env("ROCPROF_ATTACH_TOOL_LIBRARY", "librocprofiler-sdk-tool.so");
    const char* tool_lib_path = tool_lib_path_env.c_str();
    ROCP_TRACE << "Tool library path: " << tool_lib_path;

    size_t               tool_lib_path_len = strlen(tool_lib_path) + 1;
    std::vector<uint8_t> tool_lib_buffer(tool_lib_path, tool_lib_path + tool_lib_path_len);

    void* tool_lib_path_addr = nullptr;
    if(!write_data_to_target("tool library path", tool_lib_buffer, tool_lib_path_addr))
    {
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Execute the attach function with both parameters
    if(!ptrace_session->call_function("librocprofiler-register.so",
                                      "rocprofiler_register_attach",
                                      environment_buffer_addr,
                                      tool_lib_path_addr))
    {
        ROCP_ERROR << "Failed to call attach function in target process " << pid;
        ptrace_session->m_setup_status.store(ROCPROFILER_STATUS_ERROR);
        finished_setup.store(true);
        return;
    }

    // Clean up - free the tool library path memory in target process
    if(!ptrace_session->simple_munmap(tool_lib_path_addr, tool_lib_path_len))
    {
        ROCP_ERROR << "Failed to free tool library path memory in target process";
        // Continue anyway since the main operation succeeded
    }
    ROCP_TRACE << "Cleaned up tool library path memory in target process";

    // Allow main thread to continue
    finished_setup.store(true);
    if(!ptrace_session->handle_signals())
    {
        ROCP_ERROR << "Signal handling loop terminated unexepectedly for pid " << pid;
        // don't return, try to detach anyways
    }
    // Detach rocprofiler
    ROCP_TRACE << "Detaching rocprofiler from pid " << pid;
    if(!ptrace_session->call_function("librocprofiler-register.so", "rocprofiler_register_detach"))
    {
        ROCP_ERROR << "Failed to call detach function in target process";
        // don't return, try to detach anyways
    }
    ptrace_session->stop();
    ptrace_session->detach();
    ptrace_session.reset();
}

int
attach(uint32_t pid)
{
    initialize_logging();
    ptrace_thread = std::thread(handle_attach_operations, pid);
    // Wait for ptrace thread to finish setting up
    while(!finished_setup.load())
        std::this_thread::yield();

    auto status = ptrace_session->m_setup_status.load();
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        ROCP_ERROR << "ptrace session failed with error code " << ptrace_session->m_setup_status;
        ptrace_thread.join();
        finished_setup.store(false);
        return status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

int
detach()
{
    ptrace_session->detach_ptrace_session();
    ptrace_thread.join();
    finished_setup.store(false);
    return ROCPROFILER_STATUS_SUCCESS;
}

ROCPROFILER_EXTERN_C_FINI
