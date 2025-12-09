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

#pragma once

#include <rocprofiler-sdk/rocprofiler.h>

#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace common
{
// Socket path prefix for attach communication
constexpr const char* socket_path_prefix = "/tmp/rocprofiler.";
constexpr const char* socket_path_suffix = ".sock";

// Helper function to generate socket path for a specific PID
inline std::string
get_socket_path_for_pid(pid_t pid)
{
    return std::string(socket_path_prefix) + std::to_string(pid) + socket_path_suffix;
}

typedef enum rocprofiler_register_error_code_t  // NOLINT(performance-enum-size)
{
    ROCP_REG_SUCCESS = 0,
    ROCP_REG_NO_TOOLS,
    ROCP_REG_DEADLOCK,
    ROCP_REG_BAD_API_TABLE_LENGTH,
    ROCP_REG_UNSUPPORTED_API,
    ROCP_REG_INVALID_API_ADDRESS,
    ROCP_REG_ROCPROFILER_ERROR,
    ROCP_REG_EXCESS_API_INSTANCES,
    ROCP_REG_INVALID_ARGUMENT,
    ROCP_REG_ATTACHMENT_NOT_AVAILABLE,
    ROCP_REG_ERROR_CODE_END,
} rocprofiler_register_error_code_t;

// Operation codes
enum attach_op : uint32_t
{
    ATTACH_OP_ATTACH   = 0,
    ATTACH_OP_DETACH   = 1,
    ATTACH_OP_RESPONSE = 2,  // Response from server
};

// Response status codes
enum attach_status : uint32_t
{
    ATTACH_STATUS_SUCCESS = 0,
    ATTACH_STATUS_ERROR   = 1,
};

// Message header for socket communication
// Format: [op (4 bytes)][tool_lib_path_len (4 bytes)][env_len (4 bytes)][tool_lib_path][env_data]
struct attach_message_header
{
    uint32_t op                = 0;  // Operation type
    uint32_t tool_lib_path_len = 0;  // Length of tool library path string (including null terminator)
    uint32_t env_buffer_len    = 0;  // Length of environment buffer
};

// Response message from server
struct attach_response_message
{
    uint32_t op     = 0;  // Should be ATTACH_OP_RESPONSE
    uint32_t status = 0;  // Status code (attach_status)
};

// Helper function to serialize attach message
inline std::vector<uint8_t>
serialize_attach_message(uint32_t                    op,
                         const std::string&          tool_lib_path,
                         const std::vector<uint8_t>& env_buffer)
{
    attach_message_header header;
    header.op                = op;
    header.tool_lib_path_len = static_cast<uint32_t>(tool_lib_path.size() + 1);  // +1 for null
    header.env_buffer_len    = static_cast<uint32_t>(env_buffer.size());

    std::vector<uint8_t> message;
    message.reserve(sizeof(header) + header.tool_lib_path_len + header.env_buffer_len);

    // Add header
    const auto* header_bytes = reinterpret_cast<const uint8_t*>(&header);
    message.insert(message.end(), header_bytes, header_bytes + sizeof(header));

    // Add tool library path (including null terminator)
    message.insert(message.end(), tool_lib_path.begin(), tool_lib_path.end());
    message.push_back(0);  // null terminator

    // Add environment buffer
    message.insert(message.end(), env_buffer.begin(), env_buffer.end());

    return message;
}

// Helper function to serialize response message
inline std::vector<uint8_t>
serialize_response_message(uint32_t status)
{
    attach_response_message response;
    response.op     = ATTACH_OP_RESPONSE;
    response.status = status;

    std::vector<uint8_t> message(sizeof(response));
    std::memcpy(message.data(), &response, sizeof(response));
    return message;
}
}  // namespace common

namespace attach
{
class PTraceSession
{
public:
    explicit PTraceSession(int);
    ~PTraceSession();

    bool attach();
    bool detach();
    bool simple_mmap(void*& addr, size_t length) const;
    bool simple_munmap(void*& addr, size_t length) const;

    bool write(size_t addr, const std::vector<uint8_t>& data, size_t size) const;
    bool read(size_t addr, std::vector<uint8_t>& data, size_t size) const;
    bool swap(size_t                      addr,
              const std::vector<uint8_t>& in_data,
              std::vector<uint8_t>&       out_data,
              size_t                      size) const;

    int get_pid() const { return m_pid; }

    bool call_function(const std::string& library, const std::string& symbol);
    bool call_function(const std::string& library, const std::string& symbol, void* first);
    bool call_function(const std::string& library,
                       const std::string& symbol,
                       void*              first,
                       void*              second);

    bool stop() const;
    bool cont() const;
    bool handle_signals() const;
    void detach_ptrace_session();

    std::atomic<rocprofiler_status_t> m_setup_status = ROCPROFILER_STATUS_SUCCESS;

private:
    static bool find_library(void*& addr, int inpid, const std::string& library);
    bool        find_symbol(void*& addr, const std::string& library, const std::string& symbol);

    std::unordered_map<std::string, void*> m_target_library_addrs = {};
    std::unordered_map<std::string, void*> m_target_symbol_addrs  = {};

    const int         m_pid                      = -1;
    bool              m_attached                 = false;
    std::atomic<bool> m_detaching_ptrace_session = false;
};

}  // namespace attach
}  // namespace rocprofiler
