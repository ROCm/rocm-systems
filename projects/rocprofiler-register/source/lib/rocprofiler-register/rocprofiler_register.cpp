// Copyright (c) 2023 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#define GNU_SOURCE 1

#include <rocprofiler-register/rocprofiler-register.h>

#include "details/dl.hpp"
#include "details/environment.hpp"
#include "details/filesystem.hpp"
#include "details/logging.hpp"
#include "details/scope_destructor.hpp"

#include <fmt/format.h>
#include <glog/logging.h>

#include <array>
#include <atomic>
#include <bitset>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <regex>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>

#include <dlfcn.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

namespace
{
using rocprofiler_register_library_api_table_func_t =
    decltype(::rocprofiler_register_library_api_table)*;
}

extern "C" {
#pragma weak rocprofiler_configure
#pragma weak rocprofiler_set_api_table
#pragma weak rocprofiler_attach
#pragma weak rocprofiler_detach
#pragma weak rocprofiler_attach_set_api_table
#pragma weak rocprofiler_register_import_hip
#pragma weak rocprofiler_register_import_hip_static
#pragma weak rocprofiler_register_import_hip_compiler
#pragma weak rocprofiler_register_import_hip_compiler_static
#pragma weak rocprofiler_register_import_hsa
#pragma weak rocprofiler_register_import_hsa_static
#pragma weak rocprofiler_register_import_roctx
#pragma weak rocprofiler_register_import_roctx_static

typedef struct rocprofiler_client_id_t
{
    const char*    name;    ///< clients should set this value for debugging
    const uint32_t handle;  ///< internal handle
} rocprofiler_client_id_t;

typedef void (*rocprofiler_client_finalize_t)(rocprofiler_client_id_t);

typedef int (*rocprofiler_tool_initialize_t)(rocprofiler_client_finalize_t finalize_func,
                                             void*                         tool_data);

typedef void (*rocprofiler_tool_finalize_t)(void* tool_data);


rocprofiler_register_error_code_t
rocprofiler_register_attach(const char* environment_buffer,
                            const char* tool_lib_path) ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_detach() ROCPROFILER_REGISTER_PUBLIC_API;

typedef struct rocprofiler_tool_configure_result_t
{
    size_t                        size;        ///< in case of future extensions
    rocprofiler_tool_initialize_t initialize;  ///< context creation
    rocprofiler_tool_finalize_t   finalize;    ///< cleanup
    void* tool_data;  ///< data to provide to init and fini callbacks
} rocprofiler_tool_configure_result_t;

extern rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, rocprofiler_client_id_t*);

extern int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t);

extern int
rocprofiler_attach(void);

extern int
rocprofiler_detach(void);

extern int
rocprofiler_attach_set_api_table(const char*,
                                 uint64_t,
                                 uint64_t,
                                 void**,
                                 uint64_t,
                                 rocprofiler_register_library_api_table_func_t);

extern uint32_t
rocprofiler_register_import_hip(void);

extern uint32_t
rocprofiler_register_import_hip_compiler(void);

extern uint32_t
rocprofiler_register_import_hsa(void);

extern uint32_t
rocprofiler_register_import_roctx(void);

extern uint32_t
rocprofiler_register_import_hip_static(void);

extern uint32_t
rocprofiler_register_import_hip_compiler_static(void);

extern uint32_t
rocprofiler_register_import_hsa_static(void);

extern uint32_t
rocprofiler_register_import_roctx_static(void);
}

namespace
{
using namespace rocprofiler_register;
using rocprofiler_set_api_table_t        = decltype(::rocprofiler_set_api_table)*;
using rocprofiler_attach_set_api_table_t = decltype(::rocprofiler_attach_set_api_table)*;
using rocprofiler_attach_func_t          = decltype(::rocprofiler_attach)*;
using rocprofiler_detach_func_t          = decltype(::rocprofiler_detach)*;
using rocp_set_api_table_data_t          = std::tuple<void*,
                                             rocprofiler_set_api_table_t,
                                             rocprofiler_attach_func_t,
                                             rocprofiler_detach_func_t>;

using bitset_t = std::bitset<sizeof(rocprofiler_register_library_indentifier_t::handle)>;

static_assert(sizeof(bitset_t) ==
                  sizeof(rocprofiler_register_library_indentifier_t::handle),
              "bitset should be same at uint64_t");

constexpr auto rocprofiler_lib_name                = "librocprofiler-sdk.so";
constexpr auto rocprofiler_lib_register_entrypoint = "rocprofiler_set_api_table";
constexpr auto rocprofiler_attach_lib_name         = "librocprofiler-sdk-attach.so";
constexpr auto rocprofiler_attach_lib_register_entrypoint =
    "rocprofiler_attach_set_api_table";
constexpr auto rocprofiler_lib_attach_entrypoint = "rocprofiler_attach";
constexpr auto rocprofiler_lib_detach_entrypoint = "rocprofiler_detach";

constexpr auto rocprofiler_register_lib_name =
    "librocprofiler-register.so." ROCPROFILER_REGISTER_SOVERSION;

// ======================================================================================
// Socket communication types and utilities for attach/detach operations
// ======================================================================================

/// Socket path constants for Unix domain socket communication
constexpr auto socket_path_prefix = std::string_view{ "/tmp/rocprofiler." };
constexpr auto socket_path_suffix = std::string_view{ ".sock" };

/// Operation codes for attach/detach socket protocol
enum class attach_op : uint32_t
{
    attach   = 0,  ///< Request to attach profiler
    detach   = 1,  ///< Request to detach profiler
    response = 2,  ///< Response message from server
};

/// Status codes for attach/detach response
enum class attach_status : uint32_t
{
    success = 0,  ///< Operation completed successfully
    error   = 1,  ///< Operation failed with error
};

/// Message header for socket communication protocol
struct attach_message_header
{
    uint32_t op                = 0;  ///< Operation type (see attach_op)
    uint32_t tool_lib_path_len = 0;  ///< Length of tool library path string
    uint32_t env_buffer_len    = 0;  ///< Length of environment buffer
};

/// Response message sent back to client
struct attach_response_message
{
    uint32_t op     = 0;  ///< Should be attach_op::response
    uint32_t status = 0;  ///< Status code (see attach_status)
};

/// RAII wrapper for socket file descriptor management
class socket_fd_guard
{
public:
    explicit socket_fd_guard(int fd) noexcept
    : m_fd{ fd }
    {}

    ~socket_fd_guard() noexcept
    {
        if(m_fd >= 0) ::close(m_fd);
    }

    socket_fd_guard(const socket_fd_guard&)            = delete;
    socket_fd_guard& operator=(const socket_fd_guard&) = delete;
    socket_fd_guard(socket_fd_guard&& other) noexcept
    : m_fd{ std::exchange(other.m_fd, -1) }
    {}
    socket_fd_guard& operator=(socket_fd_guard&& other) noexcept
    {
        if(this != &other)
        {
            if(m_fd >= 0) ::close(m_fd);
            m_fd = std::exchange(other.m_fd, -1);
        }
        return *this;
    }

    [[nodiscard]] int    get() const noexcept { return m_fd; }
    [[nodiscard]] int    release() noexcept { return std::exchange(m_fd, -1); }
    [[nodiscard]] bool   valid() const noexcept { return m_fd >= 0; }
    [[nodiscard]]        operator bool() const noexcept { return valid(); }

private:
    int m_fd = -1;
};

/// Global state for socket monitoring thread
struct socket_monitor_state
{
    std::unique_ptr<std::thread> thread        = nullptr;
    std::atomic<bool>            running       = { false };
    std::string                  socket_path   = {};
};

auto g_socket_state = socket_monitor_state{};

/// Generates the socket path for current process using PID
[[nodiscard]] std::string
get_socket_path_for_current_process()
{
    return fmt::format("{}{}{}", socket_path_prefix, ::getpid(), socket_path_suffix);
}

/// Sends a response message to client
[[nodiscard]] bool
send_response(int _client_fd, attach_status _status)
{
    auto _response = attach_response_message{
        .op     = static_cast<uint32_t>(attach_op::response),
        .status = static_cast<uint32_t>(_status)
    };

    auto _bytes_sent = ::send(_client_fd, &_response, sizeof(_response), 0);
    return _bytes_sent == static_cast<ssize_t>(sizeof(_response));
}

/// Sends a response message with raw status code (for error codes)
[[nodiscard]] bool
send_response(int _client_fd, uint32_t _status_code)
{
    auto _response = attach_response_message{
        .op     = static_cast<uint32_t>(attach_op::response),
        .status = _status_code
    };

    auto _bytes_sent = ::send(_client_fd, &_response, sizeof(_response), 0);
    return _bytes_sent == static_cast<ssize_t>(sizeof(_response));
}

/// Receives exactly the specified number of bytes from socket
[[nodiscard]] bool
recv_all(int _fd, void* _buffer, size_t _size)
{
    auto* _buf        = static_cast<uint8_t*>(_buffer);
    auto  _total_read = size_t{ 0 };

    while(_total_read < _size)
    {
        auto _bytes_read = ::recv(_fd, _buf + _total_read, _size - _total_read, 0);

        if(_bytes_read <= 0)
        {
            if(_bytes_read == 0)
            {
                LOG(ERROR) << "Connection closed by client";
            }
            else if(errno != EINTR)
            {
                LOG(ERROR) << "Failed to read from client: " << std::strerror(errno);
            }
            return false;
        }
        _total_read += static_cast<size_t>(_bytes_read);
    }
    return true;
}

/// Handles the ATTACH operation from client
void
handle_attach_op(int                         _client_fd,
                 std::string_view            _tool_lib_path,
                 const std::vector<uint8_t>& _env_buffer)
{
    LOG(INFO) << "Received ATTACH operation";
    LOG(INFO) << "Tool library path: " << _tool_lib_path;

    // Parse and log environment buffer contents
    if(!_env_buffer.empty() && _env_buffer.size() >= sizeof(uint32_t))
    {
        auto        _var_count = *reinterpret_cast<const uint32_t*>(_env_buffer.data());
        const auto* _position =
            reinterpret_cast<const char*>(_env_buffer.data() + sizeof(uint32_t));
        const auto* _end =
            reinterpret_cast<const char*>(_env_buffer.data() + _env_buffer.size());

        LOG(INFO) << "Environment variables (" << _var_count << " pairs):";

        for(uint32_t _i = 0; _i < _var_count && _position < _end; ++_i)
        {
            const auto* _name = _position;
            _position += std::strlen(_name) + 1;
            if(_position >= _end) break;

            const auto* _value = _position;
            _position += std::strlen(_value) + 1;

            LOG(INFO) << "  " << _name << "=" << _value;
        }
    }

    // Store tool_lib_path and environment buffer for attach operation
    auto _stored_tool_lib_path = std::string{ _tool_lib_path };
    LOG(INFO) << "Stored tool_lib_path: " << _stored_tool_lib_path;

    auto _stored_env_buffer =
        std::string{ reinterpret_cast<const char*>(_env_buffer.data()), _env_buffer.size() };
    LOG(INFO) << "Stored environment buffer (" << _stored_env_buffer.size() << " bytes)";

    auto _attach_result =
        rocprofiler_register_attach(_stored_env_buffer.c_str(), _stored_tool_lib_path.c_str());

    // Send response with the actual error code
    if(_attach_result != ROCP_REG_SUCCESS)
    {
        LOG(ERROR) << "rocprofiler_register_attach failed with error: " << _attach_result
                   << " (" << rocprofiler_register_error_string(_attach_result) << ")";

        if(send_response(_client_fd, static_cast<uint32_t>(_attach_result)))
        {
            LOG(INFO) << "Sent ATTACH error response with code: " << _attach_result;
        }
        else
        {
            LOG(ERROR) << "Failed to send error response";
        }
    }
    else
    {
        if(send_response(_client_fd, attach_status::success))
        {
            LOG(INFO) << "Sent ATTACH success response";
        }
        else
        {
            LOG(ERROR) << "Failed to send response";
        }
    }
}

/// Handles the DETACH operation from client
void
handle_detach_op(int _client_fd)
{
    LOG(INFO) << "Received DETACH operation";

    auto _detach_result = rocprofiler_register_detach();

    // Send response with the actual error code
    if(_detach_result != ROCP_REG_SUCCESS)
    {
        LOG(ERROR) << "rocprofiler_register_detach failed with error: " << _detach_result
                   << " (" << rocprofiler_register_error_string(_detach_result) << ")";

        if(send_response(_client_fd, static_cast<uint32_t>(_detach_result)))
        {
            LOG(INFO) << "Sent DETACH error response with code: " << _detach_result;
        }
        else
        {
            LOG(ERROR) << "Failed to send error response";
        }
    }
    else
    {
        if(send_response(_client_fd, attach_status::success))
        {
            LOG(INFO) << "Sent DETACH success response";
        }
        else
        {
            LOG(ERROR) << "Failed to send response";
        }
    }
}

/// Main function for socket monitoring thread
void
socket_monitor_thread()
{
    // Generate socket path with current process PID
    g_socket_state.socket_path = get_socket_path_for_current_process();
    const auto* _socket_path   = g_socket_state.socket_path.c_str();

    // Create Unix domain socket
    auto _server_fd = socket_fd_guard{ ::socket(AF_UNIX, SOCK_STREAM, 0) };
    if(!_server_fd)
    {
        LOG(ERROR) << "Failed to create socket: " << std::strerror(errno);
        return;
    }

    // Remove existing socket file if it exists
    ::unlink(_socket_path);

    // Set up server address
    auto _server_addr         = sockaddr_un{};
    _server_addr.sun_family   = AF_UNIX;
    std::strncpy(_server_addr.sun_path, _socket_path, sizeof(_server_addr.sun_path) - 1);

    // Bind socket to the path
    if(::bind(_server_fd.get(),
              reinterpret_cast<sockaddr*>(&_server_addr),
              sizeof(_server_addr)) == -1)
    {
        LOG(ERROR) << "Failed to bind socket: " << std::strerror(errno);
        return;
    }

    // Listen for connections
    constexpr auto max_pending_connections = 5;
    if(::listen(_server_fd.get(), max_pending_connections) == -1)
    {
        LOG(ERROR) << "Failed to listen on socket: " << std::strerror(errno);
        ::unlink(_socket_path);
        return;
    }

    LOG(INFO) << "Socket thread started, listening on " << _socket_path;

    g_socket_state.running = true;

    while(g_socket_state.running)
    {
        auto      _client_addr = sockaddr_un{};
        socklen_t _client_len  = sizeof(_client_addr);

        // Accept client connection
        auto _client_fd = socket_fd_guard{
            ::accept(_server_fd.get(), reinterpret_cast<sockaddr*>(&_client_addr), &_client_len)
        };

        if(!_client_fd)
        {
            if(errno != EINTR && g_socket_state.running)
            {
                LOG(ERROR) << "Failed to accept connection: " << std::strerror(errno);
            }
            continue;
        }

        LOG(INFO) << "Client connected";

        // Read message header
        auto _header = attach_message_header{};
        if(!recv_all(_client_fd.get(), &_header, sizeof(_header)))
        {
            LOG(ERROR) << "Failed to read message header";
            continue;
        }

        LOG(INFO) << "Received header: op=" << _header.op
                  << ", tool_lib_path_len=" << _header.tool_lib_path_len
                  << ", env_buffer_len=" << _header.env_buffer_len;

        // Read tool library path
        auto _tool_lib_path = std::string{};
        if(_header.tool_lib_path_len > 0)
        {
            auto _path_buffer = std::vector<char>(_header.tool_lib_path_len);
            if(!recv_all(_client_fd.get(), _path_buffer.data(), _header.tool_lib_path_len))
            {
                LOG(ERROR) << "Failed to read tool library path";
                continue;
            }
            _tool_lib_path = std::string{ _path_buffer.data() };
        }

        // Read environment buffer
        auto _env_buffer = std::vector<uint8_t>{};
        if(_header.env_buffer_len > 0)
        {
            _env_buffer.resize(_header.env_buffer_len);
            if(!recv_all(_client_fd.get(), _env_buffer.data(), _header.env_buffer_len))
            {
                LOG(ERROR) << "Failed to read environment buffer";
                continue;
            }
        }

        // Handle the operation based on op code
        auto _should_exit = false;
        switch(static_cast<attach_op>(_header.op))
        {
            case attach_op::attach:
            {
                handle_attach_op(_client_fd.get(), _tool_lib_path, _env_buffer);
                break;
            }
            case attach_op::detach:
            {
                LOG(INFO) << "Detaching profiler and exiting socket thread...";
                handle_detach_op(_client_fd.get());
                _should_exit = true;
                break;
            }
            default:
            {
                LOG(ERROR) << "Unknown operation: " << _header.op;
                send_response(_client_fd.get(), attach_status::error);
                break;
            }
        }

        LOG(INFO) << "Client disconnected";

        // Exit the loop if DETACH was received
        if(_should_exit)
        {
            LOG(INFO) << "DETACH received, stopping socket monitor thread";
            g_socket_state.running = false;
            break;
        }
    }

    ::unlink(_socket_path);
    LOG(INFO) << "Socket thread terminated";
}

// ======================================================================================
// End of socket communication utilities
// ======================================================================================

enum rocp_reg_supported_library  // NOLINT(performance-enum-size)
{
    ROCP_REG_HSA = 0,
    ROCP_REG_HIP,
    ROCP_REG_ROCTX,
    ROCP_REG_HIP_COMPILER,
    ROCP_REG_RCCL,
    ROCP_REG_ROCDECODE,
    ROCP_REG_ROCJPEG,
    ROCP_REG_ROCATTACH,
    ROCP_REG_LAST,
};

template <size_t>
struct supported_library_trait
{
    static constexpr bool              specialized  = false;
    static constexpr auto              value        = ROCP_REG_LAST;
    static constexpr const char* const common_name  = nullptr;
    static constexpr const char* const symbol_name  = nullptr;
    static constexpr const char* const library_name = nullptr;
};

template <size_t Idx>
struct rocp_reg_error_message;

#define ROCP_REG_DEFINE_LIBRARY_TRAITS(ENUM, NAME, SYM_NAME, LIB_NAME)                   \
    template <>                                                                          \
    struct supported_library_trait<ENUM>                                                 \
    {                                                                                    \
        static constexpr bool specialized  = true;                                       \
        static constexpr auto value        = ENUM;                                       \
        static constexpr auto common_name  = NAME;                                       \
        static constexpr auto symbol_name  = SYM_NAME;                                   \
        static constexpr auto library_name = LIB_NAME;                                   \
    };

#define ROCP_REG_DEFINE_ERROR_MESSAGE(ENUM, MSG)                                         \
    template <>                                                                          \
    struct rocp_reg_error_message<ENUM>                                                  \
    {                                                                                    \
        static constexpr auto value = MSG;                                               \
    };

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HSA,
                               "hsa",
                               "rocprofiler_register_import_hsa",
                               "libhsa-runtime64.so.[2-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HIP,
                               "hip",
                               "rocprofiler_register_import_hip",
                               "libamdhip64.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCTX,
                               "roctx",
                               "rocprofiler_register_import_roctx",
                               "libroctx64.so.[4-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_HIP_COMPILER,
                               "hip_compiler",
                               "rocprofiler_register_import_hip_compiler",
                               "libamdhip64.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_RCCL,
                               "rccl",
                               "rocprofiler_register_import_rccl",
                               "librccl.so.[6-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCDECODE,
                               "rocdecode",
                               "rocprofiler_register_import_rocdecode",
                               "librocdecode.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCJPEG,
                               "rocjpeg",
                               "rocprofiler_register_import_rocjpeg",
                               "librocjpeg.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_LIBRARY_TRAITS(ROCP_REG_ROCATTACH,
                               "rocattach",
                               "rocprofiler_register_import_attach",
                               "librocprofiler-sdk-attach.so.[0-9]($|\\.[0-9\\.]+)")

ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_SUCCESS, "Success")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_NO_TOOLS, "rocprofiler-register found no tools")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_DEADLOCK, "rocprofiler-register deadlocked")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_BAD_API_TABLE_LENGTH,
                              "Library passed an invalid number of API tables")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_UNSUPPORTED_API, "Library's API is not supported")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_INVALID_API_ADDRESS,
                              "Invalid API address (secure mode enabled)")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_ROCPROFILER_ERROR,
                              "Unspecified rocprofiler-register error")
ROCP_REG_DEFINE_ERROR_MESSAGE(
    ROCP_REG_EXCESS_API_INSTANCES,
    "Too many instances of the same library API were registered")
ROCP_REG_DEFINE_ERROR_MESSAGE(
    ROCP_REG_INVALID_ARGUMENT,
    "rocprofiler-register API function was provided an invalid argument")
ROCP_REG_DEFINE_ERROR_MESSAGE(ROCP_REG_ATTACHMENT_NOT_AVAILABLE,
                              "rocprofiler-register attach was invoked, but the "
                              "attachment library was never loaded.")

auto
get_this_library_path()
{
    auto _this_lib_path = binary::get_linked_path(rocprofiler_register_lib_name,
                                                  { RTLD_NOLOAD | RTLD_LAZY });
    LOG_IF(FATAL, !_this_lib_path)
        << rocprofiler_register_lib_name
        << " could not locate itself in the list of loaded libraries";
    return fs::path{ *_this_lib_path }.parent_path().string();
}

template <size_t Idx, size_t... Tail>
constexpr auto
rocprofiler_register_error_string(rocprofiler_register_error_code_t _ec,
                                  std::index_sequence<Idx, Tail...>)
{
    if(_ec == Idx) return rocp_reg_error_message<Idx>::value;

    if constexpr(sizeof...(Tail) > 0)
    {
        return rocprofiler_register_error_string(_ec, std::index_sequence<Tail...>{});
    }
    else
    {
        return "rocprofiler_register_unknown_error";
    }
}

struct rocp_import
{
    rocp_reg_supported_library library_idx  = ROCP_REG_LAST;
    std::string_view           common_name  = {};
    std::string_view           symbol_name  = {};
    std::string_view           library_name = {};
};

template <size_t... Idx>
auto rocp_reg_get_imports(std::index_sequence<Idx...>)
{
    auto _data        = std::vector<rocp_import>{};
    auto _import_scan = [&_data](auto _info) {
        if(_info.specialized)
        {
            _data.emplace_back(rocp_import{
                _info.value, _info.common_name, _info.symbol_name, _info.library_name });
        }
    };

    (_import_scan(supported_library_trait<Idx>{}), ...);
    return _data;
}

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(std::string _rocp_reg_lib);

struct rocp_scan_data
{
    void*                       handle           = nullptr;
    rocprofiler_set_api_table_t set_api_table_fn = nullptr;
    rocprofiler_attach_func_t   attach_fn        = nullptr;
    rocprofiler_detach_func_t   detach_fn        = nullptr;
};

auto existing_scanned_data = rocp_scan_data{};

rocp_scan_data
rocp_reg_scan_for_tools()
{
    auto* _configure_func = dlsym(RTLD_DEFAULT, "rocprofiler_configure");
    auto  _rocp_tool_libs = common::get_env("ROCP_TOOL_LIBRARIES", std::string{});
    auto  _rocp_reg_lib = common::get_env("ROCPROFILER_REGISTER_LIBRARY", std::string{});
    bool  _force_tool =
        common::get_env("ROCPROFILER_REGISTER_FORCE_LOAD",
                        !_rocp_reg_lib.empty() || !_rocp_tool_libs.empty());

    bool _found_tool =
        (rocprofiler_configure != nullptr || _configure_func != nullptr || _force_tool);

    static void*                       rocprofiler_lib_handle    = nullptr;
    static rocprofiler_set_api_table_t rocprofiler_lib_config_fn = nullptr;
    static rocprofiler_attach_func_t   rocprofiler_lib_attach_fn = nullptr;
    static rocprofiler_detach_func_t   rocprofiler_lib_detach_fn = nullptr;

    if(_found_tool)
    {
        if(rocprofiler_lib_handle && rocprofiler_lib_config_fn)
            return rocp_scan_data{ rocprofiler_lib_handle,
                                   rocprofiler_lib_config_fn,
                                   rocprofiler_lib_attach_fn,
                                   rocprofiler_lib_detach_fn };

        if(_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;

        std::tie(rocprofiler_lib_handle,
                 rocprofiler_lib_config_fn,
                 rocprofiler_lib_attach_fn,
                 rocprofiler_lib_detach_fn) = rocp_load_rocprofiler_lib(_rocp_reg_lib);

        LOG_IF(FATAL, !rocprofiler_lib_config_fn)
            << rocprofiler_lib_register_entrypoint << " not found. Tried to dlopen "
            << _rocp_reg_lib;
    }
    else if(_found_tool && rocprofiler_set_api_table)
    {
        rocprofiler_lib_config_fn = &rocprofiler_set_api_table;
        rocprofiler_lib_attach_fn = &rocprofiler_attach;
        rocprofiler_lib_detach_fn = &rocprofiler_detach;
    }

    return rocp_scan_data{ rocprofiler_lib_handle,
                           rocprofiler_lib_config_fn,
                           rocprofiler_lib_attach_fn,
                           rocprofiler_lib_detach_fn };
}

void*
get_library_handle(std::string_view _rocp_reg_lib)
{
    void* rocprofiler_lib_handle = nullptr;

    if(_rocp_reg_lib.empty()) return nullptr;

    auto _rocp_reg_lib_path       = fs::path{ _rocp_reg_lib };
    auto _rocp_reg_lib_path_fname = _rocp_reg_lib_path.filename();
    auto _rocp_reg_lib_path_abs =
        (_rocp_reg_lib_path.is_absolute())
            ? _rocp_reg_lib_path
            : (fs::path{ get_this_library_path() } / _rocp_reg_lib_path);

    // check to see if the rocprofiler library is already loaded
    rocprofiler_lib_handle = dlopen(_rocp_reg_lib_path.c_str(), RTLD_NOLOAD | RTLD_LAZY);

    if(rocprofiler_lib_handle)
    {
        LOG(INFO) << "loaded " << _rocp_reg_lib << " library at "
                  << _rocp_reg_lib_path.string() << " (handle=" << rocprofiler_lib_handle
                  << ") via RTLD_NOLOAD | RTLD_LAZY";
    }

    // try to load with the given path
    if(!rocprofiler_lib_handle)
    {
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);

        if(rocprofiler_lib_handle)
        {
            LOG(INFO) << "loaded " << _rocp_reg_lib << " library at "
                      << _rocp_reg_lib_path.string()
                      << " (handle=" << rocprofiler_lib_handle
                      << ") via RTLD_GLOBAL | RTLD_LAZY";
        }
    }

    // try to load with the absoulte path
    if(!rocprofiler_lib_handle)
    {
        _rocp_reg_lib_path = _rocp_reg_lib_path_abs;
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    // try to load with the basename path
    if(!rocprofiler_lib_handle)
    {
        _rocp_reg_lib_path = _rocp_reg_lib_path_fname;
        rocprofiler_lib_handle =
            dlopen(_rocp_reg_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    LOG(INFO) << "loaded " << _rocp_reg_lib << " library at "
              << _rocp_reg_lib_path.string() << " (handle=" << rocprofiler_lib_handle
              << ")";

    LOG_IF(WARNING, rocprofiler_lib_handle == nullptr)
        << _rocp_reg_lib << " failed to load\n";

    return rocprofiler_lib_handle;
}

rocp_set_api_table_data_t
rocp_load_rocprofiler_lib(std::string _rocp_reg_lib)
{
    void*                       rocprofiler_lib_handle    = nullptr;
    rocprofiler_set_api_table_t rocprofiler_lib_config_fn = nullptr;
    rocprofiler_attach_func_t   rocprofiler_lib_attach_fn = nullptr;
    rocprofiler_detach_func_t   rocprofiler_lib_detach_fn = nullptr;

    if(rocprofiler_set_api_table)
    {
        rocprofiler_lib_config_fn = &rocprofiler_set_api_table;
        rocprofiler_lib_attach_fn = &rocprofiler_attach;
        rocprofiler_lib_detach_fn = &rocprofiler_detach;
    }

    // return if found via LD_PRELOAD
    if(rocprofiler_lib_config_fn)
        return std::make_tuple(rocprofiler_lib_handle,
                               rocprofiler_lib_config_fn,
                               rocprofiler_lib_attach_fn,
                               rocprofiler_lib_detach_fn);

    // look to see if entrypoint function is already a symbol
    *(void**) (&rocprofiler_lib_config_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_register_entrypoint);
    *(void**) (&rocprofiler_lib_attach_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_attach_entrypoint);
    *(void**) (&rocprofiler_lib_detach_fn) =
        dlsym(RTLD_DEFAULT, rocprofiler_lib_detach_entrypoint);

    // return if found via RTLD_DEFAULT
    if(rocprofiler_lib_config_fn)
    {
        return std::make_tuple(rocprofiler_lib_handle,
                               rocprofiler_lib_config_fn,
                               rocprofiler_lib_attach_fn,
                               rocprofiler_lib_detach_fn);
    }

    if(_rocp_reg_lib.empty()) _rocp_reg_lib = rocprofiler_lib_name;

    rocprofiler_lib_handle = get_library_handle(_rocp_reg_lib);

    *(void**) (&rocprofiler_lib_config_fn) =
        dlsym(rocprofiler_lib_handle, rocprofiler_lib_register_entrypoint);

    *(void**) (&rocprofiler_lib_attach_fn) =
        dlsym(rocprofiler_lib_handle, rocprofiler_lib_attach_entrypoint);

    *(void**) (&rocprofiler_lib_detach_fn) =
        dlsym(rocprofiler_lib_handle, rocprofiler_lib_detach_entrypoint);

    LOG_IF(WARNING, rocprofiler_lib_config_fn == nullptr)
        << _rocp_reg_lib << " (handle=" << rocprofiler_lib_handle << ") did not contain '"
        << rocprofiler_lib_register_entrypoint << "' symbol";

    LOG_IF(INFO, rocprofiler_lib_config_fn != nullptr)
        << "Found " << rocprofiler_lib_register_entrypoint << " symbol";

    LOG_IF(INFO, rocprofiler_lib_attach_fn != nullptr)
        << "Found " << rocprofiler_lib_attach_entrypoint << " symbol";

    LOG_IF(INFO, rocprofiler_lib_detach_fn != nullptr)
        << "Found " << rocprofiler_lib_detach_entrypoint << " symbol";

    return std::make_tuple(rocprofiler_lib_handle,
                           rocprofiler_lib_config_fn,
                           rocprofiler_lib_attach_fn,
                           rocprofiler_lib_detach_fn);
}

struct registered_library_api_table
{
    bool                               propagated     = false;
    const char*                        common_name    = nullptr;
    rocprofiler_register_import_func_t import_func    = nullptr;
    uint32_t                           lib_version    = 0;
    std::vector<void*>                 api_tables     = {};
    uint64_t                           instance_value = 0;
};

constexpr auto instance_bits     = sizeof(uint64_t) * 8;  // bits in instance_counters
constexpr auto max_instances     = instance_bits * ROCP_REG_LAST;
constexpr auto library_seq       = std::make_index_sequence<ROCP_REG_LAST>{};
auto           global_count      = std::atomic<uint32_t>{ 0 };
auto           import_info       = rocp_reg_get_imports(library_seq);
auto           instance_counters = std::array<std::atomic_uint64_t, ROCP_REG_LAST>{};
auto           registered =
    std::array<std::optional<registered_library_api_table>, max_instances>{};

struct scoped_count
{
    scoped_count()
    : value{ ++global_count }
    { }

    ~scoped_count() { --global_count; }

    scoped_count(const scoped_count&)     = delete;
    scoped_count(scoped_count&&) noexcept = delete;
    scoped_count& operator=(const scoped_count&) = delete;
    scoped_count& operator=(scoped_count&&) noexcept = delete;

    uint32_t value = 0;
};

std::optional<registered_library_api_table>*
rocp_add_registered_library_api_table(const char*                        common_name,
                                      rocprofiler_register_import_func_t import_func,
                                      uint32_t                           lib_version,
                                      void**                             api_tables,
                                      uint64_t                           api_tables_len,
                                      uint64_t                           instance_val)
{
    LOG(INFO) << fmt::format("rocprofiler-register library api table registration:\n\t-"
                             "name: {}\n\t- version: {}\n\t- # tables: {}",
                             common_name,
                             lib_version,
                             api_tables_len);

    for(auto& itr : registered)
    {
        if(!itr)
        {
            auto _tables = std::vector<void*>{};
            _tables.reserve(api_tables_len);
            for(uint64_t i = 0; i < api_tables_len; ++i)
                _tables.emplace_back(api_tables[i]);

            itr = registered_library_api_table{
                false,       common_name,        import_func,
                lib_version, std::move(_tables), instance_val
            };
            return &itr;
        }
    }

    return nullptr;
}

rocprofiler_register_error_code_t
rocp_invoke_registrations(bool invoke_all)
{
    auto _count = scoped_count{};
    if(_count.value > 1) return ROCP_REG_DEADLOCK;

    for(auto& itr : registered)
    {
        if(itr && (!itr->propagated || invoke_all))
        {
            auto _scan_result = rocp_reg_scan_for_tools();

            // rocprofiler_set_api_table has been found and we have pass the API data
            auto _activate_rocprofiler = (_scan_result.set_api_table_fn != nullptr);

            if(_activate_rocprofiler)
            {
                existing_scanned_data = _scan_result;
                auto _ret             = _scan_result.set_api_table_fn(itr->common_name,
                                                          itr->lib_version,
                                                          itr->instance_value,
                                                          itr->api_tables.data(),
                                                          itr->api_tables.size());
                if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;
                itr->propagated = true;
            }
        }
    }

    return ROCP_REG_SUCCESS;
}

void
load_environment_buffer(const char* environment_buffer)
{
    // environment_buffer is a null-character delimited list of name value pairs.
    // Each name and value is delimited separately.
    // The first 4 bytes contain a uint32_t count of pairs.

    if(!environment_buffer)
    {
        LOG(WARNING) << "Attachment was invoked with no environment variables provided "
                        "for what to trace.";
        return;
    }

    const uint32_t pair_count = *reinterpret_cast<const uint32_t*>(environment_buffer);
    const char*    position   = environment_buffer + sizeof(uint32_t);
    for(uint32_t pair_idx = 0; pair_idx < pair_count; ++pair_idx)
    {
        const char* name = position;
        position += strlen(name) + 1;
        const char* value = position;
        position += strlen(value) + 1;

        LOG(INFO) << "Attachment adding environment variable: " << name << "=" << value;
        setenv(name, value, 1);
    }
}

bool
is_attachment_library_registered()
{
    for(const auto& itr : registered)
    {
        if(std::string_view{ itr->common_name } ==
           supported_library_trait<ROCP_REG_ROCATTACH>::common_name)
        {
            return true;
        }
    }
    return false;
}

constexpr auto offset_factor = 64 / std::max<size_t>(ROCP_REG_LAST, 8);

rocprofiler_register_error_code_t
register_functor(const char*                                 common_name,
                 rocprofiler_register_import_func_t          import_func,
                 uint32_t                                    lib_version,
                 void**                                      api_tables,
                 uint64_t                                    api_table_length,
                 rocprofiler_register_library_indentifier_t* register_id)
{
    rocp_import* _import_match = nullptr;
    for(auto& itr : import_info)
    {
        if(itr.common_name == common_name)
        {
            _import_match = &itr;
            break;
        }
    }

    // not a supported library name
    if(!_import_match || _import_match->library_idx == ROCP_REG_LAST)
        return ROCP_REG_UNSUPPORTED_API;

    if(instance_counters.at(_import_match->library_idx) >= offset_factor)
        return ROCP_REG_EXCESS_API_INSTANCES;

    auto  _instance_val = instance_counters.at(_import_match->library_idx)++;
    auto& _bits         = *reinterpret_cast<bitset_t*>(&register_id->handle);
    _bits = bitset_t{ (offset_factor * _import_match->library_idx) + _instance_val };

    auto* reginfo = rocp_add_registered_library_api_table(common_name,
                                                          import_func,
                                                          lib_version,
                                                          api_tables,
                                                          api_table_length,
                                                          _instance_val);

    LOG_IF(WARNING, !reginfo) << fmt::format(
        "rocprofiler-register failed to create registration info for "
        "{} version {} (instance {})",
        common_name,
        lib_version,
        _instance_val);

    return ROCP_REG_SUCCESS;
};
}  // namespace

extern "C" {
rocprofiler_register_error_code_t
rocprofiler_register_library_api_table(
    const char*                                 common_name,
    rocprofiler_register_import_func_t          import_func,
    uint32_t                                    lib_version,
    void**                                      api_tables,
    uint64_t                                    api_table_length,
    rocprofiler_register_library_indentifier_t* register_id)
{
    if(api_table_length < 1) return ROCP_REG_BAD_API_TABLE_LENGTH;

    rocprofiler_register::logging::initialize();

    // rocprofiler-register is disabled via environment
    if(!common::get_env("ROCPROFILER_REGISTER_ENABLED", true))
    {
        LOG(INFO) << "rocprofiler-register disabled via ROCPROFILER_REGISTER_ENABLED=0";
        return ROCP_REG_NO_TOOLS;
    }

    auto _count = scoped_count{};
    if(_count.value > 1) return ROCP_REG_DEADLOCK;

    auto _scan_result = rocp_reg_scan_for_tools();

    // rocprofiler library is dlopened and we have the functor to pass the API data
    auto _activate_rocprofiler = (_scan_result.set_api_table_fn != nullptr);

#if defined(ROCP_REG_DEFAULT_ATTACHMENT) && ROCP_REG_DEFAULT_ATTACHMENT != 0
    constexpr auto default_attachment_enabled = true;
#else
    constexpr auto default_attachment_enabled = false;
#endif

    auto _attachment_enabled =
        common::get_env("ROCP_TOOL_ATTACH", default_attachment_enabled);

    rocp_import* _import_match = nullptr;
    for(auto& itr : import_info)
    {
        if(itr.common_name == common_name)
        {
            _import_match = &itr;
            break;
        }
    }

    // not a supported library name
    if(!_import_match || _import_match->library_idx == ROCP_REG_LAST)
        return ROCP_REG_UNSUPPORTED_API;

    if(import_func != nullptr &&
       common::get_env<bool>("ROCPROFILER_REGISTER_SECURE", false))
    {
        auto _import_func_addr  = reinterpret_cast<uintptr_t>(import_func);
        auto _segment_addresses = binary::get_segment_addresses();
        auto _in_address_range  = [](uintptr_t                                 _addr,
                                    const std::vector<binary::address_range>& _range) {
            for(auto ritr : _range)
            {
                if(_addr >= ritr.start && _addr < ritr.last) return true;
            }
            return false;
        };

        // check that the address of the import function is within the expected library
        // name
        bool _valid_addr = false;
        for(const auto& itr : _segment_addresses)
        {
            if(_in_address_range(_import_func_addr, itr.ranges))
            {
                if(std::regex_search(fs::path{ itr.filepath }.filename().string(),
                                     std::regex{ _import_match->library_name.data() }))
                {
                    _valid_addr = true;
                }
            }
        }

        // the library provided
        if(!_valid_addr) return ROCP_REG_INVALID_API_ADDRESS;
    }

    // if ROCP_REG_LAST > 8, then we can no longer encode 8 instances per lib
    // because we ran out of bits (i.e. max of 8 * 8 = 64)
    static_assert((offset_factor * ROCP_REG_LAST) <= sizeof(uint64_t) * 8,
                  "ROCP_REG_LAST has exceeded the max allowable size");

    // too many instances of the same library
    if(instance_counters.at(_import_match->library_idx) >= offset_factor)
        return ROCP_REG_EXCESS_API_INSTANCES;

    auto  _instance_val = instance_counters.at(_import_match->library_idx)++;
    auto& _bits         = *reinterpret_cast<bitset_t*>(&register_id->handle);
    _bits = bitset_t{ (offset_factor * _import_match->library_idx) + _instance_val };

    // if attachment is enabled the HSA API table should be forwarded to the attachment
    // library
    if(!_activate_rocprofiler && _attachment_enabled &&
       _import_match->library_idx == ROCP_REG_HSA)
    {
        void* attachlibrary = get_library_handle(rocprofiler_attach_lib_name);
        if(!attachlibrary)
        {
            LOG(ERROR)
                << "Proxy queues for attachment are enabled, but the attach library "
                   "was not found or able to be loaded. The attaching profiler will not "
                   "be able to profile anything that requires proxy queues.";
            return ROCP_REG_NO_TOOLS;
        }
        rocprofiler_attach_set_api_table_t rocprofiler_attach_set_api_table_fn;
        *(void**) (&rocprofiler_attach_set_api_table_fn) =
            dlsym(attachlibrary, rocprofiler_attach_lib_register_entrypoint);

        if(!rocprofiler_attach_set_api_table_fn)
        {
            LOG(ERROR)
                << "Proxy queues for attachment are enabled, but the attach library's "
                   "entry point was not found. The attaching profiler will not be able "
                   "to profile anything that requires proxy queues.";
            return ROCP_REG_NO_TOOLS;
        }

        // Pass a functor to the attach library that it can use to pass back its own API
        // table to us. This approach simplifies the interface and avoids having to modify
        // the deadlock protection of this function.

        auto _ret = rocprofiler_attach_set_api_table_fn(common_name,
                                                        lib_version,
                                                        _instance_val,
                                                        api_tables,
                                                        api_table_length,
                                                        &register_functor);
        if(_ret != 0)
        {
            LOG(ERROR) << "Proxy queues for attachment are enabled, but attach library "
                          "registration returned an error: "
                       << _ret
                       << ". The attaching profiler may not be able to profile anything "
                          "that requires proxy queues.";
            return ROCP_REG_ROCPROFILER_ERROR;
        }

        LOG(INFO) << "Successfully registered for proxy queue creation";
    }

    auto* reginfo = rocp_add_registered_library_api_table(common_name,
                                                          import_func,
                                                          lib_version,
                                                          api_tables,
                                                          api_table_length,
                                                          _instance_val);

    LOG_IF(WARNING, !reginfo) << fmt::format(
        "rocprofiler-register failed to create registration info for "
        "{} version {} (instance {})",
        common_name,
        lib_version,
        _instance_val);

    if(_bits.to_ulong() != register_id->handle)
        throw std::runtime_error("error encoding register_id");

    if(_activate_rocprofiler)
    {
        auto _ret = _scan_result.set_api_table_fn(
            common_name, lib_version, _instance_val, api_tables, api_table_length);
        if(_ret != 0) return ROCP_REG_ROCPROFILER_ERROR;

        if(reginfo) (*reginfo)->propagated = true;
    }
    else
    {
        return ROCP_REG_NO_TOOLS;
    }

    return ROCP_REG_SUCCESS;
}

const char*
rocprofiler_register_error_string(rocprofiler_register_error_code_t _ec)
{
    return rocprofiler_register_error_string(
        _ec, std::make_index_sequence<ROCP_REG_ERROR_CODE_END>{});
}

rocprofiler_register_error_code_t
rocprofiler_register_iterate_registration_info(
    rocprofiler_register_registration_info_cb_t callback,
    void*                                       data)
{
    for(const auto& itr : registered)
    {
        if(itr)
        {
            auto _info = rocprofiler_register_registration_info_t{
                .size             = sizeof(rocprofiler_register_registration_info_t),
                .common_name      = itr->common_name,
                .lib_version      = itr->lib_version,
                .api_table_length = itr->api_tables.size()
            };
            // invoke callback and break if the caller does not return zero
            if(callback(&_info, data) != ROCP_REG_SUCCESS) break;
        }
    }

    return ROCP_REG_SUCCESS;
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_invoke_nonpropagated_registrations()
{
    return rocp_invoke_registrations(false);
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations() ROCPROFILER_REGISTER_PUBLIC_API;

// This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_invoke_prestore_loads() ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_invoke_all_registrations()
{
    return rocp_invoke_registrations(true);
}

rocprofiler_register_error_code_t
rocprofiler_register_thread_init() ROCPROFILER_REGISTER_PUBLIC_API;

rocprofiler_register_error_code_t
rocprofiler_register_thread_init()
{
    // check if thread is already running
    if(g_socket_state.thread && g_socket_state.running.load())
    {
        LOG(INFO) << "Socket monitoring thread is already running";
        return ROCP_REG_SUCCESS;
    }

    try
    {
        // create and start the socket monitoring thread
        g_socket_state.thread = std::make_unique<std::thread>(socket_monitor_thread);

        // detach the thread so it can run independently
        g_socket_state.thread->detach();

        LOG(INFO) << "Socket monitoring thread started successfully";
        return ROCP_REG_SUCCESS;
    }
    catch(const std::exception& _e)
    {
        LOG(ERROR) << "Failed to start socket monitoring thread: " << _e.what();
        return ROCP_REG_ROCPROFILER_ERROR;
    }
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_attach(const char* environment_buffer, const char* tool_lib_path)
{
    // If the attachment library has not been loaded when attach is called, tracing
    // that relies on proxy queues will fail (e.g. kernel tracing).
    // Log error and abort.
    if(!is_attachment_library_registered())
    {
        LOG(ERROR)
            << "rocprofiler-register attach was invoked, but the rocprofiler-attach "
               "library was never loaded. Start the app with environment variable "
               "ROCP_TOOL_ATTACH=1 or build rocprofiler-register with cmake option "
               "ROCP_REG_DEFAULT_ATTACHMENT=ON";
        return ROCP_REG_ATTACHMENT_NOT_AVAILABLE;
    }

    static auto prev_tool_lib_path = std::string{};

    // tool_lib_path is declared with non-null attribute
    if(!prev_tool_lib_path.empty() && prev_tool_lib_path != tool_lib_path)
    {
        LOG(WARNING) << "rocprofiler_register_attach invoked with a different "
                        "tool_lib_path ("
                     << tool_lib_path
                     << ") than a previous attach (previous=" << prev_tool_lib_path
                     << "). This is not supported.";
        return ROCP_REG_INVALID_ARGUMENT;
    }

    LOG(INFO) << "rocprofiler_register_attach started with tool_lib_path: "
              << tool_lib_path;

    // Set default tool library path if not provided
    setenv("ROCPROFILER_REGISTER_TOOL_ATTACHED", "1", 1);

    LOG_IF(FATAL, tool_lib_path == nullptr)
        << "ROCP_TOOL_LIBRARIES is set, but tool_lib_path is NULL. "
           "This is not supported. Please provide a valid tool library path.";

    // TODO: should save old environment variables if they get overwritten and restore
    // them on detach
    // load_environment_buffer(environment_buffer);

    // Use provided path. Must come after load_environment_buffer to ensure override
    setenv("ROCP_TOOL_LIBRARIES", tool_lib_path, 1);
    LOG(INFO) << "Using provided tool library: " << tool_lib_path;

    // TODO: should save old environment variables if they get overwritten and restore
    // them on detach
    load_environment_buffer(environment_buffer);

    // No previous tool library was attached
    if(prev_tool_lib_path.empty())
    {
        auto status = rocprofiler_register_invoke_all_registrations();
        if(status != ROCP_REG_SUCCESS)
        {
            LOG(ERROR) << "error during invoke_all_registrations: " << status;
            return status;
        }
        prev_tool_lib_path = tool_lib_path;
    }

    if(existing_scanned_data.attach_fn == nullptr) return ROCP_REG_NO_TOOLS;

    LOG(INFO) << "rocprofiler-sdk attach starting...";
    auto _ret = existing_scanned_data.attach_fn();

    LOG(INFO) << "rocprofiler-sdk attach completed.";

    return (_ret == 0) ? ROCP_REG_SUCCESS : ROCP_REG_ROCPROFILER_ERROR;
}

//
//  This function can be invoked by ptrace
rocprofiler_register_error_code_t
rocprofiler_register_detach()
{
    LOG(INFO) << "rocprofiler_register_detach started";

    if(!is_attachment_library_registered())
    {
        LOG(ERROR)
            << "rocprofiler-register detach was invoked, but the rocprofiler-attach "
               "library was never loaded. Start the app with environment variable "
               "ROCP_TOOL_ATTACH=1 or build rocprofiler-register with cmake option "
               "ROCP_REG_DEFAULT_ATTACHMENT=ON";
        return ROCP_REG_ATTACHMENT_NOT_AVAILABLE;
    }

    if(existing_scanned_data.detach_fn)
    {
        LOG(INFO) << "rocprofiler-sdk detach starting...";
        existing_scanned_data.detach_fn();
        LOG(INFO) << "rocprofiler-sdk detach completed.";
    }
    else
    {
        LOG(ERROR) << "detach entry point is NULL";
        return ROCP_REG_NO_TOOLS;
    }

    return ROCP_REG_SUCCESS;
    // auto _scan_result = rocp_reg_scan_for_tools();
    // if(!_scan_result.detach_fn) return ROCP_REG_NO_TOOLS;

    // LOG(INFO) << "rocprofiler-sdk detach starting...";
    // auto _ret = _scan_result.detach_fn();

    // LOG(INFO) << "rocprofiler-sdk detach completed.";
    // return (_ret == 0) ? ROCP_REG_SUCCESS : ROCP_REG_ROCPROFILER_ERROR;
}
}
