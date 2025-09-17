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
#include "details/filesystem.hpp"

#include "lib/common/logging.hpp"

#include <dlfcn.h>
#include <fcntl.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/ptrace.h>
#include <sys/stat.h>
#include <sys/user.h>
#include <sys/wait.h>
#include <unistd.h>

#include <fstream>
#include <type_traits>

#define AT_ENTRY 9 /* Entry point of program */

// ptrace memory operations use "word length" which is dependent on system architecture.
static_assert(sizeof(void*) == 8);

// In addition, this file uses x64 assembly which is inherently platform dependent.
#ifndef __x86_64__
static_assert(false);
#endif

namespace
{
/* Copied from glibc's elf.h.  */
typedef struct
{
    uint64_t a_type; /* Entry type */
    union
    {
        uint64_t a_val; /* Integer value */
                        /* We use to have pointer elements added here.  We cannot do that,
                       though, since it does not work when using 32-bit definitions
                       on 64-bit platforms and vice versa.  */
    } a_un;
} Elf64_auxv_t;

// Very limited list of operations for logging only.
constexpr const char*
ptrace_op_name(__ptrace_request op)
{
    switch(op)
    {
        case PTRACE_SEIZE: return "PTRACE_SEIZE";
        case PTRACE_DETACH: return "PTRACE_DETACH";
        case PTRACE_POKEDATA: return "PTRACE_POKEDATA";
        case PTRACE_PEEKDATA: return "PTRACE_PEEKDATA";
        case PTRACE_INTERRUPT: return "PTRACE_INTERRUPT";
        case PTRACE_GETREGS: return "PTRACE_GETREGS";
        case PTRACE_SETREGS: return "PTRACE_SETREGS";
        case PTRACE_CONT: return "PTRACE_CONT";
        default: return "unknown op";
    }
}

// Boilerplate around ptrace calls.
// If an error occurs, logs the error and returns false.
#define PTRACE_CALL(op, pid, addr, data)                                                           \
    ROCP_TRACE << "ptrace call params(" << ptrace_op_name(op) << "(" << op << "), " << pid << ", " \
               << (uint64_t) addr << ", " << (uint64_t) data << ")";                               \
    if(errno = 0, ptrace(op, pid, addr, data); errno != 0)                                         \
    {                                                                                              \
        ROCP_ERROR << "ptrace call failed. errno: " << errno << " - " << strerror(errno)           \
                   << " params(" << ptrace_op_name(op) << "(" << op << "), " << pid << ", "        \
                   << (uint64_t) addr << ", " << (uint64_t) data << ")";                           \
        return false;                                                                              \
    }

// Changes the order of parameters for PEEKDATA so it can be used like other operations.
// value should be uint64_t
#define PTRACE_PEEK(pid, addr, read_value)                                                         \
    static_assert(std::is_same<decltype(read_value), uint64_t>::value);                            \
    ROCP_TRACE << "ptrace call params(PTRACE_PEEKDATA(2), " << pid << ", " << (uint64_t) addr      \
               << ", 0)";                                                                          \
    if(errno = 0, read_value = ptrace(PTRACE_PEEKDATA, pid, addr, NULL); errno != 0)               \
    {                                                                                              \
        ROCP_ERROR << "ptrace call failed. errno: " << errno << " params(PTRACE_PEEKDATA(2), "     \
                   << pid << ", " << (uint64_t) addr << ", 0)";                                    \
        return false;                                                                              \
    }

using open_modes_vec_t = std::vector<int>;

void
get_auxv_entry(int pid, size_t& entry_addr)
{
    char      filename[PATH_MAX];
    int       fd{};
    const int auxv_size = sizeof(Elf64_auxv_t);
    char      buf[sizeof(Elf64_auxv_t)]; /* The larger of the two.  */

    snprintf(filename, sizeof filename, "/proc/%d/auxv", pid);

    fd = open(filename, O_RDONLY);
    if(fd < 0) ROCP_ERROR << "Unable to open auxv file " << filename;

    entry_addr = 0;
    while(read(fd, buf, auxv_size) == auxv_size && entry_addr == 0)
    {
        Elf64_auxv_t* const aux = (Elf64_auxv_t*) buf;

        if(aux->a_type == AT_ENTRY)
        {
            entry_addr = aux->a_un.a_val;
        }
    }

    close(fd);

    if(entry_addr == 0)
    {
        ROCP_ERROR << "Unexpected mising AT_ENTRY for " << filename;
    }
    ROCP_TRACE << "Entry address found to be " << entry_addr << " from " << filename;
}

std::optional<std::string>
get_linked_path(std::string_view _name, open_modes_vec_t&& _open_modes)
{
    const open_modes_vec_t default_link_open_modes = {(RTLD_LAZY | RTLD_NOLOAD)};
    if(_name.empty()) return fs::current_path().string();

    if(_open_modes.empty()) _open_modes = default_link_open_modes;

    void* _handle = nullptr;
    bool  _noload = false;
    for(auto _mode : _open_modes)
    {
        _handle = dlopen(_name.data(), _mode);
        _noload = (_mode & RTLD_NOLOAD) == RTLD_NOLOAD;
        if(_handle) break;
    }

    if(_handle)
    {
        struct link_map* _link_map = nullptr;
        dlinfo(_handle, RTLD_DI_LINKMAP, &_link_map);
        if(_link_map != nullptr && !std::string_view{_link_map->l_name}.empty())
        {
            return fs::absolute(fs::path{_link_map->l_name}).string();
        }
        if(_noload == false) dlclose(_handle);
    }

    return std::nullopt;
}

auto
get_this_library_path()
{
    auto _this_lib_path = get_linked_path("librocprofv3-attach.so.1", {RTLD_NOLOAD | RTLD_LAZY});
    LOG_IF(FATAL, !_this_lib_path) << "librocprofv3-attach.so.1"
                                   << " could not locate itself in the list of loaded libraries";
    return fs::path{*_this_lib_path}.parent_path().string();
}

void*
get_library_handle(std::string_view _lib_name)
{
    void* _lib_handle = nullptr;

    if(_lib_name.empty()) return nullptr;

    auto _lib_path       = fs::path{_lib_name};
    auto _lib_path_fname = _lib_path.filename();
    auto _lib_path_abs =
        (_lib_path.is_absolute()) ? _lib_path : (fs::path{get_this_library_path()} / _lib_path);

    // check to see if the rocprofiler library is already loaded
    _lib_handle = dlopen(_lib_path.c_str(), RTLD_NOLOAD | RTLD_LAZY);

    if(_lib_handle)
    {
        LOG(INFO) << "loaded " << _lib_name << " library at " << _lib_path.string()
                  << " (handle=" << _lib_handle << ") via RTLD_NOLOAD | RTLD_LAZY";
    }

    // try to load with the given path
    if(!_lib_handle)
    {
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);

        if(_lib_handle)
        {
            LOG(INFO) << "loaded " << _lib_name << " library at " << _lib_path.string()
                      << " (handle=" << _lib_handle << ") via RTLD_GLOBAL | RTLD_LAZY";
        }
    }

    // try to load with the absoulte path
    if(!_lib_handle)
    {
        _lib_path   = _lib_path_abs;
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    // try to load with the basename path
    if(!_lib_handle)
    {
        _lib_path   = _lib_path_fname;
        _lib_handle = dlopen(_lib_path.c_str(), RTLD_GLOBAL | RTLD_LAZY);
    }

    LOG(INFO) << "loaded " << _lib_name << " library at " << _lib_path.string()
              << " (handle=" << _lib_handle << ")";

    LOG_IF(WARNING, _lib_handle == nullptr) << _lib_name << " failed to load\n";

    return _lib_handle;
}

}  // namespace

namespace rocprofiler
{
namespace attach
{
PTraceSession::PTraceSession(int _pid)
: m_pid{_pid}
{}

PTraceSession::~PTraceSession()
{
    if(m_attached)
    {
        detach();
    }
}

bool
PTraceSession::attach()
{
    PTRACE_CALL(PTRACE_SEIZE, m_pid, NULL, NULL);
    ROCP_INFO << "Successfully attached to pid " << m_pid;
    m_attached = true;
    return true;
}

bool
PTraceSession::detach()
{
    m_attached = false;
    PTRACE_CALL(PTRACE_DETACH, m_pid, NULL, NULL);
    ROCP_INFO << "Detached from pid " << m_pid;
    return true;
}

// pre-cond: process must be stopped
bool
PTraceSession::write(size_t addr, const std::vector<uint8_t>& data, size_t size) const
{
    constexpr size_t word_size = sizeof(void*);
    size_t           word_iter = 0;
    for(word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        uint64_t     word;
        std::memcpy(&word, data.data() + offset, word_size);
        PTRACE_CALL(PTRACE_POKEDATA, m_pid, addr + offset, word);
    }

    // If not divisible, get the last word to do a partial write correctly.
    size_t remainder = size % word_size;
    if(remainder != 0u)
    {
        const size_t offset    = (word_iter * word_size);
        uint64_t     last_word = 0;
        PTRACE_PEEK(m_pid, addr + offset, last_word);
        std::memcpy(&last_word, data.data() + offset, remainder);
        PTRACE_CALL(PTRACE_POKEDATA, m_pid, addr + offset, last_word);
    }
    ROCP_TRACE << "ptrace wrote " << size << " bytes at " << addr;
    return true;
}

// pre-cond: process must be stopped
bool
PTraceSession::read(size_t addr, std::vector<uint8_t>& data, size_t size) const
{
    data.clear();
    data.resize(size);
    constexpr size_t word_size = sizeof(void*);
    size_t           word_iter = 0;
    for(word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        uint64_t     word   = 0;
        PTRACE_PEEK(m_pid, addr + offset, word);
        std::memcpy(data.data() + offset, &word, word_size);
    }
    size_t remainder = size % word_size;
    if(remainder != 0u)
    {
        const size_t offset    = (word_iter * word_size);
        uint64_t     last_word = 0;
        PTRACE_PEEK(m_pid, addr + offset, last_word);
        std::memcpy(data.data() + offset, &last_word, remainder);
    }
    ROCP_TRACE << "ptrace read " << size << " bytes at " << addr;
    return true;
}

// pre-cond: process must be stopped
bool
PTraceSession::swap(size_t                      addr,
                    const std::vector<uint8_t>& in_data,
                    std::vector<uint8_t>&       out_data,
                    size_t                      size) const
{
    if(!read(addr, out_data, size))
    {
        return false;
    }
    return write(addr, in_data, size);
}

bool
PTraceSession::simple_mmap(void*& addr, size_t length) const
{
    if(!m_attached)
    {
        ROCP_ERROR << "simple_mmap called while not attached";
        return false;
    }

    if(!stop())
    {
        return false;
    }

    // Create a system call to mmap:
    // mmap(NULL, length, prot, flags, -1, 0);
    // Get entry address for safe injection of op codes
    size_t entry_addr{0};
    get_auxv_entry(m_pid, entry_addr);

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &oldregs);
    // Set register file for call
    struct user_regs_struct newregs = oldregs;

    newregs.rax = 9;                            // calling convention: syscall ID for mmap
    newregs.rdi = 0;                            // addr
    newregs.rsi = length;                       // length
    newregs.rdx = PROT_READ | PROT_WRITE;       // prot
    newregs.r10 = MAP_PRIVATE | MAP_ANONYMOUS;  // flags
    newregs.r8  = -1;                           // fd (unused)
    newregs.r9  = 0;                            // offset
    newregs.rip = entry_addr;
    newregs.rsp = oldregs.rsp - 128;  // move sp by 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);

    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &newregs);

    // x64 assembly to perform a syscall and breakpoint when done
    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({0x0f, 0x05, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    if(!swap(entry_addr, new_code, old_code, 3))
    {
        return false;
    }

    ROCP_TRACE << "Attempting to execute mmap syscall";
    // Resume execution
    if(!cont())
    {
        return false;
    }

    // Wait for int3 breakpoint to be hit
    int status;
    if(waitpid(m_pid, &status, WUNTRACED) == -1)
    {
        return false;
    }

    // Get registers to see mmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &returnregs);

    // Write in old opcodes
    if(!write(entry_addr, old_code, 3))
    {
        return false;
    }

    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &oldregs);
    // Restart execution
    if(!cont())
    {
        return false;
    }

    addr = reinterpret_cast<void*>(returnregs.rax);  // NOLINT(performance-no-int-to-ptr)
    return true;
}

bool
PTraceSession::simple_munmap(void*& addr, size_t length) const
{
    if(!m_attached)
    {
        ROCP_ERROR << "simple_munmap called while not attached";
        return false;
    }

    // Stop the process
    if(!stop())
    {
        return false;
    }

    // Create a system call to mumap:
    // mumap(NULL, length, prot, flags, -1, 0);
    // Get entry address for safe injection of op codes
    size_t entry_addr{0};
    get_auxv_entry(m_pid, entry_addr);

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &oldregs);
    // Set register file for call
    struct user_regs_struct newregs = oldregs;

    newregs.rax = 11;                              // calling convention: syscall ID for mumap
    newregs.rdi = reinterpret_cast<size_t>(addr);  // addr
    newregs.rsi = length;                          // length
    newregs.rip = entry_addr;
    newregs.rsp = oldregs.rsp - 128;  // move sp by 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);
    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &newregs);

    // x64 assembly to perform a syscall and breakpoint when done
    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({0x0f, 0x05, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    if(!swap(entry_addr, new_code, old_code, 3))
    {
        return false;
    }

    ROCP_TRACE << "Attempting to execute munmap syscall";
    // Restart execution
    if(!cont())
    {
        return false;
    }

    // Wait for int3 breakpoint to be hit
    int status;
    if(waitpid(m_pid, &status, WUNTRACED) == -1)
    {
        return false;
    }

    // Get registers to see munmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &returnregs);

    // Write in old opcodes
    if(!write(entry_addr, old_code, 3))
    {
        return false;
    }
    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &oldregs);
    // Restart execution
    if(!cont())
    {
        return false;
    }

    return true;
}

bool
PTraceSession::call_function(const std::string& library, const std::string& symbol)
{
    return call_function(library, symbol, nullptr);
}

// This supports calling a dynamically loaded function with at most 1 parameter.
// More parameters could be supported, but this is good enough for now.
// Correctly implementing this would require duplicating the x64 calling convention. Probably not
// worth it.
bool
PTraceSession::call_function(const std::string& library,
                             const std::string& symbol,
                             void*              first_param)
{
    if(!m_attached)
    {
        ROCP_ERROR << "call_function called while not attached";
        return false;
    }

    // Stop the process
    if(!stop())
    {
        return false;
    }

    void* target_addr;
    if(!find_symbol(target_addr, library, symbol))
    {
        return false;
    }

    // Get entry address for safe injection of op codes
    size_t entry_addr{0};
    get_auxv_entry(m_pid, entry_addr);

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &oldregs);

    // Construct registers to call a function with 1 parameter
    // symbol(first_param)
    struct user_regs_struct newregs = oldregs;
    newregs.rax                     = reinterpret_cast<size_t>(target_addr);  // target function
    newregs.rdi                     = reinterpret_cast<size_t>(first_param);  // first parameter
    newregs.rip                     = entry_addr;
    newregs.rsp = oldregs.rsp - 128;  // move sp by 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);

    // x64 assembly to call a function by register and breakpoint when done
    // ff d0  call rax
    // cc     int3
    std::vector<uint8_t> new_code({0xff, 0xd0, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    if(!swap(entry_addr, new_code, old_code, 3))
    {
        return false;
    }
    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &newregs);

    ROCP_TRACE << "Attempting to execute " << library << "::" << symbol << "(" << first_param
               << ")";
    // Restart execution
    if(!cont())
    {
        return false;
    }

    // Wait for int3 to be hit
    if(waitpid(m_pid, nullptr, WSTOPPED) == -1)
    {
        return false;
    }

    // Get registers to see return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &returnregs);

    // Write in old opcodes
    if(!write(entry_addr, old_code, 3))
    {
        return false;
    }
    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &oldregs);
    // Restart execution
    if(!cont())
    {
        return false;
    }

    return true;
}

// This supports calling a dynamically loaded function with at most 2 parameters.
// Uses x64 calling convention: RDI for first param, RSI for second param
bool
PTraceSession::call_function(const std::string& library,
                             const std::string& symbol,
                             void*              first_param,
                             void*              second_param)
{
    if(!m_attached)
    {
        ROCP_ERROR << "call_function called while not attached";
        return false;
    }

    // Stop the process
    if(!stop())
    {
        return false;
    }

    void* target_addr = nullptr;
    if(!find_symbol(target_addr, library, symbol))
    {
        return false;
    }

    // Get entry address for safe injection of op codes
    size_t entry_addr{0};
    get_auxv_entry(m_pid, entry_addr);

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &oldregs);

    // Construct registers to call a function with 2 parameters
    // symbol(first_param, second_param)
    struct user_regs_struct newregs = oldregs;
    newregs.rax                     = reinterpret_cast<size_t>(target_addr);   // target function
    newregs.rdi                     = reinterpret_cast<size_t>(first_param);   // first parameter
    newregs.rsi                     = reinterpret_cast<size_t>(second_param);  // second parameter
    newregs.rip                     = entry_addr;
    newregs.rsp = oldregs.rsp - 128;  // move sp by 128 to not clobber redlined functions
    newregs.rsp -= (newregs.rsp % 16);

    // x64 assembly to call a function by register and breakpoint when done
    // ff d0  call rax
    // cc     int3
    std::vector<uint8_t> new_code({0xff, 0xd0, 0xcc});
    std::vector<uint8_t> old_code;

    // Write in new opcodes
    if(!swap(entry_addr, new_code, old_code, 3))
    {
        return false;
    }
    // Set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &newregs);

    ROCP_TRACE << "Attempting to execute " << library << "::" << symbol << "(" << first_param
               << ", " << second_param << ")";
    // Restart execution
    if(!cont())
    {
        return false;
    }

    // Wait for int3 to be hit
    if(waitpid(m_pid, nullptr, WSTOPPED) == -1)
    {
        return false;
    }

    // Get registers to see return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, m_pid, NULL, &returnregs);

    // Write in old opcodes
    if(!write(entry_addr, old_code, 3))
    {
        return false;
    }
    // Restore register file
    PTRACE_CALL(PTRACE_SETREGS, m_pid, NULL, &oldregs);
    // Restart execution
    if(!cont())
    {
        return false;
    }

    return true;
}

bool
PTraceSession::find_library(void*& addr, int inpid, const std::string& library)
{
    std::stringstream searchname;
    searchname << inpid << "::" << library;
    // TODO: add this back
    // if (target_library_addrs.find(searchname.str()) != target_library_addrs.end())
    //{
    //    return target_library_addrs[searchname.str()];
    //}

    // uses "maps" file to find where library has been loaded in target process
    // does not require this process to be attached
    std::stringstream filename;
    filename << "/proc/" << inpid << "/maps";
    std::ifstream maps(filename.str().c_str());

    if(!maps)
    {
        ROCP_ERROR << "Couldn't open " << filename.str();
        return false;
    }

    std::string line;
    while(std::getline(maps, line))
    {
        if(line.find(library) != std::string::npos)
        {
            ROCP_TRACE << "entry in pid " << inpid << " maps file is: " << line;
            break;
        }
    }

    if(!maps)
    {
        ROCP_ERROR << "Couldn't find library " << library << " in " << filename.str();
        return false;
    }

    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    addr = reinterpret_cast<void*>(std::stoull(line, nullptr, 16));
    //  target_library_addrs[searchname.str()] = addr;
    return true;
}

bool
PTraceSession::find_symbol(void*& addr, const std::string& library, const std::string& symbol)
{
    auto searchname = std::stringstream{};
    searchname << library << "::" << symbol;
    if(auto itr = m_target_symbol_addrs.find(searchname.str()); itr != m_target_symbol_addrs.end())
    {
        ROCP_TRACE << "found symbol for " << searchname.str() << " at " << itr->second;
        return itr->second != nullptr;
    }

    void* libraryaddr = nullptr;
    void* symboladdr  = nullptr;

    // Load the library in our process to determine the offset of the requested symbol from the
    // start address of the library
    addr        = nullptr;
    libraryaddr = get_library_handle(library);

    if(!libraryaddr)
    {
        ROCP_ERROR << "host couldn't dlopen " << library;
        return false;
    }

    symboladdr = dlsym(libraryaddr, symbol.c_str());
    if(!symboladdr)
    {
        ROCP_ERROR << "host couldn't dlsym " << symbol;
        return false;
    }

    // Find the start address of the library in our process
    void* hostlibraryaddr;
    if(!find_library(hostlibraryaddr, getpid(), library))
    {
        ROCP_ERROR << "couldn't determine where " << library << " was loaded for host";
        return false;
    }

    // Caluclate the offset
    size_t offset =
        reinterpret_cast<size_t>(symboladdr) - reinterpret_cast<size_t>(hostlibraryaddr);
    ROCP_TRACE << "offset of " << symbol << " into " << library << " calculated as " << offset;

    // Find the start address of the library in the target process
    void* targetlibraryaddr;
    if(!find_library(targetlibraryaddr, m_pid, library))
    {
        ROCP_ERROR << "couldn't determine where " << library << " was loaded for target";
        return false;
    }

    // Calculate address of symbol in the target process using the offset
    // NOLINTNEXTLINE(performance-no-int-to-ptr)
    addr = reinterpret_cast<void*>(reinterpret_cast<size_t>(targetlibraryaddr) + offset);
    m_target_symbol_addrs[searchname.str()] = addr;
    ROCP_TRACE << "found symbol for " << searchname.str() << " at " << addr;
    return true;
}

bool
PTraceSession::stop() const
{
    if(!m_attached)
    {
        ROCP_ERROR << "stop called while not attached";
        return false;
    }

    // Stop the process
    PTRACE_CALL(PTRACE_INTERRUPT, m_pid, NULL, NULL);

    // Wait for the stop
    if(waitpid(m_pid, nullptr, WSTOPPED) == -1)
    {
        return false;
    }
    ROCP_TRACE << "ptrace stopped pid " << m_pid;
    return true;
}

bool
PTraceSession::cont() const
{
    if(!m_attached)
    {
        ROCP_ERROR << "cont called while not attached";
        return false;
    }

    PTRACE_CALL(PTRACE_CONT, m_pid, NULL, NULL);
    ROCP_TRACE << "ptrace resumed pid " << m_pid;
    return true;
}

bool
PTraceSession::handle_signals() const
{
    while(!m_detaching_ptrace_session.load())
    {
        int status{0};
        if(waitpid(m_pid, &status, WNOHANG) == -1)
        {
            ROCP_ERROR << "waitpid failed in handle_signal for pid " << m_pid;
            return false;
        }
        if(status != 0 && WIFEXITED(status))
        {
            ROCP_ERROR << "process " << m_pid << " exited, status=" << WEXITSTATUS(status);
            return false;
        }
        else if(status != 0 && WIFSIGNALED(status))
        {
            ROCP_ERROR << "process " << m_pid << " killed by signal " << WTERMSIG(status);
            return false;
        }
        else if(status != 0 && WIFSTOPPED(status))
        {
            auto sig = WSTOPSIG(status);
            ROCP_TRACE << "process " << m_pid << "stopped by signal " << sig;
            PTRACE_CALL(PTRACE_CONT, m_pid, NULL, sig);
        }
        std::this_thread::yield();
    }
    return true;
}

void
PTraceSession::detach_ptrace_session()
{
    m_detaching_ptrace_session.store(true);
}

}  // namespace attach
}  // namespace rocprofiler
