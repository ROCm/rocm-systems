#include "ptrace_session.hpp"

#include "lib/common/logging.hpp"

#include <dlfcn.h>
#include <sys/ptrace.h>
#include <sys/mman.h>
#include <sys/user.h>
#include <sys/wait.h>

#include <fstream>

#define PTRACE_CALL(op, pid, addr, data) \
ROCP_INFO<< "ptrace call params(" << op << ", " << pid << ", " << (size_t)addr << ", " << (size_t)data << ")"; \
if (errno=0, ptrace(op, pid, addr, data); errno != 0)         \
{                                        \
    ROCP_ERROR << "ptrace call failed. errno: " << errno << " params(" << op << ", " << pid << ", " << (size_t)addr << ", " << (size_t)data << ")";  \
    return false;                        \
}

#define PTRACE_PEEK(pid, addr) \
[&]() { \
    size_t retval; \
    ROCP_INFO << "ptrace call params(2 , " << pid << ", " << (size_t)addr << ", 0)"; \
    if (errno=0, retval=ptrace(PTRACE_PEEKDATA, pid, addr, NULL); errno != 0)         \
    {                                        \
        ROCP_ERROR << "ptrace call failed. errno: " << errno << " params(2, " << pid << ", " << (size_t)addr << ", 0)";  \
    }\
    return retval; \
}();

namespace rocprofiler {
namespace attach {

PTraceSession::PTraceSession(size_t _pid) :
  pid(_pid),
  attached(false)
{

}

PTraceSession::~PTraceSession()
{
    if (attached)
    {
        detach();
    }
}

bool PTraceSession::attach()
{
    PTRACE_CALL(PTRACE_SEIZE, pid, NULL, NULL);

    // trace log - attached
    attached = true;
    return true;
}

bool PTraceSession::detach()
{
    attached = false;
    PTRACE_CALL(PTRACE_DETACH, pid, NULL, NULL);

    // trace log - detached
    return true;
}

bool PTraceSession::write(size_t addr, const std::vector<uint8_t>& data, size_t size)
{
    constexpr size_t word_size = sizeof(void*);
    size_t word_iter = 0;
    for (word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        size_t word = *reinterpret_cast<const size_t*>(data.data()+offset);
        PTRACE_CALL(PTRACE_POKEDATA, pid, addr + offset, word);
    }
    size_t remainder = size % word_size;
    if (remainder)
    {
        const size_t offset = (word_iter * word_size);
        size_t last_word = PTRACE_PEEK(pid, addr + offset);
        {
            uint8_t* last_word_writer = reinterpret_cast<uint8_t*>(&last_word);
            for (size_t byte_iter = 0; byte_iter < remainder; ++byte_iter)
            {
                last_word_writer[byte_iter] = data[offset + byte_iter];
            }
        }
        PTRACE_CALL(PTRACE_POKEDATA, pid, addr + offset, last_word);
    }
    return true;
}

bool PTraceSession::read(size_t addr, std::vector<uint8_t>& data, size_t size)
{
    data.clear();
    data.resize(size);
    constexpr size_t word_size = sizeof(void*);
    size_t word_iter = 0;
    for (word_iter = 0; word_iter < (size / word_size); ++word_iter)
    {
        const size_t offset = (word_iter * word_size);
        size_t word = PTRACE_PEEK(pid, addr + offset);
        uint8_t* word_reader = reinterpret_cast<uint8_t*>(&word);
        for (size_t byte_iter = 0; byte_iter < word_size; ++byte_iter)
        {
            *(data.data() + offset + byte_iter) = *word_reader;
        }
    }
    size_t remainder = size % word_size;
    if (remainder)
    {
        const size_t offset = (word_iter * word_size);
        size_t last_word = PTRACE_PEEK(pid, addr + offset);
        uint8_t* last_word_reader = reinterpret_cast<uint8_t*>(&last_word);

        for (size_t byte_iter = 0; byte_iter < remainder; ++byte_iter)
        {
            data[offset + byte_iter] = last_word_reader[byte_iter];
        }
    }
    return true;
}

bool PTraceSession::swap(
    size_t addr,
    const std::vector<uint8_t>& in_data,
    std::vector<uint8_t>& out_data,
    size_t size)
{
    if (!read(addr, out_data, size))
    {
        return false;
    }
    return write(addr, in_data, size);
}


bool PTraceSession::simple_mmap(void*& addr, size_t length)
{
    if (!attached)
    {
        // log - called while not attached
        return false;
    }

    // Stop the process
    PTRACE_CALL(PTRACE_INTERRUPT, pid, NULL, NULL);

    // Wait for the stop
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &oldregs);

    ROCP_TRACE << "oldrip: " << oldregs.rip;

    // create a system call to mmap:
    // mmap(NULL, length, prot, flags, -1, 0);
    struct user_regs_struct newregs = oldregs;
    newregs.rax = 9; // ID for mmap
    newregs.rcx = oldregs.rip;
    newregs.rdi = 0; // addr
    newregs.rsi = (length == 0) ? PAGE_SIZE : length; //length
    newregs.rdx = PROT_READ | PROT_WRITE; // prot
    newregs.r10 = MAP_PRIVATE | MAP_ANONYMOUS; // flags
    newregs.r8 = -1; // fd (unused)
    newregs.r9 = 0; // offset

    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({ 0x0f, 0x05, 0xcc });
    std::vector<uint8_t> old_code;

    // write in new opcodes
    if (!swap(oldregs.rip, new_code, old_code, 3))
    {
        return false;
    }
    // set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &newregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    // wait for int3 to be hit
    int status;
    if (waitpid(pid, &status, WUNTRACED) == -1)
    {
        return false;
    }
    ROCP_TRACE << "stopped for " << status;
    ROCP_TRACE << "stop signal " << ((WSTOPSIG(status)));
    ROCP_TRACE << "is stopped? " << ((WIFSTOPPED(status)) ? "yes" : "no");
    ROCP_TRACE << "is trap?    " << ((WSTOPSIG(status) == SIGTRAP) ? "yes" : "no");
    ROCP_TRACE << "is dumped?  " << ((WCOREDUMP(status)) ? "yes" : "no");
    ROCP_TRACE << "is cont?    " << ((WIFCONTINUED(status)) ? "yes" : "no");

    // get registers to see mmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &returnregs);

    ROCP_TRACE << "afterrip: " << returnregs.rip;

    // write in old opcodes
    if (!write(oldregs.rip, old_code, 3))
    {
        return false;
    }
    // restore register file
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &oldregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    addr = reinterpret_cast<void*>(returnregs.rax);
    return true;
}

bool PTraceSession::simple_munmap(void*& addr, size_t length)
{
    if (!attached)
    {
        // log - called while not attached
        return false;
    }

    // Stop the process
    PTRACE_CALL(PTRACE_INTERRUPT, pid, NULL, NULL);

    // Wait for the stop
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &oldregs);

    // create a system call to mumap:
    // mumap(NULL, length, prot, flags, -1, 0);
    struct user_regs_struct newregs = oldregs;
    newregs.rax = 11; // ID for mumap
    newregs.rdi = reinterpret_cast<size_t>(addr); // addr
    newregs.rsi = (length == 0) ? PAGE_SIZE : length; //length

    // 0f 05  syscall
    // cc     int3
    std::vector<uint8_t> new_code({ 0x0f, 0x05, 0xcc });
    std::vector<uint8_t> old_code;

    // write in new opcodes
    if (!swap(oldregs.rip, new_code, old_code, 3))
    {
        return false;
    }
    // set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &newregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    // wait for int3 to be hit
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }

    // get registers to see mmap's return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &returnregs);

    // write in old opcodes
    if (!write(oldregs.rip, old_code, 3))
    {
        return false;
    }
    // restore register file
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &oldregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    return true;
}

bool PTraceSession::call_function(const std::string& library, const std::string& symbol)
{
    return call_function(library, symbol, nullptr);
}

// This supports calling a dynamically loaded function with 1 parameter
// More could be supported, perhaps even with ..., but this is good enough for our use.
// Correctly implementing this would require reimplementing the x64 calling convention.
// Probably not worth it.
bool PTraceSession::call_function(const std::string& library, const std::string& symbol, void* first)
{
    if (!attached)
    {
        // log - called while not attached
        return false;
    }

    // Stop the process
    PTRACE_CALL(PTRACE_INTERRUPT, pid, NULL, NULL);

    // Wait for the stop
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }

    // Save current register file
    struct user_regs_struct oldregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &oldregs);

    void* target_addr;
    find_symbol(target_addr, library, symbol);

    // construct registers to call a function with 1 parameter
    // symbol(first)
    struct user_regs_struct newregs = oldregs;
    newregs.rax = reinterpret_cast<size_t>(target_addr); // target function
    newregs.rdi = reinterpret_cast<size_t>(first); // first parameter

    // ff d0  call rax
    // cc     int3
    std::vector<uint8_t> new_code({ 0xff, 0xd0, 0xcc });
    std::vector<uint8_t> old_code;

    // write in new opcodes
    if (!swap(oldregs.rip, new_code, old_code, 3))
    {
        return false;
    }
    // set syscall registers
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &newregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    // wait for int3 to be hit
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }

    // get registers to see return values
    struct user_regs_struct returnregs;
    PTRACE_CALL(PTRACE_GETREGS, pid, NULL, &returnregs);

    // write in old opcodes
    if (!write(oldregs.rip, old_code, 3))
    {
        return false;
    }
    // restore register file
    PTRACE_CALL(PTRACE_SETREGS, pid, NULL, &oldregs);
    // restart execution
    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);

    return true;
}

size_t PTraceSession::get_pid()
{
    return pid;
}

bool PTraceSession::find_library(void*& addr, int inpid, const std::string& library)
{
    std::stringstream searchname;
    searchname << inpid << "::" << library;
    // TODO: add this back
    //if (target_library_addrs.find(searchname.str()) != target_library_addrs.end())
    //{
    //    return target_library_addrs[searchname.str()];
    //}

    // uses "maps" file to find where library has been loaded in target process
    // does not require this process to be attached
    std::stringstream filename;
    filename << "/proc/" << inpid << "/maps";
    std::ifstream maps(filename.str().c_str());

    std::string line;
    while (std::getline(maps, line))
    {
        if (line.find(library) != std::string::npos)
        {
            ROCP_TRACE << "entry in target maps file is " << line;
            break;
        }
    }

    if (!maps)
    {
        return false;
    }

    addr = reinterpret_cast<void*>(std::stoull(line, nullptr, 16));
    target_library_addrs[searchname.str()] = addr;
    return true;
}

bool PTraceSession::find_symbol(void*& addr, const std::string& library, const std::string& symbol)
{
    std::stringstream searchname;
    searchname << library << "::" << symbol;
    ROCP_TRACE << "find_symbol called for " << searchname.str();
    if (target_symbol_addrs.find(searchname.str()) != target_symbol_addrs.end())
    {
        ROCP_TRACE << "found symbol for " << searchname.str() << " at " << target_symbol_addrs[searchname.str()];
        return target_symbol_addrs[searchname.str()];
    }
    void* libraryaddr;
    void* symboladdr;
    addr = nullptr;
    libraryaddr = dlopen(library.c_str(), RTLD_LAZY);

    if (!libraryaddr)
    {
        ROCP_ERROR << "host couldn't dlopen " << library;
        return false;
    }

    symboladdr = dlsym(libraryaddr, symbol.c_str());
    if (!symboladdr)
    {
        ROCP_ERROR << "host couldn't dlsym " << symbol;
        return false;
    }

    void* hostlibraryaddr;
    if (!find_library(hostlibraryaddr, getpid(), library))
    {
        ROCP_ERROR << "couldn't determine where " << library << " was loaded for host";
        return false;
    }

    size_t offset = reinterpret_cast<size_t>(symboladdr) - reinterpret_cast<size_t>(hostlibraryaddr);
    ROCP_TRACE << "offset of " << symbol << " into " << library << " calculated as " << offset;


    void* targetlibraryaddr;
    if (!find_library(targetlibraryaddr, pid, library))
    {
        ROCP_ERROR << "couldn't determine where " << library << " was loaded for target";
        return false;
    }

    addr = reinterpret_cast<void*>(reinterpret_cast<size_t>(targetlibraryaddr) + offset);
    target_symbol_addrs[searchname.str()] = addr;
    ROCP_TRACE << "found symbol for " << searchname.str() << " at " << addr;
    return true;
}

bool PTraceSession::stop()
{
    if (!attached)
    {
        return false;
    }

    // Stop the process
    PTRACE_CALL(PTRACE_INTERRUPT, pid, NULL, NULL);

    // Wait for the stop
    if (waitpid(pid, 0, WSTOPPED) == -1)
    {
        return false;
    }
    return true;
}

bool PTraceSession::cont()
{
    if (!attached)
    {
        return false;
    }

    PTRACE_CALL(PTRACE_CONT, pid, NULL, NULL);
    return true;
}

}
}
