/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#if !defined(_GNU_SOURCE) || !defined(_XOPEN_SOURCE)
#define _XOPEN_SOURCE 700
#endif

#include "code_printing.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <hsa/amd_hsa_elf.h>

#include <amd-dbgapi/amd-dbgapi.h>

#include <cxxabi.h>
#include <elfutils/libdw.h>

/// From rocr_debug_agent
namespace amd::debug_agent {

enum class log_level_t {
  /* Print no messages.  */
  none = 0,
  /* Print error messages.  */
  error = 1,
  /* Print error, and warning messages.  */
  warning = 2,
  /* Print error, warning, and info messages.  */
  info = 3,
  /* Print error, warning, info, and verbose  messages.  */
  verbose = 4
};

static log_level_t log_level = log_level_t::warning;

static std::ofstream agent_out;

namespace detail {

/* A macro instead of a variadic template so that the __VAR_ARGS__ are not
   evaluated unless the log level indicated they are needed.  */
static void log(log_level_t level, const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format (printf, 2, 3)))
#endif // defined(__GNUC__)
    ;

static void
log(log_level_t level, const char *format, ...)
{
    va_list va;

    agent_out << "rocm-debug-agent: ";

    if (level == log_level_t::error) {
        agent_out << "error: ";
    } else if (level == log_level_t::warning) {
        agent_out << "warning: ";
    }

    va_start(va, format);
    size_t size = vsnprintf(NULL, 0, format, va);
    va_end(va);

    va_start(va, format);
    std::string str(size, '\0');
    vsprintf(&str[0], format, va);
    va_end(va);

    agent_out << str << std::endl;
}

} // namespace detail

#define agent_log(level, format, ...)                                   \
    do {                                                                \
        if (level <= amd::debug_agent::log_level) {                     \
            amd::debug_agent::detail::log(level, format, ##__VA_ARGS__); \
        }                                                               \
    } while (0)

static void
set_log_level(log_level_t level)
{
    log_level = level;
    switch (level) {
    case log_level_t::none:
        amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_NONE);
        break;
    case log_level_t::verbose:
        amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_VERBOSE);
        break;
    case log_level_t::info:
        amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_INFO);
        break;
    case log_level_t::warning:
        amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_WARNING);
        break;
    case log_level_t::error:
        amd_dbgapi_set_log_level(AMD_DBGAPI_LOG_LEVEL_FATAL_ERROR);
        break;
    }
}

/* A macro instead of a variadic template so that format is still a string
   literal when passed to agent_log.  */
#define agent_warning(format, ...)                              \
    agent_log(log_level_t::warning, format, ##__VA_ARGS__)

#define agent_error(format, ...)                                        \
    do {                                                                \
        agent_log(log_level_t::error, format, ##__VA_ARGS__);           \
        abort();                                                        \
    } while (false)

#define agent_assert_fail(assertion, file, line)                        \
    [](){                                                               \
        agent_error("%s:%d: Assertion `%s' failed.", file, line, assertion); \
    }()

#define DBGAPI_CHECK(expr)                                              \
    do {                                                                \
        if (amd_dbgapi_status_t status = (expr);                        \
            status != AMD_DBGAPI_STATUS_SUCCESS)                        \
        {                                                               \
            agent_error("%s:%d: %s failed (rc=%d)", __FILE__, __LINE__, #expr, \
                        status);                                        \
        }                                                               \
    } while (false)

#define DEBUG_AGENT_ASSERTION_ENABLED 1

#if defined(DEBUG_AGENT_ASSERTION_ENABLED)
#define agent_assert(expr)                                              \
    ((void)((expr) ? 0 : (agent_assert_fail(#expr, __FILE__, __LINE__), 0)))
#else // !defined(DEBUG_AGENT_ASSERTION_ENABLED)
#define agent_assert(expr) ((void)0)
#endif // !defined(DEBUG_AGENT_ASSERTION_ENABLED)

code_object_t::code_object_t(amd_dbgapi_code_object_id_t code_object_id)
: m_code_object_id(code_object_id)
{
    if (amd_dbgapi_code_object_get_info(
          code_object_id, AMD_DBGAPI_CODE_OBJECT_INFO_LOAD_ADDRESS,
          sizeof (m_load_address), &m_load_address)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_warning("could not get the code object's load address");
        return;
    }

    char *value;
    if (amd_dbgapi_code_object_get_info(m_code_object_id,
                                        AMD_DBGAPI_CODE_OBJECT_INFO_URI_NAME,
                                        sizeof (value), &value)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_warning("could not get the code object's URI");
        return;
    }

    m_uri.assign(value);
    free(value);
}

code_object_t::code_object_t(code_object_t &&rhs)
: m_load_address(rhs.m_load_address)
, m_mem_size(rhs.m_mem_size)
, m_uri(std::move(rhs.m_uri))
, m_code_object_id(rhs.m_code_object_id)
, m_elf_amdgpu_machine(rhs.m_elf_amdgpu_machine)
{
    m_fd = rhs.m_fd;
    rhs.m_fd.reset();
}

code_object_t::~code_object_t()
{
    if (m_fd) {
        ::close(*m_fd);
    }
}

std::optional<code_object_t::symbol_info_t>
code_object_t::find_symbol(amd_dbgapi_global_address_t address)
{
    /* Load the symbol table.  */
    load_symbol_map();

    if (auto it = m_symbol_map->upper_bound(address);
        it != m_symbol_map->begin())
    {
        if (auto &&[symbol_value, symbol] = *std::prev(it);
            address < (symbol_value + symbol.second))
        {
            std::string symbol_name = symbol.first;

            if (int status; auto *demangled_name = abi::__cxa_demangle(
                  symbol_name.c_str(), nullptr, nullptr, &status))
            {
                symbol_name = demangled_name;
                free(demangled_name);
            }

            return symbol_info_t{ std::move(symbol_name), symbol_value,
                                  symbol.second };
        }
    }

    return {};
}

void
code_object_t::open()
{
    const std::string protocol_delim{ "://" };

    size_t protocol_end = m_uri.find(protocol_delim);
    std::string protocol = m_uri.substr(0, protocol_end);
    protocol_end += protocol_delim.length();

    std::transform(protocol.begin(), protocol.end(), protocol.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    std::string path;
    size_t path_end = m_uri.find_first_of("#?", protocol_end);
    if (path_end != std::string::npos) {
        path = m_uri.substr(protocol_end, path_end++ - protocol_end);
    } else {
        path = m_uri.substr(protocol_end);
    }

    /* %-decode the string.  */
    std::string decoded_path;
    decoded_path.reserve(path.length());
    for (size_t i = 0; i < path.length(); ++i)
        if (path[i] == '%' && std::isxdigit(path[i + 1])
            && std::isxdigit(path[i + 2]))
        {
            decoded_path += std::stoi(path.substr (i + 1, 2), 0, 16);
            i += 2;
        } else {
            decoded_path += path[i];
        }

    /* Tokenize the query/fragment.  */
    std::vector<std::string> tokens;
    size_t pos, last = path_end;
    while ((pos = m_uri.find('&', last)) != std::string::npos)
    {
        tokens.emplace_back(m_uri.substr(last, pos - last));
        last = pos + 1;
    }
    if (last != std::string::npos) {
        tokens.emplace_back(m_uri.substr(last));
    }

    /* Create a tag-value map from the tokenized query/fragment.  */
    std::unordered_map<std::string, std::string> params;
    std::for_each(tokens.begin(), tokens.end(), [&](std::string &token){
        size_t delim = token.find('=');
        if (delim != std::string::npos) {
            params.emplace(token.substr(0, delim), token.substr(delim + 1));
        }
    });

    std::vector<char> buffer;
    try {
        size_t offset{ 0 }, size{ 0 };

        if (auto offset_it = params.find("offset"); offset_it != params.end()) {
            offset = std::stoul(offset_it->second, nullptr, 0);
        }

        if (auto size_it = params.find("size"); size_it != params.end()) {
            if (!(size = std::stoul(size_it->second, nullptr, 0))) {
                return;
            }
        }

        if (protocol == "file") {
            std::ifstream file(decoded_path, std::ios::in | std::ios::binary);
            if (!file) {
                agent_warning("could not open `%s'", decoded_path.c_str());
                return;
            }

            if (!size) {
                file.ignore(std::numeric_limits<std::streamsize>::max());
                size_t bytes = file.gcount();
                file.clear();

                if (bytes < offset) {
                    agent_warning("invalid uri `%s' (file size < offset)",
                                  decoded_path.c_str());
                    return;
                }
                size = bytes - offset;
            }

            file.seekg(offset, std::ios_base::beg);
            buffer.resize(size);
            file.read(&buffer[0], size);
        } else if (protocol == "memory") {
            if (!offset || !size) {
                agent_warning("invalid uri `%s' (offset and size must be != 0",
                              m_uri.c_str ());
                return;
            }

            amd_dbgapi_process_id_t process_id;
            if (amd_dbgapi_code_object_get_info(
                  m_code_object_id, AMD_DBGAPI_CODE_OBJECT_INFO_PROCESS,
                  sizeof(process_id), &process_id)
                != AMD_DBGAPI_STATUS_SUCCESS)
                agent_error("could not get the process from the agent");

            buffer.resize(size);
            if (amd_dbgapi_read_memory(process_id, AMD_DBGAPI_WAVE_NONE, AMD_DBGAPI_LANE_NONE,
                                       AMD_DBGAPI_ADDRESS_SPACE_GLOBAL, offset,
                                       &size, buffer.data())
                != AMD_DBGAPI_STATUS_SUCCESS)
            {
                agent_warning("could not read memory at 0x%lx", offset);
                return;
            }
        } else {
            agent_warning("\"%s\" protocol not supported", protocol.c_str());
            return;
        }
    }
    catch (...)
    {}

    int fd =
#if HAVE_MEMFD_CREATE
      ::memfd_create(m_uri.c_str(), MFD_ALLOW_SEALING | MFD_CLOEXEC);
#else // !HAVE_MEMFD_CREATE
    ::open("/tmp", O_TMPFILE | O_RDWR, 0666);
#endif // !HAVE_MEMFD_CREATE
    if (fd == -1) {
        agent_warning("could not create a temporary file for code object");
        return;
    }

    if (size_t size = ::write(fd, buffer.data(), buffer.size());
        size != buffer.size())
    {
        agent_warning("could not write to the temporary file");
        return;
    }

    ::lseek(fd, 0, SEEK_SET);

    /* Calculate the size of the code object as loaded in memory.  Its size is
       the distance of the end of the highest segment from the load address.  */
    std::unique_ptr<Elf, void (*)(Elf *)> elf (
      elf_begin(fd, ELF_C_READ, nullptr), [](Elf *elf){ elf_end(elf); });
    if (!elf) {
        agent_warning("elf_begin failed for `%s'", m_uri.c_str());
        return;
    }

    Elf64_Ehdr *ehdr = elf64_getehdr(elf.get());
    if (!ehdr) {
        agent_warning("elf64_getehdr failed for `%s'", m_uri.c_str());
        return;
    }
    m_elf_amdgpu_machine = ehdr->e_flags & ELF::EF_AMDGPU_MACH;

    size_t phnum;
    if (elf_getphdrnum(elf.get(), &phnum) != 0) {
        agent_warning("elf_getphdrnum failed for `%s'", m_uri.c_str());
        return;
    }

    for (size_t i = 0; i < phnum; ++i) {
        GElf_Phdr phdr_mem;
        GElf_Phdr *phdr = gelf_getphdr(elf.get(), i, &phdr_mem);
        if (!phdr) {
            agent_warning("gelf_getphdr failed for `%s'", m_uri.c_str());
            return;
        }

        if (phdr->p_type == PT_LOAD) {
            m_mem_size = std::max(m_mem_size, phdr->p_vaddr + phdr->p_memsz);
        }
    }

    m_fd.emplace(fd);
}

static amd_dbgapi_callbacks_t dbgapi_callbacks = {
    .allocate_memory = malloc,
    .deallocate_memory = free,

    .get_os_pid =
      [](amd_dbgapi_client_process_id_t client_process_id, pid_t *pid){
          *pid = getpid();
          return AMD_DBGAPI_STATUS_SUCCESS;
      },

    .insert_breakpoint =
      [](amd_dbgapi_client_process_id_t client_process_id,
         amd_dbgapi_global_address_t address,
         amd_dbgapi_breakpoint_id_t breakpoint_id)
      {
         return AMD_DBGAPI_STATUS_SUCCESS;
      },

    .remove_breakpoint =
      [](amd_dbgapi_client_process_id_t client_process_id,
         amd_dbgapi_breakpoint_id_t breakpoint_id)
      {
         return AMD_DBGAPI_STATUS_SUCCESS;
      },

    .log_message =
      [](amd_dbgapi_log_level_t level, const char *message) {
          agent_out << "rocm-dbgapi: " << message << std::endl;
      }
};

static std::optional<std::reference_wrapper<const std::vector<std::string>>>
get_source_file_index(const std::string &file_name)
{
    static std::unordered_map<std::string, std::vector<std::string>> file_map;

    if (auto it = file_map.find(file_name); it != file_map.end()) {
        return it->second;
    }

    std::ifstream file(file_name);
    if (!file) {
        return std::nullopt;
    }

    auto [it, success]
      = file_map.emplace(file_name, std::vector<std::string>{});
    agent_assert(success && "emplace should have succeeded");

    auto &lines = it->second;
    std::string line;

    while (std::getline(file, line)) {
        lines.emplace_back(line);
    }

    return lines;
}

void
code_object_t::load_symbol_map()
{
    agent_assert(is_open() && "code object is not opened");

    if (m_symbol_map.has_value()) {
        return;
    }

    m_symbol_map.emplace();

    std::unique_ptr<Elf, void (*)(Elf *)> elf (
      elf_begin(*m_fd, ELF_C_READ, nullptr),
      [](Elf *elf){ elf_end(elf); });

    if (!elf) {
        return;
    }

    /* Slurp the symbol table.  */
    Elf_Scn *scn = nullptr;
    while ((scn = elf_nextscn(elf.get(), scn)) != nullptr) {
        GElf_Shdr shdr_mem;
        GElf_Shdr *shdr = gelf_getshdr(scn, &shdr_mem);
        if (shdr->sh_type != SHT_SYMTAB && shdr->sh_type != SHT_DYNSYM) {
            continue;
        }

        Elf_Data *data = elf_getdata(scn, nullptr);
        if (!data) {
            continue;
        }

        size_t symbol_count
          = data->d_size / gelf_fsize(elf.get(), ELF_T_SYM, 1, EV_CURRENT);
        for (size_t j = 0; j < symbol_count; ++j) {
            GElf_Sym sym_mem;
            GElf_Sym *sym = gelf_getsym(data, j, &sym_mem);

            if (GELF_ST_TYPE(sym->st_info) != STT_FUNC
                || sym->st_shndx == SHN_UNDEF)
            {
                continue;
            }

            std::string symbol_name{ elf_strptr(elf.get(), shdr->sh_link,
                                                sym->st_name) };

            auto [it, success] = m_symbol_map->emplace(
              m_load_address + sym->st_value,
              std::make_pair(symbol_name, sym->st_size));

            /* If there already was a symbol defined at this address, but this
               new symbol covers a larger address range, replace the old symbol
               with this new one.  */
            if (!success && sym->st_size > it->second.second) {
                it->second = std::make_pair(symbol_name, sym->st_size);
            }
        }
    }

    /* TODO: If we did not see a symbtab, check the dynamic segment.  */
}

void
code_object_t::load_debug_info()
{
    agent_assert(is_open() && "code object is not opened");

    if (m_line_number_map.has_value() && m_pc_ranges_map.has_value()) {
        return;
    }

    m_line_number_map.emplace();
    m_pc_ranges_map.emplace();

    std::unique_ptr<Dwarf, void (*)(Dwarf *)> dbg(
      dwarf_begin(*m_fd, DWARF_C_READ), [](Dwarf *dbg){ dwarf_end(dbg); }
    );

    if (!dbg) {
        return;
    }

    Dwarf_Off cu_offset{ 0 }, next_offset;
    size_t header_size;

    while (!dwarf_nextcu(dbg.get(), cu_offset, &next_offset, &header_size,
                         nullptr, nullptr, nullptr))
    {
        Dwarf_Die die;
        if (!dwarf_offdie(dbg.get(), cu_offset + header_size, &die)) {
            continue;
        }

        ptrdiff_t offset = 0;
        Dwarf_Addr base, start{ 0 }, end{ 0 };

        /* dwarf_ranges returns a single contiguous range
           (DW_AT_low_pc/DW_AT_high_pc), or a series of non-contiguous ranges
           (DW_AT_ranges). */
        while ((offset = dwarf_ranges(&die, offset, &base, &start, &end) > 0)) {
            m_pc_ranges_map->emplace(m_load_address + start,
                                     m_load_address + end);
        }

        Dwarf_Lines *lines;
        size_t line_count;
        if (dwarf_getsrclines(&die, &lines, &line_count)) {
            continue;
        }

        for (size_t i = 0; i < line_count; ++i) {
            Dwarf_Addr addr;
            int line_number;

            if (Dwarf_Line *line = dwarf_onesrcline(lines, i);
                line && !dwarf_lineaddr(line, &addr)
                && !dwarf_lineno(line, &line_number) && line_number)
            {
                m_line_number_map->emplace(
                  m_load_address + addr,
                  std::make_pair(dwarf_linesrc(line, nullptr, nullptr),
                                 line_number));
            }
        }

        cu_offset = next_offset;
    }
}

void
code_object_t::disassemble_around(
  amd_dbgapi_architecture_id_t architecture_id,
  amd_dbgapi_global_address_t pc)
{
    amd_dbgapi_process_id_t process_id;
    if (amd_dbgapi_code_object_get_info(m_code_object_id,
                                        AMD_DBGAPI_CODE_OBJECT_INFO_PROCESS,
                                        sizeof(process_id), &process_id)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_error("could not get the process from the agent");
    }

    amd_dbgapi_size_t largest_instruction_size;
    if (amd_dbgapi_architecture_get_info(
          architecture_id,
          AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
          sizeof(largest_instruction_size), &largest_instruction_size)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_error("could not get the instruction size from the architecture");
    }

    /* Load the line number table, and low/high pc for all CUs.  */
    load_debug_info();

    constexpr int context_byte_size = 24;
    amd_dbgapi_global_address_t start_pc;

    /* Try to find a line number that precedes `pc` by `context_byte_size` bytes.
       If we don't have a line number map, simply start the disassembly from the
       current pc.  */

    if (auto it = m_line_number_map->upper_bound(pc);
        it != m_line_number_map->begin())
    {
        do {
            it = std::prev(it);
            if ((pc - it->first) >= context_byte_size) {
                break;
            }
        } while (it != m_line_number_map->begin());

        start_pc = it->first;
    } else {
        /* Don't print any instructions before the current pc.  The instructions
           are of variable size so we can't reliably tell if we'll land on a
           valid instruction.  */
        start_pc = pc;
    }

    amd_dbgapi_global_address_t end_pc = pc + context_byte_size;

    /* If pc is included in a [lowpc,highpc] interval, clamp start_pc and
       end_pc.  */

    if (auto it = m_pc_ranges_map->upper_bound(pc);
        it != m_pc_ranges_map->begin())
    {
        if (auto [low_pc, high_pc] = *std::prev(it); pc < high_pc) {
            start_pc = std::max(start_pc, low_pc);
            end_pc = std::min(end_pc, high_pc);
        }
    }

    auto symbol = find_symbol(pc);

    agent_out << std::endl << "Disassembly";
    if (symbol) {
        agent_out << " for function " << symbol->m_name;
    }
    agent_out << ":" << std::endl;

    agent_out << "    code object: " << m_uri << std::endl;
    agent_out << "    loaded at: "
              << "[0x" << std::hex << m_load_address << "-"
              << "0x" << std::hex << (m_load_address + m_mem_size) << "]"
              << std::endl;

    /* Remember the start_pc address to print the first source line.  */
    amd_dbgapi_global_address_t saved_start_pc{ start_pc };

    /* Now that we know start_pc is a valid instruction address, skip ahead until
       the distance between start_pc and pc is <= context_byte_size.  */
    while ((pc - start_pc) > context_byte_size)
    {
        std::vector<uint8_t> buffer(largest_instruction_size);

        amd_dbgapi_size_t size = buffer.size();
        if (amd_dbgapi_read_memory(
              process_id, AMD_DBGAPI_WAVE_NONE, AMD_DBGAPI_LANE_NONE,
              AMD_DBGAPI_ADDRESS_SPACE_GLOBAL, start_pc, &size, buffer.data())
            != AMD_DBGAPI_STATUS_SUCCESS)
        {
            break;
        }

        if (amd_dbgapi_disassemble_instruction(
              architecture_id, start_pc, &size, buffer.data(), nullptr,
              amd_dbgapi_symbolizer_id_t{}, nullptr)
            != AMD_DBGAPI_STATUS_SUCCESS)
        {
            break;
        }

        if ((pc - (start_pc + size)) < context_byte_size) {
            break;
        }

        start_pc += size;
    }

    std::string prev_file_name;
    size_t prev_line_number{ 0 };
    amd_dbgapi_global_address_t addr{ start_pc };

    while (addr < end_pc)
    {
        if (auto it
            = m_line_number_map->find(addr == start_pc ? saved_start_pc : addr);
            it != m_line_number_map->end())
        {
            const std::string &file_name = it->second.first;
            size_t line_number = it->second.second;

            if (file_name != prev_file_name || line_number != prev_line_number) {
                agent_out << std::endl;
            }

            if (file_name != prev_file_name) {
                agent_out << file_name << ":" << std::endl;
            }

            /* If the source line for `addr` is a different line than the
               previous one printed, then print it.  If the previous line printed
               is in the same file and an earlier line, and if all the lines
               between it and the source line for `addr` have no associated
               instructions (indicated by their being no entries in the line
               number map that mention them), then display those lines as well as
               a source line block.  That allows the disassembly to show all the
               source file lines, including those that have no associated code.
            */
            if (file_name != prev_file_name || line_number != prev_line_number)
            {
                size_t first_line = line_number;
                size_t last_line = line_number;

                /* Find the first line to print between prev_line_number and
                   line_number that does not appear in the line number table.
                */
                if (file_name == prev_file_name
                    && (line_number + 1) > prev_line_number)
                {
                    while (--first_line > prev_line_number) {
                        if (std::find_if(
                              m_line_number_map->begin(),
                              m_line_number_map->end(),
                              [first_line, &file_name](
                                const std::remove_reference_t<decltype (
                                *m_line_number_map)>::value_type
                                &value)
                              {
                                  return file_name == value.second.first
                                    && first_line == value.second.second;
                              })
                            != m_line_number_map->end())
                            break;
                    }
                    /* First is either prev_line_number, or a line associated
                       with another address, so start at the next line.  */
                    ++first_line;
                }

                for (size_t line = first_line; line <= last_line; ++line) {
                    agent_out << std::setfill(' ') << std::setw(8) << std::left
                              << std::dec << line;

                    if (auto lines = get_source_file_index(file_name); !lines) {
                        agent_out << file_name << ": No such file or directory.";
                    } else if (line && line <= lines->get().size()) {
                        agent_out << lines->get()[line - 1];
                    }

                    agent_out << std::endl;
                }
            }

            prev_file_name = file_name;
            prev_line_number = line_number;

            /* If the start_pc address is not the begining of a line number
               block, then print ... to show that the following instruction is
               not the first in the block.  */
            if (addr == start_pc && start_pc != saved_start_pc) {
                agent_out << "    ..." << std::endl;
            }
        }

        std::vector<uint8_t> buffer(largest_instruction_size);

        amd_dbgapi_size_t size = buffer.size();
        if (amd_dbgapi_read_memory(
              process_id, AMD_DBGAPI_WAVE_NONE, AMD_DBGAPI_LANE_NONE,
              AMD_DBGAPI_ADDRESS_SPACE_GLOBAL, addr, &size, buffer.data())
            != AMD_DBGAPI_STATUS_SUCCESS)
        {
            agent_out << "Cannot access memory at address 0x" << std::hex << addr
                      << std::endl;
            break;
        }

        auto symbolizer = [](amd_dbgapi_symbolizer_id_t symbolizer_id,
                             amd_dbgapi_global_address_t address,
                             char **symbol_text)
        {
            auto &code_object = *reinterpret_cast<code_object_t *>(symbolizer_id);
            std::stringstream ss;

            ss << "0x" << std::hex << address;

            if (auto &&symbol = code_object.find_symbol(address)) {
                ss << " <" << symbol->m_name;
                ss << "+" << std::dec << (address - symbol->m_value);
                ss << ">";
            }

            *symbol_text = strdup(ss.str().c_str());
            return AMD_DBGAPI_STATUS_SUCCESS;
        };

        char *value;
        if (amd_dbgapi_disassemble_instruction(
              architecture_id, addr, &size, buffer.data(), &value,
              reinterpret_cast<amd_dbgapi_symbolizer_id_t>(this), symbolizer)
            != AMD_DBGAPI_STATUS_SUCCESS)
            agent_error("amd_dbgapi_disassemble_instruction failed");

        std::string instruction(value);
        free(value);

        agent_out << ((addr == pc) ? " => " : "    ");

        agent_out << "0x" << std::hex << addr;
        if (symbol) {
            agent_out << " <";
            if (addr >= symbol->m_value) {
                agent_out << "+" << std::dec << (addr - symbol->m_value);
            } else {
                agent_out << "-" << std::dec << (symbol->m_value - addr);
            }
            agent_out << ">";
        }

        agent_out << ":    " << instruction << std::endl;

        addr += size;
    }

    /* If the end_pc address (addr) is not the beginning of a new line number
       block, then print ... to show that the previous instruction was
       not the last of the instructions associated with the previous source ine
       printed.  */
    if (auto it = m_line_number_map->find(addr);
        it == m_line_number_map->end())
        agent_out << "    ..." << std::endl;

    agent_out << std::endl << "End of disassembly." << std::endl;
}

void
code_object_t::disassemble_kernel(
  amd_dbgapi_architecture_id_t architecture_id,
  amd_dbgapi_global_address_t addr,
  bool const print_src)
{
    amd_dbgapi_process_id_t process_id;
    if (amd_dbgapi_code_object_get_info(m_code_object_id,
                                        AMD_DBGAPI_CODE_OBJECT_INFO_PROCESS,
                                        sizeof(process_id), &process_id)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_error("could not get the process from the agent");
    }

    amd_dbgapi_size_t largest_instruction_size;
    if (amd_dbgapi_architecture_get_info(
          architecture_id,
          AMD_DBGAPI_ARCHITECTURE_INFO_LARGEST_INSTRUCTION_SIZE,
          sizeof(largest_instruction_size), &largest_instruction_size)
        != AMD_DBGAPI_STATUS_SUCCESS)
    {
        agent_error("could not get the instruction size from the architecture");
    }

    /* Load the line number table, and low/high pc for all CUs.  */
    load_debug_info();

    amd_dbgapi_global_address_t start_addr = addr;

    auto symbol = find_symbol(start_addr);
    if (!symbol) {
        agent_out << std::endl << "No symbol found at address "
                  << std::hex << std::showbase << start_addr
                  << std::endl;
        return;
    }

    amd_dbgapi_global_address_t end_addr = addr + symbol->m_size;

    agent_out << std::endl << "Dump of assembler code";
    if (symbol) {
        agent_out << " for function " << symbol->m_name;
    }
    agent_out << ":" << std::endl;

    agent_out << "    code object: " << m_uri << std::endl;
    agent_out << "    loaded at: "
              << "[0x" << std::hex << m_load_address << "-"
              << "0x" << std::hex << (m_load_address + m_mem_size) << "]"
              << std::endl;

    /* Remember the start_pc address to print the first source line.  */
    amd_dbgapi_global_address_t saved_start_addr{ addr };

    std::string prev_file_name;
    size_t prev_line_number{ 0 };

    while (addr < end_addr)
    {
        if (!print_src) {
            goto do_disassemble;
        }

        if (auto it
            = m_line_number_map->find(addr == start_addr ? saved_start_addr : addr);
            it != m_line_number_map->end())
        {
            const std::string &file_name = it->second.first;
            size_t line_number = it->second.second;

            if (file_name != prev_file_name || line_number != prev_line_number) {
                agent_out << std::endl;
            }

            if (file_name != prev_file_name) {
                agent_out << file_name << ":" << std::endl;
            }

            /* If the source line for `addr` is a different line than the
               previous one printed, then print it.  If the previous line printed
               is in the same file and an earlier line, and if all the lines
               between it and the source line for `addr` have no associated
               instructions (indicated by their being no entries in the line
               number map that mention them), then display those lines as well as
               a source line block.  That allows the disassembly to show all the
               source file lines, including those that have no associated code.
            */
            if (file_name != prev_file_name || line_number != prev_line_number)
            {
                size_t first_line = line_number;
                size_t last_line = line_number;

                /* Find the first line to print between prev_line_number and
                   line_number that does not appear in the line number table.
                */
                if (file_name == prev_file_name
                    && (line_number + 1) > prev_line_number)
                {
                    while (--first_line > prev_line_number) {
                        if (std::find_if(
                              m_line_number_map->begin(),
                              m_line_number_map->end(),
                              [first_line, &file_name](
                                const std::remove_reference_t<decltype (
                                *m_line_number_map)>::value_type
                                &value)
                              {
                                  return file_name == value.second.first
                                    && first_line == value.second.second;
                              })
                            != m_line_number_map->end())
                            break;
                    }
                    /* First is either prev_line_number, or a line associated
                       with another address, so start at the next line.  */
                    ++first_line;
                }

                for (size_t line = first_line; line <= last_line; ++line) {
                    agent_out << std::setfill(' ') << std::setw(8) << std::left
                              << std::dec << line;

                    if (auto lines = get_source_file_index(file_name); !lines) {
                        agent_out << file_name << ": No such file or directory.";
                    } else if (line && line <= lines->get().size()) {
                        agent_out << lines->get()[line - 1];
                    }

                    agent_out << std::endl;
                }
            }

            prev_file_name = file_name;
            prev_line_number = line_number;

            /* If the start_pc address is not the begining of a line number
               block, then print ... to show that the following instruction is
               not the first in the block.  */
            if (addr == start_addr && start_addr != saved_start_addr) {
                agent_out << "    ..." << std::endl;
            }
        }

do_disassemble:
        std::vector<uint8_t> buffer(largest_instruction_size);

        amd_dbgapi_size_t size = buffer.size();
        if (amd_dbgapi_read_memory(
              process_id, AMD_DBGAPI_WAVE_NONE, AMD_DBGAPI_LANE_NONE,
              AMD_DBGAPI_ADDRESS_SPACE_GLOBAL, addr, &size, buffer.data())
            != AMD_DBGAPI_STATUS_SUCCESS)
        {
            agent_out << "Cannot access memory at address 0x" << std::hex << addr
                      << std::endl;
            break;
        }

        auto symbolizer = [](amd_dbgapi_symbolizer_id_t symbolizer_id,
                             amd_dbgapi_global_address_t address,
                             char **symbol_text)
        {
            auto &code_object = *reinterpret_cast<code_object_t *>(symbolizer_id);
            std::stringstream ss;

            ss << "0x" << std::hex << address;

            if (auto &&symbol = code_object.find_symbol(address)) {
                ss << " <" << symbol->m_name;
                ss << "+" << std::dec << (address - symbol->m_value);
                ss << ">";
            }

            *symbol_text = strdup(ss.str().c_str());
            return AMD_DBGAPI_STATUS_SUCCESS;
        };

        char *value;
        if (amd_dbgapi_disassemble_instruction(
              architecture_id, addr, &size, buffer.data(), &value,
              reinterpret_cast<amd_dbgapi_symbolizer_id_t>(this), symbolizer)
            != AMD_DBGAPI_STATUS_SUCCESS)
            agent_error("amd_dbgapi_disassemble_instruction failed");

        std::string instruction(value);
        free(value);

        agent_out << "    ";

        agent_out << "0x" << std::hex << addr;
        if (symbol) {
            agent_out << " <";
            if (addr >= symbol->m_value) {
                agent_out << "+" << std::dec << (addr - symbol->m_value);
            } else {
                agent_out << "-" << std::dec << (symbol->m_value - addr);
            }
            agent_out << ">";
        }

        agent_out << ":    " << instruction << std::endl;

        addr += size;
    }

    /* If the end_pc address (addr) is not the beginning of a new line number
       block, then print ... to show that the previous instruction was
       not the last of the instructions associated with the previous source ine
       printed.  */
    if (auto it = m_line_number_map->find(addr);
        it == m_line_number_map->end())
        agent_out << "    ..." << std::endl;

    agent_out << std::endl << "End of assembler dump." << std::endl;
}

bool
code_object_t::save(const std::string &directory) const
{
    agent_assert(is_open() && "code object is not opened");

    std::string name{ m_uri };

    size_t pos{};
    while ((pos = name.find_first_of(":/#?&="), pos) != std::string::npos) {
        name[pos] = '_';
    }

    std::string file_path = directory + '/' + name;
    std::ofstream file(file_path, std::ios::out | std::ios::binary);
    std::vector<char> buffer(lseek(*m_fd, 0, SEEK_END));

    ::lseek(*m_fd, 0, SEEK_SET);
    if (size_t size = ::read(*m_fd, buffer.data(), buffer.size());
        size != buffer.size())
    {
        return false;
    }

    file.write(buffer.data(), buffer.size());
    file.close();

    return file.good();
}

} // namespace amd::debug_agent

using namespace amd::debug_agent;

std::tuple
  < amd_dbgapi_process_id_t
  , std::map<amd_dbgapi_global_address_t, code_object_t>
  >
init_disassembly()
{
    set_log_level(log_level_t::warning);
    if (!agent_out.is_open()) {
        agent_out.copyfmt(std::cerr);
        agent_out.clear(std::cerr.rdstate());
        agent_out.basic_ios<char>::rdbuf(std::cerr.rdbuf());
    }

    DBGAPI_CHECK(amd_dbgapi_initialize(&dbgapi_callbacks));

    amd_dbgapi_process_id_t process_id;
    DBGAPI_CHECK(amd_dbgapi_process_attach(
                   (amd_dbgapi_client_process_id_t)&process_id,
                   &process_id));

    /* Check the runtime state.  */
    for (;;) {
        amd_dbgapi_event_id_t event_id;
        amd_dbgapi_event_kind_t event_kind;

        DBGAPI_CHECK(amd_dbgapi_process_next_pending_event(
                       process_id, &event_id, &event_kind));

        if (event_kind == AMD_DBGAPI_EVENT_KIND_RUNTIME) {
            amd_dbgapi_runtime_state_t runtime_state;

            DBGAPI_CHECK(amd_dbgapi_event_get_info(
                           event_id, AMD_DBGAPI_EVENT_INFO_RUNTIME_STATE,
                           sizeof(runtime_state), &runtime_state));

            switch (runtime_state) {
            case AMD_DBGAPI_RUNTIME_STATE_LOADED_SUCCESS:
                break;

            case AMD_DBGAPI_RUNTIME_STATE_UNLOADED:
                agent_error("invalid runtime state %d", runtime_state);

            case AMD_DBGAPI_RUNTIME_STATE_LOADED_ERROR_RESTRICTION:
                agent_error("unable to enable GPU debugging due to a "
                            "restriction error");
                break;
            }
        }

        /* No more events.  */
        if (event_kind == AMD_DBGAPI_EVENT_KIND_NONE) {
            break;
        }

        DBGAPI_CHECK(amd_dbgapi_event_processed(event_id));
    }

    auto ret = std::make_tuple(
      process_id,
      std::map<amd_dbgapi_global_address_t, code_object_t>{});
    auto &code_object_map = std::get<1>(ret);

    amd_dbgapi_code_object_id_t *code_objects_id;
    size_t code_object_count;
    DBGAPI_CHECK(amd_dbgapi_process_code_object_list(
                   process_id, &code_object_count, &code_objects_id, nullptr));

    for (size_t i = 0; i < code_object_count; ++i) {
        code_object_t code_object(code_objects_id[i]);

        code_object.open();
        if (!code_object.is_open()) {
            agent_warning("could not open code_object_%ld",
                          code_objects_id[i].handle);
            continue;
        }

        code_object_map.emplace(code_object.load_address(),
                                std::move(code_object));
    }
    free(code_objects_id);

    return ret;
}

void
disassemble(
  disassembly_mode const mode,
  amd_dbgapi_process_id_t const process_id,
  std::map<amd_dbgapi_global_address_t, code_object_t> &code_object_map,
  uint64_t const addr)
{
    /* This function is not thread-safe and not re-entrant.  */
    static std::mutex lock;
    if (!lock.try_lock()) {
        return;
    }
    /* Make sure the lock is released when this function returns.  */
    std::scoped_lock sl(std::adopt_lock, lock);

    code_object_t *code_object_found{ nullptr };
    if (auto it = code_object_map.upper_bound(addr);
        it != code_object_map.begin())
    {
        if (auto &&[load_address, code_object] = *std::prev(it);
            (addr - load_address) <= code_object.mem_size())
        {
            code_object_found = &code_object;
        }
    }

    if (code_object_found) {
        amd_dbgapi_architecture_id_t architecture_id;
        DBGAPI_CHECK(
          amd_dbgapi_get_architecture(
            code_object_found->elf_amdgpu_machine(),
            &architecture_id));
        switch (mode) {
        case disassembly_mode::KERNEL:
            code_object_found->disassemble_kernel(architecture_id, addr);
            break;
        case disassembly_mode::AROUND:
            code_object_found->disassemble_around(architecture_id, addr);
            break;
        }
    } else {
        // TODO: Add disassembly even if we did not find a code object
    }
}

void
print_pc_context(
  amd_dbgapi_process_id_t const process_id,
  std::map<amd_dbgapi_global_address_t, code_object_t> &code_object_map,
  amd_dbgapi_global_address_t const pc)
{
    disassemble(
      disassembly_mode::AROUND, process_id, code_object_map, pc);
}
