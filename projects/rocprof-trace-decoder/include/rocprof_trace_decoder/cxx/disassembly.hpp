// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// ============================================================================
// Disassembly backend selection
// ============================================================================
//
// The translation unit that includes this header must select exactly one of:
//
//   ROCPROF_TRACE_DECODER_USE_LLVM    — LLVM-C disassembler (no ROCm needed)
//   ROCPROF_TRACE_DECODER_USE_COMGR   — amd_comgr disassembler (legacy)
//   (neither)                         — disassembly disabled; ctor throws.
//                                       ELF parsing for symbols + va2fo still
//                                       works, so callers that only need
//                                       address translation are fine.
//
// `ROCPROF_TRACE_DECODER_COMGR_DISABLED` is a legacy compatibility alias for
// "no comgr"; it is honored but new code should prefer the explicit USE_*
// macros.
// ============================================================================

#if defined(ROCPROF_TRACE_DECODER_COMGR_DISABLED) && defined(ROCPROF_TRACE_DECODER_USE_COMGR)
#    error "ROCPROF_TRACE_DECODER_COMGR_DISABLED contradicts ROCPROF_TRACE_DECODER_USE_COMGR"
#endif

#if defined(ROCPROF_TRACE_DECODER_USE_LLVM) && defined(ROCPROF_TRACE_DECODER_USE_COMGR)
#    error "ROCPROF_TRACE_DECODER_USE_LLVM and ROCPROF_TRACE_DECODER_USE_COMGR are mutually exclusive"
#endif

#if defined(ROCPROF_TRACE_DECODER_USE_LLVM) || defined(ROCPROF_TRACE_DECODER_USE_COMGR)
#    define ROCPROF_TRACE_DECODER_HAS_DISASM 1
#else
#    define ROCPROF_TRACE_DECODER_HAS_DISASM 0
#endif

#ifdef ROCPROF_TRACE_DECODER_USE_COMGR
#    include <amd_comgr/amd_comgr.h>
#endif

#ifdef ROCPROF_TRACE_DECODER_USE_LLVM
#    include <llvm-c/Disassembler.h>
#    include <llvm-c/Target.h>
#endif

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ROCPROF_TRACE_DECODER_USE_COMGR
#    define THROW_COMGR(call)                                                                                          \
        if (amd_comgr_status_s status = call)                                                                          \
        {                                                                                                              \
            const char* reason = "";                                                                                   \
            amd_comgr_status_string(status, &reason);                                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << " code: " << status << " failed: " << reason << "\n";          \
            throw std::exception();                                                                                    \
        }

#    define RETURN_COMGR(call)                                                                                         \
        if (amd_comgr_status_s status = call)                                                                          \
        {                                                                                                              \
            const char* reason = "";                                                                                   \
            amd_comgr_status_string(status, &reason);                                                                  \
            std::cerr << __FILE__ << ':' << __LINE__ << " code: " << status << " failed: " << reason << "\n";          \
            return AMD_COMGR_STATUS_ERROR;                                                                             \
        }
#endif

namespace rocprof_trace_decoder
{
namespace codeobj
{
struct FallbackOp
{
    const char* str{};
    size_t size{};
};

inline FallbackOp get_fallback_op_gfx1250(uint32_t header)
{
    if ((header >> 7) == 0) return {"v_vop_generic ", 4};

    switch (header)
    {
        case 0b11001100: return {"v_vop3p_generic ", 8};
        case 0b11001111: return {"v_vopd3_generic ", 12};
        default: break;
    }

    header >>= 2;

    switch (header)
    {
        case 0b110001: return {"buffer_load_generic ", 12};
        case 0b111011: return {"global_load_generic ", 12};
        case 0b110100: return {"image_load_generic ", 12};
        case 0b110010: return {"v_vopd_generic ", 8};
        case 0b110101: return {"v_vopsd_generic ", 8};
        case 0b110110: return {"ds_generic ", 8};
        case 0b111101: return {"s_load_generic ", 8};
        default: break;
    }

    header >>= 4;
    if (header == 0b10) return {"s_sop ", 4};

    return {nullptr, 0};
}

// ---------------------------------------------------------------------------
// CodeObjectBinary — pure file I/O, no backend dependency.
// ---------------------------------------------------------------------------
class CodeObjectBinary
{
public:
    CodeObjectBinary(std::string _uri) : m_uri(std::move(_uri))
    {
        const std::string protocol_delim{"://"};

        size_t protocol_end = m_uri.find(protocol_delim);
        std::string protocol = m_uri.substr(0, protocol_end);
        protocol_end += protocol_delim.length();

        std::transform(
            protocol.begin(), protocol.end(), protocol.begin(), [](unsigned char c) { return std::tolower(c); }
        );

        std::string path;
        size_t path_end = m_uri.find_first_of("#?", protocol_end);
        if (path_end != std::string::npos) { path = m_uri.substr(protocol_end, path_end++ - protocol_end); }
        else { path = m_uri.substr(protocol_end); }

        /* %-decode the string.  */
        std::string decoded_path;
        decoded_path.reserve(path.length());
        for (size_t i = 0; i < path.length(); ++i)
        {
            if (path[i] == '%' && i + 2 < path.length() && std::isxdigit(path[i + 1]) != 0 &&
                std::isxdigit(path[i + 2]) != 0)
            {
                decoded_path += std::stoi(path.substr(i + 1, 2), nullptr, 16);
                i += 2;
            }
            else { decoded_path += path[i]; }
        }

        /* Tokenize the query/fragment.  */
        std::vector<std::string> tokens;
        size_t pos, last = path_end;
        while ((pos = m_uri.find('&', last)) != std::string::npos)
        {
            tokens.emplace_back(m_uri.substr(last, pos - last));
            last = pos + 1;
        }
        if (last != std::string::npos) { tokens.emplace_back(m_uri.substr(last)); }

        /* Create a tag-value map from the tokenized query/fragment.  */
        std::unordered_map<std::string, std::string> params;
        std::for_each(
            tokens.begin(),
            tokens.end(),
            [&](std::string& token)
            {
                size_t delim = token.find('=');
                if (delim != std::string::npos) { params.emplace(token.substr(0, delim), token.substr(delim + 1)); }
            }
        );

        buffer = std::vector<char>{};
        size_t offset = 0;
        size_t size = 0;

        if (auto offset_it = params.find("offset"); offset_it != params.end())
        {
            auto parsed_offset = std::stoull(offset_it->second, nullptr, 0);
            if (parsed_offset > std::numeric_limits<size_t>::max()) throw std::out_of_range("URI offset too large");
            offset = static_cast<size_t>(parsed_offset);
        }

        if (auto size_it = params.find("size"); size_it != params.end())
        {
            auto parsed_size = std::stoull(size_it->second, nullptr, 0);
            if (parsed_size > std::numeric_limits<size_t>::max()) throw std::out_of_range("URI size too large");
            if ((size = static_cast<size_t>(parsed_size)) == 0) return;
        }

        if (protocol == "memory") throw std::runtime_error(protocol + " protocol not supported!");

        std::ifstream file(decoded_path, std::ios::in | std::ios::binary);
        if (!file || !file.is_open()) throw std::runtime_error("could not open " + decoded_path);

        if (size == 0)
        {
            file.ignore(std::numeric_limits<std::streamsize>::max());
            size_t bytes = file.gcount();
            file.clear();

            if (bytes < offset) throw std::runtime_error("invalid uri " + decoded_path);

            size = bytes - offset;
        }

        file.seekg(offset, std::ios_base::beg);
        buffer.resize(size);
        file.read(buffer.data(), size);
    }

    std::string m_uri;
    std::vector<char> buffer;
};

struct SymbolInfo
{
    std::string name{};
    uint64_t faddr = 0;
    uint64_t vaddr = 0;
    uint64_t mem_size = 0;
};

// ---------------------------------------------------------------------------
// Inline ELF64-LE parsing helpers — used by all backends. Self-contained;
// no external ELF/HSA headers required (so this works on Windows and on
// systems without ROCm headers installed). Only what's needed for AMDGPU
// code objects emitted by the HSA loader.
// ---------------------------------------------------------------------------
namespace elf_inline
{

// EI_* indices and constants
constexpr uint8_t EI_NIDENT = 16;
constexpr uint8_t ELFMAG0 = 0x7f;
constexpr uint8_t ELFMAG1 = 'E';
constexpr uint8_t ELFMAG2 = 'L';
constexpr uint8_t ELFMAG3 = 'F';
constexpr uint8_t ELFCLASS64 = 2;
constexpr uint8_t ELFDATA2LSB = 1;

// e_machine
constexpr uint16_t EM_AMDGPU = 224;

// p_type
constexpr uint32_t PT_LOAD = 1;

// sh_type
constexpr uint32_t SHT_SYMTAB = 2;
constexpr uint32_t SHT_STRTAB = 3;
constexpr uint32_t SHT_DYNSYM = 11;

// st_info -> ELF64_ST_TYPE
constexpr uint8_t STT_FUNC = 2;

// AMDGPU e_flags (low byte = machine identifier). Only gfx9 and newer —
// the SQTT trace decoder doesn't target older ASICs. Values mirror the
// ROCm vendor enum in <hsa/amd_hsa_elf.h>; AMD's clang emits these and
// they take precedence over upstream LLVM's table when the two diverge
// (e.g. 0x049 is gfx1250 in ROCm but gfx1201 in upstream LLVM 18).
// Unknown values return nullptr; the caller raises a clear error.
constexpr uint32_t EF_AMDGPU_MACH_MASK = 0x0ff;

inline const char* amdgpu_mach_to_gfx(uint32_t mach)
{
    switch (mach & EF_AMDGPU_MACH_MASK)
    {
        case 0x02c: return "gfx900";
        case 0x02d: return "gfx902";
        case 0x02e: return "gfx904";
        case 0x02f: return "gfx906";
        case 0x030: return "gfx908";
        case 0x031: return "gfx909";
        case 0x032: return "gfx90c";
        case 0x033: return "gfx1010";
        case 0x034: return "gfx1011";
        case 0x035: return "gfx1012";
        case 0x036: return "gfx1030";
        case 0x037: return "gfx1031";
        case 0x038: return "gfx1032";
        case 0x039: return "gfx1033";
        case 0x03d: return "gfx1035";
        case 0x03e: return "gfx1034";
        case 0x03f: return "gfx90a";
        case 0x040: return "gfx940";
        case 0x041: return "gfx1100";
        case 0x042: return "gfx1013";
        case 0x043: return "gfx1150";
        case 0x044: return "gfx1103";
        case 0x045: return "gfx1036";
        case 0x046: return "gfx1101";
        case 0x047: return "gfx1102";
        case 0x048: return "gfx1200";
        case 0x049: return "gfx1250";
        case 0x04a: return "gfx1151";
        case 0x04b: return "gfx941";
        case 0x04c: return "gfx942";
        case 0x04e: return "gfx1201";
        case 0x04f: return "gfx950";
        case 0x051: return "gfx9-generic";
        case 0x052: return "gfx10-1-generic";
        case 0x053: return "gfx10-3-generic";
        case 0x054: return "gfx11-generic";
        case 0x055: return "gfx1152";
        case 0x058: return "gfx1153";
        case 0x059: return "gfx12-generic";
        case 0x05f: return "gfx9-4-generic";
        default: return nullptr; // unknown — caller must handle
    }
}

#pragma pack(push, 1)
struct Elf64Ehdr
{
    uint8_t e_ident[EI_NIDENT];
    uint16_t e_type;
    uint16_t e_machine;
    uint32_t e_version;
    uint64_t e_entry;
    uint64_t e_phoff;
    uint64_t e_shoff;
    uint32_t e_flags;
    uint16_t e_ehsize;
    uint16_t e_phentsize;
    uint16_t e_phnum;
    uint16_t e_shentsize;
    uint16_t e_shnum;
    uint16_t e_shstrndx;
};
struct Elf64Phdr
{
    uint32_t p_type;
    uint32_t p_flags;
    uint64_t p_offset;
    uint64_t p_vaddr;
    uint64_t p_paddr;
    uint64_t p_filesz;
    uint64_t p_memsz;
    uint64_t p_align;
};
struct Elf64Shdr
{
    uint32_t sh_name;
    uint32_t sh_type;
    uint64_t sh_flags;
    uint64_t sh_addr;
    uint64_t sh_offset;
    uint64_t sh_size;
    uint32_t sh_link;
    uint32_t sh_info;
    uint64_t sh_addralign;
    uint64_t sh_entsize;
};
struct Elf64Sym
{
    uint32_t st_name;
    uint8_t st_info;
    uint8_t st_other;
    uint16_t st_shndx;
    uint64_t st_value;
    uint64_t st_size;
};
#pragma pack(pop)

inline bool is_valid_elf64(const char* data, uint64_t size)
{
    if (size < sizeof(Elf64Ehdr)) return false;
    auto* eh = reinterpret_cast<const Elf64Ehdr*>(data);
    return eh->e_ident[0] == ELFMAG0 && eh->e_ident[1] == ELFMAG1 && eh->e_ident[2] == ELFMAG2 &&
           eh->e_ident[3] == ELFMAG3 && eh->e_ident[4] == ELFCLASS64 && eh->e_ident[5] == ELFDATA2LSB;
}

inline const Elf64Ehdr* elf_header(const char* data) { return reinterpret_cast<const Elf64Ehdr*>(data); }

inline const Elf64Phdr* phdr_at(const char* data, uint16_t i)
{
    auto* eh = elf_header(data);
    return reinterpret_cast<const Elf64Phdr*>(data + eh->e_phoff + i * eh->e_phentsize);
}

inline const Elf64Shdr* shdr_at(const char* data, uint16_t i)
{
    auto* eh = elf_header(data);
    return reinterpret_cast<const Elf64Shdr*>(data + eh->e_shoff + i * eh->e_shentsize);
}

// vaddr -> file offset, using PT_LOAD segments. Returns nullopt if the address
// isn't covered or falls in a NOBITS region (filesz < memsz).
inline std::optional<uint64_t> va_to_fo(const char* data, uint64_t size, uint64_t va)
{
    if (!is_valid_elf64(data, size)) return std::nullopt;
    auto* eh = elf_header(data);
    if (eh->e_phoff + uint64_t(eh->e_phnum) * eh->e_phentsize > size) return std::nullopt;

    for (uint16_t i = 0; i < eh->e_phnum; ++i)
    {
        auto* ph = phdr_at(data, i);
        if (ph->p_type != PT_LOAD) continue;
        if (va < ph->p_vaddr || va >= ph->p_vaddr + ph->p_memsz) continue;
        uint64_t delta = va - ph->p_vaddr;
        if (delta >= ph->p_filesz) return std::nullopt; // NOBITS tail
        return ph->p_offset + delta;
    }
    return std::nullopt;
}

// Walk SHT_SYMTAB (preferred) or SHT_DYNSYM, emitting STT_FUNC entries.
// Visitor: void(std::string name, uint64_t vaddr, uint64_t size).
template <class Visitor> inline void for_each_func_symbol(const char* data, uint64_t size, Visitor&& v)
{
    if (!is_valid_elf64(data, size)) return;
    auto* eh = elf_header(data);
    if (eh->e_shoff + uint64_t(eh->e_shnum) * eh->e_shentsize > size) return;

    auto walk = [&](uint32_t which)
    {
        for (uint16_t i = 0; i < eh->e_shnum; ++i)
        {
            auto* sh = shdr_at(data, i);
            if (sh->sh_type != which) continue;
            if (sh->sh_entsize == 0 || sh->sh_link >= eh->e_shnum) continue;
            if (sh->sh_offset + sh->sh_size > size) continue;

            auto* str_sh = shdr_at(data, sh->sh_link);
            if (str_sh->sh_type != SHT_STRTAB) continue;
            if (str_sh->sh_offset + str_sh->sh_size > size) continue;

            const char* str_base = data + str_sh->sh_offset;
            uint64_t nsym = sh->sh_size / sh->sh_entsize;
            for (uint64_t k = 0; k < nsym; ++k)
            {
                auto* sym = reinterpret_cast<const Elf64Sym*>(data + sh->sh_offset + k * sh->sh_entsize);
                if ((sym->st_info & 0xf) != STT_FUNC) continue;
                if (sym->st_name >= str_sh->sh_size) continue;

                const char* nm = str_base + sym->st_name;
                // Bounded length — stop at NUL or end of strtab.
                size_t maxlen = str_sh->sh_size - sym->st_name;
                size_t nlen = ::strnlen(nm, maxlen);
                v(std::string(nm, nlen), sym->st_value, sym->st_size);
            }
            return; // First matching section wins (avoids dup with DYNSYM).
        }
    };

    walk(SHT_SYMTAB);
    walk(SHT_DYNSYM);
}

inline uint32_t amdgpu_mach(const char* data, uint64_t size)
{
    if (!is_valid_elf64(data, size)) return 0;
    auto* eh = elf_header(data);
    if (eh->e_machine != EM_AMDGPU) return 0;
    return eh->e_flags & EF_AMDGPU_MACH_MASK;
}

} // namespace elf_inline

// ---------------------------------------------------------------------------
// DisassemblyInstance
//
// API contract (preserved across all backends so code_printing.hpp need not
// change):
//   ctor(buffer, size)              — build per-codeobj state; throw on failure
//   ~DisassemblyInstance()          — release backend resources
//   ReadInstruction(faddr) -> {asm, size}
//   GetKernelMap() -> map<vaddr, SymbolInfo>&
//   va2fo(vaddr) -> optional<faddr>
//
// ELF parsing (symbols + va2fo) is always done inline so callers that don't
// disassemble (e.g. funcmap-only consumers) work even with no backend.
// ---------------------------------------------------------------------------
class DisassemblyInstance
{
public:
    DisassemblyInstance(const char* codeobj_data, uint64_t codeobj_size)
    {
        buffer.assign(codeobj_data, codeobj_data + codeobj_size);

        if (!elf_inline::is_valid_elf64(buffer.data(), buffer.size()))
            throw std::runtime_error("DisassemblyInstance: not a valid 64-bit LE ELF");

#if defined(ROCPROF_TRACE_DECODER_USE_COMGR)
        THROW_COMGR(amd_comgr_create_data(AMD_COMGR_DATA_KIND_EXECUTABLE, &data));
        THROW_COMGR(amd_comgr_set_data(data, buffer.size(), buffer.data()));

        size_t isa_size = 128;
        std::string input_isa{};
        input_isa.resize(isa_size);
        THROW_COMGR(amd_comgr_get_data_isa_name(data, &isa_size, input_isa.data()));

        THROW_COMGR(amd_comgr_create_disassembly_info(
            input_isa.data(),
            &DisassemblyInstance::comgr_memory_callback,
            &DisassemblyInstance::comgr_inst_callback,
            [](uint64_t, void*) {},
            &info
        ));
        if (input_isa.find("gfx125") != std::string::npos) gfxip = 125;

#elif defined(ROCPROF_TRACE_DECODER_USE_LLVM)
        const char* gfx = elf_inline::amdgpu_mach_to_gfx(elf_inline::amdgpu_mach(buffer.data(), buffer.size()));
        if (!gfx)
        {
            std::cerr << "DisassemblyInstance: unknown AMDGPU mach 0x" << std::hex
                      << elf_inline::amdgpu_mach(buffer.data(), buffer.size()) << std::dec
                      << " — add it to amdgpu_mach_to_gfx() in disassembly.hpp\n";
            throw std::runtime_error("DisassemblyInstance: unrecognised AMDGPU mach");
        }
        ensure_llvm_amdgpu_inited();
        // Triple `amdgcn-amd-amdhsa`, CPU = gfx target, no extra features.
        // The disassembler doesn't need xnack/sramecc settings — those affect
        // codegen, not the bytewise decode of opcodes.
        dc = LLVMCreateDisasmCPU("amdgcn-amd-amdhsa", gfx, nullptr, 0, nullptr, nullptr);
        if (!dc)
        {
            std::cerr << "DisassemblyInstance: LLVMCreateDisasmCPU(amdgcn-amd-amdhsa, " << gfx
                      << ") failed — is LLVM built with the AMDGPU disassembler?\n";
            throw std::runtime_error("DisassemblyInstance: LLVM disasm context creation failed");
        }
        // Print operands in numeric form (matches comgr default behavior closer
        // than the AT&T-style symbolic operands).
        LLVMSetDisasmOptions(dc, LLVMDisassembler_Option_PrintImmHex);
#else
            // No backend selected — disassembly is unavailable. ELF helpers below
            // still work, but ReadInstruction() will throw.
#endif
    }

    ~DisassemblyInstance()
    {
#if defined(ROCPROF_TRACE_DECODER_USE_COMGR)
        amd_comgr_release_data(data);
        amd_comgr_destroy_disassembly_info(info);
#elif defined(ROCPROF_TRACE_DECODER_USE_LLVM)
        if (dc) LLVMDisasmDispose(dc);
#endif
    }

    DisassemblyInstance(const DisassemblyInstance&) = delete;
    DisassemblyInstance& operator=(const DisassemblyInstance&) = delete;

    std::pair<std::string, size_t> ReadInstruction(uint64_t faddr)
    {
#if defined(ROCPROF_TRACE_DECODER_USE_COMGR)
        uint64_t size_read;
        uint64_t addr_in_buffer = reinterpret_cast<uint64_t>(buffer.data()) + faddr;

        auto _status = amd_comgr_disassemble_instruction(info, addr_in_buffer, (void*) this, &size_read);

        if (_status != AMD_COMGR_STATUS_SUCCESS)
        {
            if (faddr + 4 > buffer.size()) THROW_COMGR(_status);

            uint32_t read = 0;
            std::memcpy(&read, buffer.data() + faddr, 4);

            FallbackOp fallback{};

            if (gfxip == 125) fallback = get_fallback_op_gfx1250(read >> 24);
            if (fallback.str == nullptr || fallback.size == 0) THROW_COMGR(_status);

            this->last_instruction = fallback.str;
            size_read = fallback.size;
        }

        return {std::move(this->last_instruction), size_read};
#elif defined(ROCPROF_TRACE_DECODER_USE_LLVM)
        if (!dc) throw std::runtime_error("DisassemblyInstance: LLVM disasm not initialised");
        if (faddr >= buffer.size()) throw std::runtime_error("DisassemblyInstance: faddr OOB");

        char out[256] = {};
        uint64_t pc = faddr; // Treat file offset as the disasm PC; we
                             // only need a relative position for sizing.
        size_t inst_size = LLVMDisasmInstruction(
            dc, reinterpret_cast<uint8_t*>(buffer.data()) + faddr, buffer.size() - faddr, pc, out, sizeof(out)
        );
        if (inst_size == 0)
        {
            // Mirror amd_comgr behaviour: surface as throw, callers downgrade
            // to "no instruction here" via existing try/catch.
            throw std::runtime_error("DisassemblyInstance: LLVM failed to decode instruction");
        }
        // Strip leading whitespace to match comgr inst_callback behaviour.
        const char* p = out;
        while (*p == '\t' || *p == ' ') ++p;
        return {std::string(p), inst_size};
#else
        (void) faddr;
        throw std::runtime_error("DisassemblyInstance: built without a disassembly backend");
#endif
    }

    std::map<uint64_t, SymbolInfo>& GetKernelMap()
    {
        symbol_map.clear();
        elf_inline::for_each_func_symbol(
            buffer.data(),
            buffer.size(),
            [this](std::string name, uint64_t vaddr, uint64_t size)
            {
                if (auto faddr = va2fo(vaddr)) symbol_map[vaddr] = SymbolInfo{std::move(name), *faddr, vaddr, size};
            }
        );
        return symbol_map;
    }

    std::optional<uint64_t> va2fo(uint64_t va) const { return elf_inline::va_to_fo(buffer.data(), buffer.size(), va); }

    int gfxip{};
    std::vector<char> buffer{};
    std::map<uint64_t, SymbolInfo> symbol_map{};

#if defined(ROCPROF_TRACE_DECODER_USE_COMGR)
private:
    static uint64_t comgr_memory_callback(uint64_t from, char* to, uint64_t size, void* user_data)
    {
        DisassemblyInstance& instance = *static_cast<DisassemblyInstance*>(user_data);
        int64_t copysize =
            reinterpret_cast<int64_t>(instance.buffer.data()) + instance.buffer.size() - static_cast<int64_t>(from);
        copysize = std::min<int64_t>(size, copysize);
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        std::memcpy(to, (char*) from, copysize);
        return copysize;
    }

    static void comgr_inst_callback(const char* instruction, void* user_data)
    {
        DisassemblyInstance& instance = *static_cast<DisassemblyInstance*>(user_data);
        if (!instruction) return;
        while (*instruction == '\t' || *instruction == ' ') instruction++;
        instance.last_instruction = instruction;
    }

    std::string last_instruction{};
    amd_comgr_disassembly_info_t info{};
    amd_comgr_data_t data{};
#endif

#if defined(ROCPROF_TRACE_DECODER_USE_LLVM)
private:
    static void ensure_llvm_amdgpu_inited()
    {
        // LLVM's Initialize* functions are idempotent and cheap; calling them
        // once per process is the standard pattern. std::call_once is the
        // simplest way to make this safe across threads / multiple
        // DisassemblyInstance ctors.
        static std::once_flag flag;
        std::call_once(
            flag,
            []
            {
                LLVMInitializeAMDGPUTargetInfo();
                LLVMInitializeAMDGPUTargetMC();
                LLVMInitializeAMDGPUDisassembler();
            }
        );
    }

    LLVMDisasmContextRef dc{nullptr};
#endif
};

} // namespace codeobj
} // namespace rocprof_trace_decoder
