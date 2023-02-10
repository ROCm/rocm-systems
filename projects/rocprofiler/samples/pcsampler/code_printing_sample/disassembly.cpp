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

#include <algorithm>
#include <atomic>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cinttypes>
#include <cstdint>
#include <cstdio>

#include <sys/mman.h>

#include <hsa/hsa.h>
#include <amd-dbgapi/amd-dbgapi.h>
#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa_ven_amd_loader.h>

#include "rocprofiler.h"

#include "code_printing.hpp"
#include "program.hpp"

struct libc_freer {
    void operator()(char *p) { free(p); }
};

namespace util {

template <typename T, typename... Ts>
static void
hash_combine(size_t &hsh, T const& v, Ts const&... rest)
{
    hsh ^= std::hash<T>{}(v) + 0x9e3779b9 + (hsh << 6) + (hsh >> 2);
    (hash_combine(hsh, rest), ...);
}

} // namespace util

[[maybe_unused]]
static inline bool
operator==(hsa_executable_t const &l, hsa_executable_t const &r)
{
    return l.handle == r.handle;
}

[[maybe_unused]]
static inline bool
operator==(
  rocprofiler_kernel_dispatch_id_t const &l,
  rocprofiler_kernel_dispatch_id_t const &r)
{
    return l.value == r.value;
}

static inline bool
operator==(amd_dbgapi_process_id_t const &l, amd_dbgapi_process_id_t const &r)
{
    return l.handle == r.handle;
}

static inline bool
operator!=(amd_dbgapi_process_id_t const &l, amd_dbgapi_process_id_t const &r)
{
    return !(l == r);
}

namespace std {

template <>
struct hash<hsa_executable_t> {
    size_t operator()(hsa_executable_t const &v) const {
        size_t ret = 0;
        util::hash_combine(ret, v.handle);
        return ret;
    }
};

template <>
struct hash<rocprofiler_kernel_dispatch_id_t> {
    size_t operator()(rocprofiler_kernel_dispatch_id_t const &v) const {
        size_t ret = 0;
        util::hash_combine(ret, v.value);
        return ret;
    }
};

} // namespace std

struct disassembly_ctx_t {
    disassembly_ctx_t();
    ~disassembly_ctx_t();

    void disassemble_kernels(bool const reinitialize);
    void init();
    bool inited() const;
    void reset();

    amd_dbgapi_process_id_t process_id;
    std::map
      < amd_dbgapi_global_address_t
      , amd::debug_agent::code_object_t
      > codeobjs;
};

disassembly_ctx_t::disassembly_ctx_t()
: process_id(AMD_DBGAPI_PROCESS_NONE)
, codeobjs()
{}

disassembly_ctx_t::~disassembly_ctx_t()
{
    reset();
}

void
disassembly_ctx_t::disassemble_kernels(bool const reinitialize)
{
    if (reinitialize) {
        reset();
    }
    if (!inited()) {
        init();
    }

    auto it = codeobjs.begin();
    auto const end = codeobjs.end();
    auto const pred = [](decltype(*it) &x){
        /*
         * A lame filter for the kernels in the current file, because nothing
         * else in this little demo will have the URL prefix of `file://`.
         */
        return x.second.m_uri.find("file://", 0, 7) != std::string::npos;
    };
    while (end != (it = std::find_if(it, end, pred))) {
        auto &codeobj = it->second;
        codeobj.load_symbol_map();
        if (!codeobj.m_symbol_map) {
            fputs(PROGNAME ": error: failed to load symbol map\n", stderr);
            break;
        }

        for (auto const &sym : *codeobj.m_symbol_map) {
            auto const &addr = sym.first;
            ::disassemble(disassembly_mode::KERNEL, process_id, codeobjs, addr);
        }

        ++it;
    }
}

inline void
disassembly_ctx_t::init()
{
    std::tie(process_id, codeobjs) = init_disassembly();
}

inline bool
disassembly_ctx_t::inited() const
{
    return AMD_DBGAPI_PROCESS_NONE != process_id;
}

void
disassembly_ctx_t::reset()
{
    codeobjs.clear();
    if (AMD_DBGAPI_PROCESS_NONE.handle != process_id.handle) {
        amd_dbgapi_process_detach(process_id);
        amd_dbgapi_finalize();
        process_id = AMD_DBGAPI_PROCESS_NONE;
    }
}

static disassembly_ctx_t g_dis;

void
disassembly_disassemble_kernels(bool const reinitialize)
{
    g_dis.disassemble_kernels(reinitialize);
}

void
disassembly_print_pc_sample_context(amd_dbgapi_global_address_t const pc)
{
    if (!g_dis.inited()) {
        g_dis.init();
    }
    print_pc_context(g_dis.process_id, g_dis.codeobjs, pc);
}
