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

#include "lib/rocprofiler-sdk/hsa/memory_snapshot.hpp"

#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/code_object/code_object.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace memory_snapshot
{
namespace
{
// 1 MB page granularity for dirty-page diffing (design Section 4.3). At 4 bytes per page, 256 GB
// of HBM produces ~1 MB of checksum data, comfortably kept in host RAM.
constexpr size_t kPageBytes = 1ULL << 20;
// Default staging buffer size; matches Kerncap's KERNCAP_SNAPSHOT_CHUNK_BYTES default.
constexpr size_t kDefaultChunkBytes = 64ULL << 20;
constexpr size_t kMinChunkBytes     = 64ULL << 10;
// Skip implausibly large allocations (matches Kerncap's 1 TB guard in kerncap.hip:789).
constexpr size_t kMaxRegionBytes = 1ULL << 40;

// MurmurHash3 x86_32 (public domain, Austin Appleby). Per-page content hash for the dirty set.
uint32_t
murmur3_x86_32(const void* key, size_t len, uint32_t seed = 0)
{
    const auto*    data    = static_cast<const uint8_t*>(key);
    const size_t   nblocks = len / 4;
    uint32_t       h1      = seed;
    constexpr auto c1      = 0xcc9e2d51U;
    constexpr auto c2      = 0x1b873593U;

    const auto* blocks = reinterpret_cast<const uint32_t*>(data);
    for(size_t i = 0; i < nblocks; ++i)
    {
        uint32_t k1 = blocks[i];
        k1 *= c1;
        k1 = (k1 << 15) | (k1 >> 17);
        k1 *= c2;
        h1 ^= k1;
        h1 = (h1 << 13) | (h1 >> 19);
        h1 = h1 * 5 + 0xe6546b64U;
    }

    const uint8_t* tail = data + nblocks * 4;
    uint32_t       k1   = 0;
    switch(len & 3U)
    {
        case 3: k1 ^= static_cast<uint32_t>(tail[2]) << 16; [[fallthrough]];
        case 2: k1 ^= static_cast<uint32_t>(tail[1]) << 8; [[fallthrough]];
        case 1:
            k1 ^= static_cast<uint32_t>(tail[0]);
            k1 *= c1;
            k1 = (k1 << 15) | (k1 >> 17);
            k1 *= c2;
            h1 ^= k1;
            break;
        default: break;
    }

    h1 ^= static_cast<uint32_t>(len);
    h1 ^= h1 >> 16;
    h1 *= 0x85ebca6bU;
    h1 ^= h1 >> 13;
    h1 *= 0xc2b2ae35U;
    h1 ^= h1 >> 16;
    return h1;
}

hsa_status_t
dma_copy(void* dst, const void* src, size_t n)
{
    auto* core = get_core_table();
    if(!core || !core->hsa_memory_copy_fn) return HSA_STATUS_ERROR;
    return core->hsa_memory_copy_fn(dst, src, n);
}

size_t
resolve_chunk_bytes()
{
    size_t chunk = kDefaultChunkBytes;
    if(const char* env = std::getenv("ROCPROFILER_REPLAY_SNAPSHOT_CHUNK_BYTES"))
    {
        try
        {
            long long v = std::stoll(env);
            if(v >= static_cast<long long>(kMinChunkBytes)) chunk = static_cast<size_t>(v);
        } catch(const std::exception&)
        {}
    }
    return chunk;
}

std::string
resolve_output_dir()
{
    if(const char* env = std::getenv("ROCPROFILER_REPLAY_SNAPSHOT_DIR")) return env;
    return std::string("/tmp/rocprofiler_replay_") + std::to_string(getpid());
}
}  // namespace

Snapshot::Snapshot()
: output_dir_{resolve_output_dir()}
, chunk_bytes_{resolve_chunk_bytes()}
{
    mkdir(output_dir_.c_str(), 0755);
}

Snapshot::~Snapshot() { cleanup(); }

size_t
Snapshot::snap()
{
    auto inventory = memory_tracker::snap_inventory();

    blocks_.clear();
    blocks_.reserve(inventory.size());

    std::vector<char> buf(chunk_bytes_);  // staging buffer allocated ONCE, reused per region
    size_t            ok_count = 0;

    for(const auto& [ptr, info] : inventory)
    {
        if(info.size == 0 || info.size > kMaxRegionBytes) continue;

        mem_block blk;
        blk.gpu_addr = ptr;
        blk.size     = info.size;
        blk.is_vmem  = info.is_vmem;

        char path[256];
        std::snprintf(path,
                      sizeof(path),
                      "%s/region_%" PRIxPTR ".bin",
                      output_dir_.c_str(),
                      reinterpret_cast<uintptr_t>(ptr));
        blk.path = path;

        std::ofstream out(blk.path, std::ios::binary);
        if(!out)
        {
            ROCP_WARNING << "replay snapshot: failed to open " << blk.path << " for writing";
            continue;
        }

        bool region_ok = true;
        for(size_t off = 0; off < blk.size; off += chunk_bytes_)
        {
            size_t n   = std::min(chunk_bytes_, blk.size - off);
            void*  src = static_cast<char*>(ptr) + off;
            if(dma_copy(buf.data(), src, n) != HSA_STATUS_SUCCESS)
            {
                ROCP_WARNING << "replay snapshot: hsa_memory_copy failed for region " << blk.path
                             << " at offset " << off;
                region_ok = false;
                break;
            }
            out.write(buf.data(), static_cast<std::streamsize>(n));

            // Hash each 1 MB page in this chunk into checksums1 (pre-kernel state).
            for(size_t poff = 0; poff < n; poff += kPageBytes)
            {
                size_t plen = std::min(kPageBytes, n - poff);
                blk.checksums1.push_back(murmur3_x86_32(buf.data() + poff, plen));
            }
        }
        out.close();

        if(region_ok)
        {
            blocks_.push_back(std::move(blk));
            ++ok_count;
        }
        else
        {
            std::remove(blk.path.c_str());
        }
    }

    snap_module_variables(buf);

    ROCP_INFO << "replay snapshot: captured " << ok_count << "/" << inventory.size() << " regions";
    return ok_count;
}

size_t
Snapshot::snap_module_variables(std::vector<char>& staging)
{
    vars_.clear();

    // Discover variable symbols across all loaded executables (no D2H yet). The HSA iterate
    // callback cannot capture, so collect into a ctx passed by void*.
    struct discovered
    {
        uint64_t address;
        size_t   size;
    };
    std::vector<discovered> found;

    auto* core = get_core_table();
    if(!core || !core->hsa_executable_iterate_symbols_fn ||
       !core->hsa_executable_symbol_get_info_fn)
        return 0;

    code_object::iterate_loaded_code_objects([&](const code_object::hsa::code_object& co) {
        auto exec = co.hsa_executable;
        core->hsa_executable_iterate_symbols_fn(
            exec,
            [](hsa_executable_t, hsa_executable_symbol_t sym, void* data) -> hsa_status_t {
                auto* out  = static_cast<std::vector<discovered>*>(data);
                auto* core = get_core_table();

                hsa_symbol_kind_t kind{};
                if(core->hsa_executable_symbol_get_info_fn(
                       sym, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &kind) != HSA_STATUS_SUCCESS ||
                   kind != HSA_SYMBOL_KIND_VARIABLE)
                    return HSA_STATUS_SUCCESS;

                uint64_t addr = 0;
                size_t   size = 0;
                if(core->hsa_executable_symbol_get_info_fn(
                       sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_ADDRESS, &addr) !=
                       HSA_STATUS_SUCCESS ||
                   core->hsa_executable_symbol_get_info_fn(
                       sym, HSA_EXECUTABLE_SYMBOL_INFO_VARIABLE_SIZE, &size) != HSA_STATUS_SUCCESS)
                    return HSA_STATUS_SUCCESS;

                // Per-variable 1 GiB cap (Kerncap kerncap.hip:1018).
                if(addr == 0 || size == 0 || size > (1ULL << 30)) return HSA_STATUS_SUCCESS;

                out->push_back(discovered{addr, size});
                return HSA_STATUS_SUCCESS;
            },
            &found);
    });

    if(found.empty()) return 0;

    std::string vardir = output_dir_ + "/module_variables";
    mkdir(vardir.c_str(), 0755);

    size_t ok = 0;
    for(const auto& d : found)
    {
        char path[256];
        std::snprintf(
            path, sizeof(path), "%s/module_var_%" PRIx64 ".bin", vardir.c_str(), d.address);

        std::ofstream out(path, std::ios::binary);
        if(!out) continue;

        bool var_ok = true;
        for(size_t off = 0; off < d.size; off += chunk_bytes_)
        {
            size_t n   = std::min(chunk_bytes_, d.size - off);
            void*  src = reinterpret_cast<void*>(d.address + off);
            if(dma_copy(staging.data(), src, n) != HSA_STATUS_SUCCESS)
            {
                var_ok = false;
                break;
            }
            out.write(staging.data(), static_cast<std::streamsize>(n));
        }
        out.close();

        if(var_ok)
        {
            vars_.push_back(var_block{d.address, d.size, path});
            ++ok;
        }
        else
        {
            std::remove(path);
        }
    }

    ROCP_INFO << "replay snapshot: captured " << ok << "/" << found.size() << " module variables";
    return ok;
}

size_t
Snapshot::restore()
{
    std::vector<char> cur(chunk_bytes_);   // current GPU bytes for the chunk
    std::vector<char> disk(chunk_bytes_);  // snapshot bytes read back from disk
    size_t            dirty_pages_written = 0;

    for(const auto& blk : blocks_)
    {
        std::ifstream in(blk.path, std::ios::binary);
        if(!in)
        {
            ROCP_WARNING << "replay restore: failed to open " << blk.path << " for reading";
            continue;
        }

        size_t page_index = 0;
        for(size_t off = 0; off < blk.size; off += chunk_bytes_)
        {
            size_t n   = std::min(chunk_bytes_, blk.size - off);
            void*  gpu = static_cast<char*>(blk.gpu_addr) + off;

            // (1) DMA CURRENT GPU memory into staging -- not the on-disk snapshot (Section 3.3
            // bug 3). (2)/(3) hash fresh into a local checksums2 and diff against checksums1
            // every call (Section 3.3 bug 2).
            if(dma_copy(cur.data(), gpu, n) != HSA_STATUS_SUCCESS)
            {
                ROCP_WARNING << "replay restore: hsa_memory_copy (D2H) failed for region "
                             << blk.path << " at offset " << off;
                page_index += (n + kPageBytes - 1) / kPageBytes;
                continue;
            }

            // Read the corresponding snapshot bytes for this chunk from disk.
            in.seekg(static_cast<std::streamoff>(off), std::ios::beg);
            in.read(disk.data(), static_cast<std::streamsize>(n));

            // Walk pages, collecting maximal contiguous dirty runs and writing them back in one
            // DMA.
            size_t run_start = n;  // sentinel "no open run"
            auto   flush_run = [&](size_t run_end) {
                if(run_start == n) return;
                void*  dst = static_cast<char*>(blk.gpu_addr) + off + run_start;
                size_t len = run_end - run_start;
                if(dma_copy(dst, disk.data() + run_start, len) != HSA_STATUS_SUCCESS)
                {
                    ROCP_WARNING << "replay restore: hsa_memory_copy (H2D) failed writing dirty "
                                    "range for region "
                                 << blk.path;
                }
                run_start = n;
            };

            for(size_t poff = 0; poff < n; poff += kPageBytes)
            {
                size_t   plen      = std::min(kPageBytes, n - poff);
                uint32_t cur_hash  = murmur3_x86_32(cur.data() + poff, plen);
                uint32_t orig_hash = (page_index < blk.checksums1.size())
                                         ? blk.checksums1[page_index]
                                         : (cur_hash ^ 1U);  // force dirty if checksum missing
                bool     dirty     = (cur_hash != orig_hash);
                if(dirty)
                {
                    if(run_start == n) run_start = poff;
                    ++dirty_pages_written;
                }
                else
                {
                    flush_run(poff);
                }
                ++page_index;
            }
            flush_run(n);
        }
    }

    // Module variables are restored by full overwrite (small, and the design specifies full
    // overwrite rather than dirty-page diffing for them).
    restore_module_variables(disk);

    ROCP_INFO << "replay restore: wrote back " << dirty_pages_written << " dirty page(s)";
    return dirty_pages_written;
}

size_t
Snapshot::restore_module_variables(std::vector<char>& staging)
{
    size_t ok = 0;
    for(const auto& v : vars_)
    {
        std::ifstream in(v.path, std::ios::binary);
        if(!in) continue;

        bool var_ok = true;
        for(size_t off = 0; off < v.size; off += chunk_bytes_)
        {
            size_t n = std::min(chunk_bytes_, v.size - off);
            in.read(staging.data(), static_cast<std::streamsize>(n));
            void* dst = reinterpret_cast<void*>(v.address + off);
            if(dma_copy(dst, staging.data(), n) != HSA_STATUS_SUCCESS)
            {
                var_ok = false;
                break;
            }
        }
        if(var_ok) ++ok;
    }
    return ok;
}

void
Snapshot::cleanup()
{
    for(const auto& blk : blocks_)
    {
        if(!blk.path.empty()) std::remove(blk.path.c_str());
    }
    for(const auto& v : vars_)
    {
        if(!v.path.empty()) std::remove(v.path.c_str());
    }
    blocks_.clear();
    vars_.clear();
}
}  // namespace memory_snapshot
}  // namespace hsa
}  // namespace rocprofiler
