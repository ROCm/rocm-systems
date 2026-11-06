// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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

// ctrl_v3: aqlprofile v2 API test with flexible PMC CLI
//
// Usage:
//   ctrl_v3 --list-gpus                                           list all GPU agents and exit
//   ctrl_v3 --pmc SQ_WAVES,GRBM_COUNT                             named counters (built-in table)
//   ctrl_v3 --pmc-raw 4:0:4,4:0:5                                 raw block:idx:event_id triples (comma-separated)
//   ctrl_v3 --pmc-raw SQ:0:4,SQ:0:5                               raw triples, block given by name instead of number
//   ctrl_v3 --pmc-scan                                            scan all valid events
//   ctrl_v3 --pmc-scan --block 6                                  scan only SQ block events (by number)
//   ctrl_v3 --pmc-scan --block SQ                                 scan only SQ block events (by name)
//   ctrl_v3 --list-blocks                                         list supported block numbers and names and exit
//   ctrl_v3 --list-blocks-gfx gfx1250                             list supported blocks for gfxXXXX (stub agent, no GPU)
//   ctrl_v3 --list-events                                         list all valid events and exit
//   ctrl_v3 --list-events --block 6                               list only SQ block events (by number) and exit
//   ctrl_v3 --list-events --block SQ                              list only SQ block events (by name) and exit
//   ctrl_v3 --list-dimensions                                     list coordinate dimensions for all valid events
//   ctrl_v3 --list-dimensions --block SQ                          list coordinate dimensions for SQ block events
//   ctrl_v3 --pmc SQ_WAVES,SQ_CYCLES,GRBM_COUNT --pmc-batch 2     split counters across dispatches (2 per batch)
//   ctrl_v3 --pmc-raw 6:0:4,6:0:2,4:0:0 --pmc-batch 2             raw events also support --pmc-batch batching
//   ctrl_v3 --gpu 1 --iterations 3 --verbose                      multi-iteration with result dump
//   ctrl_v3 --pmc SQ_WAVES --dry-run                              build+decode packets, skip GPU dispatch (real GPU)
//   ctrl_v3 --pmc SQ_WAVES --dry-run-gfx gfx1250                  build+decode packets without any GPU (stub agent)

// These must come before ctrl/test_kernel.h which uses std::string and malloc
#include <cstdlib>
#include <string>

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ven_amd_aqlprofile.h"
#include "lib/aqlprofile/aqlprofile.hpp"

#include "ctrl/test_kernel.h"
#include "simple_convolution/simple_convolution.h"
#include "util/hsa_rsrc_factory.h"
#include "util/kfd_pmc_enable.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <sstream>
#include <vector>

#ifdef CTRL_V3_EMBEDDED_HSACO
#include "embedded_hsaco.h"
#endif

// ---------------------------------------------------------------------------
// aqlprofile_pmc_event_t ordering for std::map
// ---------------------------------------------------------------------------

struct PmcEventCmp {
    bool operator()(const aqlprofile_pmc_event_t& a, const aqlprofile_pmc_event_t& b) const {
        if (a.block_name  != b.block_name)  return a.block_name  < b.block_name;
        if (a.block_index != b.block_index) return a.block_index < b.block_index;
        return a.event_id < b.event_id;
    }
};

using ResultMap   = std::map<aqlprofile_pmc_event_t, uint64_t, PmcEventCmp>;
// per-instance storage: event -> [(counter_id, value), ...]
using InstanceMap = std::map<aqlprofile_pmc_event_t,
                             std::vector<std::pair<uint64_t, uint64_t>>,
                             PmcEventCmp>;

// ---------------------------------------------------------------------------
// Built-in counter table for --pmc <name> support.
// aqlprofile v2's iterate_event_ids yields coordinate-dimension metadata,
// not named counter strings, so name→event mapping is hardcoded here.
// Use --pmc-raw for any counter not listed here.
// ---------------------------------------------------------------------------

struct BuiltinCounter {
    const char*                          name;
    hsa_ven_amd_aqlprofile_block_name_t  block;
    int                                  event_id;
};

static const BuiltinCounter kBuiltin[] = {
    {"GRBM_COUNT",          HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM,  0},
    {"GRBM_GUI_ACTIVE",     HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM,  2},
    {"SQ_CYCLES",           HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,    2},
    {"SQ_BUSY_CYCLES",      HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,    3},
    {"SQ_WAVES",            HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,    4},
    {"SQ_ITEMS",            HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,   14},
    {"SQ_WAVE_READY",       HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,   47},
    {"CPC_ALWAYS_COUNT",    HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC,   0},
    {"CPF_ALWAYS_COUNT",    HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF,   0},
    {"SPI_CSN_BUSY",        HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI,  46},
    {"SPI_CSN_THREADGROUPS",HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI,  47},
    {"TCC_CYCLE",           HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC,   1},
    {"TCC_REQ",             HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC,   3},
};
static const int kBuiltinCount = static_cast<int>(sizeof(kBuiltin) / sizeof(kBuiltin[0]));

// Look up a built-in counter; returns nullptr if not found.
static const BuiltinCounter* FindBuiltin(const char* name) {
    for (int i = 0; i < kBuiltinCount; i++)
        if (strcmp(kBuiltin[i].name, name) == 0) return &kBuiltin[i];
    return nullptr;
}

// Block number → name table (covers HSA vendor enum + aqlprofile v2 extensions).
struct BlockInfo {
    int         number;
    const char* name;
};
static const BlockInfo kBlocks[] = {
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC,    "CPC"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF,    "CPF"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS,    "GDS"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM,   "GRBM"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, "GRBMSE"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI,    "SPI"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ,     "SQ"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS,   "SQCS"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM,   "SRBM"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX,     "SX"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA,     "TA"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA,    "TCA"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC,    "TCC"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP,    "TCP"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD,     "TD"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB,  "MCARB"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB,  "MCHUB"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM,"MCMCBVM"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ,  "MCSEQ"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, "MCVML2"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, "MCXBAR"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC,    "ATC"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2,  "ATCL2"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA,   "GCEA"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB,    "RPB"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA,   "SDMA"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A,   "GL1A"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C,   "GL1C"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A,   "GL2A"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C,   "GL2C"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR,    "GCR"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GUS,    "GUS"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_UMC,    "UMC"},
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MMEA,   "MMEA"},
    // aqlprofile v2 extensions
    {AQLPROFILE_BLOCK_NAME_CPG,        "CPG"},
    {AQLPROFILE_BLOCK_NAME_RLC,        "RLC"},
    {AQLPROFILE_BLOCK_NAME_CHA,        "CHA"},
    {AQLPROFILE_BLOCK_NAME_CHC,        "CHC"},
    {AQLPROFILE_BLOCK_NAME_GC_CANE,    "GC_CANE"},
    {AQLPROFILE_BLOCK_NAME_GC_FFBM,    "GC_FFBM"},
    {AQLPROFILE_BLOCK_NAME_GC_L2TLB,   "GC_L2TLB"},
    {AQLPROFILE_BLOCK_NAME_GC_UTCL1,   "GC_UTCL1"},
    {AQLPROFILE_BLOCK_NAME_GC_UTCL2,   "GC_UTCL2"},
    {AQLPROFILE_BLOCK_NAME_GC_VML2,    "GC_VML2"},
    {AQLPROFILE_BLOCK_NAME_GC_VML2_SPM,"GC_VML2_SPM"},
    {AQLPROFILE_BLOCK_NAME_GCEA_SE,    "GCEA_SE"},
    {AQLPROFILE_BLOCK_NAME_GRBMH,      "GRBMH"},
    {AQLPROFILE_BLOCK_NAME_SQG,        "SQG"},
    {AQLPROFILE_BLOCK_NAME_SP,         "SP"},
    {AQLPROFILE_BLOCK_NAME_GLARBA,     "GLARBA"},
    {AQLPROFILE_BLOCK_NAME_GLARBC,     "GLARBC"},
    {AQLPROFILE_BLOCK_NAME_GRBMA,      "GRBMA"},
    {AQLPROFILE_BLOCK_NAME_GC_NHTTLB,  "GC_NHTTLB"},
};
static const int kBlockCount = static_cast<int>(sizeof(kBlocks) / sizeof(kBlocks[0]));

static const char* BlockName(int block_number) {
    for (int i = 0; i < kBlockCount; i++)
        if (kBlocks[i].number == block_number) return kBlocks[i].name;
    return nullptr;
}

// Resolve a block token from --block / --pmc-raw to a numeric block id.
// Accepts a decimal number or a block name (exact match against kBlocks).
// Returns -1 if the token is a name that isn't found.
static int ResolveBlockToken(const char* tok) {
    if (tok[0] >= '0' && tok[0] <= '9') return atoi(tok);
    for (int i = 0; i < kBlockCount; i++)
        if (strcmp(kBlocks[i].name, tok) == 0) return kBlocks[i].number;
    return -1;
}

// Query the hardware counter slot count for a block via aqlprofile_get_pmc_info.
// Returns 0 on failure.
static uint32_t QueryBlockCounters(aqlprofile_agent_handle_t agent,
                                   hsa_ven_amd_aqlprofile_block_name_t block,
                                   int first_valid_event_id) {
    aqlprofile_pmc_event_t ev{};
    ev.block_name  = block;
    ev.block_index = 0;
    ev.event_id    = first_valid_event_id;

    aqlprofile_pmc_profile_t profile{};
    profile.agent       = agent;
    profile.events      = &ev;
    profile.event_count = 1;

    uint32_t count = 0;
    hsa_status_t st = aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_COUNTERS, &count);
    return (st == HSA_STATUS_SUCCESS) ? count : 0;
}

// Query the number of instances (max block_index + 1) for a block via
// aqlprofile_get_pmc_info / AQLPROFILE_INFO_BLOCK_ID (same API
// rocprofiler::aql::get_query_info() uses). Returns 0 on failure (unknown
// block name or query not supported), in which case callers should fall
// back to a conservative scan bound.
static uint32_t QueryBlockInstances(aqlprofile_agent_handle_t agent, int block_number) {
    const char* name = BlockName(block_number);
    if (!name) return 0;

    aqlprofile_pmc_profile_t profile{};
    profile.agent       = agent;
    profile.events      = nullptr;
    profile.event_count = 0;

    hsa_ven_amd_aqlprofile_id_query_t query{name, 0, 0};
    hsa_status_t st = aqlprofile_get_pmc_info(&profile, AQLPROFILE_INFO_BLOCK_ID, &query);
    return (st == HSA_STATUS_SUCCESS) ? query.instance_count : 0;
}

// Look up a name for a known event (block_index ignored — matches on block+event_id).
static const char* BuiltinName(const aqlprofile_pmc_event_t& ev) {
    for (int i = 0; i < kBuiltinCount; i++)
        if (kBuiltin[i].block == ev.block_name && kBuiltin[i].event_id == (int)ev.event_id)
            return kBuiltin[i].name;
    return nullptr;
}

// Binary-search the real max valid event_id for a block via aqlprofile_validate_pmc_event
// (block_index 0, which is always valid for any block the driver recognizes). Relies on
// aqlprofile's event_id_max bounds check (core/pm4_factory.h GetBlockInfo) actually being
// enforced, and assumes validity is monotonic: event_ids 0..max are valid, beyond that
// invalid. Returns -1 if event_id 0 itself doesn't validate (unknown/unsupported block).
static int FindMaxEventId(aqlprofile_agent_handle_t agent,
                           hsa_ven_amd_aqlprofile_block_name_t block,
                           int probe_ceiling = 4096) {
    aqlprofile_pmc_event_flags_t flags{};
    flags.raw = 0;

    auto validate = [&](int eid) {
        aqlprofile_pmc_event_t ev{};
        ev.block_name  = block;
        ev.block_index = 0;
        ev.event_id    = eid;
        ev.flags       = flags;
        bool ok = false;
        aqlprofile_validate_pmc_event(agent, &ev, &ok);
        return ok;
    };

    if (!validate(0)) return -1;

    // Exponential search for a (lo valid, hi invalid) bracket.
    int lo = 0, hi = 1;
    while (hi < probe_ceiling && validate(hi)) {
        lo = hi;
        hi *= 2;
    }
    if (hi >= probe_ceiling)
        return validate(probe_ceiling) ? probe_ceiling : lo;

    // Binary search the exact valid/invalid boundary within (lo, hi).
    while (lo + 1 < hi) {
        int mid = lo + (hi - lo) / 2;
        if (validate(mid)) lo = mid;
        else hi = mid;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// PMC memory callbacks (mirrors test/integration/counter.cpp)
// ---------------------------------------------------------------------------

struct MemCtx {
    hsa_agent_t            gpu_agent;
    hsa_amd_memory_pool_t  cpu_pool;
    hsa_amd_memory_pool_t  kernarg_pool;
    // non-null when --dry-run: intercepts copies to record PM4 DWORDs
    std::vector<std::vector<uint32_t>>* captured = nullptr;
};

static hsa_status_t PmcAlloc(void** ptr, size_t size,
                               aqlprofile_buffer_desc_flags_t flags, void* ud) {
    auto* ctx = static_cast<MemCtx*>(ud);
    auto pool = (flags.memory_hint == AQLPROFILE_MEMORY_HINT_DEVICE_UNCACHED)
                 ? ctx->kernarg_pool : ctx->cpu_pool;
    hsa_status_t s = hsa_amd_memory_pool_allocate(pool, size,
                                                   HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, ptr);
    if (s != HSA_STATUS_SUCCESS) return s;
    hsa_amd_memory_fill(*ptr, 0u, size / sizeof(uint32_t));
    return hsa_amd_agents_allow_access(1, &ctx->gpu_agent, nullptr, *ptr);
}

static void PmcFree(void* ptr, void* /*ud*/) {
    if (ptr) hsa_amd_memory_pool_free(ptr);
}

static hsa_status_t PmcCopy(void* dst, const void* src, size_t size, void* ud) {
    if (size == 0) return HSA_STATUS_SUCCESS;
    auto* ctx = static_cast<MemCtx*>(ud);
    if (ctx && ctx->captured && size >= sizeof(uint32_t)) {
        const uint32_t* dwords = static_cast<const uint32_t*>(src);
        size_t n = size / sizeof(uint32_t);
        ctx->captured->emplace_back(dwords, dwords + n);
    }
    return hsa_memory_copy(dst, src, size);
}

// ---------------------------------------------------------------------------
// Dry-run: malloc-based memory callbacks (no HSA / no GPU required)
// ---------------------------------------------------------------------------

static hsa_status_t DryRunAlloc(void** ptr, size_t size,
                                 aqlprofile_buffer_desc_flags_t /*flags*/, void* /*ud*/) {
    *ptr = calloc(1, size);
    return *ptr ? HSA_STATUS_SUCCESS : HSA_STATUS_ERROR_OUT_OF_RESOURCES;
}

static void DryRunFree(void* ptr, void* /*ud*/) { free(ptr); }

static hsa_status_t DryRunCopy(void* dst, const void* src, size_t size, void* ud) {
    if (size == 0) return HSA_STATUS_SUCCESS;
    memcpy(dst, src, size);
    // Record each copied buffer for later decode
    auto* captured = static_cast<std::vector<std::vector<uint32_t>>*>(ud);
    const uint32_t* dwords = static_cast<const uint32_t*>(src);
    size_t n = size / sizeof(uint32_t);
    captured->emplace_back(dwords, dwords + n);
    return HSA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// PM4 packet decoder: prints a raw DWORD stream to stdout
// ---------------------------------------------------------------------------

static const char* Pm4OpcodeName(uint32_t opcode) {
    switch (opcode) {
        case 0x10: return "NOP";
        case 0x14: return "EVENT_WRITE";
        case 0x04: return "SET_SH_REG";
        case 0x76: return "SET_SH_REG_INDEX";
        case 0x1D: return "SET_PCONFIG_REG";
        case 0x37: return "SET_UCONFIG_REG";
        case 0x79: return "SET_UCONFIG_REG_INDEX";
        case 0x40: return "COPY_DATA";
        case 0x46: return "RELEASE_MEM";
        case 0x3C: return "WAIT_REG_MEM";
        case 0x3F: return "INDIRECT_BUFFER";
        case 0x49: return "ACQUIRE_MEM";
        case 0x27: return "WRITE_DATA";
        case 0x58: return "PRED_EXEC";
        case 0x7C: return "LOAD_UCONFIG_REG";
        case 0x12: return "INDIRECT_BUFFER_CIK";
        case 0x18: return "SURFACE_SYNC";
        default:   return nullptr;
    }
}

static void DecodePm4Buffer(const char* label, const uint32_t* dwords, size_t count) {
    printf("  [%s]  %zu DWORDs\n", label, count);
    size_t i = 0;
    while (i < count) {
        uint32_t hdr = dwords[i];
        uint32_t type = hdr >> 30;
        if (type == 3) {
            uint32_t opcode = (hdr >> 8) & 0xFF;
            uint32_t body   = ((hdr >> 16) & 0x3FFF) + 1; // count field is bits[29:16], encodes body length - 1
            const char* name = Pm4OpcodeName(opcode);
            if (name)
                printf("    [%4zu] PKT3 %-22s  op=0x%02X  len=%u  hdr=0x%08X\n",
                       i, name, opcode, body + 1, hdr);
            else {
                fprintf(stderr, "Warning: unknown PM4 opcode 0x%02X\n", opcode);
                printf("    [%4zu] PKT3 op=0x%02X              len=%u  hdr=0x%08X\n",
                       i, opcode, body + 1, hdr);
            }
            for (uint32_t j = 0; j < body && i + 1 + j < count; j++)
                printf("           [%4zu]   0x%08X\n", i + 1 + j, dwords[i + 1 + j]);
            i += 1 + body;
        } else if (type == 2) {
            printf("    [%4zu] PKT2 (filler)  0x%08X\n", i, hdr);
            i++;
        } else if (type == 0) {
            uint32_t body = (hdr & 0x3FFF) + 1;
            printf("    [%4zu] PKT0 reg=0x%04X  len=%u  hdr=0x%08X\n",
                   i, (hdr >> 16) & 0x3FFF, body + 1, hdr);
            for (uint32_t j = 0; j < body && i + 1 + j < count; j++)
                printf("           [%4zu]   0x%08X\n", i + 1 + j, dwords[i + 1 + j]);
            i += 1 + body;
        } else {
            // type == 1 or unrecognised — advance one word to avoid infinite loop
            printf("    [%4zu] UNKNOWN  0x%08X\n", i, hdr);
            i++;
        }
    }
}

static void DecodeBatch(int batch_num,
                        const std::vector<std::vector<uint32_t>>& captured) {
    // aqlprofile copies: [read_cmd, start_cmd, stop_cmd] in that order
    // (Read is built first, then Start, then Stop in counters.cpp)
    static const char* kLabels[] = {"read", "start", "stop"};
    printf("\n--- Batch %d: %zu command buffer(s) captured ---\n",
           batch_num, captured.size());
    for (size_t k = 0; k < captured.size(); k++) {
        const char* lbl = (k < 3) ? kLabels[k] : "extra";
        DecodePm4Buffer(lbl, captured[k].data(), captured[k].size());
    }
}

// ---------------------------------------------------------------------------
// Data callback for aqlprofile_pmc_iterate_data
// ---------------------------------------------------------------------------

struct DataCbCtx {
    ResultMap*   totals;
    InstanceMap* instances; // nullptr when not in verbose mode
};

static hsa_status_t DataCb(aqlprofile_pmc_event_t event, uint64_t counter_id,
                             uint64_t value, void* ud) {
    auto* ctx = static_cast<DataCbCtx*>(ud);
    (*ctx->totals)[event] += value;
    if (ctx->instances)
        (*ctx->instances)[event].emplace_back(counter_id, value);
    return HSA_STATUS_SUCCESS;
}

// ---------------------------------------------------------------------------
// Stub agent creation: build an aqlprofile_agent_handle_t from a gfxXXXX
// string alone, no HSA / no GPU required. Shared by --dry-run-gfx and
// --list-blocks-gfx.
// ---------------------------------------------------------------------------

static hsa_status_t CreateStubAgent(const std::string& gfxip, aqlprofile_agent_handle_t* out) {
    aqlprofile_agent_info_v1_t stub_info{};
    stub_info.agent_gfxip         = gfxip.c_str();
    stub_info.xcc_num              = 1;
    stub_info.se_num               = 1;
    stub_info.cu_num               = 16;
    stub_info.shader_arrays_per_se = 1;
    stub_info.domain               = 0;
    stub_info.location_id          = 0;
    return aqlprofile_register_agent_info(out, &stub_info, AQLPROFILE_AGENT_VERSION_V1);
}

// Probe every block number against the given agent and print the supported
// ones. Works for both real HSA-backed agents and stub (gfx-only) agents.
static int PrintSupportedBlocks(aqlprofile_agent_handle_t agent, const char* label) {
    printf("Supported PMC blocks for %s (block:name):\n", label);
    aqlprofile_pmc_event_flags_t flags{};
    flags.raw = 0;
    int count = 0;
    for (int block = 0; block < (int)AQLPROFILE_BLOCKS_NUMBER; block++) {
        // Probe a small range of event IDs — some blocks have no event at ID 0.
        bool supported = false;
        for (int eid = 0; eid <= 16 && !supported; eid++) {
            aqlprofile_pmc_event_t ev{};
            ev.block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(block);
            ev.block_index = 0;
            ev.event_id    = eid;
            ev.flags       = flags;
            aqlprofile_validate_pmc_event(agent, &ev, &supported);
        }
        if (supported) {
            const char* name = BlockName(block);
            if (name)
                printf("  %3d  %s\n", block, name);
            else
                printf("  %3d  (unknown)\n", block);
            count++;
        }
    }
    printf("Total: %d block(s)\n", count);
    return count;
}

// ---------------------------------------------------------------------------
// Enumerate valid events by scanning block/index/event_id with validate
// (mirrors the AQLPROFILE_SCAN loop in test.cpp but uses v2 validate API).
// ---------------------------------------------------------------------------

static void ScanValidEvents(aqlprofile_agent_handle_t agent,
                             std::vector<aqlprofile_pmc_event_t>& out,
                             int block_index_max_fallback = 16,
                             int event_id_max_fallback = 128,
                             int scan_step       = 1,
                             int block_filter    = -1) {
    aqlprofile_pmc_event_flags_t flags{};
    flags.raw = 0;

    int block_start = (block_filter >= 0) ? block_filter : 0;
    int block_end   = (block_filter >= 0) ? block_filter + 1 : (int)AQLPROFILE_BLOCKS_NUMBER;

    for (int block = block_start; block < block_end; block++) {
        // Prefer the real instance count from the driver; fall back to the
        // caller-supplied bound only if the block is unknown or the query fails.
        uint32_t queried_instances = QueryBlockInstances(agent, block);
        int block_index_max = (queried_instances > 0)
                                   ? static_cast<int>(queried_instances)
                                   : block_index_max_fallback;

        // Prefer the real max event_id discovered via binary search against
        // validate_pmc_event; fall back to the caller-supplied bound if the
        // block doesn't validate at all (handled by the eid==0 check below anyway).
        int found_max_event_id = FindMaxEventId(
            agent, static_cast<hsa_ven_amd_aqlprofile_block_name_t>(block));
        int event_id_max = (found_max_event_id >= 0) ? found_max_event_id : event_id_max_fallback;

        for (int idx = 0; idx < block_index_max; idx++) {
            for (int eid = 0; eid <= event_id_max; eid += scan_step) {
                aqlprofile_pmc_event_t ev{};
                ev.block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(block);
                ev.block_index = idx;
                ev.event_id    = eid;
                ev.flags       = flags;

                bool ok = false;
                aqlprofile_validate_pmc_event(agent, &ev, &ok);
                if (ok) {
                    out.push_back(ev);
                } else if (eid == 0) {
                    // No event at id=0 for this block/index — skip remaining event_ids
                    if (idx == 0) goto next_block;
                    goto next_index;
                }
            }
            next_index:;
        }
        next_block:;
    }
}

// ---------------------------------------------------------------------------
// Usage
// ---------------------------------------------------------------------------

static void PrintUsage(const char* prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "PMC event selection (at least one required unless --list-events):\n"
        "  --pmc <name>[,<name>...]    Named counters from the built-in table\n"
        "  --pmc-raw <b>:<i>:<e>[,...] Raw block:index:event_id triple(s), comma-separated; repeatable;\n"
        "                              block may be a number or a name (see --list-blocks)\n"
        "  --pmc-scan                  Scan all valid events via validate_pmc_event\n"
        "\n"
        "Options:\n"
        "  --pmc-batch <n>             Max events per dispatch (default: all in one batch)\n"
        "  --block <n|name>            Restrict --list-events/--pmc-scan to one block (number or name)\n"
        "  --gpu <idx>                 GPU agent index (default: 0)\n"
        "  --iterations <n>            Kernel dispatches per batch (default: 1)\n"
        "  --list-blocks               Print supported block numbers and names and exit\n"
        "  --list-blocks-gfx <gfxXXXX> Print supported blocks for gfxXXXX (stub agent, no GPU)\n"
        "  --list-dimensions           Print coordinate dimensions for valid events and exit\n"
        "  --list-events               Print all valid events (scan + validate) and exit\n"
        "  --list-gpus                 Print all GPU agents and exit\n"
        "  --dry-run                   Build+decode start/stop/read packets, skip GPU dispatch\n"
        "  --dry-run-gfx <gfxXXXX>     Build+decode packets using stub agent (no GPU required)\n"
        "  --verbose                   Print per-event counter values\n"
        "  --help                      Show this message\n"
        "\n"
        "Built-in counter names for --pmc:\n",
        prog);
    for (int i = 0; i < kBuiltinCount; i++)
        fprintf(stderr, "  %s\n", kBuiltin[i].name);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char* argv[]) {
    // ===== Parse CLI =====
    uint32_t gpu_idx    = 0;
    int      iterations = 1;
    int      pmc_batch  = 0;   // 0 = all events in one batch
    int      scan_step  = 1;
    int      block_filter = -1; // -1 = all blocks
    bool     list_events     = false;
    bool     list_blocks     = false;
    bool     list_dimensions = false;
    bool     list_gpus       = false;
    bool     verbose         = false;
    bool     pmc_scan    = false;
    bool     dry_run     = false;       // --dry-run: skip dispatch, decode packets (real GPU)
    std::string dry_run_gfx;            // --dry-run-gfx <gfxXXXX>: stub agent, decode packets
    std::string list_blocks_gfx;        // --list-blocks-gfx <gfxXXXX>: stub agent, list blocks only
    std::vector<std::string>            pmc_names;
    std::vector<aqlprofile_pmc_event_t> raw_events;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            PrintUsage(argv[0]);
            return 0;
        } else if (arg == "--list-events") {
            list_events = true;
        } else if (arg == "--list-blocks") {
            list_blocks = true;
        } else if (arg == "--list-blocks-gfx" && i + 1 < argc) {
            list_blocks = true;
            list_blocks_gfx = argv[++i];
        } else if (arg == "--list-dimensions") {
            list_dimensions = true;
        } else if (arg == "--list-gpus") {
            list_gpus = true;
        } else if (arg == "--pmc-scan") {
            pmc_scan = true;
        } else if (arg == "--verbose") {
            verbose = true;
        } else if (arg == "--gpu" && i + 1 < argc) {
            gpu_idx = static_cast<uint32_t>(atoi(argv[++i]));
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = atoi(argv[++i]);
            if (iterations < 1) iterations = 1;
        } else if (arg == "--pmc-batch" && i + 1 < argc) {
            pmc_batch = atoi(argv[++i]);
            if (pmc_batch < 1) pmc_batch = 0;
        } else if (arg == "--pmc-scan-step" && i + 1 < argc) {
            scan_step = atoi(argv[++i]);
            if (scan_step < 1) scan_step = 1;
        } else if (arg == "--block" && i + 1 < argc) {
            const char* block_arg = argv[++i];
            block_filter = ResolveBlockToken(block_arg);
            if (block_filter < 0) {
                fprintf(stderr, "Error: unknown block name '%s'\n", block_arg);
                return 1;
            }
        } else if (arg == "--pmc" && i + 1 < argc) {
            std::istringstream ss(argv[++i]);
            for (std::string tok; std::getline(ss, tok, ','); )
                if (!tok.empty()) pmc_names.push_back(tok);
        } else if (arg == "--dry-run") {
            dry_run = true;
        } else if (arg == "--dry-run-gfx" && i + 1 < argc) {
            dry_run_gfx = argv[++i];
        } else if (arg == "--pmc-raw" && i + 1 < argc) {
            std::istringstream ss(argv[++i]);
            for (std::string tok; std::getline(ss, tok, ','); ) {
                if (tok.empty()) continue;
                size_t p1 = tok.find(':');
                size_t p2 = (p1 == std::string::npos) ? std::string::npos : tok.find(':', p1 + 1);
                if (p1 == std::string::npos || p2 == std::string::npos) {
                    fprintf(stderr, "Error: --pmc-raw expects block:idx:id, got '%s'\n", tok.c_str());
                    return 1;
                }
                std::string block_tok = tok.substr(0, p1);
                int block_name_int = ResolveBlockToken(block_tok.c_str());
                if (block_name_int < 0) {
                    fprintf(stderr, "Error: --pmc-raw unknown block '%s' in '%s'\n",
                            block_tok.c_str(), tok.c_str());
                    return 1;
                }
                int block_index = atoi(tok.substr(p1 + 1, p2 - p1 - 1).c_str());
                int event_id    = atoi(tok.substr(p2 + 1).c_str());
                aqlprofile_pmc_event_t e{};
                e.block_name  = static_cast<hsa_ven_amd_aqlprofile_block_name_t>(block_name_int);
                e.block_index = block_index;
                e.event_id    = event_id;
                raw_events.push_back(e);
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", arg.c_str());
            PrintUsage(argv[0]);
            return 1;
        }
    }

    // ===== Mutual-exclusion enforcement =====

    // Hard conflicts: error and exit
    if (dry_run && !dry_run_gfx.empty()) {
        fprintf(stderr, "Error: --dry-run and --dry-run-gfx are mutually exclusive\n");
        return 1;
    }
    if (!dry_run_gfx.empty() && pmc_scan) {
        fprintf(stderr, "Error: --dry-run-gfx and --pmc-scan are mutually exclusive "
                        "(scan requires a real GPU agent for event validation)\n");
        return 1;
    }
    if (list_gpus && (list_blocks || list_events || list_dimensions)) {
        fprintf(stderr, "Error: --list-gpus cannot be combined with "
                        "--list-blocks / --list-events / --list-dimensions\n");
        return 1;
    }
    if (list_gpus && (!pmc_names.empty() || !raw_events.empty() || pmc_scan)) {
        fprintf(stderr, "Error: --list-gpus cannot be combined with "
                        "--pmc / --pmc-raw / --pmc-scan\n");
        return 1;
    }
    if (list_gpus && (dry_run || !dry_run_gfx.empty())) {
        fprintf(stderr, "Error: --list-gpus cannot be combined with "
                        "--dry-run / --dry-run-gfx\n");
        return 1;
    }
    if (!list_blocks_gfx.empty() && (dry_run || !dry_run_gfx.empty())) {
        fprintf(stderr, "Error: --list-blocks-gfx and --dry-run / --dry-run-gfx "
                        "are mutually exclusive\n");
        return 1;
    }

    // Meaningless combinations: warn but continue
    if (!dry_run_gfx.empty() && gpu_idx != 0) {
        fprintf(stderr, "Warning: --gpu is ignored when --dry-run-gfx is set "
                        "(no HSA agent is used)\n");
    }
    if (!dry_run_gfx.empty() && iterations != 1) {
        fprintf(stderr, "Warning: --iterations is ignored when --dry-run-gfx is set "
                        "(no GPU dispatch occurs)\n");
    }
    if (dry_run && iterations != 1) {
        fprintf(stderr, "Warning: --iterations is ignored when --dry-run is set "
                        "(no GPU dispatch occurs)\n");
    }
    if (!list_blocks_gfx.empty() && gpu_idx != 0) {
        fprintf(stderr, "Warning: --gpu is ignored when --list-blocks-gfx is set "
                        "(no HSA agent is used)\n");
    }

    // ===== --dry-run-gfx: stub agent path (no HSA / no GPU) =====
    if (!dry_run_gfx.empty()) {
        // Build event list from CLI args (--pmc / --pmc-raw only; --pmc-scan needs a real agent)
        std::vector<aqlprofile_pmc_event_t> stub_events;
        aqlprofile_pmc_event_flags_t zero_flags_stub{};
        zero_flags_stub.raw = 0;
        for (auto& name : pmc_names) {
            const BuiltinCounter* bc = FindBuiltin(name.c_str());
            if (!bc) {
                fprintf(stderr, "Error: unknown counter name '%s'\n", name.c_str());
                return 1;
            }
            aqlprofile_pmc_event_t ev{};
            ev.block_name  = bc->block;
            ev.block_index = 0;
            ev.event_id    = bc->event_id;
            ev.flags       = zero_flags_stub;
            stub_events.push_back(ev);
        }
        for (auto& ev : raw_events)
            stub_events.push_back(ev);
        if (stub_events.empty()) {
            fprintf(stderr, "Error: no events specified\n");
            PrintUsage(argv[0]);
            return 1;
        }

        printf("Dry-run (stub agent): gfx=%s  events=%zu\n",
               dry_run_gfx.c_str(), stub_events.size());

        aqlprofile_agent_handle_t stub_agent{};
        hsa_status_t st = CreateStubAgent(dry_run_gfx, &stub_agent);
        if (st != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "Error: aqlprofile_register_agent_info failed: %d\n",
                    static_cast<int>(st));
            return 1;
        }

        size_t batch_size_stub = (pmc_batch > 0) ? static_cast<size_t>(pmc_batch) : stub_events.size();
        int    total_batches   = 0;

        for (size_t start = 0; start < stub_events.size(); start += batch_size_stub) {
            size_t end = std::min(start + batch_size_stub, stub_events.size());
            std::vector<aqlprofile_pmc_event_t> batch(stub_events.begin() + start,
                                                       stub_events.begin() + end);
            total_batches++;

            aqlprofile_pmc_profile_t profile{};
            profile.agent       = stub_agent;
            profile.events      = batch.data();
            profile.event_count = static_cast<uint32_t>(batch.size());

            std::vector<std::vector<uint32_t>> captured;
            aqlprofile_handle_t pmc_handle{};
            aqlprofile_pmc_aql_packets_t pkts{};

            st = aqlprofile_pmc_create_packets(&pmc_handle, &pkts, profile,
                                               &DryRunAlloc, &DryRunFree,
                                               &DryRunCopy, &captured);
            if (st != HSA_STATUS_SUCCESS) {
                fprintf(stderr, "Error: aqlprofile_pmc_create_packets failed: %d\n",
                        static_cast<int>(st));
                return 1;
            }
            DecodeBatch(total_batches, captured);
            aqlprofile_pmc_delete_packets(pmc_handle);
        }
        printf("\nDone (dry-run-gfx, %d batch(es))\n", total_batches);
        return 0;
    }

    // ===== --list-blocks-gfx: stub agent path (no HSA / no GPU) =====
    if (!list_blocks_gfx.empty()) {
        aqlprofile_agent_handle_t stub_agent{};
        hsa_status_t st = CreateStubAgent(list_blocks_gfx, &stub_agent);
        if (st != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "Error: aqlprofile_register_agent_info failed: %d\n",
                    static_cast<int>(st));
            return 1;
        }
        PrintSupportedBlocks(stub_agent, list_blocks_gfx.c_str());
        return 0;
    }

    // ===== HSA init via HsaRsrcFactory =====
    HsaRsrcFactory* factory = HsaRsrcFactory::Create();
    if (!factory) {
        fprintf(stderr, "Error: HsaRsrcFactory::Create() failed\n");
        return 1;
    }

    // ===== --list-gpus: print all GPU agent info and exit =====
    if (list_gpus) {
        uint32_t ngpu = factory->GetCountOfGpuAgents();
        printf("GPU agents (%u):\n", ngpu);
        for (uint32_t i = 0; i < ngpu; i++) {
            const AgentInfo* info = nullptr;
            if (!factory->GetGpuAgentInfo(i, &info)) continue;
            uint32_t bdf = info->bdf_id;
            printf("  [%u] %-20s  gfxip=%-8s  cu=%-4u  se=%-2u  xcc=%-2u  sa/se=%-2u"
                   "  wave_size=%-4u  BDF=%04x:%02x:%02x.%x\n",
                   i, info->name, info->gfxip,
                   info->cu_num, info->se_num, info->xcc_num, info->shader_arrays_per_se,
                   info->max_wave_size,
                   info->domain, (bdf >> 8) & 0xff, (bdf >> 3) & 0x1f, bdf & 0x7);
        }
        factory->Destroy();
        return 0;
    }

    const AgentInfo* ai = nullptr;
    if (!factory->GetGpuAgentInfo(gpu_idx, &ai)) {
        fprintf(stderr, "Error: GPU agent %u not found\n", gpu_idx);
        factory->Destroy();
        return 1;
    }
    // ai->gfxip is a short form (e.g. "gfx12"); ai->name holds the full HSA agent name
    // (e.g. "gfx1200:sramecc+:xnack-").  Strip the feature suffix to get "gfx1200",
    // which is what aqlprofile and the hsaco filenames both expect.
    std::string agent_name(ai->name);
    {
        auto colon = agent_name.find(':');
        if (colon != std::string::npos) agent_name = agent_name.substr(0, colon);
    }
    printf("Using GPU agent[%u]: %s\n", gpu_idx, agent_name.c_str());

    // ===== Register agent with aqlprofile v2 =====
    aqlprofile_agent_info_v1_t aql_info{};
    aql_info.agent_gfxip         = agent_name.c_str();
    aql_info.xcc_num              = ai->xcc_num;
    aql_info.se_num               = ai->se_num;
    aql_info.cu_num               = ai->cu_num;
    aql_info.shader_arrays_per_se = ai->shader_arrays_per_se;
    aql_info.domain               = ai->domain;
    aql_info.location_id          = ai->bdf_id;

    aqlprofile_agent_handle_t aql_agent{};
    hsa_status_t st = aqlprofile_register_agent_info(&aql_agent, &aql_info, AQLPROFILE_AGENT_VERSION_V1);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Error: aqlprofile_register_agent failed: %d\n", static_cast<int>(st));
        factory->Destroy();
        return 1;
    }

    // ===== --list-blocks: probe every block number and print supported ones, then exit =====
    if (list_blocks) {
        PrintSupportedBlocks(aql_agent, agent_name.c_str());
        factory->Destroy();
        return 0;
    }

    // ===== --list-events: scan all events, validate, print, exit =====
    if (list_events) {
        if (block_filter >= 0)
            printf("Valid PMC events for %s  block=%d (block:idx:id  name):\n",
                   agent_name.c_str(), block_filter);
        else
            printf("Valid PMC events for %s (block:idx:id  name):\n", agent_name.c_str());
        std::vector<aqlprofile_pmc_event_t> all;
        ScanValidEvents(aql_agent, all, 16, 128, 1, block_filter);
        for (auto& ev : all) {
            const char* name = BuiltinName(ev);
            printf("  %3d:%-3d:%-5d  %s\n",
                   static_cast<int>(ev.block_name),
                   static_cast<int>(ev.block_index),
                   static_cast<int>(ev.event_id),
                   name ? name : "");
        }
        printf("Total: %zu events\n", all.size());
        factory->Destroy();
        return 0;
    }

    // ===== --list-dimensions: for each valid event in block, print coordinate dimensions =====
    if (list_dimensions) {
        struct DimEntry { int position; int id; int extent; int coordinate; char name[64]; };
        struct DimCbCtx { std::vector<DimEntry> dims; };

        auto dim_cb = [](int position, int id, int extent, int coordinate,
                         const char* name, void* ud) -> hsa_status_t {
            auto* ctx = static_cast<DimCbCtx*>(ud);
            DimEntry e{};
            e.position   = position;
            e.id         = id;
            e.extent     = extent;
            e.coordinate = coordinate;
            snprintf(e.name, sizeof(e.name), "%s", name ? name : "");
            ctx->dims.push_back(e);
            return HSA_STATUS_SUCCESS;
        };

        if (block_filter >= 0)
            printf("PMC event dimensions for %s  block=%d:\n", agent_name.c_str(), block_filter);
        else
            printf("PMC event dimensions for %s:\n", agent_name.c_str());

        std::vector<aqlprofile_pmc_event_t> all;
        ScanValidEvents(aql_agent, all, 16, 128, 1, block_filter);

        // Collect dims per event, then group by (block, idx).
        // If all event_ids in a group share the same dims, print once; otherwise per-event.
        struct EventDims {
            aqlprofile_pmc_event_t      ev;
            std::vector<DimEntry>       dims;
        };
        std::vector<EventDims> collected;
        for (auto& ev : all) {
            DimCbCtx ctx{};
            aqlprofile_iterate_event_coord(aql_agent, ev, 0, dim_cb, &ctx);
            if (!ctx.dims.empty())
                collected.push_back({ev, std::move(ctx.dims)});
        }

        auto dims_equal = [](const std::vector<DimEntry>& a, const std::vector<DimEntry>& b) {
            if (a.size() != b.size()) return false;
            for (size_t i = 0; i < a.size(); i++)
                if (a[i].id != b[i].id || a[i].extent != b[i].extent) return false;
            return true;
        };

        auto print_dims = [](const std::vector<DimEntry>& dims) {
            for (auto& d : dims)
                printf("    dim[%d] (id, name)=(%-2d, %-12s)  extent=%d\n",
                       d.position, d.id, d.name, d.extent);
        };

        size_t i = 0;
        while (i < collected.size()) {
            // Find the end of the current (block, idx) group
            size_t j = i + 1;
            while (j < collected.size() &&
                   collected[j].ev.block_name  == collected[i].ev.block_name &&
                   collected[j].ev.block_index == collected[i].ev.block_index)
                j++;

            // Check if all events in the group share the same dims
            bool uniform = true;
            for (size_t k = i + 1; k < j && uniform; k++)
                if (!dims_equal(collected[k].dims, collected[i].dims)) uniform = false;

            const char* bname = BlockName(static_cast<int>(collected[i].ev.block_name));
            if (uniform) {
                // Print one summary line for the whole group
                printf("  block=%-3d (%s)  idx=%-3d  ids=%d..%d\n",
                       static_cast<int>(collected[i].ev.block_name),
                       bname ? bname : "?",
                       static_cast<int>(collected[i].ev.block_index),
                       static_cast<int>(collected[i].ev.event_id),
                       static_cast<int>(collected[j-1].ev.event_id));
                print_dims(collected[i].dims);
            } else {
                // Dims differ by event_id — print each one individually
                for (size_t k = i; k < j; k++) {
                    const char* cname = BuiltinName(collected[k].ev);
                    printf("  block=%-3d (%s)  idx=%-3d  id=%-5d  %s\n",
                           static_cast<int>(collected[k].ev.block_name),
                           bname ? bname : "?",
                           static_cast<int>(collected[k].ev.block_index),
                           static_cast<int>(collected[k].ev.event_id),
                           cname ? cname : "");
                    print_dims(collected[k].dims);
                }
            }
            i = j;
        }
        printf("Total: %zu events\n", all.size());
        factory->Destroy();
        return 0;
    }

    // ===== Build event list =====
    std::vector<aqlprofile_pmc_event_t> events;
    aqlprofile_pmc_event_flags_t zero_flags{};
    zero_flags.raw = 0;

    if (pmc_scan) {
        ScanValidEvents(aql_agent, events, 16, 128, scan_step, block_filter);
    }
    for (auto& name : pmc_names) {
        const BuiltinCounter* bc = FindBuiltin(name.c_str());
        if (!bc) {
            fprintf(stderr,
                    "Error: unknown counter name '%s'. Use --pmc-raw for raw triples "
                    "or --help to see the built-in list.\n",
                    name.c_str());
            factory->Destroy();
            return 1;
        }
        aqlprofile_pmc_event_t ev{};
        ev.block_name  = bc->block;
        ev.block_index = 0;
        ev.event_id    = bc->event_id;
        ev.flags       = zero_flags;
        events.push_back(ev);
    }
    for (auto& ev : raw_events)
        events.push_back(ev);

    if (events.empty()) {
        fprintf(stderr, "Error: no events specified\n");
        PrintUsage(argv[0]);
        factory->Destroy();
        return 1;
    }

    // ===== Validate events (skip invalid, warn) =====
    if (!pmc_scan) {
        // scan path already ran validate; skip redundant check for non-scan events
        std::vector<aqlprofile_pmc_event_t> valid;
        valid.reserve(events.size());
        int skipped = 0;
        for (auto& ev : events) {
            bool ok = false;
            aqlprofile_validate_pmc_event(aql_agent, &ev, &ok);
            if (ok) {
                valid.push_back(ev);
            } else {
                const char* name = BuiltinName(ev);
                fprintf(stderr, "  skip (invalid): %d:%d:%d %s\n",
                        static_cast<int>(ev.block_name),
                        static_cast<int>(ev.block_index),
                        static_cast<int>(ev.event_id),
                        name ? name : "");
                skipped++;
            }
        }
        if (skipped > 0)
            printf("Skipped %d invalid event(s)\n", skipped);
        events = std::move(valid);
    }
    if (events.empty()) {
        fprintf(stderr, "Error: no valid events after validation\n");
        factory->Destroy();
        return 1;
    }

    // When --pmc-scan is active and --pmc-batch wasn't set, query the hardware counter
    // slot count per block and use the minimum as the batch size so we never exceed
    // physical limits.
    size_t batch_size;
    if (pmc_scan && pmc_batch == 0) {
        // Collect the distinct blocks in the event list and query each.
        std::map<int, uint32_t> block_counters;
        for (auto& ev : events) {
            int block = static_cast<int>(ev.block_name);
            if (block_counters.find(block) == block_counters.end()) {
                uint32_t cnt = QueryBlockCounters(aql_agent, ev.block_name, ev.event_id);
                if (cnt > 0) block_counters[block] = cnt;
            }
        }
        if (!block_counters.empty()) {
            uint32_t min_cnt = UINT32_MAX;
            for (auto& [block, cnt] : block_counters) {
                if (verbose)
                    printf("  block %-12s hw counters: %u\n",
                           BlockName(block) ? BlockName(block) : "?", cnt);
                if (cnt < min_cnt) min_cnt = cnt;
            }
            batch_size = static_cast<size_t>(min_cnt);
        } else {
            batch_size = events.size();
        }
    } else {
        batch_size = (pmc_batch > 0) ? static_cast<size_t>(pmc_batch) : events.size();
    }
    printf("Profiling %zu event(s) in batch(es) of %zu\n", events.size(), batch_size);

    // ===== Memory callback context =====
    // HsaRsrcFactory leaves cpu_pool/kern_arg_pool zeroed on GPU AgentInfo (only gpu_pool
    // is populated there).  The CPU-accessible pools live on the CPU agent — same source
    // used by the integration test's FindGlobalPool / AgentInfo::iterate_agents().
    const AgentInfo* cpu_ai = nullptr;
    if (!factory->GetCpuAgentInfo(0, &cpu_ai)) {
        fprintf(stderr, "Error: CPU agent not found\n");
        factory->Destroy();
        return 1;
    }
    MemCtx mem_ctx{ai->dev_id, cpu_ai->cpu_pool, cpu_ai->kern_arg_pool};

    // ===== Force the perfmon clock on =====
    // Non-exclusive (lock=0, perfcount_enable=1): forces the perfmon clock on for this
    // process without requiring CAP_PERFMON, matching how rocprofiler-sdk's dispatch
    // counting path behaves. Reports (rather than silently skips) unsupported kernels/drivers;
    // see kfd_pmc_enable.h.
    kfd_pmc::EnableDispatchPerfmonClock(ai->driver_node_id);

    // ===== Create a single-queue for strict PMC packet ordering =====
    hsa_queue_t* queue = nullptr;
    st = hsa_queue_create(ai->dev_id, 1024, HSA_QUEUE_TYPE_SINGLE,
                           nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
    if (st != HSA_STATUS_SUCCESS) {
        fprintf(stderr, "Error: hsa_queue_create failed: %d\n", static_cast<int>(st));
        factory->Destroy();
        return 1;
    }

    // ===== Set up simple_convolution kernel =====
    simple_convolution kern;

    for (auto& [buf_id, des] : kern.GetMemMap()) {
        switch (des.id) {
            case TestKernel::LOCAL_DES_ID:
                des.ptr = factory->AllocateLocalMemory(ai, des.size);
                break;
            case TestKernel::KERNARG_DES_ID:
                des.ptr = factory->AllocateKernArgMemory(ai, des.size);
                if (des.ptr) memset(des.ptr, 0, des.size);
                break;
            case TestKernel::SYS_DES_ID:
                des.ptr = factory->AllocateSysMemory(ai, des.size);
                if (des.ptr) memset(des.ptr, 0, des.size);
                break;
            default:
                break;
        }
        if (!des.ptr && des.id != TestKernel::NULL_DES_ID && des.id != TestKernel::REFOUT_DES_ID) {
            fprintf(stderr, "Error: failed to allocate kernel buffer %u\n", buf_id);
            hsa_queue_destroy(queue);
            factory->Destroy();
            return 1;
        }
    }
    kern.Init();

    std::string hsaco_path = agent_name + "_" + kern.Name() + ".hsaco";
    hsa_executable_t hsa_exec{};
    hsa_executable_symbol_t kern_sym{};
    bool loaded = false;
#ifdef CTRL_V3_EMBEDDED_HSACO
    {
        EmbeddedHsaco blob = FindEmbeddedHsaco(agent_name.c_str());
        if (blob.data) {
            printf("Loading embedded hsaco for %s (%u bytes)\n",
                   agent_name.c_str(), blob.size);
            loaded = factory->LoadAndFinalizeMemory(ai, blob.data, blob.size,
                                                    kern.SymbName().c_str(),
                                                    &hsa_exec, &kern_sym);
            if (!loaded)
                fprintf(stderr, "Warning: embedded hsaco load failed, trying file\n");
        }
    }
#endif
    if (!loaded) {
        loaded = factory->LoadAndFinalize(ai, hsaco_path.c_str(), kern.SymbName().c_str(),
                                          &hsa_exec, &kern_sym);
    }
    if (!loaded) {
        fprintf(stderr, "Error: failed to load hsaco for %s (tried embedded + '%s')\n",
                agent_name.c_str(), hsaco_path.c_str());
        hsa_queue_destroy(queue);
        factory->Destroy();
        return 1;
    }

    uint64_t code_handle      = 0;
    uint32_t group_seg_size   = 0;
    uint32_t private_seg_size = 0;
    hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                    &code_handle);
    hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                    &group_seg_size);
    hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                                    &private_seg_size);

    // Build kernel dispatch packet template (no completion signal — read_packet carries it)
    constexpr uint32_t kWorkgroupSize = 64;
    hsa_kernel_dispatch_packet_t dispatch_pkt{};
    dispatch_pkt.header  = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    dispatch_pkt.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
    dispatch_pkt.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
    dispatch_pkt.setup                = 1;
    dispatch_pkt.grid_size_x          = kern.GetGridSize();
    dispatch_pkt.grid_size_y          = 1;
    dispatch_pkt.grid_size_z          = 1;
    dispatch_pkt.workgroup_size_x     = kWorkgroupSize;
    dispatch_pkt.workgroup_size_y     = 1;
    dispatch_pkt.workgroup_size_z     = 1;
    dispatch_pkt.kernel_object        = code_handle;
    dispatch_pkt.kernarg_address      = kern.GetKernargPtr();
    dispatch_pkt.group_segment_size   = group_seg_size;
    dispatch_pkt.private_segment_size = private_seg_size;

    hsa_signal_t signal{};
    factory->CreateSignal(1, &signal);

    // ===== Batch loop =====
    ResultMap   totals;
    InstanceMap all_instances;
    for (auto& ev : events) totals[ev] = 0;
    int total_batches = 0;

    for (size_t start = 0; start < events.size(); start += batch_size) {
        size_t end = std::min(start + batch_size, events.size());
        std::vector<aqlprofile_pmc_event_t> batch(events.begin() + start, events.begin() + end);
        total_batches++;
        if (verbose)
            printf("Batch %d: events[%zu..%zu)\n", total_batches, start, end);

        aqlprofile_pmc_profile_t profile{};
        profile.agent       = aql_agent;
        profile.events      = batch.data();
        profile.event_count = static_cast<uint32_t>(batch.size());

        aqlprofile_handle_t pmc_handle{};
        aqlprofile_pmc_aql_packets_t pkts{};

        // For --dry-run, attach a capture vector so PmcCopy records PM4 DWORDs
        std::vector<std::vector<uint32_t>> dry_captured;
        mem_ctx.captured = dry_run ? &dry_captured : nullptr;

        st = aqlprofile_pmc_create_packets(&pmc_handle, &pkts, profile,
                                            &PmcAlloc, &PmcFree, &PmcCopy, &mem_ctx);
        if (st != HSA_STATUS_SUCCESS) {
            fprintf(stderr, "Error: aqlprofile_pmc_create_packets failed: %d "
                            "(try --pmc-batch to reduce events per dispatch)\n",
                    static_cast<int>(st));
            hsa_signal_destroy(signal);
            hsa_executable_destroy(hsa_exec);
            hsa_queue_destroy(queue);
            factory->Destroy();
            return 1;
        }

        if (dry_run) {
            DecodeBatch(total_batches, dry_captured);
            aqlprofile_pmc_delete_packets(pmc_handle);
            continue;
        }

        // v2 requires manual header initialization after packet creation
        pkts.start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        pkts.stop_packet.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
        pkts.read_packet.header  = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

        for (int iter = 0; iter < iterations; iter++) {
            hsa_signal_store_relaxed(signal, 1);
            // Signal on read_packet: fires when PMC data is in memory and ready for readback
            pkts.read_packet.completion_signal = signal;

            HsaRsrcFactory::Submit(queue, &pkts.start_packet);
            HsaRsrcFactory::Submit(queue, &dispatch_pkt);
            HsaRsrcFactory::Submit(queue, &pkts.stop_packet);
            HsaRsrcFactory::Submit(queue, &pkts.read_packet);

            hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                       UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
            pkts.read_packet.completion_signal.handle = 0;
        }

        InstanceMap batch_instances;
        DataCbCtx   cb_ctx{&totals, verbose ? &batch_instances : nullptr};
        aqlprofile_pmc_iterate_data(pmc_handle, DataCb, &cb_ctx);
        if (verbose)
            for (auto& [ev, pairs] : batch_instances)
                for (auto& p : pairs)
                    all_instances[ev].push_back(p);

        aqlprofile_pmc_delete_packets(pmc_handle);
    }

    // ===== Print results =====
    if (dry_run) {
        printf("\nDone (dry-run, %d batch(es), no GPU dispatch)\n", total_batches);
        hsa_signal_destroy(signal);
        hsa_executable_destroy(hsa_exec);
        hsa_queue_destroy(queue);
        factory->Destroy();
        return 0;
    }

    printf("\n=== PMC Results  gpu[%u]=%s  batches=%d  iterations=%d ===\n",
           gpu_idx, agent_name.c_str(), total_batches, iterations);
    for (auto& ev : events) {
        const char* name = BuiltinName(ev);
        char name_buf[64];
        if (!name) {
            snprintf(name_buf, sizeof(name_buf), "%d:%d:%d",
                     static_cast<int>(ev.block_name),
                     static_cast<int>(ev.block_index),
                     static_cast<int>(ev.event_id));
            name = name_buf;
        }
        const char* block_name = BlockName(static_cast<int>(ev.block_name));
        char block_label[32];
        if (block_name)
            snprintf(block_label, sizeof(block_label), "%-3d (%s)",
                     static_cast<int>(ev.block_name), block_name);
        else
            snprintf(block_label, sizeof(block_label), "%-3d",
                     static_cast<int>(ev.block_name));
        printf("  %-40s  block=%-12s  idx=%-3d  id=%-5d  %llu\n",
               name,
               block_label,
               static_cast<int>(ev.block_index),
               static_cast<int>(ev.event_id),
               static_cast<unsigned long long>(totals[ev]));

        if (verbose) {
            auto it = all_instances.find(ev);
            if (it != all_instances.end()) {
                struct CoordEntry { int pos; int id; int coordinate; char name[32]; };
                auto coord_cb = [](int pos, int id, int /*extent*/, int coordinate,
                                   const char* nm, void* ud) -> hsa_status_t {
                    auto* v = static_cast<std::vector<CoordEntry>*>(ud);
                    CoordEntry c{};
                    c.pos = pos; c.id = id; c.coordinate = coordinate;
                    snprintf(c.name, sizeof(c.name), "%s", nm ? nm : "");
                    v->push_back(c);
                    return HSA_STATUS_SUCCESS;
                };
                for (auto& [cid, val] : it->second) {
                    std::vector<CoordEntry> coords;
                    aqlprofile_iterate_event_coord(aql_agent, ev, cid, coord_cb, &coords);
                    printf("    instance[%2llu]", static_cast<unsigned long long>(cid));
                    for (auto& c : coords)
                        printf("  %s=%d", c.name, c.coordinate);
                    printf("  -> %llu\n", static_cast<unsigned long long>(val));
                }
            }
        }
    }

    // ===== Cleanup =====
    hsa_signal_destroy(signal);
    hsa_executable_destroy(hsa_exec);
    hsa_queue_destroy(queue);
    factory->Destroy();

    return 0;
}
