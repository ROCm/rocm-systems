// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "trie.h"

#include <cstddef>
#include <new>

namespace
{
struct TrieEntry
{
    std::string_view instruction;
    InstCategory type;
};

constexpr TrieEntry type_dict[] = {
    {"s_waitcnt",        InstCategory::IMMED},
    {"s_wait_idle",      InstCategory::IMMED},
    {"s_wait_dep",       InstCategory::SKIP },
    {"s_wait_kmcnt",     InstCategory::IMMED},
    {"s_wait_loadcnt",   InstCategory::IMMED},
    {"s_wait_storecnt",  InstCategory::IMMED},
    {"s_wait_bvhcnt",    InstCategory::IMMED},
    {"s_wait_expcnt",    InstCategory::IMMED},
    {"s_wait_dscnt",     InstCategory::IMMED},
    {"s_wait_samplecnt", InstCategory::IMMED},
    {"s_wait_event",     InstCategory::IMMED},
    {"s_wait_alu",       InstCategory::SKIP },
    {"s_wait_xcnt",      InstCategory::IMMED},
    {"s_wait_tensor",    InstCategory::IMMED},
    {"s_wait_async",     InstCategory::IMMED},
    {"s_nop",            InstCategory::IMMED},
    {"s_sleep",          InstCategory::IMMED},
    {"s_wakeup",         InstCategory::IMMED},
    {"s_sendmsg",        InstCategory::IMMED},
    {"s_setprio ",       InstCategory::IMMED},
    {"s_setprio_inc",    InstCategory::SALU },
    {"s_set_inst",       InstCategory::IMMED},
    {"s_inst_prefetch",  InstCategory::IMMED},
    {"s_setkill",        InstCategory::IMMED},
    {"s_version",        InstCategory::IMMED},
    {"s_trap",           InstCategory::IMMED},
    {"s_barrie",         InstCategory::IMMED},
    {"s_endpgm",         InstCategory::IMMED},
    {"s_icache_inv",     InstCategory::IMMED},
    {"s_dcache_inv",     InstCategory::IMMED},
    {"s_incperflvl",     InstCategory::IMMED},
    {"s_sethalt",        InstCategory::IMMED},
    {"s_decperflvl",     InstCategory::IMMED},
    {"s_setvskip",       InstCategory::IMMED},
    {"s_ttrace",         InstCategory::IMMED},
    {"s_clause",         InstCategory::IMMED},
    {"s_call",           InstCategory::SALU },
    {"s_cvt",            InstCategory::SALU },

    {"s_load",           InstCategory::SMEM },
    {"s_buffer",         InstCategory::SMEM },
    {"s_atomic",         InstCategory::SMEM },
    {"s_atc",            InstCategory::SMEM },
    {"s_scratch",        InstCategory::SMEM },
    {"s_store",          InstCategory::SMEM },
    {"s_dcache",         InstCategory::SMEM },
    {"s_getreg",         InstCategory::SALU },
    {"s_setreg",         InstCategory::SALU },
    {"s_memrealtime",    InstCategory::SMEM },

    {"buffer_l",         InstCategory::VMEM },
    {"buffer_s",         InstCategory::VMEM },
    {"buffer_a",         InstCategory::VMEM },
    {"buffer_g",         InstCategory::VMEM },
    {"buffer_i",         InstCategory::VMEM },
    {"buffer_w",         InstCategory::VMEM },
    {"buffer_p",         InstCategory::VMEM },
    {"buffer_d",         InstCategory::VMEM },
    {"buffer_nop",       InstCategory::SKIP },
    {"tbuffer_",         InstCategory::VMEM },

    {"flat_",            InstCategory::FLAT },
    {"global_",          InstCategory::VMEM },
    {"scratch_",         InstCategory::VMEM },
    {"cluster_",         InstCategory::VMEM },
    {"tensor_",          InstCategory::VMEM },

    {"ds_",              InstCategory::LDS  },
    {"lds_",             InstCategory::LDS  },
    {"dds_",             InstCategory::VMEM },
    {"s_add",            InstCategory::SALU },
    {"s_sub",            InstCategory::SALU },
    {"s_and",            InstCategory::SALU },
    {"s_or",             InstCategory::SALU },
    {"s_pack",           InstCategory::SALU },
    {"s_prefetch",       InstCategory::SMEM },

    {"s_round",          InstCategory::SALU },
    {"s_denorm",         InstCategory::SALU },
    {"s_ctz",            InstCategory::SALU },
    {"s_cls",            InstCategory::SALU },
    {"s_cmp",            InstCategory::SALU },
    {"s_bit",            InstCategory::SALU },
    {"s_set_vgpr",       InstCategory::SKIP },
    {"s_mov",            InstCategory::SALU },
    {"s_monitor",        InstCategory::IMMED},
    {"s_mul",            InstCategory::SALU },
    {"s_ash",            InstCategory::SALU },
    {"s_bf",             InstCategory::SALU },
    {"s_lsh",            InstCategory::SALU },
    {"s_min",            InstCategory::SALU },
    {"s_max",            InstCategory::SALU },
    {"s_abs",            InstCategory::SALU },
    {"s_cmov",           InstCategory::SALU },
    {"s_not",            InstCategory::SALU },
    {"s_nor",            InstCategory::SALU },
    {"s_wqm",            InstCategory::SALU },
    {"s_quadmask",       InstCategory::SALU },
    {"s_brev",           InstCategory::SALU },
    {"s_bcnt",           InstCategory::SALU },
    {"s_f",              InstCategory::SALU },
    {"s_sext",           InstCategory::SALU },
    {"s_cselect",        InstCategory::SALU },
    {"s_xor",            InstCategory::SALU },
    {"s_xnor",           InstCategory::SALU },
    {"s_nand",           InstCategory::SALU },
    {"s_delay",          InstCategory::SKIP },

    {"s_getpc",          InstCategory::SALU },
    {"s_setpc",          InstCategory::SALU },
    {"s_swappc",         InstCategory::SALU },
    {"s_get_pc",         InstCategory::SALU },
    {"s_set_pc",         InstCategory::SALU },
    {"s_swap_pc",        InstCategory::SALU },

    {"image_bvh",        InstCategory::BVH  },
    {"image_l",          InstCategory::VMEM },
    {"image_s",          InstCategory::VMEM },
    {"image_a",          InstCategory::VMEM },
    {"image_m",          InstCategory::VMEM },
    {"image_g",          InstCategory::VMEM },
};
} // namespace

Trie::Trie()
{
    for (const auto& entry : type_dict) add_type(entry.instruction, entry.type);
}

InstCategory Trie::type_from_trie(const std::string_view inst) const
{
    if (inst.find("v_") == 0) return InstCategory::VALU;

    const Node* trie = &root;
    for (char c : inst)
    {
        auto it = trie->paths.find(c);
        if (it != trie->paths.end()) trie = it->second.get();
        if (trie->type != InstCategory::LAST) return trie->type;
    }

    if (inst.find("s_") == 0) return InstCategory::SALU;

    return InstCategory::DONT_KNOW;
}

void Trie::add_type(std::string_view inst_header, InstCategory type)
{
    Node* trie = &root;
    for (char c : inst_header)
    {
        auto [it, inserted] = trie->paths.try_emplace(c);
        if (inserted) it->second = std::make_unique<Node>();
        trie = it->second.get();
    }
    trie->type = type;
}

Trie& get_instruction_trie()
{
    alignas(Trie) static std::byte storage[sizeof(Trie)];
    static Trie* trie = ::new (static_cast<void*>(storage)) Trie{};
    return *trie;
}
