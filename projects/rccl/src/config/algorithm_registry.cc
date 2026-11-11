/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2015-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/
#include "config/algorithm_registry.h"
#include "nccl_tuner.h"  // NCCL_ALGO_*, NCCL_PROTO_*, NCCL_NUM_PROTOCOLS
#include "sym_kernels.h" // ncclSymkKernelId_*
#include "tuning.h"      // NCCL_TUNING_SYM_KERNEL_ID_OFFSET
#include "os.h"          // strncasecmp (portable across Linux/Windows)
#include <cstring>       // strlen

#define ALGBIT(p) (1ull << (p))
#define GENBIT(a, p) ALGBIT((a) * NCCL_NUM_PROTOCOLS + (p))
#define SYMBIT(k) ALGBIT(NCCL_TUNING_SYM_KERNEL_ID_OFFSET + (k))
#define F_AR (1u << ncclFuncAllReduce)
#define F_AG (1u << ncclFuncAllGather)
#define F_RS (1u << ncclFuncReduceScatter)
#define F_BC (1u << ncclFuncBroadcast)
#define F_RD (1u << ncclFuncReduce)
#define F_ALL (F_AR | F_AG | F_RS | F_BC | F_RD) // RING: all general collectives

// Each row's mask bit IS its cost-model tuning id, so the mask lines up 1:1 with
// src/tuning/cost_model.cc modelMap[] (collMask mirrors modelMap[].enabled[func]):
//   general    -> bit (algo*NCCL_NUM_PROTOCOLS + proto), the [0, NCCL_TUNING_SYM_KERNEL_ID_OFFSET) space
//   symmetric  -> bit (NCCL_TUNING_SYM_KERNEL_ID_OFFSET + symKernelId)
// Commented rows are algo/proto combinations that do not exist today; they mark where a
// future kernel would slot into the tuning-id space.
// IMPORTANT: this table need must be consistent with the modelMap in src/tuning/cost_model.cc
// clang-format off:  hand-aligned table
static constexpr struct ncclAlgRegEntry algRegistry[] = {
  // name / mask / collMask / algo / proto / symkKernelId
  {"TREE_LL",                GENBIT(NCCL_ALGO_TREE, NCCL_PROTO_LL),              F_AR,               NCCL_ALGO_TREE,           NCCL_PROTO_LL,     -1},
  {"TREE_LL128",             GENBIT(NCCL_ALGO_TREE, NCCL_PROTO_LL128),           F_AR,               NCCL_ALGO_TREE,           NCCL_PROTO_LL128,  -1},
  {"TREE_SIMPLE",            GENBIT(NCCL_ALGO_TREE, NCCL_PROTO_SIMPLE),          F_AR,               NCCL_ALGO_TREE,           NCCL_PROTO_SIMPLE, -1},
  {"TREE_NAN",               GENBIT(NCCL_ALGO_TREE, NCCL_PROTO_NAN),             F_AR,               NCCL_ALGO_TREE,           NCCL_PROTO_NAN,    -1},
  {"RING_LL",                GENBIT(NCCL_ALGO_RING, NCCL_PROTO_LL),              F_ALL,              NCCL_ALGO_RING,           NCCL_PROTO_LL,     -1},
  {"RING_LL128",             GENBIT(NCCL_ALGO_RING, NCCL_PROTO_LL128),           F_ALL,              NCCL_ALGO_RING,           NCCL_PROTO_LL128,  -1},
  {"RING_SIMPLE",            GENBIT(NCCL_ALGO_RING, NCCL_PROTO_SIMPLE),          F_ALL,              NCCL_ALGO_RING,           NCCL_PROTO_SIMPLE, -1},
  {"RING_NAN",               GENBIT(NCCL_ALGO_RING, NCCL_PROTO_NAN),             F_AR | F_AG | F_RS, NCCL_ALGO_RING,           NCCL_PROTO_NAN,    -1},
  // COLLNET_DIRECT x LL / LL128 / NAN  -- combos not implemented
  {"COLLNET_DIRECT_SIMPLE",  GENBIT(NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE), F_AR | F_AG | F_RS, NCCL_ALGO_COLLNET_DIRECT, NCCL_PROTO_SIMPLE, -1},
  // COLLNET_CHAIN x LL / LL128 / NAN   -- combos not implemented
  {"COLLNET_CHAIN_SIMPLE",   GENBIT(NCCL_ALGO_COLLNET_CHAIN, NCCL_PROTO_SIMPLE), F_AR,               NCCL_ALGO_COLLNET_CHAIN,  NCCL_PROTO_SIMPLE, -1},
  // NVLS x LL / LL128 / NAN            -- combos not implemented
  {"NVLS_SIMPLE",            GENBIT(NCCL_ALGO_NVLS, NCCL_PROTO_SIMPLE),          F_AR | F_AG | F_RS, NCCL_ALGO_NVLS,           NCCL_PROTO_SIMPLE, -1},
  // NVLS_TREE x LL / LL128 / NAN       -- combos not implemented
  {"NVLSTREE_SIMPLE",        GENBIT(NCCL_ALGO_NVLS_TREE, NCCL_PROTO_SIMPLE),     F_AR,               NCCL_ALGO_NVLS_TREE,      NCCL_PROTO_SIMPLE, -1},
  // PAT x LL / LL128 / NAN             -- combos not implemented
  {"PAT_SIMPLE",             GENBIT(NCCL_ALGO_PAT, NCCL_PROTO_SIMPLE),           F_AG | F_RS,        NCCL_ALGO_PAT,            NCCL_PROTO_SIMPLE, -1},
  // Symmetric AllReduce kernels
  {"SYMK_AGxLL_R",           SYMBIT(ncclSymkKernelId_AllReduce_AGxLL_R),               F_AR, -1, -1, ncclSymkKernelId_AllReduce_AGxLL_R},
  {"SYMK_AGxLLMC_R",         SYMBIT(ncclSymkKernelId_AllReduce_AGxLLMC_R),             F_AR, -1, -1, ncclSymkKernelId_AllReduce_AGxLLMC_R},
  {"SYMK_RSxTmaLD_AGxTmaST", SYMBIT(ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST),     F_AR, -1, -1, ncclSymkKernelId_AllReduce_RSxTmaLD_AGxTmaST},
  {"SYMK_RSxLD_AGxST",       SYMBIT(ncclSymkKernelId_AllReduce_RSxLD_AGxST),           F_AR, -1, -1, ncclSymkKernelId_AllReduce_RSxLD_AGxST},
  {"SYMK_RSxLDMC_AGxSTMC",   SYMBIT(ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC),       F_AR, -1, -1, ncclSymkKernelId_AllReduce_RSxLDMC_AGxSTMC},
  // Symmetric AllGather kernels
  {"SYMK_LL",                SYMBIT(ncclSymkKernelId_AllGather_LL),                    F_AG, -1, -1, ncclSymkKernelId_AllGather_LL},
  {"SYMK_LLMC",              SYMBIT(ncclSymkKernelId_AllGather_LLMC),                  F_AG, -1, -1, ncclSymkKernelId_AllGather_LLMC},
  {"SYMK_TmaST",             SYMBIT(ncclSymkKernelId_AllGather_TmaST),                 F_AG, -1, -1, ncclSymkKernelId_AllGather_TmaST},
  {"SYMK_ST",                SYMBIT(ncclSymkKernelId_AllGather_ST),                    F_AG, -1, -1, ncclSymkKernelId_AllGather_ST},
  {"SYMK_TmaSTMC",           SYMBIT(ncclSymkKernelId_AllGather_TmaSTMC),               F_AG, -1, -1, ncclSymkKernelId_AllGather_TmaSTMC},
  {"SYMK_STMC",              SYMBIT(ncclSymkKernelId_AllGather_STMC),                  F_AG, -1, -1, ncclSymkKernelId_AllGather_STMC},
  {"SYMK_RailRing_LsaST",    SYMBIT(ncclSymkKernelId_AllGather_RailRing_LsaST),        F_AG, -1, -1, ncclSymkKernelId_AllGather_RailRing_LsaST},
  {"SYMK_RailRing_LsaSTMC",  SYMBIT(ncclSymkKernelId_AllGather_RailRing_LsaSTMC),      F_AG, -1, -1, ncclSymkKernelId_AllGather_RailRing_LsaSTMC},
  {"SYMK_HierLsaST",         SYMBIT(ncclSymkKernelId_AllGather_HierLsaST),             F_AG, -1, -1, ncclSymkKernelId_AllGather_HierLsaST},
  {"SYMK_HierLsaTmaST",      SYMBIT(ncclSymkKernelId_AllGather_HierLsaTmaST),          F_AG, -1, -1, ncclSymkKernelId_AllGather_HierLsaTmaST},
  // Symmetric ReduceScatter kernels
  {"SYMK_LL",                SYMBIT(ncclSymkKernelId_ReduceScatter_LL),                F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_LL},
  {"SYMK_TmaLD",             SYMBIT(ncclSymkKernelId_ReduceScatter_TmaLD),             F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_TmaLD},
  {"SYMK_LD",                SYMBIT(ncclSymkKernelId_ReduceScatter_LD),                F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_LD},
  {"SYMK_LDMC",              SYMBIT(ncclSymkKernelId_ReduceScatter_LDMC),              F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_LDMC},
  {"SYMK_RailA2A_LsaLD",     SYMBIT(ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD),     F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLD},
  {"SYMK_RailA2A_LsaLDMC",   SYMBIT(ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC),   F_RS, -1, -1, ncclSymkKernelId_ReduceScatter_RailA2A_LsaLDMC},
};
// clang-format on
static constexpr int algRegistryCount = (int)(sizeof(algRegistry) / sizeof(algRegistry[0]));

// OR of every selectable algorithm (the real rows) -- the base the parser negates ('^') against.
// The registry is compile-time constant, so this reduces to a single constant (no runtime loop).
static constexpr uint64_t computeAllAlgBits() {
  uint64_t m = 0;
  for (int i = 0; i < algRegistryCount; i++) m |= algRegistry[i].mask;
  return m;
}
static constexpr uint64_t kAllAlgBits = computeAllAlgBits();

uint64_t ncclAlgAllBits() {
  return kAllAlgBits;
}

uint64_t ncclAlgTagMask(const char* tag) {
  if (tag == NULL || tag[0] == '\0') return 0;
  // A tag matches a registry name only from its start (case-insensitive), so "RING" selects
  // RING_LL/LL128/SIMPLE but not SYMK_RailRing_LsaSTMC, while "NVLS" still spans NVLS_SIMPLE and
  // NVLSTREE_SIMPLE. strncasecmp over the tag length is that prefix test: a shorter name hits its
  // '\0' before the tag ends and compares unequal.
  size_t tagLen = strlen(tag);
  uint64_t m = 0;
  for (int i = 0; i < algRegistryCount; i++) {
    if (strncasecmp(algRegistry[i].name, tag, tagLen) == 0) m |= algRegistry[i].mask;
  }
  return m;
}

uint64_t ncclAlgValidForFuncMask(ncclFunc_t func) {
  uint64_t m = 0;
  for (int i = 0; i < algRegistryCount; i++) {
    if (algRegistry[i].collMask & (1u << func)) m |= algRegistry[i].mask;
  }
  return m;
}

const char* ncclAlgNameForGeneral(int algo, int proto) {
  for (int i = 0; i < algRegistryCount; i++) {
    if (algRegistry[i].algo == algo && algRegistry[i].proto == proto) return algRegistry[i].name;
  }
  return NULL;
}

const char* ncclAlgNameForSymk(int symkKernelId) {
  for (int i = 0; i < algRegistryCount; i++) {
    if (algRegistry[i].symkKernelId == symkKernelId) return algRegistry[i].name;
  }
  return NULL;
}
