/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-FileCopyrightText: Copyright (c) 2023, Meta Platforms, Inc. and affiliates.
 * SPDX-License-Identifier: Apache-2.0 and BSD-3
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

#ifndef NCCL_TUNER_H_
#define NCCL_TUNER_H_

#include "nccl.h"
#include "nccl_common.h"

#include "tuner/tuner_v6.h"
#include "tuner/tuner_v5.h"
#include "tuner/tuner_v4.h"
#include "tuner/tuner_v3.h"
#include "tuner/tuner_v2.h"

typedef ncclTuner_v6_t ncclTuner_t;
typedef ncclNvlDomainInfo_v6_t ncclNvlDomainInfo_t;

#define NCCL_TUNER_PLUGIN_SYMBOL "ncclTunerPlugin_v6"

#define NCCL_ALGO_UNDEF -1
#define NCCL_ALGO_TREE 0
#define NCCL_ALGO_RING 1
#define NCCL_ALGO_COLLNET_DIRECT 2
#define NCCL_ALGO_COLLNET_CHAIN 3
#define NCCL_ALGO_NVLS 4
#define NCCL_ALGO_NVLS_TREE 5
#define NCCL_ALGO_PAT 6
#define NCCL_NUM_ALGORITHMS NCCL_NUM_ALGORITHMS_V5 // Tree/Ring/CollNet*/PAT

#define NCCL_PROTO_UNDEF -1
#define NCCL_PROTO_LL 0
#define NCCL_PROTO_LL128 1
#define NCCL_PROTO_SIMPLE 2
// NaN-flag protocol (src/device/prims_nan.h): the payload doubles as the ready
// flag, so there is no flag lane on the wire. Kept last, and NCCL_NUM_PROTOCOLS
// is deliberately not tied to NCCL_NUM_PROTOCOLS_V5 any more: the v5 tuner plugin
// ABI must keep exactly three protocol columns.
#define NCCL_PROTO_NAN 3
#define NCCL_NUM_PROTOCOLS 4 // Simple/LL/LL128/NaN

#define NCCL_TUNING_IGNORE -1.0
#define NCCL_ALGO_PROTO_IGNORE NCCL_TUNING_IGNORE

#define NCCL_NUM_UNROLLS 6 // 1/2/4/8/16/32
#define NCCL_UNROLL_1 0
#define NCCL_UNROLL_2 1
#define NCCL_UNROLL_4 2
#define NCCL_UNROLL_8 3
#define NCCL_UNROLL_16 4
#define NCCL_UNROLL_32 5

#define NCCL_NUM_FLOATS 6 // half/float/double/rccl_bfloat16/rccl_float8/rccl_bfloat8

#define NCCL_HW_NVLINK 0
#define NCCL_HW_PCI 1
#define NCCL_HW_NET 2
#define NCCL_NUM_HW_LINKS NCCL_NUM_HW_LINKS_V5

#define NCCL_VOLTA_COMPCAP_IDX 0
#define NCCL_AMPERE_COMPCAP_IDX 1
#define NCCL_HOPPER_COMPCAP_IDX 2
#define NCCL_BLACKWELL_COMPCAP_IDX 3
#define NCCL_NUM_COMPCAPS NCCL_NUM_COMPCAPS_V5

#define NCCL_TUNING_SCALE_1NODE 0
#define NCCL_TUNING_SCALE_2NODES 1
#define NCCL_TUNING_SCALE_4NODES 2
#define NCCL_NUM_TUNING_SCALES NCCL_NUM_TUNING_SCALES_V5

// Internal mirror of ncclTunerConstants_v5_t widened to NCCL_NUM_PROTOCOLS. The
// v5 struct is frozen at three protocol columns because external tuner plugins
// compile against it, so anything that crosses the plugin boundary marshals
// through ncclTunerConstantsFromV5 / ncclTunerConstantsToV5 instead of aliasing.
typedef struct {
  double baseLatencies[NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
  double hwLatencies[NCCL_NUM_HW_LINKS][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];

  double llMaxBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double perChMaxRingLL128Bws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double perChMaxTreeLL128Bws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double perChMaxTreeBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double perChMaxNVLSTreeBws[NCCL_NUM_COMPCAPS][NCCL_NUM_TUNING_SCALES];
  double bwRatio[2][NCCL_NUM_ALGORITHMS][NCCL_NUM_PROTOCOLS];
} ncclTunerConstants_t;

#ifdef __cplusplus
// Narrow to the plugin-visible layout. Protocols beyond NCCL_NUM_PROTOCOLS_V5 are
// dropped; no plugin knows about them.
inline void ncclTunerConstantsToV5(ncclTunerConstants_v5_t* dst, const ncclTunerConstants_t* src) {
  for (int a = 0; a < NCCL_NUM_ALGORITHMS_V5; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS_V5; p++) {
      dst->baseLatencies[a][p] = src->baseLatencies[a][p];
      for (int hw = 0; hw < NCCL_NUM_HW_LINKS_V5; hw++) dst->hwLatencies[hw][a][p] = src->hwLatencies[hw][a][p];
      for (int n = 0; n < 2; n++) dst->bwRatio[n][a][p] = src->bwRatio[n][a][p];
    }
  }
  for (int c = 0; c < NCCL_NUM_COMPCAPS_V5; c++) {
    for (int s = 0; s < NCCL_NUM_TUNING_SCALES_V5; s++) {
      dst->llMaxBws[c][s] = src->llMaxBws[c][s];
      dst->perChMaxRingLL128Bws[c][s] = src->perChMaxRingLL128Bws[c][s];
      dst->perChMaxTreeLL128Bws[c][s] = src->perChMaxTreeLL128Bws[c][s];
      dst->perChMaxTreeBws[c][s] = src->perChMaxTreeBws[c][s];
      dst->perChMaxNVLSTreeBws[c][s] = src->perChMaxNVLSTreeBws[c][s];
    }
  }
}

// Widen back after the plugin has written its values. Columns the plugin cannot
// see keep whatever the caller had in them.
inline void ncclTunerConstantsFromV5(ncclTunerConstants_t* dst, const ncclTunerConstants_v5_t* src) {
  for (int a = 0; a < NCCL_NUM_ALGORITHMS_V5; a++) {
    for (int p = 0; p < NCCL_NUM_PROTOCOLS_V5; p++) {
      dst->baseLatencies[a][p] = src->baseLatencies[a][p];
      for (int hw = 0; hw < NCCL_NUM_HW_LINKS_V5; hw++) dst->hwLatencies[hw][a][p] = src->hwLatencies[hw][a][p];
      for (int n = 0; n < 2; n++) dst->bwRatio[n][a][p] = src->bwRatio[n][a][p];
    }
  }
  for (int c = 0; c < NCCL_NUM_COMPCAPS_V5; c++) {
    for (int s = 0; s < NCCL_NUM_TUNING_SCALES_V5; s++) {
      dst->llMaxBws[c][s] = src->llMaxBws[c][s];
      dst->perChMaxRingLL128Bws[c][s] = src->perChMaxRingLL128Bws[c][s];
      dst->perChMaxTreeLL128Bws[c][s] = src->perChMaxTreeLL128Bws[c][s];
      dst->perChMaxTreeBws[c][s] = src->perChMaxTreeBws[c][s];
      dst->perChMaxNVLSTreeBws[c][s] = src->perChMaxNVLSTreeBws[c][s];
    }
  }
}
#endif

#endif
