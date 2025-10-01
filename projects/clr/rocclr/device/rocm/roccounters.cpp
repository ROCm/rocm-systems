/* Copyright (c) 2017 - 2025 Advanced Micro Devices, Inc.

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

#include "device/rocm/roccounters.hpp"
#include "device/rocm/rocvirtual.hpp"
#include <array>

namespace amd::roc {

hsa_status_t PerfCounterCallback(hsa_ven_amd_aqlprofile_info_type_t info_type,
                                 hsa_ven_amd_aqlprofile_info_data_t* info_data,
                                 void* callback_data) {
  typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> passed_data_t;

  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
    reinterpret_cast<passed_data_t*>(callback_data)->push_back(*info_data);
  }

  return HSA_STATUS_SUCCESS;
}

/*
 Converting from ORCA cmndefs.h to ROCR hsa_ven_amd_aqlprofile.h
 Note that some blocks are not defined in cmndefs.h
*/

struct BlockNameIdMapType {
  hsa_ven_amd_aqlprofile_block_name_t name;
  int value;
};

static constexpr BlockNameIdMapType viBlockIdOrcaToRocr[] = {
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CB0 - 0
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // CB1 - 1
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // CB2 - 2
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // CB3 - 3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0},     // CPF - 4
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // DB0 - 5
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // DB1 - 6
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // DB2 - 7
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // DB3 - 8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0},    // GRBM - 9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, 0},  // GRBMSE - 10
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SU - 11
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SC - 12
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0},     // SPI - 13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0},      // SQ - 14
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_ES - 15
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_GS - 16
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_VS - 17
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_PS - 18
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_LS - 19
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_HS - 20
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, 0},    // SQ_CS - 21
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, 0},      // SX - 22
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0},      // TA0 - 23
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 1},      // TA1 - 24
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 2},      // TA2 - 25
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 3},      // TA3 - 26
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 4},      // TA4 - 27
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 5},      // TA5 - 28
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 6},      // TA6 - 29
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 7},      // TA7 - 30
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 8},      // TA8 - 31
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 9},      // TA9 - 32
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0a},   // TA10 - 33
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0b},   // TA11 - 34
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0c},   // TA12 - 35
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0d},   // TA13 - 36
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0e},   // TA14 - 37
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0f},   // TA15 - 38
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, 0},     // TCA0 - 39
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, 1},     // TCA1 - 40
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0},     // TCC0 - 41
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 1},     // TCC1 - 42
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2},     // TCC2 - 43
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 3},     // TCC3 - 44
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 4},     // TCC4 - 45
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 5},     // TCC5 - 46
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 6},     // TCC6 - 47
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 7},     // TCC7 - 48
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 8},     // TCC8 - 49
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 9},     // TCC9 - 50
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0a},  // TCC10 - 51
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0b},  // TCC11 - 52
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0c},  // TCC12 - 53
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0d},  // TCC13 - 54
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0e},  // TCC14 - 55
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0f},  // TCC15 - 56
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0},      // TD0 - 57
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 1},      // TD1 - 58
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 2},      // TD2 - 59
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 3},      // TD3 - 60
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 4},      // TD4 - 61
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 5},      // TD5 - 62
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 6},      // TD6 - 63
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 7},      // TD7 - 64
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 8},      // TD8 - 65
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 9},      // TD9 - 66
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0a},   // TD10 - 67
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0b},   // TD11 - 68
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0c},   // TD12 - 69
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0d},   // TD13 - 70
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0e},   // TD14 - 71
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0f},   // TD15 - 72
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0},     // TCP0 - 73
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1},     // TCP1 - 74
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 2},     // TCP2 - 75
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 3},     // TCP3 - 76
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 4},     // TCP4 - 77
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 5},     // TCP5 - 78
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 6},     // TCP6 - 79
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 7},     // TCP7 - 80
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 8},     // TCP8 - 81
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 9},     // TCP9 - 82
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0a},  // TCP10 - 83
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0b},  // TCP11 - 84
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0c},  // TCP12 - 85
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0d},  // TCP13 - 86
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0e},  // TCP14 - 87
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0f},  // TCP15 - 88
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, 0},     // GDS - 89
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // VGT - 90
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // IA - 91
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCSEQ, 0},   // MC - 92
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SRBM, 0},    // SRBM - 93
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // WD - 94
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CPG - 95
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0},     // CPC - 96
};

// The number of counters per block has been increased for gfx9 but this table may not reflect all
// of them
// as compute may not use all of them.
static constexpr size_t gfx9BlockIdOrcaToRocrSize = 125;
static constexpr BlockNameIdMapType gfx9BlockIdOrcaToRocr[gfx9BlockIdOrcaToRocrSize] = {
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CB0
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // CB1
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // CB2
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // CB3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0},     // CPF
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // DB0
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // DB1
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // DB2
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // DB3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0},    // GRBM
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, 0},  // GRBMSE
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SU
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SC
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0},     // SPI
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0},      // SQ
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_ES
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_GS
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_VS
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_PS
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_LS
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_HS
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, 0},    // SQ_CS
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, 0},      // SX
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0},      // TA0
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 1},      // TA1
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 2},      // TA2
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 3},      // TA3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 4},      // TA4
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 5},      // TA5
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 6},      // TA6
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 7},      // TA7
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 8},      // TA8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 9},      // TA9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0a},   // TA10
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0b},   // TA11
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0c},   // TA12
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0d},   // TA13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0e},   // TA14
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0f},   // TA15
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, 0},     // TCA0
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCA, 1},     // TCA1
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0},     // TCC0
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 1},     // TCC1
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2},     // TCC2
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 3},     // TCC3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 4},     // TCC4
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 5},     // TCC5
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 6},     // TCC6
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 7},     // TCC7
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 8},     // TCC8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 9},     // TCC9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0a},  // TCC10
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0b},  // TCC11
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0c},  // TCC12
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0d},  // TCC13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0e},  // TCC14
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 0x0f},  // TCC15
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0},      // TD0
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 1},      // TD1
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 2},      // TD2
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 3},      // TD3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 4},      // TD4
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 5},      // TD5
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 6},      // TD6
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 7},      // TD7
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 8},      // TD8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 9},      // TD9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0a},   // TD10
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0b},   // TD11
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0c},   // TD12
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0d},   // TD13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0e},   // TD14
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0f},   // TD15
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0},     // TCP0
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1},     // TCP1
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 2},     // TCP2
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 3},     // TCP3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 4},     // TCP4
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 5},     // TCP5
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 6},     // TCP6
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 7},     // TCP7
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 8},     // TCP8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 9},     // TCP9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0a},  // TCP10
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0b},  // TCP11
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0c},  // TCP12
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0d},  // TCP13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0e},  // TCP14
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0f},  // TCP15
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, 0},     // GDS - 89
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // VGT - 90
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // IA - 91
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // WD - 92
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CPG - 93
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0},     // CPC - 94
                                                    // blocks that are not defined in GSL
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0},     // ATC - 97
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0},   // ATCL2  - 98
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0},  // MCVML2 - 99
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // EA - 100
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // EA - 101
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // EA - 102
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // EA - 103
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 4},      // EA - 104
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 5},      // EA - 105
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 6},      // EA - 106
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 7},      // EA - 107
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 8},      // EA - 108
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 9},      // EA - 109
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0a},   // EA - 110
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0b},   // EA - 111
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0c},   // EA - 112
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0d},   // EA - 113
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0e},   // EA - 114
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0f},   // EA - 115
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // RPB - 116
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // RMI - 117
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // RMI - 118
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // RMI - 119
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // RMI - 120
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 4},      // RMI - 121
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 5},      // RMI - 122
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 6},      // RMI - 123
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 7},      // RMI - 124
};

static constexpr size_t gfx10BlockIdOrcaToRocrSize = 139;
static constexpr BlockNameIdMapType gfx10BlockIdOrcaToRocr[gfx10BlockIdOrcaToRocrSize] = {
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CB0      - 0
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // CB1      - 1
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // CB2      - 2
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // CB3      - 3
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0},     // CPF      - 4
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // DB0      - 5
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // DB1      - 6
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // DB2      - 7
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // DB3      - 8
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0},    // GRBM     - 9
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBMSE, 0},  // GRBMSE   - 10
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SU    - 11
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SC0   - 12
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PA_SC1   - 13
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0},     // SPI      - 14
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0},      // SQ       - 15
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_ES    - 16
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_GS    - 17
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_VS    - 18
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_PS    - 19
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_LS    - 20
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // SQ_HS    - 21
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQCS, 0},    // SQ_CS    - 22
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SX, 0},      // SX       - 23
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0},      // TA0      - 24
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 1},      // TA1      - 25
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 2},      // TA2      - 26
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 3},      // TA3      - 27
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 4},      // TA4      - 28
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 5},      // TA5      - 29
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 6},      // TA6      - 30
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 7},      // TA7      - 31
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 8},      // TA8      - 32
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 9},      // TA9      - 33
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0a},   // TA10     - 34
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0b},   // TA11     - 35
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0c},   // TA12     - 36
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0d},   // TA13     - 37
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0e},   // TA14     - 38
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0x0f},   // TA15     - 39
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0},      // TD0      - 40
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 1},      // TD1      - 41
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 2},      // TD2      - 42
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 3},      // TD3      - 43
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 4},      // TD4      - 44
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 5},      // TD5      - 45
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 6},      // TD6      - 46
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 7},      // TD7      - 47
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 8},      // TD8      - 48
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 9},      // TD9      - 49
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0a},   // TD10     - 50
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0b},   // TD11     - 51
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0c},   // TD12     - 52
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0d},   // TD13     - 53
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0e},   // TD14     - 54
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0x0f},   // TD15     - 55
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0},     // TCP0     - 56
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1},     // TCP1     - 57
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 2},     // TCP2     - 58
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 3},     // TCP3     - 59
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 4},     // TCP4     - 60
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 5},     // TCP5     - 61
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 6},     // TCP6     - 62
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 7},     // TCP7     - 63
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 8},     // TCP8     - 64
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 9},     // TCP9     - 65
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0a},  // TCP10    - 66
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0b},  // TCP11    - 67
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0c},  // TCP12    - 68
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0d},  // TCP13    - 69
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0e},  // TCP14    - 70
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0x0f},  // TCP15    - 71
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, 0},     // GDS      - 72
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CPG      - 73
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0},     // CPC      - 74
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0},     // ATC      - 75
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0},   // ATCL2    - 76
    {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0},  // MCVML2   - 77
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // EA       - 78
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // EA       - 79
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // EA       - 80
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // EA       - 81
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 4},      // EA       - 82
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 5},      // EA       - 83
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 6},      // EA       - 84
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 7},      // EA       - 85
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 8},      // EA       - 86
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 9},      // EA       - 87
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0a},   // EA       - 88
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0b},   // EA       - 89
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0c},   // EA       - 90
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0d},   // EA       - 91
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0e},   // EA       - 92
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0f},   // EA       - 93
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // RPB      - 94
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // RMI0     - 95
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // RMI1     - 96
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GE       - 97
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GL1A     - 98
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GL1C     - 99
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GL1CG0   - 100
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // GL1CG1   - 101
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // GL1CG2   - 102
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // GL1CG3   - 103
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GL2A0    - 104
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // GL2A1    - 105
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // GL2A2    - 106
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // GL2A3    - 107
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GL2C0    - 108
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 1},      // GL2C1    - 109
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 2},      // GL2C2    - 110
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 3},      // GL2C3    - 111
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 4},      // GL2C4    - 112
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 5},      // GL2C5    - 113
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 6},      // GL2C6    - 114
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 7},      // GL2C7    - 115
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 8},      // GL2C8    - 116
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 9},      // GL2C9    - 117
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0a},   // GL2C10   - 118
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0b},   // GL2C11   - 119
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0c},   // GL2C12   - 120
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0d},   // GL2C13   - 121
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0e},   // GL2C14   - 122
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x0f},   // GL2C15   - 123
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x10},   // GL2C16   - 124
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x11},   // GL2C17   - 125
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x12},   // GL2C18   - 126
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x13},   // GL2C19   - 127
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x14},   // GL2C20   - 128
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x15},   // GL2C21   - 129
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x16},   // GL2C22   - 130
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0x17},   // GL2C23   - 131
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CHA      - 132
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CHC      - 133
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // CHCG     - 134
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GUS      - 135
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // GCR      - 136
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // PH       - 137
    {HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER, 0},      // UTCL1    - 138
};

//! Constructor for the ROC PerfCounter object
PerfCounter::PerfCounter(
    const Device& device,   //!< A ROC device object
    uint32_t blockIndex,    //!< HW block index
    uint32_t counterIndex,  //!< Counter index (Counter register) within the block
    uint32_t eventIndex)    //!< Event index (Counter selection) for profiling
    : roc_device_(device), profileRef_(nullptr) {
  info_.blockIndex_ = blockIndex;      // Block name + block index
  info_.counterIndex_ = counterIndex;  // Ignored as not being used in PPT library
  info_.eventIndex_ = eventIndex;      // Counter Event Selection (counter_id)

  // these block indices are valid for the SI (Gfx8) & Gfx9 devices
  switch (roc_device_.isa().versionMajor()) {
    case (9):
      gfxVersion_ = ROC_GFX9;
      if (blockIndex < gfx9BlockIdOrcaToRocrSize) {
        auto p = gfx9BlockIdOrcaToRocr[blockIndex];
        event_.block_name = p.name;
        event_.block_index = p.value;
      }
      break;
    case (10):
      gfxVersion_ = ROC_GFX10;
      if (blockIndex < gfx10BlockIdOrcaToRocrSize) {
        auto p = gfx10BlockIdOrcaToRocr[blockIndex];
        event_.block_name = p.name;
        event_.block_index = p.value;
      }
      break;
    default:
      gfxVersion_ = ROC_UNSUPPORTED;
      event_.block_name = HSA_VEN_AMD_AQLPROFILE_BLOCKS_NUMBER;
      event_.block_index = 0;
      break;
  }
  event_.counter_id = eventIndex;
}

void PerfCounter::setProfile(PerfCounterProfile* profileRef) {
  profileRef->perfCounters().push_back(this);
  profileRef->addEvent(event_);

  if (profileRef_ != nullptr) {
    profileRef_->release();
  }
  profileRef_ = profileRef;
  profileRef->retain();
}

uint64_t PerfCounter::getInfo(uint64_t infoType) const {
  switch (infoType) {
    case CL_PERFCOUNTER_GPU_BLOCK_INDEX: {
      // Return the GPU block index
      return info()->blockIndex_;
    }
    case CL_PERFCOUNTER_GPU_COUNTER_INDEX: {
      // Return the GPU counter index
      return info()->counterIndex_;
    }
    case CL_PERFCOUNTER_GPU_EVENT_INDEX: {
      // Return the GPU event index
      return info()->eventIndex_;
    }
    case CL_PERFCOUNTER_DATA: {
      const hsa_ven_amd_aqlprofile_profile_t* profile = profileRef_->profile();

      std::vector<hsa_ven_amd_aqlprofile_info_data_t> data;
      profileRef_->api()->hsa_ven_amd_aqlprofile_iterate_data(profile, PerfCounterCallback, &data);

      uint64_t result = 0;
      for (const auto& it : data) {
        if (it.pmc_data.event.block_name == event_.block_name &&
            it.pmc_data.event.block_index == event_.block_index &&
            it.pmc_data.event.counter_id == event_.counter_id) {
          result += it.pmc_data.result;
        }
      }
      return result;
    }
    default:
      LogError("Wrong PerfCounter::getInfo parameter");
  }
  return 0;
}

PerfCounter::~PerfCounter() {
  if (profileRef_ != nullptr) {
    profileRef_->release();
    profileRef_ = nullptr;
  }
}


bool PerfCounterProfile::initialize() {
  // save the current command and output buffer information
  hsa_ven_amd_aqlprofile_descriptor_t cmd_buf = profile_.command_buffer;
  hsa_ven_amd_aqlprofile_descriptor_t out_buf = profile_.output_buffer;

  // determine the required buffer sizes for the profiling events
  profile_.events = &events_[0];
  profile_.event_count = events_.size();
  profile_.command_buffer = {nullptr, 0};
  profile_.output_buffer = {nullptr, 0};

  if (api_.hsa_ven_amd_aqlprofile_start(&profile_, nullptr) != HSA_STATUS_SUCCESS) {
    LogError("Start hsa aql profile counter failed");
    return false;
  }

  const uint32_t alignment = amd::Os::pageSize();  // use page alignment

  if (cmd_buf.ptr != nullptr && cmd_buf.size != profile_.command_buffer.size) {
    roc_device_.memFree(cmd_buf.ptr, cmd_buf.size);
    cmd_buf.ptr = nullptr;
  }

  if (cmd_buf.ptr == nullptr) {
    void* buf_ptr = roc_device_.hostAlloc(profile_.command_buffer.size, alignment,
                                          Device::MemorySegment::kAtomics);
    if (buf_ptr != nullptr) {
      profile_.command_buffer.ptr = buf_ptr;
    } else {
      LogError("Failed to allocate profile counter command buffer");
      return false;
    }
  }

  if (out_buf.ptr != nullptr && out_buf.size != profile_.output_buffer.size) {
    roc_device_.memFree(out_buf.ptr, out_buf.size);
    out_buf.ptr = nullptr;
  }

  if (out_buf.ptr == nullptr) {
    void* buf_ptr = roc_device_.hostAlloc(profile_.output_buffer.size, alignment,
                                          Device::MemorySegment::kAtomics);
    if (buf_ptr != nullptr) {
      profile_.output_buffer.ptr = buf_ptr;
    } else {
      roc_device_.hostFree(profile_.command_buffer.ptr, profile_.command_buffer.size);
      LogError("Failed to allocate profile counter output buffer");
      return false;
    }
  }

  // create the completion signal
  if (Hsa::signal_create(1, 0, nullptr, &completionSignal_) != HSA_STATUS_SUCCESS) {
    LogError("Failed to create signal for profile counter");
    return false;
  }

  return true;
}

hsa_ext_amd_aql_pm4_packet_t* PerfCounterProfile::createStartPacket() {
  profile_.events = &events_[0];
  profile_.event_count = events_.size();

  // set up the profile aql packets for capturing performance counter
  if (api_.hsa_ven_amd_aqlprofile_start(&profile_, &prePacket_) != HSA_STATUS_SUCCESS) {
    DevLogError("Cannot Start AQL Profile \n");
    return nullptr;
  }

  return &prePacket_;
}

hsa_ext_amd_aql_pm4_packet_t* PerfCounterProfile::createStopPacket() {
  profile_.events = &events_[0];
  profile_.event_count = events_.size();

  // set up the profile aql packets for post-capturing performance counter
  // and create the completion signal
  if (api_.hsa_ven_amd_aqlprofile_stop(&profile_, &postPacket_) != HSA_STATUS_SUCCESS) {
    DevLogError("Cannot Stop AQL Profile \n");
    return nullptr;
  }

  postPacket_.completion_signal = completionSignal_;

  return &postPacket_;
}

PerfCounterProfile::~PerfCounterProfile() {
  if (completionSignal_.handle != 0) {
    Hsa::signal_destroy(completionSignal_);
  }

  if (profile_.command_buffer.ptr) {
    roc_device_.memFree(profile_.command_buffer.ptr, profile_.command_buffer.size);
  }

  if (profile_.output_buffer.ptr) {
    roc_device_.memFree(profile_.output_buffer.ptr, profile_.output_buffer.size);
  }
}

}  // namespace amd::roc
