// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <string>
#include <unordered_map>
#include <vector>

// Expected values for GFX908. GFX908 was chosen because it is not the first
// arch defined in the XML and it is also an arch that inherits values (from gfx9)
// Layout is: {name, block, event, expression, description}.
static const std::unordered_map<std::string, std::vector<std::vector<std::string>>> basic_gfx908 = {
    {"gfx908",
     {{"SQ_INSTS_VMEM_WR",
       "SQ",
       "28",
       "<None>",
       "The number of VMEM (GPU Memory) write instructions issued (including FLAT/scratch memory). "
       "The value is returned per-SE (aggregate of values in SIMDs in the SE)."},
      {"SQ_INSTS_VMEM_RD",
       "SQ",
       "29",
       "<None>",
       "The number of VMEM (GPU Memory) read instructions issued (including FLAT/scratch memory). "
       "The value is returned per-SE (aggregate of values in SIMDs in the SE)."},
      {"SQ_INSTS_SALU",
       "SQ",
       "31",
       "<None>",
       "Total Number of SALU (Scalar ALU) instructions issued. This value is returned per-SE "
       "(aggregate of values in SIMDs in the SE). See AMD ISAs for more information on SALU "
       "instructions."},
      {"SQ_INSTS_SMEM",
       "SQ",
       "32",
       "<None>",
       "Total number of SMEM (Scalar Memory Read) instructions issued. This value is returned "
       "per-SE (aggregate of values in SIMDs in the SE). See AMD ISAs for more information on SMEM "
       "instructions."},
      {"SQ_INSTS_FLAT",
       "SQ",
       "33",
       "<None>",
       "Total number of FLAT instructions issued. When used in combination with "
       "SQ_ACTIVE_INST_FLAT (cycle count for executing instructions) the average latency of FLAT "
       "instruction execution can be calculated (SQ_ACTIVE_INST_FLAT / SQ_INSTS). This value is "
       "returned per-SE (aggregate of values in SIMDs in the SE)."},
      {"SQ_INSTS_FLAT_LDS_ONLY",
       "SQ",
       "34",
       "<None>",
       "Total number of FLAT instructions issued that read/wrote only from/to LDS (scratch "
       "memory). Values are only populated if EARLY_TA_DONE is enabled. This value is returned "
       "per-SE (aggregate of values in SIMDs in the SE)."},
      {"SQ_INSTS_LDS",
       "SQ",
       "35",
       "<None>",
       "Total number of LDS instructions issued (including FLAT). This value is returned per-SE "
       "(aggregate of values in SIMDs in the SE). See AMD ISAs for more information on LDS "
       "instructions."},
      {"SQ_INSTS_GDS",
       "SQ",
       "36",
       "<None>",
       "Total number of GDS (global data sync) instructions issued. This value is returned per-SE "
       "(aggregate of values in SIMDs in the SE). See AMD ISAs for more information on GDS (global "
       "data sync) instructions."},
      {"SQ_WAIT_INST_LDS",
       "SQ",
       "64",
       "<None>",
       "Number of wave-cycles spent waiting for LDS instruction issue. In units of 4 cycles. "
       "(per-simd, nondeterministic)"},
      {"SQ_ACTIVE_INST_VALU",
       "SQ",
       "72",
       "<None>",
       "Number of cycles each wave spends working on a VALU instructions. This value represents "
       "the number of cycles each wave spends executing vector ALU instructions. On MI200 "
       "platforms, there are 4 VALUs per CU. High values indicates a large amount of time spent "
       "executing vector instructions. This value is returned on a per-SE (aggregate of values in "
       "SIMDs in the SE) basis with units in quad-cycles(4 cycles)."},
      {"SQ_INST_CYCLES_SALU",
       "SQ",
       "85",
       "<None>",
       "The number of cycles needed to execute non-memory read scalar operations (SALU). This "
       "value is returned on a per-SE (aggregate of values in SIMDs in the SE) basis with units in "
       "quad-cycles(4 cycles)."},
      {"SQ_THREAD_CYCLES_VALU",
       "SQ",
       "86",
       "<None>",
       "Number of thread-cycles used to execute VALU operations (similar to INST_CYCLES_VALU but "
       "multiplied by # of active threads). (per-simd)"},
      {"SQ_LDS_BANK_CONFLICT",
       "SQ",
       "94",
       "<None>",
       "The number of cycles LDS (local data store) is stalled by bank conflicts. This value is "
       "returned on a per-SE (aggregate of values in SIMDs in the SE) basis."},
      {"TCC_HIT", "TCC", "17", "<None>", "Number of cache hits."},
      {"TCC_MISS", "TCC", "19", "<None>", "Number of cache misses. UC reads count as misses."},
      {"TCC_EA_WRREQ",
       "TCC",
       "26",
       "<None>",
       "Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq interface. "
       "Atomics may travel over the same interface and are generally classified as write requests. "
       "This does not include probe commands."},
      {"TCC_EA_WRREQ_64B",
       "TCC",
       "27",
       "<None>",
       "Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq "
       "interface."},
      {"TCC_EA_WRREQ_STALL",
       "TCC",
       "30",
       "<None>",
       "Number of cycles a write request was stalled."},
      {"TCC_EA_RDREQ",
       "TCC",
       "38",
       "<None>",
       "Number of TCC/EA read requests (either 32-byte or 64-byte)"},
      {"TCC_EA_RDREQ_32B", "TCC", "39", "<None>", "Number of 32-byte TCC/EA read requests"},
      {"GRBM_COUNT", "GRBM", "0", "<None>", "Tie High - Count Number of Clocks"},
      {"GRBM_GUI_ACTIVE", "GRBM", "2", "<None>", "The GUI is Active"},
      {"SQ_WAVES",
       "SQ",
       "4",
       "<None>",
       "Count number of waves sent to distributed sequencers (SQs). This value represents the "
       "number of waves that are sent to each SQ. This only counts new waves sent since the start "
       "of collection (for dispatch profiling this is the timeframe of kernel execution, for agent "
       "profiling it is the timeframe between start_context and read counter data). A sum of all "
       "SQ_WAVES values will give the total number of waves started by the application during the "
       "collection timeframe. Returns one value per-SE (aggregates of SIMD values)."},
      {"SQ_INSTS_VALU",
       "SQ",
       "26",
       "<None>",
       "The number of VALU (Vector ALU) instructions issued. The value is returned per-SE "
       "(aggregate of values in SIMDs in the SE). See AMD ISAs for more information on VALU "
       "instructions."},
      {"TA_TA_BUSY",
       "TA",
       "15",
       "<None>",
       "TA block is busy. Perf_Windowing not supported for this counter."},
      {"TA_FLAT_READ_WAVEFRONTS",
       "TA",
       "101",
       "<None>",
       "Number of flat opcode reads processed by the TA."},
      {"TA_FLAT_WRITE_WAVEFRONTS",
       "TA",
       "102",
       "<None>",
       "Number of flat opcode writes processed by the TA."},
      {"TCP_TCP_TA_DATA_STALL_CYCLES",
       "TCP",
       "6",
       "<None>",
       "TCP stalls TA data interface. Now Windowed."}}}};

static const std::unordered_map<std::string, std::vector<std::vector<std::string>>> derived_gfx908 =
    {{"gfx908",
      {{"MAX_WAVE_SIZE", "", "", "wave_front_size", "Max wave size constant"},
       {"SE_NUM", "", "", "array_count/simd_arrays_per_engine", "SE_NUM"},
       {"SIMD_NUM", "", "", "simd_count", "SIMD Number"},
       {"CU_NUM", "", "", "simd_count/simd_per_cu", "CU_NUM"},
       {"GPUBusy",
        "",
        "",
        "100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)",
        "The percentage of time GPU was busy."},
       {"Wavefronts", "", "", "reduce(SQ_WAVES,sum)", "Total wavefronts."},
       {"VALUInsts",
        "",
        "",
        "reduce(SQ_INSTS_VALU,sum)/reduce(SQ_WAVES,sum)",
        "The average number of vector ALU instructions executed per work-item (affected by flow "
        "control)."},
       {"SALUInsts",
        "",
        "",
        "reduce(SQ_INSTS_SALU,sum)/reduce(SQ_WAVES,sum)",
        "The average number of scalar ALU instructions executed per work-item (affected by flow "
        "control)."},
       {"SFetchInsts",
        "",
        "",
        "reduce(SQ_INSTS_SMEM,sum)/reduce(SQ_WAVES,sum)",
        "The average number of scalar fetch instructions from the video memory executed per "
        "work-item (affected by flow control)."},
       {"GDSInsts",
        "",
        "",
        "reduce(SQ_INSTS_GDS,sum)/reduce(SQ_WAVES,sum)",
        "The average number of GDS read or GDS write instructions executed per work item "
        "(affected by flow control)."},
       {"MemUnitBusy",
        "",
        "",
        "100*reduce(TA_TA_BUSY,max)/reduce(GRBM_GUI_ACTIVE,max)",
        "The percentage of GPUTime the memory unit is active. The result includes the stall "
        "time (MemUnitStalled). This is measured with all extra fetches and writes and any "
        "cache or memory effects taken into account. Value range: 0% to 100% (fetch-bound)."},
       {"ALUStalledByLDS",
        "",
        "",
        "400*reduce(SQ_WAIT_INST_LDS,sum)/reduce(SQ_WAVES,sum)/reduce(GRBM_GUI_ACTIVE,max)",
        "The percentage of GPUTime ALU units are stalled by the LDS input queue being full or "
        "the output queue being not ready. If there are LDS bank conflicts, reduce them. "
        "Otherwise, try reducing the number of LDS accesses if possible. Value range: 0% "
        "(optimal) to 100% (bad)."},
       {"GPU_UTIL",
        "",
        "",
        "100*reduce(GRBM_GUI_ACTIVE,max)/reduce(GRBM_COUNT,max)",
        "Percentage of the time that GUI is active"},
       {"SQ_WAVES_sum",
        "",
        "",
        "reduce(SQ_WAVES,sum)",
        "Gives the total number of waves currently enqueued by the application during the "
        "collection timeframe (for dispatch profiling this is the timeframe of kernel execution, "
        "for agent profiling it is the timeframe between start_context and read counter data). See "
        "SQ_WAVES for more details."},
       {"TCC_HIT_sum",
        "",
        "",
        "reduce(TCC_HIT,sum)",
        "Number of cache hits. Sum over TCC instances."},
       {"TCC_MISS_sum",
        "",
        "",
        "reduce(TCC_MISS,sum)",
        "Number of cache misses. UC reads count as misses. Sum over TCC instances."},
       {"TCC_EA_RDREQ_32B_sum",
        "",
        "",
        "reduce(TCC_EA_RDREQ_32B,sum)",
        "Number of 32-byte TCC/EA read requests. Sum over TCC instances."},
       {"TCC_EA_RDREQ_sum",
        "",
        "",
        "reduce(TCC_EA_RDREQ,sum)",
        "Number of TCC/EA read requests (either 32-byte or 64-byte). Sum over TCC instances."},
       {"TCC_EA_WRREQ_sum",
        "",
        "",
        "reduce(TCC_EA_WRREQ,sum)",
        "Number of transactions (either 32-byte or 64-byte) going over the TC_EA_wrreq "
        "interface. Sum over TCC instances."},
       {"TCC_EA_WRREQ_64B_sum",
        "",
        "",
        "reduce(TCC_EA_WRREQ_64B,sum)",
        "Number of 64-byte transactions going (64-byte write or CMPSWAP) over the TC_EA_wrreq "
        "interface. Sum over TCC instances."},
       {"TCC_WRREQ_STALL_max",
        "",
        "",
        "reduce(TCC_EA_WRREQ_STALL,max)",
        "Number of cycles a write request was stalled. Max over TCC instances."},
       {"TA_BUSY_avr",
        "",
        "",
        "reduce(TA_TA_BUSY,avr)",
        "TA block is busy. Average over TA instances."},
       {"TA_BUSY_max",
        "",
        "",
        "reduce(TA_TA_BUSY,max)",
        "TA block is busy. Max over TA instances."},
       {"TA_BUSY_min",
        "",
        "",
        "reduce(TA_TA_BUSY,min)",
        "TA block is busy. Min over TA instances."},
       {"TA_FLAT_READ_WAVEFRONTS_sum",
        "",
        "",
        "reduce(TA_FLAT_READ_WAVEFRONTS,sum)",
        "Number of flat opcode reads processed by the TA. Sum over TA instances."},
       {"TA_FLAT_WRITE_WAVEFRONTS_sum",
        "",
        "",
        "reduce(TA_FLAT_WRITE_WAVEFRONTS,sum)",
        "Number of flat opcode writes processed by the TA. Sum over TA instances."},
       {"TCP_TCP_TA_DATA_STALL_CYCLES_sum",
        "",
        "",
        "reduce(TCP_TCP_TA_DATA_STALL_CYCLES,sum)",
        "Total number of TCP stalls TA data interface."},
       {"TCP_TCP_TA_DATA_STALL_CYCLES_max",
        "",
        "",
        "reduce(TCP_TCP_TA_DATA_STALL_CYCLES,max)",
        "Maximum number of TCP stalls TA data interface."},
       {"FETCH_SIZE",
        "",
        "",
        "(TCC_EA_RDREQ_32B_sum*32+(TCC_EA_RDREQ_sum-TCC_EA_RDREQ_32B_sum)*64)/1024",
        "The total kilobytes fetched from the video memory. This is measured with all extra "
        "fetches and any cache or memory effects taken into account."},
       {"WRITE_SIZE",
        "",
        "",
        "((TCC_EA_WRREQ_sum-TCC_EA_WRREQ_64B_sum)*32+TCC_EA_WRREQ_64B_sum*64)/1024",
        "The total kilobytes written to the video memory. This is measured with all extra "
        "fetches and any cache or memory effects taken into account."},
       {"WRITE_REQ_32B",
        "",
        "",
        "TCC_EA_WRREQ_64B_sum*2+(TCC_EA_WRREQ_sum-TCC_EA_WRREQ_64B_sum)",
        "The total number of 32-byte effective memory writes."},
       {"VFetchInsts",
        "",
        "",
        "(reduce(SQ_INSTS_VMEM_RD,sum)-TA_FLAT_READ_WAVEFRONTS_sum)/reduce(SQ_WAVES,sum)",
        "The average number of vector fetch instructions from the video memory executed per "
        "work-item (affected by flow control). Excludes FLAT instructions that fetch from video "
        "memory."},
       {"VWriteInsts",
        "",
        "",
        "(reduce(SQ_INSTS_VMEM_WR,sum)-TA_FLAT_WRITE_WAVEFRONTS_sum)/reduce(SQ_WAVES,sum)",
        "The average number of vector write instructions to the video memory executed per "
        "work-item (affected by flow control). Excludes FLAT instructions that write to video "
        "memory."},
       {"FlatVMemInsts",
        "",
        "",
        "(reduce(SQ_INSTS_FLAT,sum)-reduce(SQ_INSTS_FLAT_LDS_ONLY,sum))/reduce(SQ_WAVES,sum)",
        "The average number of FLAT instructions that read from or write to the video memory "
        "executed per work item (affected by flow control). Includes FLAT instructions that "
        "read from or write to scratch."},
       {"LDSInsts",
        "",
        "",
        "(reduce(SQ_INSTS_LDS,sum)-reduce(SQ_INSTS_FLAT_LDS_ONLY,sum))/reduce(SQ_WAVES,sum)",
        "The average number of LDS read or LDS write instructions executed per work item "
        "(affected by flow control).  Excludes FLAT instructions that read from or write to "
        "LDS."},
       {"FlatLDSInsts",
        "",
        "",
        "reduce(SQ_INSTS_FLAT_LDS_ONLY,sum)/reduce(SQ_WAVES,sum)",
        "The average number of FLAT instructions that read or write to LDS executed per work "
        "item (affected by flow control)."},
       {"VALUUtilization",
        "",
        "",
        "100*reduce(SQ_THREAD_CYCLES_VALU,sum)/(reduce(SQ_ACTIVE_INST_VALU,sum)*MAX_WAVE_SIZE)",
        "The percentage of active vector ALU threads in a wave. A lower number can mean either "
        "more thread divergence in a wave or that the work-group size is not a multiple of 64. "
        "Value range: 0\% (bad), 100\% (ideal - no thread divergence)."},
       {"VALUBusy",
        "",
        "",
        "100*reduce(SQ_ACTIVE_INST_VALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)",
        "The percentage of GPUTime vector ALU instructions are processed. Value range: 0\% "
        "(bad) to 100\% (optimal)."},
       {"SALUBusy",
        "",
        "",
        "100*reduce(SQ_INST_CYCLES_SALU,sum)/CU_NUM/reduce(GRBM_GUI_ACTIVE,max)",
        "The percentage of GPUTime scalar ALU instructions are processed. Value range: 0% (bad) "
        "to 100% (optimal)."},
       {"FetchSize",
        "",
        "",
        "FETCH_SIZE",
        "The total kilobytes fetched from the video memory. This is measured with all extra "
        "fetches and any cache or memory effects taken into account."},
       {"WriteSize",
        "",
        "",
        "WRITE_SIZE",
        "The total kilobytes written to the video memory. This is measured with all extra "
        "fetches and any cache or memory effects taken into account."},
       {"MemWrites32B",
        "",
        "",
        "WRITE_REQ_32B",
        "The total number of effective 32B write transactions to the memory"},
       {"L2CacheHit",
        "",
        "",
        "100*reduce(TCC_HIT,sum)/(reduce(TCC_HIT,sum)+reduce(TCC_MISS,sum))",
        "The percentage of fetch, write, atomic, and other instructions that hit the data in L2 "
        "cache. Value range: 0\% (no hit) to 100\% (optimal)."},
       {"MemUnitStalled",
        "",
        "",
        "100*TCP_TCP_TA_DATA_STALL_CYCLES_max/reduce(GRBM_GUI_ACTIVE,max)/SE_NUM",
        "The percentage of GPUTime the memory unit is stalled. Try reducing the number or size "
        "of fetches and writes if possible. Value range: 0\% (optimal) to 100\% (bad)."},
       {"WriteUnitStalled",
        "",
        "",
        "100*TCC_WRREQ_STALL_max/reduce(GRBM_GUI_ACTIVE,max)",
        "The percentage of GPUTime the Write unit is stalled. Value range: 0\% to 100\% (bad)."},
       {"LDSBankConflict",
        "",
        "",
        "100*reduce(SQ_LDS_BANK_CONFLICT,sum)/reduce(GRBM_GUI_ACTIVE,max)/CU_NUM",
        "The percentage of GPUTime LDS is stalled by bank conflicts. Value range: 0\% (optimal) "
        "to 100\% (bad)."}}}};
