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


#include "hsa/hsa_ext_amd.h"
#include "aqlprofile-sdk/aql_profile_v2.h"
#include <stdlib.h>

#include <string>
#include <thread>

#include "ctrl/run_kernel.h"
#include "pgen/test_pgen_pcsmp.h"
#include "pgen/test_pgen_pmc.h"
#include "pgen/test_pgen_spm.h"
#include "pgen/test_pgen_sqtt.h"
#include "simple_convolution/simple_convolution.h"

char** pmc_argv(unsigned argc, const hsa_ven_amd_aqlprofile_event_t* events) {
  const int argv_pmc_size = 32;
  static unsigned argc_pmc = 0;
  static char* argv_arr = NULL;
  static char** argv_pmc = NULL;

  if (argc > argc_pmc) {
    argc_pmc = argc;
    argv_arr = reinterpret_cast<char*>(realloc(argv_arr, argc_pmc * argv_pmc_size));
    if (argv_pmc) delete argv_pmc;
    argv_pmc = new char*[argc + 1];
  }
  for (unsigned i = 0; i < argc; ++i) {
    char* argv_ptr = argv_arr + (i * argv_pmc_size);
    snprintf(argv_ptr, argv_pmc_size, "%d:%d:%d", events[i].block_name, events[i].block_index,
             events[i].counter_id);
    argv_pmc[i] = argv_ptr;
  }
  argv_pmc[argc] = NULL;
  return argv_pmc;
}

int main(int argc, char* argv[]) {
  bool ret_val = false;
  const bool pmc_enable = (getenv("AQLPROFILE_PMC") != NULL);
  const bool pmc_priv_enable = (getenv("AQLPROFILE_PMC_PRIV") != NULL);
  const bool sdma_enable = (getenv("AQLPROFILE_SDMA") != NULL);
  const bool sqtt_enable = (getenv("AQLPROFILE_SQTT") != NULL);
  const bool pcsmp_enable = (getenv("AQLPROFILE_PCSMP") != NULL);
  const bool scan_enable = (getenv("AQLPROFILE_SCAN") != NULL);
  const bool trace_enable = (getenv("AQLPROFILE_TRACE") != NULL);
  const bool spm_enable = (getenv("AQLPROFILE_SPM") != NULL);
  int scan_step = 1;
  const char* step_env = getenv("AQLPROFILE_SCAN_STEP");
  if (step_env != NULL) {
    int step = atoi(step_env);
    if (step <= 0) {
      std::cerr << "Error in setting environment variable AQLPROFILE_SCAN_STEP=" << step_env
                << ", it should be greater than or equal to 1." << std::endl;
      return 1;
    }
    scan_step = step;
  }

  const char* spm_loop_env = getenv("AQLPROFILE_SPM_LOOPS");
  int spm_loops = spm_loop_env ? atoi(spm_loop_env) : 1;
  if (!spm_loops) spm_loops = 1;

  if (!trace_enable) {
    std::clog.rdbuf(NULL);
  }
  if (scan_enable) {
    std::cerr.rdbuf(NULL);
  }

  TestHsa::HsaInstantiate();
  const hsa_ven_amd_aqlprofile_event_t* events_arr;

  // Run simple convolution test
  if (pmc_enable) {
    if (argc > 1) {
      ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(argc - 1, argv + 1);
    } else if (!scan_enable) {
      int events_count = 0;
      if (TestHsa::HsaAgentName() == "gfx9") {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY_CYCLES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 47 /*WAVE_READY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 1 /*CYCLE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 3 /*REQ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 22 /*WRITEBACK*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*ME1_STALL_WAIT_ON_RCIU_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 7},  /*PDE0_CACHE_REQS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 8},  /*PDE0_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 13}, /*BANK0_4K_PTE_CACHE_MISSES*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 14}, /*BANK0_BIGK_PTE_CACHE_HITS*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 15}, /*BANK0_BIGK_PTE_CACHE_MISSES*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 0},   /*CYCLE*/
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 2},   /*BANK0_REQUESTS*/
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      } else if (TestHsa::HsaAgentName() == "gfx12") {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHA, 0, 25 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHA, 0, 0 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHC, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CHC, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 25 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0, 24 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CPG, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_CPG, 0, 51 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL2, 0, 1},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_VML2, 0, 5},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 3},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 4},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, 0, 6},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCR, 0, 22},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, 0, 1 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2A, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, 0, 1 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL2C, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GRBM, 0, 2 /*GUI_ACTIVE*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_RLC, 0, 2},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_RLC, 0, 5},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 2 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 2 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL1, 0, 1},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GC_UTCL1, 0, 2},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GCEA_SE, 0, 3},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GCEA_SE, 0, 4},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GRBMH, 0, 0 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_GRBMH, 0, 19},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 46 /*CSN_BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 47 /*CSN_NUM_THREADGROUPS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0,14 /*ALWAYS*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0, 15 /*BUSY*/},
            {(hsa_ven_amd_aqlprofile_block_name_t)AQLPROFILE_BLOCK_NAME_SQG, 0, 19 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, 0, 21 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1A, 0, 0 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, 0, 0 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GL1C, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*ALWAYS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0, 15 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0, 1 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 96 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 10 /*REQ_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 0, 14 /*REQ_WRITE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 96 /*BUSY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 10 /*REQ_READ*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 14 /*REQ_WRITE*/},
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      } else {
        const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 47 /*WAVE_READY*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 1 /*CYCLE*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 3 /*REQS*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 22 /*WRITEBACK*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
            {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*ME1_STALL_WAIT_ON_RCIU_READ*/},
        };
        events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
        events_arr = events_arr1;
      }
      ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(
          events_count, pmc_argv(events_count, events_arr));
    } else {
      const int block_index_max = 16;
      const int event_id_max = 128;
      for (unsigned i = 0; i < AQLPROFILE_BLOCKS_NUMBER; ++i) {
        for (unsigned j = 0; j < block_index_max; ++j) {
          for (unsigned k = 0; k <= event_id_max; k += scan_step) {
            fflush(stdout);
            fprintf(stderr, " %d %d %d                 \r", i, j, k);
            fflush(stderr);
            hsa_ven_amd_aqlprofile_event_t event = {(hsa_ven_amd_aqlprofile_block_name_t)i, j, k};
            if (!RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(1, pmc_argv(1, &event))) {
              if (k == 0) {
                k = event_id_max + 1;
                if (j == 0) j = block_index_max + 1;
              }
              continue;
            }
          }
        }
      }
    }
  } else if (sdma_enable) {
    int events_count = 0;
    const hsa_ven_amd_aqlprofile_event_t events_sdma[] = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 17 /*MC_WR_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 0, 19 /*MC_RD_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 17 /*MC_WR_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SDMA, 1, 19 /*MC_RD_COUNT*/},
    };
    events_count = sizeof(events_sdma) / sizeof(hsa_ven_amd_aqlprofile_event_t);
    ret_val = RunKernel<simple_convolution, TestPGenPmc<SETUP_MODE> >(
        events_count, pmc_argv(events_count, events_sdma));
  } else if (pmc_priv_enable) {
    int events_count = 0;
    if (TestHsa::HsaAgentName() == "gfx9") {
      const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 7},  /*PDE0_CACHE_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 8},  /*PDE0_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 13}, /*BANK0_4K_PTE_CACHE_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 14}, /*BANK0_BIGK_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 15}, /*BANK0_BIGK_PTE_CACHE_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATCL2, 0, 2},   /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 2},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 7},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 8},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 0},    /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GCEA, 0, 2},    /*REQS_PER_CLIENT_GROUP*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 2},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 7},     /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 8},     /**/
      };
      events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
      events_arr = events_arr1;
    } else {
      const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCVML2, 0, 2},  /*BANK0_PTE_CACHE_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 1},   /*CORRECTABLE_GECC_ERR_CNT_CHAN0*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 2},   /*CORRECTABLE_GECC_ERR_CNT_CHAN1*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCARB, 0, 3},   /*UNCORRECTABLE_GECC_ERR_CNT_CHAN0*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 0},   /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 1},   /*ACPG_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 2},   /*ACPO_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCHUB, 0, 3},   /*IH_WRRET_VLD*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 0},  /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 1},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 2},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCXBAR, 0, 3},  /**/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 0}, /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 1}, /*TLB0_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 2}, /*TLB0_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_MCMCBVM, 0, 3}, /*TLB0_MISSES*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 24},    /*ATCL2_L1_REQAS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 25},    /*ATCL2_BANK0_REQS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_ATC, 0, 26},    /*ATCL2_BANK0_HITS*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 0},     /*CYCLE*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 2},     /*RD_REQS_IN*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 7},     /*WR_REQ_QUEUE2_IN*/
          {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_RPB, 0, 8},     /*WR_REQ_QUEUE3_IN*/
      };
      events_count = sizeof(events_arr1) / sizeof(hsa_ven_amd_aqlprofile_event_t);
      events_arr = events_arr1;
    }
    ret_val = RunKernel<simple_convolution, TestPGenPmc<RUN_MODE> >(
        events_count, pmc_argv(events_count, events_arr));
  } else if (sqtt_enable) {
    ret_val = RunKernel<simple_convolution, TestPGenSqtt>(argc, argv);
  } else if (pcsmp_enable && TestHsa::HsaAgentName().substr(0, 4) != "gfx1") {
    ret_val = RunKernel<simple_convolution, TestPGenPcsmp>(argc, argv);
  } else if (spm_enable) {
    int events_count = 0;
    const hsa_ven_amd_aqlprofile_event_t events_spm[] = {
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 48 /*CSN_BUSY*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 49 /*CSN_NUM_THREADGROUPS*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 51 /*CSN_EVENT_WAVE*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 47 /*CSN_WINDOW_VALID*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 8 /*SEL_ME1_STALL_WAIT_ON_RCIU_READ*/},
    };
    events_count = sizeof(events_spm) / sizeof(hsa_ven_amd_aqlprofile_event_t);

    ret_val = RunKernel<simple_convolution, TestPGenSpm>(
        events_count, pmc_argv(events_count, events_spm), spm_loops);
  } else {
    ret_val = RunKernel<simple_convolution, TestAql>(argc, argv);
  }
  TestHsa::HsaShutdown();

  return (ret_val) ? 0 : 1;
}
