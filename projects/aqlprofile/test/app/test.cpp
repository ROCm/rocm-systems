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

const int argv_pmc_size = 32;
unsigned argc_pmc = 0;
char* argv_arr = NULL;
char** argv_pmc = NULL;

typedef struct {
  uint32_t size;  // size of buffer in bytes
  uint32_t timeout;
  uint32_t len;    // len of streamed data in spm buffer
  void* addr;      // address of spm buffer
  bool data_loss;  // OUT
} spm_buffer_params_t;

int gpu_node_id = -1;
spm_buffer_params_t spm_buffer_params[2];
std::atomic<uint32_t> spm_buffer_idx{0};  // current spm buffer in use for spm samples
std::atomic<bool> spm_check_data{
    true};                           // request to check spm data in spm_buffer at spm_buffer_idx
std::atomic<bool> test_done{false};  // is GPU kernel finished?

int get_gpu_node_id() {
  int gpu_node = -1;

#if 0
  // find a valid gpu node from /sys/class/kfd/kfd/topology/nodes
  std::string path = "/sys/class/kfd/kfd/topology/nodes";
  DIR *dir;
  struct dirent *ent;

  if ((dir = opendir(path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string dir = ent->d_name;

      if (dir.find_first_not_of("0123456789") == std::string::npos) {
        std::string file = path + "/" + ent->d_name + "/gpu_id";
        std::ifstream infile(file);
        int id;

        infile >> id;
        if (id != 0) {
          gpu_node = atoi(ent->d_name);
          break;
        }
      }
    }
    closedir(dir);
  }

        HsaSystemProperties m_SystemProperties;
        memset(&m_SystemProperties, 0, sizeof(m_SystemProperties));

        HSAKMT_STATUS status = hsaKmtAcquireSystemProperties(&m_SystemProperties);
        if (status != HSAKMT_STATUS_SUCCESS) {
          std::cerr << "Error in hsaKmtAcquireSystemProperties" << std::endl;
          return 1;
        }

        // tranverse all CPU and GPU nodes and break when a GPU node is found
        for (unsigned i = 0; i < m_SystemProperties.NumNodes; ++i) {
          HsaNodeProperties nodeProperties;
          memset(&nodeProperties, 0, sizeof(HsaNodeProperties));

          status = hsaKmtGetNodeProperties(i, &nodeProperties);
          if (status != HSAKMT_STATUS_SUCCESS) {
            std::cerr << "Error in hsaKmtAcquireSystemProperties" << std::endl;
            break;
          } else if (nodeProperties.NumFComputeCores) {
            gpu_node = i;
            break;
          }
        }
#endif

  printf("GPU node id(%d)\n", gpu_node);
  return gpu_node;
}

char** pmc_argv(unsigned argc, const hsa_ven_amd_aqlprofile_event_t* events) {
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

typedef char** pf_pmc_argv(unsigned argc, const hsa_ven_amd_aqlprofile_event_t* events);

void thread_kernel(bool* ret_val, pf_pmc_argv pmc_argv, int events_count,
                   const hsa_ven_amd_aqlprofile_event_t* events) {
  *ret_val =
      RunKernel<simple_convolution, TestPGenSpm>(events_count, pmc_argv(events_count, events));
  test_done = true;
}

void thread_spm_buffer_setup() {
#if ENABLE_SPM
  while (!test_done) {
    auto idx = (spm_buffer_idx.load() + 1) & 0x1;
    std::cout << "thread_spm_buffer_setup: " << idx << std::endl;
    hsa_status_t status =
        hsa_amd_spm_set_dest_buffer(gpu_node_id, spm_buffer_params[idx].size,
                                    &spm_buffer_params[idx].timeout, &spm_buffer_params[idx].len,
                                    spm_buffer_params[idx].addr, &spm_buffer_params[idx].data_loss);
    if (status != HSA_STATUS_SUCCESS) {
      std::cerr << "Error in initial spm setup of buffer 0" << std::endl;
      return;
    }
    // inform data saving thread there is spm data to save
    if (spm_buffer_params[idx].len != 0) spm_check_data = true;
  }
#endif

  std::cout << "Exiting thread_spm_buffer_setup ..." << std::endl;
}

void thread_spm_data_save(FILE* file) {
  while (!test_done) {
    if (spm_check_data) {
      auto buffer_idx = spm_buffer_idx.load();
      auto idx = buffer_idx & 0x1;
      std::cout << "thread_spm_data_save " << idx << " with " << spm_buffer_params[idx].len
                << " bytes" << std::endl;
      fwrite(spm_buffer_params[idx].addr, 1, spm_buffer_params[idx].len, file);

      spm_buffer_idx = buffer_idx + 1;
      spm_check_data = false;
    }
  }
  std::cout << "Exiting thread_spm_data_save ..." << std::endl;
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
  [[maybe_unused]] const bool spm_kfd_mode = (getenv("AQLPROFILE_SPM_KFD_MODE") != NULL);
  // int gpu_node_id = -1;
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

  if (!trace_enable) {
    std::clog.rdbuf(NULL);
  }
  if (scan_enable) {
    std::cerr.rdbuf(NULL);
  }

  //{
  //  hsa_status_t status = hsa_init();
  //  CHECK_STATUS("Error in hsa_init", status);
  //}

  if (spm_enable) {
#if ENABLE_SPM
    {
      hsa_status_t status = hsa_init();
      CHECK_STATUS("Error in hsa_init", status);
    }
    gpu_node_id = get_gpu_node_id();
    if (gpu_node_id == -1) {
      std::cerr << "Error in get_gpu_node_id()" << std::endl;
      return 1;
    }

    if (spm_kfd_mode) {
      hsa_status_t status = hsa_amd_spm_acquire(gpu_node_id);
      if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Error in acquiring SPM for NodeId " << gpu_node_id << std::endl;
        return 1;
      }
    } else {
#if SPM_DEBUG_TRAP
      HSAKMT_STATUS status = hsaKmtEnableDebugTrap(gpu_node_id, INVALID_QUEUEID);
#else
      HSAKMT_STATUS status = HSAKMT_STATUS_ERROR;
#endif
      if (status != HSAKMT_STATUS_SUCCESS) {
        std::cerr << "Error in enabling debug trap for NodeId " << gpu_node_id << std::endl;
        return 1;
      }
    }
#endif
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
#ifdef ENABLE_SPM
    int events_count = 0;
    const hsa_ven_amd_aqlprofile_event_t events_spm[] = {
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 0 /*NONE*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY_CYCLES*/},
        {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 4 /*WAVES*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 14 /*ITEMS*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 47 /*WAVE_READY*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SPI, 0, 47 /*CSN_WINDOW_VALID*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TA, 0, 1 /*SH_FIFO_BUSY*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TD, 0, 1 /*TD_BUSY*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 3 /*REQ*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 22 /*WRITEBACK*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCC, 2, 1 /*CYCLE*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_TCP, 1, 2 /*CORE_REG_SCLK_VLD*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPC, 0, 0 /*ALWAYS_COUNT*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_CPF, 0, 0 /*ALWAYS_COUNT*/},
        // {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_GDS, 0, 0 /*DS_ADDR_CONFL*/},
    };
    events_count = sizeof(events_spm) / sizeof(hsa_ven_amd_aqlprofile_event_t);

    if (spm_kfd_mode) {
      // open a binary file for spm samples
      const std::string spm_sample_file = "example.txt";
      FILE* file = fopen(spm_sample_file.c_str(), "wb");
      if (file == NULL) {
        std::cerr << "Error opening file " << spm_sample_file << " for spm sampling!" << std::endl;
        return 1;
      }

      // set up 2 spm sample data buffers
      const uint32_t timeout = 10000;
      const uint32_t spm_buffer_size = 0x2000000;
      spm_buffer_params[0].size = spm_buffer_size;
      spm_buffer_params[0].timeout = timeout;
      spm_buffer_params[0].len = 0;
      spm_buffer_params[0].addr = malloc(spm_buffer_size);
      if (spm_buffer_params[0].addr == NULL) {
        std::cerr << "Malloc(size) for spm buffer 0 Failed." << std::endl;
        return 1;
      }
      spm_buffer_params[0].data_loss = false;

      spm_buffer_params[1].size = spm_buffer_size;
      spm_buffer_params[1].timeout = timeout;
      spm_buffer_params[1].len = 0;
      spm_buffer_params[1].addr = malloc(spm_buffer_size);
      if (spm_buffer_params[1].addr == NULL) {
        std::cerr << "Malloc(size) for spm buffer 1 Failed." << std::endl;
        return 1;
      }
      spm_buffer_params[1].data_loss = false;

      // non-blocking set up the first spm buffer for use before GPU kernel started
      std::cout << "spm_buffer_setup 0 ..." << std::endl;
      hsa_status_t status = hsa_amd_spm_set_dest_buffer(
          gpu_node_id, spm_buffer_params[0].size, &spm_buffer_params[0].timeout,
          &spm_buffer_params[0].len, spm_buffer_params[0].addr, &spm_buffer_params[0].data_loss);
      if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Error in initial spm setup of buffer 0" << std::endl;
        return 1;
      }

      // spm threads
      std::thread k_thread(thread_kernel, &ret_val, pmc_argv, events_count, events_spm);
      std::thread buffer_setup(thread_spm_buffer_setup);
      std::thread data_save(thread_spm_data_save, file);

      k_thread.join();
      buffer_setup.join();
      data_save.join();

      // my_anaylyze(spm_buffer_params[0].addr, spm_buffer_params[0].size,
      // spm_buffer_params[0].len); my_anaylyze(spm_buffer_params[1].addr,
      // spm_buffer_params[1].size, spm_buffer_params[1].len);

      // free allocated spm buffers
      free(spm_buffer_params[0].addr);
      free(spm_buffer_params[1].addr);

      std::cout << "data in buff0: " << spm_buffer_params[0].len << " bytes" << std::endl;
      std::cout << "data in buff1: " << spm_buffer_params[1].len << " bytes" << std::endl;

      status = hsa_amd_spm_release(gpu_node_id);
      if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Error in releasing SPM for NodeId " << gpu_node_id << std::endl;
        return 1;
      }

      // close spm sample binary file
      fclose(file);
    } else {
      ret_val = RunKernel<simple_convolution, TestPGenSpm>(events_count,
                                                           pmc_argv(events_count, events_spm));
#if SPM_DEBUG_TRAP
      HSAKMT_STATUS status = hsaKmtDisableDebugTrap(gpu_node_id);
#else
      hsa_status_t status = HSA_STATUS_ERROR;
#endif
      if (status != HSA_STATUS_SUCCESS) {
        std::cerr << "Error in disabling debug trap for NodeId " << gpu_node_id << std::endl;
        return 1;
      }
    }

    // SPM data analysis: need to change command dependent on binary vs text sample files
    // The 'da_16b.py' script is checking the first SPM counter data to match expected value
    // provided by the option '-e'
    std::string command = "python3 da_16b.py -e 16 spm_dump_0.txt";

    int command_status = system(command.c_str());

    if (command_status == -1)
      std::cerr << "Cannot run \"" << command << "\"" << std::endl;
    else if (command_status == 0)
      std::cout << "SPM test passed!" << std::endl;
    else
      std::cerr << "SPM test failed!" << std::endl;
#endif
  } else {
    ret_val = RunKernel<simple_convolution, TestAql>(argc, argv);
  }
  TestHsa::HsaShutdown();

  return (ret_val) ? 0 : 1;
}
