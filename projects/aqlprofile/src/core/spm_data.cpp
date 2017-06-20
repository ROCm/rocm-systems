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

#include <dirent.h>
#include "hsa/hsa_ext_amd.h"
#include <pthread.h>
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

#include "core/aql_profile.hpp"
#include "core/logger.h"
#include "core/pm4_factory.h"

#define PTHREAD_CALL(call) \
  do {                     \
    int err = call;        \
    if (err != 0) {        \
      errno = err;         \
      perror(#call);       \
      abort();             \
    }                      \
  } while (0)

namespace spm_kfd_namespace {

int get_gpu_node_id(uint32_t gpu_ind) {
  int gpu_node = -1;
  uint32_t index = 0;

  // find a valid gpu node from /sys/class/kfd/kfd/topology/nodes
  std::string path = "/sys/class/kfd/kfd/topology/nodes";
  DIR* dir;
  struct dirent* ent;

  if ((dir = opendir(path.c_str())) != NULL) {
    while ((ent = readdir(dir)) != NULL) {
      std::string dir = ent->d_name;

      if (dir.find_first_not_of("0123456789") == std::string::npos) {
        std::string file = path + "/" + ent->d_name + "/gpu_id";
        std::ifstream infile(file);
        int id;

        infile >> id;
        if ((id != 0) && (index == gpu_ind)) {
          ++index;
          gpu_node = atoi(ent->d_name);
          break;
        }
      }
    }
    closedir(dir);
  }

  if (gpu_node == -1) {
    printf("get_gpu_node_id`error: GPU[%d] not found\n", gpu_ind);
    fflush(stdout);
    abort();
  }

  return gpu_node;
}

int get_gpu_node_id(hsa_agent_t agent) {
  const uint32_t gpu_ind = HsaRsrcFactory::Instance().GetAgentInfo(agent)->dev_index;
  return get_gpu_node_id(gpu_ind);
}

struct state_t {
  bool thread_stop;
  int node_id;
  uint32_t buf_size;
  uint32_t timeout;
  uint32_t data_size;
  void* kfd_buf;
  void* prod_buf;
  void* cons_buf;
  bool data_loss;
  bool ready;
  pthread_mutex_t work_mutex;
  pthread_cond_t work_cond;
  hsa_agent_t agent;
};

void producer_fun(state_t* state) {
  uint32_t timeout = 0;
  hsa_status_t status = HSA_STATUS_SUCCESS;
  // hsa_amd_spm_set_dest_buffer(state->agent, state->buf_size, &timeout, &(state->data_size),
  //                          state->kfd_buf, &(state->data_loss));
  if (status != HSA_STATUS_SUCCESS) {
    printf("hsa SPM Set DestBuffer init error\n");
    fflush(stdout);
    abort();
  }

  do {
    timeout = state->timeout;
    status = HSA_STATUS_SUCCESS;
    // hsa_amd_spm_set_dest_buffer(state->agent, state->buf_size, &timeout, &(state->data_size),
    //                               state->prod_buf, &(state->data_loss));
    if (status != HSA_STATUS_SUCCESS) {
      printf("hsa SPM Set DestBuffer error\n");
      fflush(stdout);
      abort();
    }

    PTHREAD_CALL(pthread_mutex_lock(&(state->work_mutex)));
    void* tmp = state->prod_buf;
    state->prod_buf = state->cons_buf;
    state->cons_buf = state->kfd_buf;
    state->kfd_buf = tmp;
    state->ready = true;
    PTHREAD_CALL(pthread_cond_signal(&(state->work_cond)));
    PTHREAD_CALL(pthread_mutex_unlock(&(state->work_mutex)));
  } while (!state->thread_stop);

  status = HSA_STATUS_SUCCESS;
  // hsa_amd_spm_set_dest_buffer(state->agent, 0, &timeout, &(state->data_size), NULL,
  //                               &(state->data_loss));
  if (status != HSA_STATUS_SUCCESS) {
    printf("hsa SPM Set DestBuffer stop error\n");
    fflush(stdout);
    abort();
  }
}

void consumer_fun(state_t* state, hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  const uint32_t sample_id = 0;
  PTHREAD_CALL(pthread_mutex_lock(&(state->work_mutex)));
  do {
    while (state->ready == false) {
      PTHREAD_CALL(pthread_cond_wait(&(state->work_cond), &(state->work_mutex)));
    }
    state->ready = false;

    hsa_ven_amd_aqlprofile_info_data_t sample_info;
    sample_info.sample_id = sample_id;
    sample_info.trace_data.ptr = state->cons_buf;
    sample_info.trace_data.size = state->data_size;

    hsa_status_t status = callback(HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA, &sample_info, data);
    if (status == HSA_STATUS_INFO_BREAK) {
      status = HSA_STATUS_SUCCESS;
      state->thread_stop = true;
      break;
    } else if (status != HSA_STATUS_SUCCESS) {
      printf("SPM consumer callback failed\n");
      abort();
    }
  } while (1);
  PTHREAD_CALL(pthread_mutex_unlock(&(state->work_mutex)));
}

void mananger_fun(const hsa_ven_amd_aqlprofile_profile_t* profile,
                  hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  state_t obj{};
  const int gpu_node_id = get_gpu_node_id(profile->agent);
  char* buf_ptr = (char*)(profile->output_buffer.ptr);
  // SPM data buffer size 256 byte aligned
  const uint32_t buf_size = (profile->output_buffer.size / 3) & ~(uint32_t(256) - 1);

  obj.timeout = 1000000;  // 1sec
  obj.node_id = gpu_node_id;
  obj.buf_size = buf_size;
  obj.kfd_buf = buf_ptr;
  obj.prod_buf = buf_ptr + buf_size;
  obj.cons_buf = buf_ptr + 2 * buf_size;
  obj.agent = profile->agent;

  PTHREAD_CALL(pthread_mutex_init(&(obj.work_mutex), NULL));
  PTHREAD_CALL(pthread_cond_init(&(obj.work_cond), NULL));

  hsa_status_t status = HSA_STATUS_SUCCESS;  // hsa_amd_spm_acquire(profile->agent);
  if (status != HSA_STATUS_SUCCESS) {
    printf("hsa SPM Acquire error\n");
    fflush(stdout);
    abort();
  }

  // spm threads
  std::thread producer(producer_fun, &obj);
  std::thread consumer(consumer_fun, &obj, callback, data);

  producer.join();
  consumer.join();

  status = HSA_STATUS_SUCCESS;  // hsa_amd_spm_release(profile->agent);
  if (status != HSA_STATUS_SUCCESS) {
    printf("hsa SPM Release error\n");
    fflush(stdout);
    abort();
  }
}

typedef std::mutex spm_mutex_t;
spm_mutex_t spm_mutex;

// Getting SPM data using driver API
hsa_status_t spm_iterate_data(const hsa_ven_amd_aqlprofile_profile_t* profile,
                              hsa_ven_amd_aqlprofile_data_callback_t callback, void* data) {
  std::lock_guard<spm_mutex_t> lck(spm_mutex);
  static std::thread* t = NULL;

  if (t == NULL) {
    // spm manager thread
    t = new std::thread(mananger_fun, profile, callback, data);
  } else {
    t->join();
  }

  return HSA_STATUS_SUCCESS;
}

}  // namespace spm_kfd_namespace
