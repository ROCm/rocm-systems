/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*******************************************************************************/

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
// Test tool used as ROC profiler library demo                               //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include <assert.h>
#include <dirent.h>
#include <hsa.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "inc/rocprofiler.h"
#include "util/hsa_rsrc_factory.h"
#include "util/xml.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))
#define KERNEL_NAME_LEN_MAX 128

// Disoatch callback data type
struct callbacks_data_t {
  rocprofiler_feature_t* features;
  unsigned feature_count;
  std::vector<uint32_t>* set;
  unsigned group_index;
  FILE* file_handle;
  int filter_on;
  std::vector<uint32_t>* gpu_index;
  std::vector<std::string>* kernel_string;
  std::vector<uint32_t>* range;
};

// kernel properties structure
struct kernel_properties_t {
  uint32_t grid_size;
  uint32_t workgroup_size;
  uint32_t lds_size;
  uint32_t scratch_size;
  uint32_t vgpr_count;
  uint32_t sgpr_count;
  uint32_t fbarrier_count;
};

// Context stored entry type
struct context_entry_t {
  bool valid;
  bool active;
  uint32_t index;
  hsa_agent_t agent;
  rocprofiler_group_t group;
  rocprofiler_feature_t* features;
  unsigned feature_count;
  rocprofiler_callback_data_t data;
  kernel_properties_t kernel_properties;
  FILE* file_handle;
};

//
const std::string rcfile_name = "rpl_rc.xml";
// verbose mode
static uint32_t verbose = 0;
// Enable tracing
static const bool trace_on = false;
// Tool is unloaded
volatile bool is_loaded = false;
// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Dispatch callback data
callbacks_data_t* callbacks_data = NULL;
// Stored contexts array
typedef std::map<uint32_t, context_entry_t> context_array_t;
context_array_t* context_array = NULL;
// Contexts collected count
volatile uint32_t context_count = 0;
volatile uint32_t context_collected = 0;
// Profiling results output file name
const char* result_prefix = NULL;
// Global results file handle
FILE* result_file_handle = NULL;
// True if a result file is opened
bool result_file_opened = false;
// Dispatch filters
// Metrics set
std::vector<uint32_t>* metrics_set = NULL;
// GPU index filter
std::vector<uint32_t>* gpu_index_vec = NULL;
//  Kernel name filter
std::vector<std::string>* kernel_string_vec = NULL;
//  DIspatch number range filter
std::vector<uint32_t>* range_vec = NULL;
// Otstanding dispatches parameters
static uint32_t CTX_OUTSTANDING_MAX = 0;
static uint32_t CTX_OUTSTANDING_MON = 0;
// to truncate kernel names
uint32_t to_truncate_names = 0;
// local SQTT buffer
bool is_sqtt_local = true;

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    abort();
  }
}

// Print profiling results output break if terminal output is enabled
void results_output_break() {
  const bool is_terminal_output = (result_file_opened == false);
  if (is_terminal_output) printf("\nROCprofiler results:\n");
}

// Filtering kernel name
std::string filtr_kernel_name(const std::string name) {
  auto rit = name.rbegin();
  auto rend = name.rend();
  uint32_t counter = 0;
  char open_token = 0;
  char close_token = 0;
  while (rit != rend) {
    if (counter == 0) {
      switch (*rit) {
        case ')':
          counter = 1;
          open_token = ')';
          close_token = '(';
          break;
        case '>':
          counter = 1;
          open_token = '>';
          close_token = '<';
          break;
      }
      if (counter == 0) break;
    } else {
      if (*rit == open_token) counter++;
      if (*rit == close_token) counter--;
    }
    ++rit;
  }
  while (rit != rend) if ((*rit == ' ') || (*rit == '	')) rit++; else break;
  auto rbeg = rit;
  while (rit != rend) if ((*rit != ' ') && (*rit != ':')) rit++; else break;
  const uint32_t pos = rend - rit;
  const uint32_t length = rit - rbeg;
  return name.substr(pos, length);
}

// Inflight submits monitoring thread
void* monitor_thr_fun(void*) {
  while (context_array != NULL) {
    sleep(CTX_OUTSTANDING_MON);
    if (pthread_mutex_lock(&mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
    }
    const uint32_t inflight = context_count - context_collected;
    std::cerr << std::flush;
    std::clog << std::flush;
    std::cout << "ROCProfiler: count(" << context_count << "), outstanding(" << inflight << "/" << CTX_OUTSTANDING_MAX << ")" << std::endl << std::flush;
    if (pthread_mutex_unlock(&mutex) != 0) {
      perror("pthread_mutex_unlock");
      abort();
    }
  }
  return NULL;
}

// Increment profiling context counter value
uint32_t next_context_count() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  ++context_count;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }
  return context_count;
}

// Allocate entry to store profiling context
context_entry_t* alloc_context_entry() {
  if (CTX_OUTSTANDING_MAX != 0) {
    while((context_count - context_collected) > CTX_OUTSTANDING_MAX) usleep(1000);
  }

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  const uint32_t index = next_context_count() - 1;
  auto ret = context_array->insert({index, context_entry_t{}});
  if (ret.second == false) {
    fprintf(stderr, "context_array corruption, index repeated %u\n", index);
    abort();
  }

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  context_entry_t* entry = &(ret.first->second);
  entry->index = index;
  return entry;
}

// Allocate entry to store profiling context
void dealloc_context_entry(context_entry_t* entry) {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  assert(context_array != NULL);
  context_array->erase(entry->index);

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }
}

// Dump trace data to file
void dump_sqtt_trace(const char* label, const uint32_t chunk, const void* data, const uint32_t& size) {
  if (result_prefix != NULL) {
    // Open SQTT file
    std::ostringstream oss;
    oss << result_prefix << "/thread_trace_" << label << "_se" << chunk << ".out";
    FILE* file = fopen(oss.str().c_str(), "w");
    if (file == NULL) {
      std::ostringstream errmsg;
      errmsg << "fopen error, file '" << oss.str().c_str() << "'";
      perror(errmsg.str().c_str());
      abort();
    }

    // Write the buffer in terms of shorts (16 bits)
    const unsigned short* ptr = reinterpret_cast<const unsigned short*>(data);
    for (uint32_t i = 0; i < (size / sizeof(short)); ++i) {
      fprintf(file, "%04x\n", ptr[i]);
    }

    // Close SQTT file
    fclose(file);
  }
}

struct trace_data_arg_t {
  FILE* file;
  const char* label;
  hsa_agent_t agent;
};

// Trace data callback for getting trace data from GPU local mamory
hsa_status_t trace_data_cb(hsa_ven_amd_aqlprofile_info_type_t info_type,
                           hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  trace_data_arg_t* arg = reinterpret_cast<trace_data_arg_t*>(data);
  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_SQTT_DATA) {
    const void* data_ptr = info_data->sqtt_data.ptr;
    const uint32_t data_size = info_data->sqtt_data.size;
    fprintf(arg->file, "    SE(%u) size(%u)\n", info_data->sample_id, data_size);

    if (is_sqtt_local) {
      HsaRsrcFactory* hsa_rsrc = &HsaRsrcFactory::Instance();
      const AgentInfo* agent_info = hsa_rsrc->GetAgentInfo(arg->agent);
      const uint32_t mem_size = data_size;
      void* buffer = hsa_rsrc->AllocateSysMemory(agent_info, mem_size);
      if(!hsa_rsrc->Memcpy(agent_info, buffer, data_ptr, mem_size)) {
        fatal("SQTT data memcopy to host failed");
      }
      dump_sqtt_trace(arg->label, info_data->sample_id, buffer, data_size);
      HsaRsrcFactory::FreeMemory(buffer);
    } else {
      dump_sqtt_trace(arg->label, info_data->sample_id, data_ptr, data_size);
    }
  } else
    status = HSA_STATUS_ERROR;
  return status;
}

// Align to specified alignment
unsigned align_size(unsigned size, unsigned alignment) {
  return ((size + alignment - 1) & ~(alignment - 1));
}

// Output profiling results for input features
void output_results(const context_entry_t* entry, const char* label) {
  FILE* file = entry->file_handle;
  const rocprofiler_feature_t* features = entry->features;
  const unsigned feature_count = entry->feature_count;
  rocprofiler_t* context = entry->group.context;

  for (unsigned i = 0; i < feature_count; ++i) {
    const rocprofiler_feature_t* p = &features[i];
    fprintf(file, "  %s ", p->name);
    switch (p->data.kind) {
      // Output metrics results
      case ROCPROFILER_DATA_KIND_INT64:
        fprintf(file, "(%lu)\n", p->data.result_int64);
        break;
      // Output trace results
      case ROCPROFILER_DATA_KIND_BYTES: {
        if (p->data.result_bytes.copy) {
          uint64_t size = 0;

          const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
          const char* end = reinterpret_cast<const char*>(ptr + p->data.result_bytes.size);
          for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
            const uint32_t chunk_size = *reinterpret_cast<const uint32_t*>(ptr);
            const char* chunk_data = ptr + sizeof(uint32_t);
            if (chunk_data >= end) fatal("SQTT data is out of the result buffer size");

            dump_sqtt_trace(label, i, chunk_data, chunk_size);
            const uint32_t off = align_size(chunk_size, sizeof(uint32_t));
            ptr = chunk_data + off;
            if (chunk_data >= end) fatal("SQTT data ptr is out of the result buffer size");
            size += chunk_size;
          }
          fprintf(file, "size(%lu)\n", size);
          HsaRsrcFactory::FreeMemory(p->data.result_bytes.ptr);
          const_cast<rocprofiler_feature_t*>(p)->data.result_bytes.size = 0;
        } else {
          fprintf(file, "(\n");
          trace_data_arg_t trace_data_arg{file, label, entry->agent};
          hsa_status_t status = rocprofiler_iterate_trace_data(context, trace_data_cb, reinterpret_cast<void*>(&trace_data_arg));
          check_status(status);
          fprintf(file, "  )\n");
        }
        break;
      }
      default:
        fprintf(stderr, "RPL-tool: undefined data kind(%u)\n", p->data.kind);
        abort();
    }
  }
}

// Output group intermeadate profiling results, created internally for complex metrics
void output_group(const context_entry_t* entry, const char* label) {
  const rocprofiler_group_t* group = &(entry->group);
  context_entry_t group_entry = *entry;
  for (unsigned i = 0; i < group->feature_count; ++i) {
    if (group->features[i]->data.kind == ROCPROFILER_DATA_KIND_INT64) {
      group_entry.features = group->features[i];
      group_entry.feature_count = 1;
      output_results(&group_entry, label);
    }
  }
}

// Dump stored context entry
bool dump_context_entry(context_entry_t* entry) {
  hsa_status_t status = HSA_STATUS_ERROR;

  volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
  while (valid->load() == false) sched_yield();

  const rocprofiler_dispatch_record_t* record = entry->data.record;
  if (record) {
    if (record->complete == 0) {
      return false;
    }
  }

  ++context_collected;

  const uint32_t index = entry->index;
  FILE* file_handle = entry->file_handle;
  const std::string nik_name = (to_truncate_names == 0) ? entry->data.kernel_name : filtr_kernel_name(entry->data.kernel_name);

  fprintf(file_handle, "dispatch[%u], gpu-id(%u), queue-id(%u), queue-index(%lu), tid(%lu), grd(%u), wgr(%u), lds(%u), scr(%u), vgpr(%u), sgpr(%u), fbar(%u), kernel-name(\"%s\")",
    index,
    HsaRsrcFactory::Instance().GetAgentInfo(entry->agent)->dev_index,
    entry->data.queue_id,
    entry->data.queue_index,
    entry->data.thread_id,
    entry->kernel_properties.grid_size,
    entry->kernel_properties.workgroup_size,
    entry->kernel_properties.lds_size,
    entry->kernel_properties.scratch_size,
    entry->kernel_properties.vgpr_count,
    entry->kernel_properties.sgpr_count,
    entry->kernel_properties.fbarrier_count,
    nik_name.c_str());
  if (record) fprintf(file_handle, ", time(%lu,%lu,%lu,%lu)",
    record->dispatch,
    record->begin,
    record->end,
    record->complete);
  fprintf(file_handle, "\n");
  fflush(file_handle);

  if (record) {
    delete record;
    entry->data.record = NULL;
  }

  rocprofiler_group_t& group = entry->group;
  if (group.context != NULL) {
    if (entry->feature_count > 0) {
      status = rocprofiler_group_get_data(&group);
      check_status(status);
      if (verbose == 1) output_group(entry, "group0-data");

      status = rocprofiler_get_metrics(group.context);
      check_status(status);
    }
    std::ostringstream oss;
    oss << index << "__" << filtr_kernel_name(entry->data.kernel_name);
    output_results(entry, oss.str().substr(0, KERNEL_NAME_LEN_MAX).c_str());
    free(const_cast<char*>(entry->data.kernel_name));

    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    rocprofiler_close(group.context);
  }

  return true;
}

// Wait for and dump all stored contexts for a given queue if not NULL
void dump_context_array(hsa_queue_t* queue) {
  bool done = false;
  while (done == false) {
    done = true;
    if (pthread_mutex_lock(&mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
    }

    if (context_array) {
      auto it = context_array->begin();
      auto end = context_array->end();
      while (it != end) {
        auto cur = it++;
        context_entry_t* entry = &(cur->second);
        volatile std::atomic<bool>* valid = reinterpret_cast<std::atomic<bool>*>(&entry->valid);
        while (valid->load() == false) sched_yield();
        if ((queue == NULL) || (entry->data.queue == queue)) {
          if (entry->active == true) {
            if (dump_context_entry(&(cur->second)) == false) done = false;
            else entry->active = false;
          }
        }
      }

      if (pthread_mutex_unlock(&mutex) != 0) {
        perror("pthread_mutex_unlock");
        abort();
      }
      if (done == false) sched_yield();
    }
  }
}

// Profiling completion handler
// Dump and delete the context entry
// Return true if the context was dumped successfully
bool context_handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  bool ret = true;
  if (entry->active == true) {
    ret = dump_context_entry(entry);
    if (ret == false) {
      fprintf(stderr, "tool error: context is not complete\n");
      abort();
    }
  }
  if (ret) dealloc_context_entry(entry);

  if (trace_on) {
    fprintf(stdout, "tool::handler: context_array %d tid %u\n", (int)(context_array->size()), GetTid());
    fflush(stdout);
  }

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  return false;
}

bool check_filter(const rocprofiler_callback_data_t* callback_data, const callbacks_data_t* tool_data) {
  bool found = true;

  std::vector<uint32_t>* range_ptr = tool_data->range;
  if (found && range_ptr) {
    found = false;
    std::vector<uint32_t>& range = *range_ptr;
    if (range.size() == 1) {
      if (context_count >= range[0]) found = true;
    } else if (range.size() == 2) {
      if ((context_count >= range[0]) && (context_count < range[1])) found = true;
    }
  }
  std::vector<uint32_t>* gpu_index = tool_data->gpu_index;
  if (found && gpu_index) {
    found = false;
    for (uint32_t i : *gpu_index) {
      if (i == callback_data->agent_index) {
        found = true;
      }
    }
  }
  std::vector<std::string>* kernel_string  = tool_data->kernel_string;
  if (found && kernel_string) {
    found = false;
    for (const std::string& s : *kernel_string) {
      if (std::string(callback_data->kernel_name).find(s) != std::string::npos) {
        found = true;
      }
    }
  }

  return found;
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* user_data,
                               rocprofiler_group_t* group) {
  // Passed tool data
  const hsa_kernel_dispatch_packet_t* packet = callback_data->packet;
  const amd_kernel_code_t* kernel_code = callback_data->kernel_code;
  callbacks_data_t* tool_data = reinterpret_cast<callbacks_data_t*>(user_data);
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;

  // Checking dispatch condition
  if (tool_data->filter_on == 1) {
    if (check_filter(callback_data, tool_data) == false) {
      next_context_count();
      return HSA_STATUS_SUCCESS;
    }
  }
  // Profiling context
  rocprofiler_t* context = NULL;
  // Context entry
  context_entry_t* entry = alloc_context_entry();
  // kernel properties
  kernel_properties_t* kernel_properties_ptr = &(entry->kernel_properties);
  uint64_t grid_size = packet->grid_size_x * packet->grid_size_y * packet->grid_size_z;
  if (grid_size > UINT32_MAX) abort();
  kernel_properties_ptr->grid_size = (uint32_t)grid_size;
  uint64_t workgroup_size = packet->workgroup_size_x * packet->workgroup_size_y * packet->workgroup_size_z;
  if (workgroup_size > UINT32_MAX) abort();
  kernel_properties_ptr->workgroup_size = (uint32_t)workgroup_size;
  kernel_properties_ptr->lds_size = packet->group_segment_size;
  kernel_properties_ptr->scratch_size = packet->private_segment_size;
  kernel_properties_ptr->vgpr_count = kernel_code->reserved_vgpr_count;
  kernel_properties_ptr->sgpr_count = kernel_code->reserved_sgpr_count;
  kernel_properties_ptr->fbarrier_count = kernel_code->workgroup_fbarrier_count;

  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = (result_prefix != NULL) ? context_handler : NULL;
  properties.handler_arg = (void*)entry;

  rocprofiler_feature_t* features = tool_data->features;
  unsigned feature_count = tool_data->feature_count;

  if (tool_data->set != NULL) {
    uint32_t set_offset = 0;
    uint32_t next_offset = 0;
    const auto entry_index = entry->index;
    if (entry_index < (tool_data->set->size() - 1)) {
      set_offset = (*(tool_data->set))[entry_index];
      next_offset = (*(tool_data->set))[entry_index + 1];
    } else {
      set_offset = tool_data->set->back();
      next_offset = feature_count;
    }
    features += set_offset;
    feature_count = next_offset - set_offset;
  }

  // Open profiling context
  status = rocprofiler_open(callback_data->agent, features, feature_count,
                            &context, 0 /*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  // Check that we have only one profiling group
  uint32_t group_count = 0;
  status = rocprofiler_group_count(context, &group_count);
  check_status(status);
  assert(group_count == 1);
  // Get group[0]
  const uint32_t group_index = 0;
  status = rocprofiler_get_group(context, group_index, group);
  check_status(status);

  // Fill profiling context entry
  entry->agent = callback_data->agent;
  entry->group = *group;
  entry->features = features;
  entry->feature_count = feature_count;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  entry->file_handle = tool_data->file_handle;
  entry->active = true;
  reinterpret_cast<std::atomic<bool>*>(&entry->valid)->store(true);

  if (trace_on) {
    fprintf(stdout, "tool::dispatch: context_array %d tid %u\n", (int)(context_array->size()), GetTid());
    fflush(stdout);
  }

  return status;
}

hsa_status_t destroy_callback(hsa_queue_t* queue, void*) {
  results_output_break();
  dump_context_array(queue);
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t info_callback(const rocprofiler_info_data_t info, void * arg) {
  const char symb = *reinterpret_cast<const char*>(arg);
  if (((symb == 'b') && (info.metric.expr == NULL)) ||
      ((symb == 'd') && (info.metric.expr != NULL)))
  {
    if (info.metric.expr != NULL) {
      fprintf(stdout, "\n  gpu-agent%d : %s : %s\n", info.agent_index, info.metric.name, info.metric.description);
      fprintf(stdout, "      %s = %s\n", info.metric.name, info.metric.expr);
    } else {
      fprintf(stdout, "\n  gpu-agent%d : %s", info.agent_index, info.metric.name);
      if (info.metric.instances > 1) fprintf(stdout, "[0-%u]", info.metric.instances - 1);
      fprintf(stdout, " : %s\n", info.metric.description);
      fprintf(stdout, "      block %s has %u counters\n", info.metric.block_name, info.metric.block_counters);
    }
    fflush(stdout);
  }
  return HSA_STATUS_SUCCESS;
}

std::string normalize_token(const std::string token, bool not_empty, std::string label) {
  const std::string space_chars_set = " \t";
  const size_t first_pos = token.find_first_not_of(space_chars_set);
  size_t norm_len = 0;
  std::string error_str = "none";
  if (first_pos != std::string::npos) {
    const size_t last_pos = token.find_last_not_of(space_chars_set);
    if (last_pos == std::string::npos) error_str = "token string error: \"" + token + "\"";
    else {
      const size_t end_pos = last_pos + 1;
      if (end_pos <= first_pos) error_str = "token string error: \"" + token + "\"";
      else norm_len = end_pos - first_pos;
    }
  }
  if (((first_pos != std::string::npos) && (norm_len == 0)) ||
      ((first_pos == std::string::npos) && not_empty)) {
    fatal(label + ": " + error_str);
  }
  return (norm_len != 0) ? token.substr(first_pos, norm_len) : std::string("");
}

int get_xml_array(xml::Xml* xml, const std::string& tag, const std::string& field, const std::string& delim, std::vector<std::string>* vec, const char* label = NULL) {
  int parse_iter = 0;
  auto nodes = xml->GetNodes(tag);
  auto rit = nodes.rbegin();
  auto rend = nodes.rend();
  while (rit != rend) {
    auto& opts = (*rit)->opts;
    if (opts.find(field) != opts.end()) break;
    ++rit;
  }
  if (rit != rend) {
    const std::string array_string = (*rit)->opts[field];
    if (label != NULL) printf("%s%s = %s\n", label, field.c_str(), array_string.c_str());
    size_t pos1 = 0;
    const size_t string_len = array_string.length();
    while (pos1 < string_len) {
      const size_t pos2 = array_string.find(delim, pos1);
      const bool found = (pos2 != std::string::npos);
      const size_t token_len = (pos2 != std::string::npos) ? pos2 - pos1 : string_len - pos1;
      const std::string token = array_string.substr(pos1, token_len);
      const std::string norm_str = normalize_token(token, found, "Tokens array parsing error, file '" + xml->GetName() + "', " + tag + "::" + field);
      if (norm_str.length() != 0) vec->push_back(norm_str);
      if (!found) break;
      pos1 = pos2 + 1;
      ++parse_iter;
    }
  }

  return parse_iter;
}

int get_xml_array(xml::Xml* xml, const std::string& tag, const std::string& field, const std::string& delim, std::vector<uint32_t>* vec, const char* label = NULL) {
  std::vector<std::string> str_vec;
  const int parse_iter = get_xml_array(xml, tag, field, delim, &str_vec, label);
  for (const std::string& str : str_vec) vec->push_back(atoi(str.c_str()));
  return parse_iter;
}

static inline void check_env_var(const char* var_name, uint32_t& val) {
  const char* str = getenv(var_name);
  if (str != NULL ) val = atol(str);
}
static inline void check_env_var(const char* var_name, uint64_t& val) {
  const char* str = getenv(var_name);
  if (str != NULL ) val = atoll(str);
}

// Tool constructor
extern "C" PUBLIC_API void OnLoadToolProp(rocprofiler_settings_t* settings)
{
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (is_loaded) return;
  is_loaded = true;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  // Loading configuration rcfile
  std::string rcpath = std::string("./") + rcfile_name;
  xml::Xml* rcfile = xml::Xml::Create(rcpath);
  const char* home_dir = getenv("HOME");
  if (rcfile == NULL && home_dir != NULL) {
    rcpath = std::string(home_dir) + "/" + rcfile_name;
    rcfile = xml::Xml::Create(rcpath);
  }
  const char* pkg_dir = getenv("ROCP_PACKAGE_DIR");
  if (rcfile == NULL && pkg_dir != NULL) {
    rcpath = std::string(pkg_dir) + "/" + rcfile_name;
    rcfile = xml::Xml::Create(rcpath);
  }
  if (rcfile != NULL) {
    // Getting defaults
    printf("ROCProfiler: rc-file '%s'\n", rcpath.c_str());
    auto defaults_list = rcfile->GetNodes("top.defaults");
    for (auto* entry : defaults_list) {
      const auto& opts = entry->opts;
      auto it = opts.find("basenames");
      if (it != opts.end()) { to_truncate_names = (it->second == "on") ? 1 : 0; }
      it = opts.find("timestamp");
      if (it != opts.end()) { settings->timestamp_on = (it->second == "on") ? 1 : 0; }
      it = opts.find("ctx-limit");
      if (it != opts.end()) { CTX_OUTSTANDING_MAX = atol(it->second.c_str()); }
      it = opts.find("heartbeat");
      if (it != opts.end()) { CTX_OUTSTANDING_MON = atol(it->second.c_str()); }
      it = opts.find("sqtt-size");
      if (it != opts.end()) {
        std::string str = normalize_token(it->second, true, "option sqtt-size");
        uint32_t multiplier = 1;
        switch (str.back()) {
          case 'K': multiplier = 1024; break;
          case 'M': multiplier = 1024 * 1024; break;
        }
        if (multiplier != 1) str = str.substr(0, str.length() - 1);
        settings->sqtt_size = strtoull(str.c_str(), NULL, 0) * multiplier;
      }
      it = opts.find("sqtt-local");
      if (it != opts.end()) { settings->sqtt_local = (it->second == "on"); }
      it = opts.find("memcopies");
      if (it != opts.end()) { settings->memcopy_tracking = (it->second == "on"); }
    }
  }
  // Enable verbose mode
  check_env_var("ROCP_VERBOSE_MODE", verbose);
  // Enable kernel names truncating
  check_env_var("ROCP_TRUNCATE_NAMES", to_truncate_names);
  // Set outstanding dispatches parameter
  check_env_var("ROCP_OUTSTANDING_MAX", CTX_OUTSTANDING_MAX);
  check_env_var("ROCP_OUTSTANDING_MON", CTX_OUTSTANDING_MON);
  // Enable timestamping
  check_env_var("ROCP_TIMESTAMP_ON", settings->timestamp_on);
  // Set data timeout
  check_env_var("ROCP_DATA_TIMEOUT", settings->timeout);
  // Set SQTT size
  check_env_var("ROCP_SQTT_SIZE", settings->sqtt_size);
  // Set SQTT local buffer
  check_env_var("ROCP_SQTT_LOCAL", settings->sqtt_local);
  // Set memcopies tracking
  check_env_var("ROCP_MCOPY_TRACKING", settings->memcopy_tracking);

  is_sqtt_local = settings->sqtt_local;

  // Printing out info
  char* info_symb = getenv("ROCP_INFO");
  if (info_symb != NULL) {
    if (*info_symb != 'b' && *info_symb != 'd') {
      fprintf(stderr, "ROCProfiler: bad info symbol '%c', ROCP_INFO env", *info_symb);
    } else {
      if (*info_symb == 'b') printf("Basic HW counters:\n");
      else printf("Derived metrics:\n");
      hsa_status_t status = rocprofiler_iterate_info(NULL, ROCPROFILER_INFO_KIND_METRIC, info_callback, info_symb);
      check_status(status);
    }
    exit(1);
  }

  // Set output file
  result_prefix = getenv("ROCP_OUTPUT_DIR");
  if (result_prefix != NULL) {
    DIR* dir = opendir(result_prefix);
    if (dir == NULL) {
      std::ostringstream errmsg;
      errmsg << "ROCProfiler: Cannot open output directory '" << result_prefix << "'";
      perror(errmsg.str().c_str());
      abort();
    }
    std::ostringstream oss;
    oss << result_prefix << "/results.txt";
    result_file_handle = fopen(oss.str().c_str(), "w");
    if (result_file_handle == NULL) {
      std::ostringstream errmsg;
      errmsg << "ROCProfiler: fopen error, file '" << oss.str().c_str() << "'";
      perror(errmsg.str().c_str());
      abort();
    }
  } else result_file_handle = stdout;

  result_file_opened = (result_prefix != NULL) && (result_file_handle != NULL);

  // Getting input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name == NULL) fatal("ROCProfiler: input is not specified, ROCP_INPUT env");
  printf("ROCProfiler: input from \"%s\"\n", xml_name);
  xml::Xml* xml = xml::Xml::Create(xml_name);
  if (xml == NULL) {
    fprintf(stderr, "ROCProfiler: Input file not found '%s'\n", xml_name);
    abort();
  }

  // Getting metrics
  std::vector<std::string> metrics_vec;
  get_xml_array(xml, "top.metric", "name", ",", &metrics_vec);

  // Metrics set
  metrics_set = new std::vector<uint32_t>;
  get_xml_array(xml, "top.metric", "set", ",", metrics_set, "  ");
  if (metrics_set->size() != 0) {
    uint32_t accum = 0;
    metrics_set->insert(metrics_set->begin(), 0);
    for (auto it = metrics_set->begin(); it != metrics_set->end(); ++it) {
      accum += *it;
      *it = accum;
    }
  }

  // Getting GPU indexes
  gpu_index_vec = new std::vector<uint32_t>;
  get_xml_array(xml, "top.metric", "gpu_index", ",", gpu_index_vec, "  ");

  // Getting kernel names
  kernel_string_vec = new std::vector<std::string>;
  get_xml_array(xml, "top.metric", "kernel", ",", kernel_string_vec, "  ");

  // Getting profiling range
  range_vec = new std::vector<uint32_t>;
  const int range_parse_iter = get_xml_array(xml, "top.metric", "range", ":", range_vec, "  ");
  if ((range_vec->size() > 2) || (range_parse_iter > 1))
  {
    fatal("Bad range format, input file " + xml->GetName());
  }
  if ((range_vec->size() == 1) && (range_parse_iter == 0)) {
    range_vec->push_back(*(range_vec->begin()) + 1);
  }

  // Getting traces
  auto traces_list = xml->GetNodes("top.trace");

  const unsigned feature_count = metrics_vec.size() + traces_list.size();
  rocprofiler_feature_t* features = new rocprofiler_feature_t[feature_count];
  memset(features, 0, feature_count * sizeof(rocprofiler_feature_t));

  printf("  %d metrics\n", (int)metrics_vec.size());
  for (unsigned i = 0; i < metrics_vec.size(); ++i) {
    const std::string& name = metrics_vec[i];
    printf("%s%s", (i == 0) ? "    " : ", ", name.c_str());
    features[i] = {};
    features[i].kind = ROCPROFILER_FEATURE_KIND_METRIC;
    features[i].name = strdup(name.c_str());
  }
  if (metrics_vec.size()) printf("\n");

  printf("  %d traces\n", (int)traces_list.size());
  unsigned index = metrics_vec.size();
  for (auto* entry : traces_list) {
    auto params_list = xml->GetNodes("top.trace.parameters");
    if (params_list.size() > 1) {
      fatal("ROCProfiler: Single input 'parameters' section is supported");
    }
    std::string name = "";
    bool to_copy_data = false;
    for (const auto& opt : entry->opts) {
      if (opt.first == "name") name = opt.second;
      else if (opt.first == "copy") to_copy_data = (opt.second == "true");
      else fatal("ROCProfiler: Bad trace property '" + opt.first + "'");
    }
    if (name == "") fatal("ROCProfiler: Bad trace properties, name is not specified");

    std::map<std::string, hsa_ven_amd_aqlprofile_parameter_name_t> parameters_dict;
    parameters_dict["TARGET_CU"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
    parameters_dict["VM_ID_MASK"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK;
    parameters_dict["MASK"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
    parameters_dict["TOKEN_MASK"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
    parameters_dict["TOKEN_MASK2"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2;
#ifdef AQLPROF_NEW_API
    parameters_dict["SE_MASK"] =
        HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_SE_MASK;
#endif

    printf("    %s (", name.c_str());
    features[index] = {};
    features[index].kind = ROCPROFILER_FEATURE_KIND_TRACE;
    features[index].name = strdup(name.c_str());
    features[index].data.result_bytes.copy = to_copy_data;

    for (auto* params : params_list) {
      const unsigned parameter_count = params->opts.size();
      rocprofiler_parameter_t* parameters = new rocprofiler_parameter_t[parameter_count];
      unsigned p_index = 0;
      for (auto& v : params->opts) {
        const std::string parameter_name = v.first;
        if (parameters_dict.find(parameter_name) == parameters_dict.end()) {
          fprintf(stderr, "ROCProfiler: unknown trace parameter '%s'\n", parameter_name.c_str());
          abort();
        }
        const uint32_t value = strtol(v.second.c_str(), NULL, 0);
        printf("\n      %s = 0x%x", parameter_name.c_str(), value);
        parameters[p_index] = {};
        parameters[p_index].parameter_name = parameters_dict[parameter_name];
        parameters[p_index].value = value;
        ++p_index;
      }

      features[index].parameters = parameters;
      features[index].parameter_count = parameter_count;
    }
    if (params_list.empty() == false) printf("\n    ");
    printf(")\n");
    fflush(stdout);
    ++index;
  }
  fflush(stdout);

  // Context array aloocation
  context_array = new context_array_t;

  // Adding dispatch observer
  rocprofiler_queue_callbacks_t callbacks_ptrs{0};
  callbacks_ptrs.dispatch = dispatch_callback;
  callbacks_ptrs.destroy = destroy_callback;

  callbacks_data = new callbacks_data_t{};
  callbacks_data->features = features;
  callbacks_data->feature_count = feature_count;
  callbacks_data->set = (metrics_set->empty()) ? NULL : metrics_set;
  callbacks_data->group_index = 0;
  callbacks_data->file_handle = result_file_handle;
  callbacks_data->gpu_index = (gpu_index_vec->empty()) ? NULL : gpu_index_vec;
  callbacks_data->kernel_string = (kernel_string_vec->empty()) ? NULL : kernel_string_vec;
  callbacks_data->range = (range_vec->empty()) ? NULL : range_vec;;
  callbacks_data->filter_on = (callbacks_data->gpu_index != NULL) ||
                              (callbacks_data->kernel_string != NULL) ||
                              (callbacks_data->range != NULL)
                              ? 1 : 0;

  rocprofiler_set_queue_callbacks(callbacks_ptrs, callbacks_data);

  xml::Xml::Destroy(xml);

  if (CTX_OUTSTANDING_MON != 0) {
    pthread_t thread;
    pthread_attr_t attr;
    int err = pthread_attr_init(&attr);
    if (err) { errno = err; perror("pthread_attr_init"); abort(); }
    err = pthread_create(&thread, &attr, monitor_thr_fun, NULL);
  }
}

// Tool destructor
extern "C" PUBLIC_API void OnUnloadTool() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }
  if (!is_loaded) return;
  is_loaded = false;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }

  // Unregister dispatch callback
  rocprofiler_remove_queue_callbacks();

  // Dump stored profiling output data
  fflush(stdout);
  if (result_file_opened) {
    printf("\nROCPRofiler:"); fflush(stdout);
    dump_context_array(NULL);
    fclose(result_file_handle);
    printf(" %u contexts collected, output directory %s\n", context_collected, result_prefix);
  } else {
    if (context_collected != context_count) {
      results_output_break();
      dump_context_array(NULL);
    }
    printf("\nROCPRofiler: %u contexts collected\n", context_collected);
  }
  fflush(stdout);

  // Cleanup
  if (callbacks_data != NULL) {
    delete[] callbacks_data->features;
    delete callbacks_data;
    callbacks_data = NULL;
  }
  delete metrics_set;
  metrics_set = NULL;
  delete gpu_index_vec;
  gpu_index_vec = NULL;
  delete kernel_string_vec;
  kernel_string_vec = NULL;
  delete range_vec;
  range_vec = NULL;
  delete context_array;
  context_array = NULL;
}

extern "C" DESTRUCTOR_API void destructor() {
  if (is_loaded == true) OnUnloadTool();
}
