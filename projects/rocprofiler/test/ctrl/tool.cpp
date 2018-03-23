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
#include <sys/syscall.h>   /* For SYS_xxx definitions */
#include <sys/types.h>
#include <unistd.h>

#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "inc/rocprofiler.h"
#include "util/xml.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))
#define KERNEL_NAME_LEN_MAX 128

// Disoatch callback data type
struct callbacks_data_t {
  rocprofiler_feature_t* features;
  unsigned feature_count;
  unsigned group_index;
  FILE* file_handle;
  int filter_on;
  std::vector<uint32_t>* gpu_index;
  std::vector<std::string>* kernel_string;
  std::vector<uint32_t>* range;
};

// Context stored entry type
struct context_entry_t {
  uint32_t valid;
  uint32_t index;
  rocprofiler_group_t group;
  rocprofiler_feature_t* features;
  unsigned feature_count;
  rocprofiler_callback_data_t data;
  FILE* file_handle;
};

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
typedef std::list<context_entry_t*> wait_list_t;
wait_list_t* wait_list = NULL;
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
// GPU index filter
std::vector<uint32_t>* gpu_index_vec = NULL;
//  Kernel name filter
std::vector<std::string>* kernel_string_vec = NULL;
//  DIspatch number range filter
std::vector<uint32_t>* range_vec = NULL;
// Otstanding dispatches parameters
#if 0
static uint32_t CTX_OUTSTANDING_MAX_DFLT = 10000;
static uint32_t CTX_OUTSTANDING_WAIT_DFLT = 1000;
static uint32_t CTX_OUTSTANDING_WAIT_MAX = 1000000;
static uint32_t CTX_OUTSTANDING_MAX = CTX_OUTSTANDING_MAX_DFLT;
static uint32_t CTX_OUTSTANDING_WAIT = CTX_OUTSTANDING_WAIT_DFLT;
#endif
static uint32_t CTX_OUTSTANDING_MAX = 0;
static uint32_t CTX_OUTSTANDING_MON = 0;

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// Error handler
void fatal(const char* msg) {
  fprintf(stderr, "%s\n", msg);
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

void* monitor_thr_fun(void*) {
  while (context_array != NULL) {
    sleep(CTX_OUTSTANDING_MON);
    if (pthread_mutex_lock(&mutex) != 0) {
      perror("pthread_mutex_lock");
      abort();
    }
    const uint32_t inflight = context_count - context_collected;
    std::cout << "ROCProfiler: count(" << context_count << "), outstanding(" << inflight << "/" << CTX_OUTSTANDING_MAX << ")" << std::endl << std::flush;
    if (pthread_mutex_unlock(&mutex) != 0) {
      perror("pthread_mutex_unlock");
      abort();
    }
  }
  return NULL;
}

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
#if 0
  uint32_t context_inflight = context_count - context_collected;
  if (context_inflight > CTX_OUTSTANDING_MAX) {
    if (trace_on) std::cout << "inflight " << context_inflight << " tid " << GetTid() << " <" << CTX_OUTSTANDING_WAIT << "usec, max " << CTX_OUTSTANDING_MAX << ">" <<  std::flush;
    usleep(CTX_OUTSTANDING_WAIT);
    if (CTX_OUTSTANDING_WAIT > CTX_OUTSTANDING_WAIT_MAX) {
      CTX_OUTSTANDING_MAX = 1 + (CTX_OUTSTANDING_MAX >> 1);
    } else {
      CTX_OUTSTANDING_WAIT = CTX_OUTSTANDING_WAIT << 1;
    }
  } else {
    CTX_OUTSTANDING_MAX = CTX_OUTSTANDING_MAX_DFLT;
    CTX_OUTSTANDING_WAIT = CTX_OUTSTANDING_WAIT_DFLT;
  }
#endif
  if (CTX_OUTSTANDING_MAX != 0) {
    while((context_count - context_collected) > CTX_OUTSTANDING_MAX) usleep(1000);
  }

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  const uint32_t index = next_context_count();
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
};

// Trace data callback for getting trace data from GPU local mamory
hsa_status_t trace_data_cb(hsa_ven_amd_aqlprofile_info_type_t info_type,
                           hsa_ven_amd_aqlprofile_info_data_t* info_data, void* data) {
  hsa_status_t status = HSA_STATUS_SUCCESS;
  trace_data_arg_t* arg = reinterpret_cast<trace_data_arg_t*>(data);
  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_SQTT_DATA) {
    fprintf(arg->file, "    SE(%u) size(%u)\n", info_data->sample_id, info_data->sqtt_data.size);
    dump_sqtt_trace(arg->label, info_data->sample_id, info_data->sqtt_data.ptr, info_data->sqtt_data.size);

  } else
    status = HSA_STATUS_ERROR;
  return status;
}

// Align to specified alignment
unsigned align_size(unsigned size, unsigned alignment) {
  return ((size + alignment - 1) & ~(alignment - 1));
}

// Output profiling results for input features
void output_results(FILE* file, const rocprofiler_feature_t* features, const unsigned feature_count,
                    rocprofiler_t* context, const char* label) {
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
          for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
            const uint32_t chunk_size = *reinterpret_cast<const uint64_t*>(ptr);
            const char* chunk_data = ptr + sizeof(uint64_t);
            dump_sqtt_trace(label, i, chunk_data, chunk_size);

            const uint32_t off = align_size(chunk_size, sizeof(uint64_t));
            ptr = chunk_data + off;
            size += chunk_size;
          }
          fprintf(file, "size(%lu)\n", size);
          if (size > p->data.result_bytes.size) fatal("SQTT data size is out of the result buffer size");
          free(p->data.result_bytes.ptr);
          const_cast<rocprofiler_feature_t*>(p)->data.result_bytes.size = 0;
        } else {
          fprintf(file, "(\n");
          trace_data_arg_t trace_data_arg{file, label};
          rocprofiler_iterate_trace_data(context, trace_data_cb, reinterpret_cast<void*>(&trace_data_arg));
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
void output_group(FILE* file, const rocprofiler_group_t* group, const char* str) {
  for (unsigned i = 0; i < group->feature_count; ++i) {
    output_results(file, group->features[i], 1, group->context, str);
  }
}

// Dump stored context profiling output data
bool dump_context(context_entry_t* entry) {
  hsa_status_t status = HSA_STATUS_ERROR;

  if (entry->valid == 0) return true;

  const rocprofiler_dispatch_record_t* record = entry->data.record;
  if (record) {
    if (record->complete == 0) {
      return false;
    }
  }

  ++context_collected;
  const uint32_t index = entry->index;
  FILE* file_handle = entry->file_handle;
  const rocprofiler_feature_t* features = entry->features;
  const unsigned feature_count = entry->feature_count;

  fprintf(file_handle, "dispatch[%u], queue_index(%lu), kernel_name(\"%s\")",
    index,
    entry->data.queue_index,
    entry->data.kernel_name);
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
    status = rocprofiler_group_get_data(&group);
    check_status(status);
    // output_group(file, group, "Group[0] data");

    status = rocprofiler_get_metrics(group.context);
    check_status(status);
    std::ostringstream oss;
    oss << index << "__" << entry->data.kernel_name;
    output_results(file_handle, features, feature_count, group.context, oss.str().substr(0, KERNEL_NAME_LEN_MAX).c_str());
    free(const_cast<char*>(entry->data.kernel_name));

    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    rocprofiler_close(group.context);
  }

  entry->valid = 0;
  return true;
}

// Dump and clean a given context entry
static inline bool dump_context_entry(context_entry_t* entry) {
  const bool ret = dump_context(entry);
  if (ret) dealloc_context_entry(entry);
  return ret;
}

// Dump waiting entries
static inline void dump_wait_list() {
  auto it = wait_list->begin();
  auto end = wait_list->end();
  while (it != end) {
    auto cur = it++;
    if (dump_context_entry(*cur)) {
      wait_list->erase(cur);
    }
  }
}

// Dump all stored contexts profiling output data
void dump_context_array() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  if (context_array) {
    if (!wait_list->empty()) dump_wait_list();

    auto it = context_array->begin();
    auto end = context_array->end();
    while (it != end) {
      auto cur = it++;
      dump_context(&(cur->second));
    }
  }

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    abort();
  }
}

// Profiling completion handler
bool handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    abort();
  }

  if (!wait_list->empty()) dump_wait_list();

  if (!dump_context_entry(entry)) {
    wait_list->push_back(entry);
  }

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
  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = (result_prefix != NULL) ? handler : NULL;
  properties.handler_arg = (void*)entry;

  if (tool_data->feature_count > 0) {
    // Open profiling context
    status = rocprofiler_open(callback_data->agent, tool_data->features, tool_data->feature_count,
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
  }

  // Fill profiling context entry
  entry->group = *group;
  entry->features = tool_data->features;
  entry->feature_count = tool_data->feature_count;
  entry->data = *callback_data;
  entry->data.kernel_name = strdup(callback_data->kernel_name);
  entry->file_handle = tool_data->file_handle;
  entry->valid = 1;

  if (trace_on) {
    fprintf(stdout, "tool::dispatch: context_array %d tid %u\n", (int)(context_array->size()), GetTid());
    fflush(stdout);
  }

  return status;
}

hsa_status_t destroy_callback(hsa_queue_t* queue, void*) {
  if (result_file_opened == false) printf("\nROCProfiler results:\n");
  dump_context_array();
  return HSA_STATUS_SUCCESS;
}

static hsa_status_t info_callback(const rocprofiler_info_data_t info, void * arg) {
  const char symb = *reinterpret_cast<const char*>(arg);
  if (((symb == 'b') && (info.metric.expr == NULL)) ||
      ((symb == 'd') && (info.metric.expr != NULL)))
  {
    printf("\n  gpu-agent%d : %s : %s\n", info.agent_index, info.metric.name, info.metric.description);
    if (info.metric.expr != NULL) printf("      %s = %s\n", info.metric.name, info.metric.expr);
  }
  return HSA_STATUS_SUCCESS;
}

void get_xml_array(xml::Xml* xml, const std::string& tag, const std::string& field, const std::string& delim, std::vector<std::string>* vec, const char* label = NULL) {
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
    while (pos1 < array_string.length()) {
      const size_t pos2 = array_string.find(delim, pos1);
      const std::string token = array_string.substr(pos1, pos2 - pos1);
      vec->push_back(token);
      if (pos2 == std::string::npos) break;
      pos1 = pos2 + 1;
    }
  }
}

void get_xml_array(xml::Xml* xml, const std::string& tag, const std::string& field, const std::string& delim, std::vector<uint32_t>* vec, const char* label = NULL) {
  std::vector<std::string> str_vec;
  get_xml_array(xml, tag, field, delim, &str_vec, label);
  for (const std::string& str : str_vec) vec->push_back(atoi(str.c_str()));
}

static inline void check_env_var(const char* var_name, uint32_t& val) {
  const char* str = getenv(var_name);
  if (str != NULL ) val = atol(str);
}

// Tool constructor
extern "C" PUBLIC_API void OnLoadTool()
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

  std::map<std::string, hsa_ven_amd_aqlprofile_parameter_name_t> parameters_dict;
  parameters_dict["COMPUTE_UNIT_TARGET"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
  parameters_dict["VM_ID_MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK;
  parameters_dict["MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
  parameters_dict["TOKEN_MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
  parameters_dict["TOKEN_MASK2"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2;

  char* info_symb = getenv("ROCP_INFO");
  if (info_symb != NULL) {
    if (*info_symb != 'b' && *info_symb != 'd') {
      fprintf(stderr, "ROCProfiler: bad info symbol '%c', ROCP_INFO env", *info_symb);
    } else {
      if (*info_symb == 'b') printf("Basic HW counters:\n");
      else printf("Derived metrics:\n");
      rocprofiler_iterate_info(NULL, ROCPROFILER_INFO_KIND_METRIC, info_callback, info_symb);
    }
    abort();
  }
  // Set outstanding dispatches parameter
  check_env_var("ROCP_OUTSTANDING_MAX", CTX_OUTSTANDING_MAX);
  check_env_var("ROCP_OUTSTANDING_MON", CTX_OUTSTANDING_MON);
#if 0
  // Set outstanding dispatches parameter
  const char* dispatches_max = getenv("ROCP_OUTSTANDING_MAX");
  const char* dispatches_wait = getenv("ROCP_OUTSTANDING_WAIT");
  const char* dispatches_wait_max = getenv("ROCP_OUTSTANDING_WAIT_MAX");
  if (dispatches_max != NULL ) {
    if (dispatches_wait == NULL) fatal("ROCP_OUTSTANDING_WAIT should be defined together with ROCP_OUTSTANDING_MAX env var");
    if (dispatches_wait_max == NULL) fatal("ROCP_OUTSTANDING_WAIT_MAX should be defined together with ROCP_OUTSTANDING_MAX env var");
    CTX_OUTSTANDING_MAX_DFLT = atol(dispatches_max);
    CTX_OUTSTANDING_WAIT_DFLT = atol(dispatches_wait);
    CTX_OUTSTANDING_WAIT_MAX = atol(dispatches_wait_max);
    CTX_OUTSTANDING_MAX = CTX_OUTSTANDING_MAX_DFLT;
    CTX_OUTSTANDING_WAIT = CTX_OUTSTANDING_WAIT_DFLT;
  }
#endif

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
  } else
    result_file_handle = stdout;

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

  // Getting GPU indexes
  gpu_index_vec = new std::vector<uint32_t>;
  get_xml_array(xml, "top.metric", "gpu_index", ",", gpu_index_vec, "  ");
  // Getting kernel names
  kernel_string_vec = new std::vector<std::string>;
  get_xml_array(xml, "top.metric", "kernel", ",", kernel_string_vec, "  ");
  // Getting profiling range
  range_vec = new std::vector<uint32_t>;
  get_xml_array(xml, "top.metric", "range", ":", range_vec, "  ");

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
    if (params_list.size() != 1) {
      fprintf(stderr, "ROCProfiler: Single input 'parameters' section is supported\n");
      abort();
    }
    const std::string& name = entry->opts["name"];
    const bool to_copy_data = (entry->opts["copy"] == "true");
    printf("    %s (\n", name.c_str());
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
        printf("      %s = 0x%x\n", parameter_name.c_str(), value);
        parameters[p_index] = {};
        parameters[p_index].parameter_name = parameters_dict[parameter_name];
        parameters[p_index].value = value;
        ++p_index;
      }

      features[index].parameters = parameters;
      features[index].parameter_count = parameter_count;
    }
    printf("    )\n");
    ++index;
  }
  fflush(stdout);

  // Context array aloocation
  context_array = new context_array_t;
  wait_list = new wait_list_t;

  // Adding dispatch observer
  rocprofiler_queue_callbacks_t callbacks_ptrs{0};
  callbacks_ptrs.dispatch = dispatch_callback;
  callbacks_ptrs.destroy = destroy_callback;

  callbacks_data = new callbacks_data_t{};
  callbacks_data->features = features;
  callbacks_data->feature_count = feature_count;
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
  printf("\nROCPRofiler: Finishing:\n"); fflush(stdout);
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

  printf("ROCPRofiler: Waiting for outstanding dispatches ..."); fflush(stdout);
  while(context_count != context_collected); usleep(1000);
  printf(".done\n"); fflush(stdout);

  // Dump stored profiling output data
  printf("ROCPRofiler: %u contexts collected", context_collected);
  if (result_file_opened) printf(", output directory %s", result_prefix);
  printf("\n"); fflush(stdout);
  //dump_context_array();
  if (wait_list) {
    if (!wait_list->empty()) {
      printf("\nWaiting for pending kernels ..."); fflush(stdout);
      while (wait_list->size() != 0) {
        usleep(1000);
        dump_wait_list();
      }
      printf(".done\n"); fflush(stdout);
    }
  }
  if (result_file_opened) fclose(result_file_handle);

  // Cleanup
  if (callbacks_data != NULL) {
    delete[] callbacks_data->features;
    delete callbacks_data;
    callbacks_data = NULL;
  }
  delete gpu_index_vec;
  gpu_index_vec = NULL;
  delete kernel_string_vec;
  kernel_string_vec = NULL;
  delete range_vec;
  range_vec = NULL;
  delete context_array;
  context_array = NULL;
  delete wait_list;
  wait_list = NULL;
}

extern "C" DESTRUCTOR_API void destructor() {
  if (is_loaded == true) OnUnloadTool();
}
