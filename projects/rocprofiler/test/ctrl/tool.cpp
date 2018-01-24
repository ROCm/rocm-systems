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
#include <sys/types.h>

#include <iostream>
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
struct dispatch_data_t {
  rocprofiler_feature_t* features;
  unsigned feature_count;
  unsigned group_index;
  FILE* file_handle;
};

// Context stored entry type
struct context_entry_t {
  int valid;
  uint32_t index;
  rocprofiler_group_t group;
  rocprofiler_feature_t* features;
  unsigned feature_count;
  rocprofiler_callback_data_t data;
  FILE* file_handle;
};

// Dispatch callbacks and context handlers synchronization
pthread_mutex_t mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
// Stored contexts array
typedef std::map<uint32_t, context_entry_t> context_array_t;
context_array_t* context_array = NULL;
// Contexts collected count
uint32_t context_count = 0;
// Profiling results output file name
const char* result_prefix = NULL;
// Global results file handle
FILE* result_file_handle = NULL;

// Check returned HSA API status
void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    exit(1);
  }
}

// Allocate entry to store profiling context
context_entry_t* alloc_context_entry() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  if (context_array == NULL) context_array = new context_array_t;
  const uint32_t index = context_count;
  auto ret = context_array->insert({index, context_entry_t{}});
  if (ret.second == false) {
    fprintf(stderr, "context_array corruption, index repeated %u\n", index);
    abort();
  }
  ++context_count;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }

  context_entry_t* entry = &(ret.first->second);
  entry->index = index;
  return entry;
}

// Allocate entry to store profiling context
void dealloc_context_entry(context_entry_t* entry) {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  assert(context_array != NULL);
  context_array->erase(entry->index);

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
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
      exit(1);
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
          if (size > p->data.result_bytes.size) {
            fprintf(stderr, "SQTT data size is out of the result buffer size\n");
            exit(1);
          }
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
void dump_context(context_entry_t* entry) {
  hsa_status_t status = HSA_STATUS_ERROR;

  if (entry->valid) {
    entry->valid = 0;
    const uint32_t index = entry->index;
    FILE* file_handle = entry->file_handle;
    const rocprofiler_feature_t* features = entry->features;
    const unsigned feature_count = entry->feature_count;

    fprintf(file_handle,
            "dispatch[%u], queue_index(%lu), kernel_object(0x%lx), kernel_name(\"%s\"):\n", index,
            entry->data.queue_index, entry->data.kernel_object, entry->data.kernel_name);

    rocprofiler_group_t group = entry->group;
    status = rocprofiler_group_get_data(&group);
    check_status(status);
    // output_group(file, group, "Group[0] data");
  
    status = rocprofiler_get_metrics(group.context);
    check_status(status);
    std::ostringstream oss;
    oss << index << "__" << entry->data.kernel_name;
    output_results(file_handle, features, feature_count, group.context, oss.str().substr(0, KERNEL_NAME_LEN_MAX).c_str());
  
    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    rocprofiler_close(group.context);
  }
}

// Dump all stored contexts profiling output data
void dump_context_array() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  if (context_array) for (auto& v : *context_array) dump_context(&v.second);

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}

// Profiling completion handler
void handler(rocprofiler_group_t group, void* arg) {
  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  if (context_array->find(entry->index) != context_array->end()) {
    dump_context(entry);
    dealloc_context_entry(entry);
  }

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}

// Kernel disoatch callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* user_data,
                               rocprofiler_group_t* group) {
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;
  // Passed tool data
  dispatch_data_t* tool_data = reinterpret_cast<dispatch_data_t*>(user_data);
  // Profiling context
  rocprofiler_t* context = NULL;
  // Context entry
  context_entry_t* entry = alloc_context_entry();
  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = (result_prefix != NULL) ? handler : NULL;
  properties.handler_arg = (void*)entry;

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

  // Fill profiling context entry
  entry->group = *group;
  entry->features = tool_data->features;
  entry->feature_count = tool_data->feature_count;
  entry->data = *callback_data;
  entry->file_handle = tool_data->file_handle;
  entry->valid = 1;

  return status;
}

// Tool constructor
CONSTRUCTOR_API void constructor()
{
  std::map<std::string, hsa_ven_amd_aqlprofile_parameter_name_t> parameters_dict;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2"] =
      HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2;

  // Set output file
  result_prefix = getenv("ROCP_OUTPUT_DIR");
  if (result_prefix != NULL) {
    DIR* dir = opendir(result_prefix);
    if (dir == NULL) {
      std::ostringstream errmsg;
      errmsg << "Cannot open output directory '" << result_prefix << "'";
      perror(errmsg.str().c_str());
      exit(1);
    }
    std::ostringstream oss;
    oss << result_prefix << "/results.txt";
    result_file_handle = fopen(oss.str().c_str(), "w");
    if (result_file_handle == NULL) {
      std::ostringstream errmsg;
      errmsg << "fopen error, file '" << oss.str().c_str() << "'";
      perror(errmsg.str().c_str());
      exit(1);
    }
  } else
    result_file_handle = stdout;

  // Getting input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name == NULL) {
    fprintf(stderr, "ROCProfiler: input is not specified, ROCP_INPUT env");
    exit(1);
  }
  printf("ROCProfiler: input from \"%s\"\n", xml_name);
  xml::Xml* xml = xml::Xml::Create(xml_name);
  if (xml == NULL) {
    fprintf(stderr, "Input file not found '%s'\n", xml_name);
    exit(1);
  }

  // Getting metrics
  auto metrics_list = xml->GetNodes("top.metric");
  std::vector<std::string> metrics_vec;
  for (auto* entry : metrics_list) {
    const std::string entry_str = entry->opts["name"];
    size_t pos1 = 0;
    while (pos1 < entry_str.length()) {
      const size_t pos2 = entry_str.find(",", pos1);
      const std::string metric_name = entry_str.substr(pos1, pos2 - pos1);
      metrics_vec.push_back(metric_name);
      if (pos2 == std::string::npos) break;
      pos1 = pos2 + 1;
    }
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
    if (params_list.size() != 1) {
      fprintf(stderr, "ROCProfiler: Single input 'parameters' section is supported\n");
      exit(1);
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
          fprintf(stderr, "ROCProfiler: unknown trace parameter %s\n", parameter_name.c_str());
          exit(1);
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

  // Adding dispatch observer
  if (feature_count) {
    dispatch_data_t* dispatch_data = new dispatch_data_t{};
    dispatch_data->features = features;
    dispatch_data->feature_count = feature_count;
    dispatch_data->group_index = 0;
    dispatch_data->file_handle = result_file_handle;
    rocprofiler_set_dispatch_callback(dispatch_callback, dispatch_data);
  }
}

// Tool destructor
DESTRUCTOR_API void destructor() {
  const bool result_file_opened = (result_prefix != NULL) && (result_file_handle != NULL);

  printf("\nROCPRofiler: %u contexts collected", context_count);
  if (result_file_opened) printf(", output directory %s", result_prefix);
  printf("\n");

  // Dump stored profiling output data
  dump_context_array();

  // Close output file
  if (result_file_opened) fclose(result_file_handle);
}
