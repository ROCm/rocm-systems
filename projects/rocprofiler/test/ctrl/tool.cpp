#include <fcntl.h>
#include <hsa.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <iostream>
#include <map>
#include <vector>

#include "inc/rocprofiler.h"
#include "util/xml.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

struct dispatch_data_t {
  rocprofiler_info_t* info;
  unsigned info_count;
  unsigned group_index;
  FILE* file_handle;
};

struct context_entry_t {
  uint32_t index;
  rocprofiler_group_t* group;
  rocprofiler_info_t* info;
  unsigned info_count;
  rocprofiler_callback_data_t data;
  FILE* file_handle;
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned context_array_size = 1;
context_entry_t* context_array = NULL;
unsigned context_array_count = 0;

const char* file_name = NULL;

void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    exit(1);
  }
}

hsa_status_t trace_data_cb(
  hsa_ven_amd_aqlprofile_info_type_t info_type,
  hsa_ven_amd_aqlprofile_info_data_t* info_data,
  void* data)
{
  FILE* file = reinterpret_cast<FILE*>(data);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_SQTT_DATA) {
    fprintf(file, "    data ptr (%p), size(%u)\n", info_data->sqtt_data.ptr, info_data->sqtt_data.size);

  } else status = HSA_STATUS_ERROR;
  return status;
}

unsigned align_size(unsigned size, unsigned alignment) { return ((size + alignment - 1) & ~(alignment - 1)); }

void print_info(FILE* file, const rocprofiler_info_t* info, const unsigned info_count, rocprofiler_t* context, const char* str) {
  if (str) fprintf(file, "%s:\n", str);
  for (unsigned i= 0; i < info_count; ++i) {
    const rocprofiler_info_t* p = &info[i];
    fprintf(file, "  %s ", p->name);
    switch (p->data.kind) {
      case ROCPROFILER_INT64:
        fprintf(file, "(%lu)\n", p->data.result_int64);
        break;
      case ROCPROFILER_BYTES: {
        if (p->data.result_bytes.copy) {
          uint64_t size = 0;

          const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
          for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
            uint64_t chunk_size = *reinterpret_cast<const uint64_t*>(ptr);
            const char* data = ptr + sizeof(uint64_t);
            chunk_size = align_size(chunk_size, sizeof(uint64_t));
            ptr = data + chunk_size;
            size += chunk_size;
          }
          fprintf(file, "size(%lu)\n", size);
          if (size > p->data.result_bytes.size) {
            fprintf(stderr, "SQTT data size is out of the result buffer size\n");
            exit(1);
          }

          const int fd = open("trace_data.sqtt", O_RDWR|O_CREAT|O_TRUNC, 0640);
          if (fd == -1) {
            perror("open 'trace_data.sqtt'");
            exit(1);
          }
          const size_t write_size = write(fd, p->data.result_bytes.ptr, size);
          if (write_size < size) {
            fprintf(stderr, "SQTT data size(%lu) write failed, returned size(%zd)\n", size, write_size);
            exit(1);
          }
        } else {
          fprintf(file, "iterate GPU local memory (\n");
          rocprofiler_iterate_trace_data(context, trace_data_cb, reinterpret_cast<void*>(file));
          fprintf(file, "  )\n");
        }
        break;
      }
      default:
        std::cout << "Bad result kind (" << p->data.kind << ")" << std::endl;
    }
  }
}

void print_group(FILE* file, const rocprofiler_group_t* group, const char* str) {
  if (str) fprintf(file, "%s:\n", str);
  for (unsigned i= 0; i < group->info_count; ++i) {
    print_info(file, group->info[i], 1, group->context, NULL);
  }
}

context_entry_t* alloc_entry() {
  context_entry_t* ptr = 0;

  if(pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  if ((context_array == NULL) || (context_array_count >= context_array_size)) {
    context_array_size *= 2;
    context_array = reinterpret_cast<context_entry_t*>(realloc(context_array, context_array_size * sizeof(context_entry_t)));
  }
  ptr = &context_array[context_array_count];
  *ptr = {};
  ptr->index = context_array_count;
  context_array_count += 1;

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }

  return ptr;
}

void dump_context(context_entry_t* entry) {
  hsa_status_t status = HSA_STATUS_ERROR;
  rocprofiler_group_t* group = entry->group;

  if (group) {
    uint32_t index = entry->index;
    const rocprofiler_info_t* info = entry->info;
    const unsigned info_count = entry->info_count;
    FILE* file_handle = entry->file_handle;

    fprintf(file_handle, "Dispatch[%u], kernel_object(0x%lx):\n", index, entry->data.kernel_object);
  
    status = rocprofiler_get_group_data(group);
    check_status(status);
    //print_group(file, group, "Group[0] data");
  
    status = rocprofiler_get_metrics(group->context);
    check_status(status);
    print_info(file_handle, info, info_count, group->context, NULL);
  
    // Finishing cleanup
    // Deleting profiling context will delete all allocated resources
    rocprofiler_close(group->context);
    entry->group = NULL;
  }
}

void dumping_data() {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  for (unsigned index = 0; index < context_array_count; ++index) {
    dump_context(&context_array[index]);
  }

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}

void handler(rocprofiler_group_t group, void* arg) {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }

  context_entry_t* entry = reinterpret_cast<context_entry_t*>(arg);
  dump_context(entry);

  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}

// profiling callback
hsa_status_t dispatch_callback(
    const rocprofiler_callback_data_t* callback_data,
    void* user_data,
    rocprofiler_group_t** group) {
  // HSA status
  hsa_status_t status = HSA_STATUS_ERROR;
  // Passed tool data
  dispatch_data_t* tool_data = reinterpret_cast<dispatch_data_t*>(user_data);
  // Profiling context
  rocprofiler_t* context = NULL;
  // Context entry
  context_entry_t* entry = alloc_entry();
  // context properties
  rocprofiler_properties_t properties{};
  properties.handler = (file_name != NULL) ? handler : NULL;
  properties.handler_arg = (void*)entry;

  // Open profiling context
  status = rocprofiler_open(0, tool_data->info, tool_data->info_count, &context, 0/*ROCPROFILER_MODE_SINGLEGROUP*/, &properties);
  check_status(status);

  rocprofiler_group_t* groups = NULL;
  uint32_t group_count = 0;
  status = rocprofiler_get_groups(context, &groups, &group_count);
  check_status(status);
  assert(group_count == 1);

  *group = &groups[0];
  entry->group = *group;
  entry->info = tool_data->info;
  entry->info_count = tool_data->info_count;
  entry->data = *callback_data;
  entry->file_handle = tool_data->file_handle;

  return status;
}

CONSTRUCTOR_API void constructor() {
  std::map<std::string, hsa_ven_amd_aqlprofile_parameter_name_t> parameters_dict;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET"] = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_COMPUTE_UNIT_TARGET;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK"] = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_VM_ID_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK"] = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK"] = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK;
  parameters_dict["HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2"] = HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_TOKEN_MASK2;

  // Set output file
  file_name = getenv("ROCP_OUTPUT");
  FILE* file_handle = NULL;
  if (file_name != NULL) {
    file_handle = fopen(file_name, "w");
    if (file_handle == NULL) {
      perror("fopen");
      exit(1);
    }
  } else file_handle = stdout;

  // Getting input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name == NULL) {
    fprintf(stderr, "ROCProfiler: input is not specified, ROCP_INPUT env");
    exit(1);
  }
  printf("ROCProfiler: input from \"%s\"\n", xml_name);
  xml::Xml* xml = new xml::Xml(xml_name);

  // Getting metrics
  auto metrics_list = xml->GetNodes("top.metric");
  std::vector<std::string> metrics_vec;
  for (auto* entry : metrics_list) {
    const std::string entry_str = entry->opts["name"];
    size_t pos1 = 0;
    while(pos1 < entry_str.length()) {
      const size_t pos2 = entry_str.find(",", pos1);
      const std::string metric_name = entry_str.substr(pos1, pos2 - pos1);
      metrics_vec.push_back(metric_name);
      if (pos2 == std::string::npos) break;
      pos1 = pos2 + 1;
    }
  }

  // Getting traces
  auto traces_list = xml->GetNodes("top.trace");

  const unsigned info_count = metrics_vec.size() + traces_list.size();
  rocprofiler_info_t* info= new rocprofiler_info_t[info_count];
  memset(info, 0, info_count * sizeof(rocprofiler_info_t));

  printf("  %d metrics\n", (int) metrics_vec.size());
  for (unsigned i = 0; i < metrics_vec.size(); ++i) {
    const std::string& name = metrics_vec[i];
    printf("%s%s", (i == 0) ? "    " : ", ", name.c_str());
    info[i] = {};
    info[i].type = ROCPROFILER_TYPE_METRIC;
    info[i].name = strdup(name.c_str());
  }
  if (metrics_vec.size()) printf("\n");

  printf("  %d traces\n", (int) traces_list.size());
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
    info[index] = {};
    info[index].type = ROCPROFILER_TYPE_TRACE;
    info[index].name = strdup(name.c_str());
    info[index].data.result_bytes.copy = to_copy_data;

    for (auto* params : params_list) {
      const unsigned parameter_count = params->opts.size();
      rocprofiler_parameter_t *parameters = new rocprofiler_parameter_t[parameter_count];
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

      info[index].parameters = parameters;
      info[index].parameter_count = parameter_count;
    }
    printf("    )\n");
    ++index;
  }

  if (info_count) {
    // Adding dispatch observer
    dispatch_data_t* dispatch_data = new dispatch_data_t{};
    dispatch_data->info = info;
    dispatch_data->info_count = info_count;
    dispatch_data->group_index = 0;
    dispatch_data->file_handle = file_handle;
    rocprofiler_set_dispatch_observer(dispatch_callback, dispatch_data);
  }
}

DESTRUCTOR_API void destructor() {
  printf("\nROCPRofiler: %u contexts collected", context_array_count);
  if (file_name == NULL) {
    printf("\n");
  } else {
    printf(", dumping to %s\n", file_name);
  }
  dumping_data();
}
