#include <hsa.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <map>
#include <vector>

#include "inc/rocprofiler.h"
#include "util/xml.h"

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Tool thread
pthread_t thread;
pthread_attr_t thr_attr;
bool thr_stop = false;

struct dispatch_data_t {
  rocprofiler_info_t* info;
  unsigned info_count;
  unsigned group_index;
};

struct context_entry_t {
  rocprofiler_group_t* group;
  rocprofiler_info_t* info;
  unsigned info_count;
  rocprofiler_callback_data_t data;
};

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
unsigned context_array_size = 1;
context_entry_t* context_array = NULL;
unsigned context_array_index = 0;

const char* file_name;
FILE* file_handle = NULL;

void check_status(hsa_status_t status) {
  if (status != HSA_STATUS_SUCCESS) {
    const char* error_string = NULL;
    rocprofiler_error_string(&error_string);
    fprintf(stderr, "ERROR: %s\n", error_string);
    exit(1);
  }
}

unsigned align_size(unsigned size, unsigned alignment) {
  return ((size + alignment - 1) & ~(alignment - 1));
}

void print_info(FILE* file, const rocprofiler_info_t* info, const unsigned info_count,
                const char* str) {
  if (str) fprintf(file, "%s:\n", str);
  for (unsigned i = 0; i < info_count; ++i) {
    const rocprofiler_info_t* p = &info[i];
    fprintf(file, "  %s ", p->name);
    switch (p->data.kind) {
      case ROCPROFILER_DATA_KIND_INT64:
        fprintf(file, "(%lu)\n", p->data.result64);
        break;
      case ROCPROFILER_BYTES: {
        fprintf(file, "(\n");
        const char* ptr = reinterpret_cast<const char*>(p->data.result_bytes.ptr);
        uint64_t size = 0;
        for (unsigned i = 0; i < p->data.result_bytes.instance_count; ++i) {
          size = *reinterpret_cast<const uint64_t*>(ptr);
          const char* data = ptr + sizeof(size);
          fprintf(file, "    data (%p), size (%lu)\n", data, size);
          size = align_size(size, sizeof(uint64_t));
          ptr = data + size;
        }
        fprintf(file, "  )\n");
        break;
      }
      default:
        std::cout << "Bad result kind (" << p->data.kind << ")" << std::endl;
    }
  }
}

void print_group(FILE* file, const rocprofiler_group_t* group, const char* str) {
  if (str) fprintf(file, "%s:\n", str);
  for (unsigned i = 0; i < group->info_count; ++i) {
    print_info(file, group->info[i], 1, NULL);
  }
}

void store_context(context_entry_t context_entry) {
  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }
  if ((context_array == NULL) || (context_array_index >= context_array_size)) {
    context_array_size *= 2;
    context_array = reinterpret_cast<context_entry_t*>(
        realloc(context_array, context_array_size * sizeof(context_entry_t)));
  }
  context_array_index += 1;
  context_array[context_array_index - 1] = context_entry;
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }
}

void dump_context(FILE* file, unsigned index) {
  hsa_status_t status = HSA_STATUS_ERROR;

  if (pthread_mutex_lock(&mutex) != 0) {
    perror("pthread_mutex_lock");
    exit(1);
  }
  context_entry_t* entry = &context_array[index];
  rocprofiler_group_t* group = entry->group;
  const rocprofiler_info_t* info = entry->info;
  const unsigned info_count = entry->info_count;
  fprintf(file, "Dispatch[%u], kernel_object(0x%lx):\n", index, entry->data.kernel_object);
  if (pthread_mutex_unlock(&mutex) != 0) {
    perror("pthread_mutex_unlock");
    exit(1);
  }

  status = rocprofiler_get_group_data(group);
  check_status(status);
  // print_group(file, group, "Group[0] data");

  status = rocprofiler_get_metrics(group->context);
  check_status(status);
  print_info(file, info, info_count, NULL);

  // Finishing cleanup
  // Deleting profiling context will delete all allocated resources
  rocprofiler_close(group->context);
}

// Provided standard profiling callback
hsa_status_t dispatch_callback(const rocprofiler_callback_data_t* callback_data, void* user_data,
                               rocprofiler_group_t** group) {
  hsa_status_t status = HSA_STATUS_ERROR;
  // Passed tool data
  dispatch_data_t* tool_data = reinterpret_cast<dispatch_data_t*>(user_data);
  // Profiling context
  rocprofiler_t* context = NULL;

  // Open profiling context
  status = rocprofiler_open(0, tool_data->info, tool_data->info_count, &context, 0, NULL);
  check_status(status);

  rocprofiler_group_t* groups = NULL;
  uint32_t group_count = 0;
  status = rocprofiler_get_groups(context, &groups, &group_count);
  check_status(status);
  assert(group_count == 1);

  *group = &groups[0];
  store_context({*group, tool_data->info, tool_data->info_count, *callback_data});

  return status;
}

void* dumping_data(void*) {
  unsigned index = 0;
  do {
    while (index < context_array_index) {
      dump_context(file_handle, index);
      ++index;
    }
  } while (!thr_stop);
  return NULL;
}

CONSTRUCTOR_API void constructor() {
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

#ifdef TOOL_THREAD
  int err = pthread_attr_init(&thr_attr);
  if (err) {
    errno = err;
    perror("pthread_attr_init");
    exit(1);
  }
  err = pthread_create(&thread, &thr_attr, dumping_data, NULL);
  if (err) {
    errno = err;
    perror("pthread_create");
    exit(1);
  }
#endif

  // Set output file
  file_name = getenv("ROCP_OUTPUT");
  if (file_name != NULL) {
    file_handle = fopen(file_name, "w");
    if (file_handle == NULL) {
      perror("fopen");
      exit(1);
    }
  } else
    file_handle = stdout;

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

  const unsigned info_count = metrics_vec.size() + traces_list.size();
  rocprofiler_info_t* info = new rocprofiler_info_t[info_count];
  memset(info, 0, info_count * sizeof(rocprofiler_info_t));

  printf("  %d metrics\n", (int)metrics_vec.size());
  for (unsigned i = 0; i < metrics_vec.size(); ++i) {
    const std::string& name = metrics_vec[i];
    printf("%s%s", (i == 0) ? "    " : ", ", name.c_str());
    info[i] = {};
    info[i].type = ROCPROFILER_TYPE_METRIC;
    info[i].name = strdup(name.c_str());
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
    printf("    %s (\n", name.c_str());
    info[index] = {};
    info[index].type = ROCPROFILER_TYPE_TRACE;
    info[index].name = strdup(name.c_str());

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
    rocprofiler_dispatch_observer(dispatch_callback, dispatch_data);
  }
}

DESTRUCTOR_API void destructor() {
  printf("\nROCPRofiler: %u contexts collected", context_array_index);
  thr_stop = true;
#ifdef TOOL_THREAD
  pthread_join(thread, NULL);
#else
  dumping_data(NULL);
#endif
}
