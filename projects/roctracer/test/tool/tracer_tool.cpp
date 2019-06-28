/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
*/

#include <sstream>
#include <string>

#include <dirent.h>
#include <stdio.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#include <inc/roctracer_hsa.h>
#include <inc/roctracer_hip.h>
#include <inc/roctracer_hcc.h>
#include <inc/ext/hsa_rt_utils.hpp>
#include <src/core/loader.h>
#include <src/core/trace_buffer.h>
#include <util/xml.h>

#define PUBLIC_API __attribute__((visibility("default")))
#define CONSTRUCTOR_API __attribute__((constructor))
#define DESTRUCTOR_API __attribute__((destructor))

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                                       \
  do {                                                                                             \
    int err = call;                                                                                \
    if (err != 0) {                                                                                \
      std::cerr << roctracer_error_string() << std::endl << std::flush;                            \
      abort();                                                                                     \
    }                                                                                              \
  } while (0)

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer* timer = NULL;
thread_local timestamp_t hsa_begin_timestamp = 0;
thread_local timestamp_t hip_begin_timestamp = 0;
bool trace_hsa = false;
bool trace_hip = false;

LOADER_INSTANTIATE();

// Global output file handle
FILE* hsa_api_file_handle = NULL;
FILE* hsa_async_copy_file_handle = NULL;
FILE* hip_api_file_handle = NULL;
FILE* hcc_activity_file_handle = NULL;

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// Error handler
void fatal(const std::string msg) {
  fflush(hsa_api_file_handle);
  fflush(hsa_async_copy_file_handle);
  fflush(hip_api_file_handle);
  fflush(hcc_activity_file_handle);
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

struct api_trace_entry_t {
  uint32_t valid;
  uint32_t type;
  uint32_t cid;
  timestamp_t begin;
  timestamp_t end;
  uint32_t pid;
  uint32_t tid;
  hsa_api_data_t data;
};

roctracer::TraceBuffer<api_trace_entry_t> api_trace_buffer(0x200000);

// HSA API callback function
void hsa_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hsa_api_data_t* data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hsa_begin_timestamp = timer->timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down) ? hsa_begin_timestamp : timer->timestamp_fn_ns();
    api_trace_entry_t* entry = api_trace_buffer.GetEntry();
    entry->valid = roctracer::TRACE_ENTRY_COMPL;
    entry->cid = cid;
    entry->begin = hsa_begin_timestamp;
    entry->end = end_timestamp;
    entry->pid = GetPid();
    entry->tid = GetTid();
    entry->data = *data;
  }
}

void hsa_api_flush_cb(api_trace_entry_t* entry) {
  std::ostringstream os;
  os << entry->begin << ":" << entry->end << " " << entry->pid << ":" << entry->tid << " " << hsa_api_data_pair_t(entry->cid, entry->data);
  fprintf(hsa_api_file_handle, "%s\n", os.str().c_str());
}

void hsa_activity_callback(
  uint32_t op,
  activity_record_t* record,
  void* arg)
{
  static uint64_t index = 0;
  fprintf(hsa_async_copy_file_handle, "%lu:%lu async-copy%lu\n", record->begin_ns, record->end_ns, index);
  index++;
}

void hip_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hsa_begin_timestamp = timer->timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = timer->timestamp_fn_ns();
    std::ostringstream oss;                                                                        \
    oss << std::dec <<
      hsa_begin_timestamp << ":" << end_timestamp << " " << GetPid() << ":" << GetTid() << " " << roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);                                                       \

    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        fprintf(hip_api_file_handle, "%s(dst(%p) src(%p) size(0x%x) kind(%u))\n",
          oss.str().c_str(),
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        fprintf(hip_api_file_handle, "%s(ptr(0x%p) size(0x%x))\n",
          oss.str().c_str(),
          *(data->args.hipMalloc.ptr),
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(hip_api_file_handle, "%s(ptr(%p))\n",
          oss.str().c_str(),
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        fprintf(hip_api_file_handle, "%s(kernel(%s) stream(%p))\n",
          oss.str().c_str(),
          roctracer::HipLoader::Instance().KernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        fprintf(hip_api_file_handle, "%s()\n", oss.str().c_str());
        break;
    }
    fflush(hip_api_file_handle);
  }
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void hcc_activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);

  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      fprintf(hcc_activity_file_handle, "%lu:%lu %d:%lu %s:%lu\n",
        record->begin_ns, record->end_ns, record->device_id, record->queue_id, name, record->correlation_id);
    } else {
#if 0
      fprintf(hip_api_file_handle, "%lu:%lu %u:%u %s()\n",
        record->begin_ns, record->end_ns, record->process_id, record->thread_id, name);
#endif
    }
    ROCTRACER_CALL(roctracer_next_record(record, &record));
  }
}

// Input parser
std::string normalize_token(const std::string& token, bool not_empty, const std::string& label) {
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
    fatal("normalize_token error, " + label + ": '" + token + "'," + error_str);
  }
  return (norm_len != 0) ? token.substr(first_pos, norm_len) : std::string("");
}

int get_xml_array(const xml::Xml::level_t* node, const std::string& field, const std::string& delim, std::vector<std::string>* vec, const char* label = NULL) {
  int parse_iter = 0;
  const auto& opts = node->opts;
  auto it = opts.find(field);
  if (it != opts.end()) {
    const std::string array_string = it->second;
    if (label != NULL) printf("%s%s = %s\n", label, field.c_str(), array_string.c_str());
    size_t pos1 = 0;
    const size_t string_len = array_string.length();
    while (pos1 < string_len) {
      const size_t pos2 = array_string.find(delim, pos1);
      const bool found = (pos2 != std::string::npos);
      const size_t token_len = (pos2 != std::string::npos) ? pos2 - pos1 : string_len - pos1;
      const std::string token = array_string.substr(pos1, token_len);
      const std::string norm_str = normalize_token(token, found, "get_xml_array");
      if (norm_str.length() != 0) vec->push_back(norm_str);
      if (!found) break;
      pos1 = pos2 + 1;
      ++parse_iter;
    }
  }
  return parse_iter;
}

// Open output file
FILE* open_output_file(const char* prefix, const char* name) {
  FILE* file_handle = NULL;
  if (prefix != NULL) {
    std::ostringstream oss;
    oss << prefix << "/" << name;
    file_handle = fopen(oss.str().c_str(), "w");
    if (file_handle == NULL) {
      std::ostringstream errmsg;
      errmsg << "ROCTracer: fopen error, file '" << oss.str().c_str() << "'";
      perror(errmsg.str().c_str());
      abort();
    }
  } else file_handle = stdout;
  return file_handle;
}

// HSA-runtime tool on-load method
extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  timer = new hsa_rt_utils::Timer(table->core_->hsa_system_get_info_fn);

  // API traces switches
  const char* trace_domain = getenv("ROCTRACER_DOMAIN");
  trace_hsa = (trace_domain == NULL) || (strncmp(trace_domain, "hsa", 3) == 0);
  trace_hip = (trace_domain == NULL) || (strncmp(trace_domain, "hip", 3) == 0);

  // Output file
  const char* output_prefix = getenv("ROCP_OUTPUT_DIR");
  if (output_prefix != NULL) {
    DIR* dir = opendir(output_prefix);
    if (dir == NULL) {
      std::ostringstream errmsg;
      errmsg << "ROCTracer: Cannot open output directory '" << output_prefix << "'";
      perror(errmsg.str().c_str());
      abort();
    }
  }

  // API trace vector
  std::vector<std::string> hsa_api_vec;

  printf("ROCTracer: "); fflush(stdout);
  // XML input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name != NULL) {
    xml::Xml* xml = xml::Xml::Create(xml_name);
    if (xml == NULL) {
      fprintf(stderr, "ROCTracer: Input file not found '%s'\n", xml_name);
      abort();
    }

    bool found = false;
    for (const auto* entry : xml->GetNodes("top.trace")) {
      auto it = entry->opts.find("name");
      if (it == entry->opts.end()) fatal("ROCTracer: trace name is missing");
      const std::string& name = it->second;

      std::vector<std::string> api_vec;
      for (const auto* node : entry->nodes) {
        if (node->tag != "parameters") fatal("ROCTracer: trace node is not supported '" + name + ":" + node->tag + "'");
        get_xml_array(node, "api", ",", &api_vec);
        break;
      }

      if (name == "HSA") {
        found = true;
        trace_hsa |= true;
        hsa_api_vec = api_vec;
      }
      if (name == "HIP") {
        found = true;
        trace_hip |= true;
      }
    }

    if (found) printf("input from \"%s\"", xml_name);
  }
  printf("\n");

  // Enable HSA API callbacks
  if (trace_hsa) {
    hsa_api_file_handle = open_output_file(output_prefix, "hsa_api_trace.txt");
    hsa_async_copy_file_handle = open_output_file(output_prefix, "async_copy_trace.txt");

    // initialize HSA tracing
    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*)table);
    roctracer::hsa_ops_properties_t ops_properties{
      table,
      reinterpret_cast<activity_async_callback_t>(hsa_activity_callback),
      NULL,
      output_prefix};
    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

    fprintf(stdout, "    HSA-trace("); fflush(stdout);
    if (hsa_api_vec.size() != 0) {
      for (unsigned i = 0; i < hsa_api_vec.size(); ++i) {
        uint32_t cid = HSA_API_ID_NUMBER;
        const char* api = hsa_api_vec[i].c_str();
        ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api, &cid));
        ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_API, cid, hsa_api_callback, NULL));
        printf(" %s", api);
      }
    } else {
      ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, NULL));
    }
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
    printf(")\n");
  }

  // Enable HIP API callbacks/activity
  if (trace_hip) {
    hip_api_file_handle = open_output_file(output_prefix, "hip_api_trace.txt");
    hcc_activity_file_handle = open_output_file(output_prefix, "hcc_ops_trace.txt");

    fprintf(stdout, "    HIP-trace()\n"); fflush(stdout);
    // Allocating tracing pool
    roctracer_properties_t properties{};
    properties.buffer_size = 0x80000;
    properties.buffer_callback_fun = hcc_activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, NULL));
  }

  return roctracer_load(table, runtime_version, failed_tool_count, failed_tool_names);
}

// HSA-runtime tool on-unload method
extern "C" PUBLIC_API void OnUnload() {
  static bool is_unloaded = false;
  if (is_unloaded) {
    return;
  }
  is_unloaded = true;
  roctracer_unload();

  if (trace_hsa) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));

    api_trace_buffer.Flush(0, hsa_api_flush_cb);

    fclose(hsa_api_file_handle);
    fclose(hsa_async_copy_file_handle);
  }
  if (trace_hip) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_flush_activity());
    ROCTRACER_CALL(roctracer_close_pool());

    fclose(hip_api_file_handle);
    fclose(hcc_activity_file_handle);
  }
}

extern "C" CONSTRUCTOR_API void constructor() {}
extern "C" DESTRUCTOR_API void destructor() { OnUnload(); }
