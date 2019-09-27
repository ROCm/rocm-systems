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

#include <cxxabi.h>  /* names denangle */
#include <dirent.h>
#include <stdio.h>
#include <sys/syscall.h>   /* For SYS_xxx definitions */

#include <inc/roctracer_hsa.h>
#include <inc/roctracer_hip.h>
#include <inc/roctracer_hcc.h>
#include <inc/roctracer_kfd.h>
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

#ifndef onload_debug
#define onload_debug false
#endif

typedef hsa_rt_utils::Timer::timestamp_t timestamp_t;
hsa_rt_utils::Timer* timer = NULL;
thread_local timestamp_t hsa_begin_timestamp = 0;
thread_local timestamp_t hip_begin_timestamp = 0;
thread_local timestamp_t kfd_begin_timestamp = 0;
bool trace_hsa_api = false;
bool trace_hsa_activity = false;
bool trace_hip = false;
bool trace_kfd = false;

LOADER_INSTANTIATE();

// Global output file handle
FILE* hsa_api_file_handle = NULL;
FILE* hsa_async_copy_file_handle = NULL;
FILE* hip_api_file_handle = NULL;
FILE* hcc_activity_file_handle = NULL;
FILE* kfd_api_file_handle = NULL;

static inline uint32_t GetPid() { return syscall(__NR_getpid); }
static inline uint32_t GetTid() { return syscall(__NR_gettid); }

// Error handler
void fatal(const std::string msg) {
  fflush(hsa_api_file_handle);
  fflush(hsa_async_copy_file_handle);
  fflush(hip_api_file_handle);
  fflush(hcc_activity_file_handle);
  fflush(kfd_api_file_handle);
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

// KFD API callback function
void kfd_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const kfd_api_data_t* data = reinterpret_cast<const kfd_api_data_t*>(callback_data);
  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    kfd_begin_timestamp = timer->timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = timer->timestamp_fn_ns();
    std::ostringstream os;
    os << kfd_begin_timestamp << ":" << end_timestamp << " " << GetPid() << ":" << GetTid() << " " << kfd_api_data_pair_t(cid, *data);
    fprintf(kfd_api_file_handle, "%s\n", os.str().c_str());
  }
}
// C++ symbol demangle
static inline const char* cxx_demangle(const char* symbol) {
  size_t funcnamesize;
  int status;
  const char* ret = (symbol != NULL) ? abi::__cxa_demangle(symbol, NULL, &funcnamesize, &status) : symbol;
  return (ret != NULL) ? ret : symbol;
}

struct hsa_api_trace_entry_t {
  uint32_t valid;
  uint32_t type;
  uint32_t cid;
  timestamp_t begin;
  timestamp_t end;
  uint32_t pid;
  uint32_t tid;
  hsa_api_data_t data;
};

void hsa_api_flush_cb(hsa_api_trace_entry_t* entry);
roctracer::TraceBuffer<hsa_api_trace_entry_t>::flush_prm_t hsa_flush_prm[1] = {{0, hsa_api_flush_cb}};
roctracer::TraceBuffer<hsa_api_trace_entry_t> hsa_api_trace_buffer("HSA API", 0x200000, hsa_flush_prm, 1);

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
    hsa_api_trace_entry_t* entry = hsa_api_trace_buffer.GetEntry();
    entry->valid = roctracer::TRACE_ENTRY_COMPL;
    entry->type = 0;
    entry->cid = cid;
    entry->begin = hsa_begin_timestamp;
    entry->end = end_timestamp;
    entry->pid = GetPid();
    entry->tid = GetTid();
    entry->data = *data;
  }
}

void hsa_api_flush_cb(hsa_api_trace_entry_t* entry) {
  std::ostringstream os;
  os << entry->begin << ":" << entry->end << " " << entry->pid << ":" << entry->tid << " " << hsa_api_data_pair_t(entry->cid, entry->data);
  fprintf(hsa_api_file_handle, "%s\n", os.str().c_str()); fflush(hsa_api_file_handle);
}

void hsa_activity_callback(
  uint32_t op,
  activity_record_t* record,
  void* arg)
{
  static uint64_t index = 0;
  fprintf(hsa_async_copy_file_handle, "%lu:%lu async-copy%lu\n", record->begin_ns, record->end_ns, index); fflush(hsa_async_copy_file_handle);
  index++;
}

struct hip_api_trace_entry_t {
  uint32_t valid;
  uint32_t type;
  uint32_t domain;
  uint32_t cid;
  timestamp_t begin;
  timestamp_t end;
  uint32_t pid;
  uint32_t tid;
  hip_api_data_t data;
  const char* name;
  void* ptr;
};

void hip_api_flush_cb(hip_api_trace_entry_t* entry);
roctracer::TraceBuffer<hip_api_trace_entry_t>::flush_prm_t hip_flush_prm[1] = {{0, hip_api_flush_cb}};
roctracer::TraceBuffer<hip_api_trace_entry_t> hip_api_trace_buffer("HIP", 0x200000, hip_flush_prm, 1);

void hip_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);

  if (data->phase == ACTIVITY_API_PHASE_ENTER) {
    hip_begin_timestamp = timer->timestamp_fn_ns();
  } else {
    const timestamp_t end_timestamp = timer->timestamp_fn_ns();
    hip_api_trace_entry_t* entry = hip_api_trace_buffer.GetEntry();
    entry->valid = roctracer::TRACE_ENTRY_COMPL;
    entry->type = 0;
    entry->cid = cid;
    entry->domain = domain;
    entry->begin = hip_begin_timestamp;
    entry->end = end_timestamp;
    entry->pid = GetPid();
    entry->tid = GetTid();
    entry->data = *data;
    entry->name = NULL;
    entry->ptr = NULL;

    switch (cid) {
      case HIP_API_ID_hipMalloc:
        entry->ptr = *(data->args.hipMalloc.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
        const hipFunction_t f = data->args.hipModuleLaunchKernel.f;
        if (f != NULL) {
          entry->name = strdup(roctracer::HipLoader::Instance().KernelNameRef(f));
        }
    }
  }
}

void mark_api_callback(
    uint32_t domain,
    uint32_t cid,
    const void* callback_data,
    void* arg)
{
  (void)arg;
  const char* name = reinterpret_cast<const char*>(callback_data);

  const timestamp_t timestamp = timer->timestamp_fn_ns();
  hip_api_trace_entry_t* entry = hip_api_trace_buffer.GetEntry();
  entry->valid = roctracer::TRACE_ENTRY_COMPL;
  entry->type = 0;
  entry->cid = 0;
  entry->domain = domain;
  entry->begin = timestamp;
  entry->end = timestamp + 1;
  entry->pid = GetPid();
  entry->tid = GetTid();
  entry->data = {};
  entry->name = name;
  entry->ptr = NULL;
}

void hip_api_flush_cb(hip_api_trace_entry_t* entry) {
  const uint32_t domain = entry->domain;
  const uint32_t cid = entry->cid;
  const hip_api_data_t* data = &(entry->data);
  const timestamp_t begin_timestamp = entry->begin;
  const timestamp_t end_timestamp = entry->end;
  std::ostringstream oss;                                                                        \

  const char* str = (domain != ACTIVITY_DOMAIN_EXT_API) ? roctracer_op_string(domain, cid, 0) : strdup("MARK");
  oss << std::dec <<
    begin_timestamp << ":" << end_timestamp << " " << entry->pid << ":" << entry->tid << " " << str;

  if (domain == ACTIVITY_DOMAIN_HIP_API) {
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
        fprintf(hip_api_file_handle, "%s(ptr(%p) size(0x%x))\n",
          oss.str().c_str(),
          entry->ptr,
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(hip_api_file_handle, "%s(ptr(%p))\n",
          oss.str().c_str(),
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
      case HIP_API_ID_hipExtModuleLaunchKernel:
      case HIP_API_ID_hipHccModuleLaunchKernel:
        fprintf(hip_api_file_handle, "%s(kernel(%s) stream(%p))\n",
          oss.str().c_str(),
          cxx_demangle(entry->name),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        fprintf(hip_api_file_handle, "%s()\n", oss.str().c_str());
    }
  } else {
    fprintf(hip_api_file_handle, "%s(%s)\n", oss.str().c_str(), entry->name);
  }

  fflush(hip_api_file_handle);
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
      fflush(hcc_activity_file_handle);
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
    oss << prefix << "/" << GetPid() << "_" << name;
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

void close_output_file(FILE* file_handle) {
  if ((file_handle != NULL) && (file_handle != stdout)) fclose(file_handle);
}

// HSA-runtime tool on-load method
extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  if (onload_debug) { printf("TOOL OnLoad\n"); fflush(stdout); }
  timer = new hsa_rt_utils::Timer(table->core_->hsa_system_get_info_fn);

  // API traces switches
  const char* trace_domain = getenv("ROCTRACER_DOMAIN");
  if (trace_domain != NULL) {
    if (strncmp(trace_domain, "hsa", 3) == 0) {
      trace_hsa_api = true;
      trace_hsa_activity = true;
    }
    if (strncmp(trace_domain, "hip", 3) == 0) {
      trace_hip = true;
    }
    if (strncmp(trace_domain, "sys", 3) == 0) {
      trace_hsa_api = true;
      trace_hip = true;
    }
  }

  trace_kfd = (trace_domain == NULL) || (strncmp(trace_domain, "kfd", 3) == 0);

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
  std::vector<std::string> kfd_api_vec;

  printf("ROCTracer (pid=%d): ", (int)GetPid()); fflush(stdout);
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
        trace_hsa_api = true;
        hsa_api_vec = api_vec;
      }
      if (name == "KFD") {
        found = true;
        trace_kfd = true;
        kfd_api_vec = api_vec;
      }
      if (name == "GPU") {
        found = true;
        trace_hsa_activity = true;
      }
      if (name == "HIP") {
        found = true;
        trace_hip = true;
      }
    }

    if (found) printf("input from \"%s\"", xml_name);
  }
  printf("\n");

  // Enable HSA API callbacks
  if (trace_hsa_api) {
    hsa_api_file_handle = open_output_file(output_prefix, "hsa_api_trace.txt");

    // initialize HSA tracing
    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*)table);

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
    printf(")\n");
  }

  if (trace_kfd) {
    kfd_api_file_handle = open_output_file(output_prefix, "kfd_api_trace.txt");
    // initialize KFD tracing
    roctracer_set_properties(ACTIVITY_DOMAIN_KFD_API, NULL);

    printf("    KFD-trace(");
    if (kfd_api_vec.size() != 0) {
      for (unsigned i = 0; i < kfd_api_vec.size(); ++i) {
        uint32_t cid = KFD_API_ID_NUMBER;
        const char* api = kfd_api_vec[i].c_str();
        ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_KFD_API, api, &cid));
        ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_KFD_API, cid, kfd_api_callback, NULL));
        printf(" %s", api);
      }
    } else {
      ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_KFD_API, kfd_api_callback, NULL));
    }
    printf(")\n");
  }
  if (trace_hsa_activity) {
    hsa_async_copy_file_handle = open_output_file(output_prefix, "async_copy_trace.txt");

    // initialize HSA tracing
    roctracer::hsa_ops_properties_t ops_properties{
      table,
      reinterpret_cast<activity_async_callback_t>(hsa_activity_callback),
      NULL,
      output_prefix};
    roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

    fprintf(stdout, "    HSA-activity-trace()\n"); fflush(stdout);
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));
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

    roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, (void*)mark_api_callback);
  }

  if (onload_debug) { printf("TOOL OnLoad end\n"); fflush(stdout); }
  return roctracer_load(table, runtime_version, failed_tool_count, failed_tool_names);
}

// tool unload method
void tool_unload(bool destruct) {
  static bool is_unloaded = false;

  if (onload_debug) { printf("TOOL tool_unload (%d, %d)\n", (int)destruct, (int)is_unloaded); fflush(stdout); }
  if (destruct == false) return;
  if (is_unloaded == true) return;
  is_unloaded = true;
  roctracer_unload(destruct);

  if (trace_hsa_api) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

    hsa_api_trace_buffer.Flush();
    close_output_file(hsa_api_file_handle);
  }
  if (trace_hsa_activity) {
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HSA_OPS));

    close_output_file(hsa_async_copy_file_handle);
  }
  if (trace_hip) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_flush_activity());
    ROCTRACER_CALL(roctracer_close_pool());

    hip_api_trace_buffer.Flush();
    close_output_file(hip_api_file_handle);
    close_output_file(hcc_activity_file_handle);
  }

  if (trace_kfd) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_KFD_API));
    fclose(kfd_api_file_handle);
  }
  if (onload_debug) { printf("TOOL tool_unload end\n"); fflush(stdout); }
}

// HSA-runtime on-unload method
extern "C" PUBLIC_API void OnUnload() {
  if (onload_debug) { printf("TOOL OnUnload\n"); fflush(stdout); }
  tool_unload(false);
  if (onload_debug) { printf("TOOL OnUnload end\n"); fflush(stdout); }
}

extern "C" CONSTRUCTOR_API void constructor() {
  if (onload_debug) { printf("TOOL constructor ...end\n"); fflush(stdout); }
}
extern "C" DESTRUCTOR_API void destructor() {
  if (onload_debug) { printf("TOOL destructor\n"); fflush(stdout); }
  tool_unload(true);
  if (onload_debug) { printf("TOOL destructor end\n"); fflush(stdout); }
}
