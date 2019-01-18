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

#include <stdio.h>

#include <inc/roctracer_hsa.h>
#include <inc/roctracer_hip.h>
#include <inc/roctracer_hcc.h>
#include <inc/ext/hsa_rt_utils.hpp>
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

// Error handler
void fatal(const std::string msg) {
  fflush(stdout);
  fprintf(stderr, "%s\n\n", msg.c_str());
  fflush(stderr);
  abort();
}

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
    std::ostringstream os;
    os << '(' << hsa_begin_timestamp << ":" << end_timestamp << ") " << hsa_api_data_pair_t(cid, *data);
    fprintf(stdout, "%s\n", os.str().c_str());
  }
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
    fprintf(stdout, "(%lu:%lu) %s(", hsa_begin_timestamp, end_timestamp, roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0));
    switch (cid) {
      case HIP_API_ID_hipMemcpy:
        fprintf(stdout, "dst(%p) src(%p) size(0x%x) kind(%u)",
          data->args.hipMemcpy.dst,
          data->args.hipMemcpy.src,
          (uint32_t)(data->args.hipMemcpy.sizeBytes),
          (uint32_t)(data->args.hipMemcpy.kind));
        break;
      case HIP_API_ID_hipMalloc:
        fprintf(stdout, "ptr(0x%p) size(0x%x)",
          *(data->args.hipMalloc.ptr),
          (uint32_t)(data->args.hipMalloc.size));
        break;
      case HIP_API_ID_hipFree:
        fprintf(stdout, "ptr(%p)",
          data->args.hipFree.ptr);
        break;
      case HIP_API_ID_hipModuleLaunchKernel:
        fprintf(stdout, "kernel(\"%s\") stream(%p)",
          hipKernelNameRef(data->args.hipModuleLaunchKernel.f),
          data->args.hipModuleLaunchKernel.stream);
        break;
      default:
        break;
    }
    fprintf(stdout, ")\n"); fflush(stdout);
  }
}

// Activity tracing callback
//   hipMalloc id(3) correlation_id(1): begin_ns(1525888652762640464) end_ns(1525888652762877067)
void activity_callback(const char* begin, const char* end, void* arg) {
  const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
  const roctracer_record_t* end_record = reinterpret_cast<const roctracer_record_t*>(end);

  fprintf(stdout, "\tActivity records:\n"); fflush(stdout);
  while (record < end_record) {
    const char * name = roctracer_op_string(record->domain, record->op, record->kind);
    fprintf(stdout, "\t%s\tcorrelation_id(%lu) time_ns(%lu:%lu)",
      name,
      record->correlation_id,
      record->begin_ns,
      record->end_ns
    );
    if (record->domain == ACTIVITY_DOMAIN_HIP_API) {
      fprintf(stdout, " process_id(%u) thread_id(%u)",
        record->process_id,
        record->thread_id
      );
    } else if (record->domain == ACTIVITY_DOMAIN_HCC_OPS) {
      fprintf(stdout, " device_id(%d) queue_id(%lu)",
        record->device_id,
        record->queue_id
      );
    } else {
      fprintf(stderr, "Bad domain %d\n", record->domain);
      abort();
    }
    if (record->op == hc::HSA_OP_ID_COPY) fprintf(stdout, " bytes(0x%zx)", record->bytes);
    fprintf(stdout, "\n");
    fflush(stdout);
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

// HSA-runtime tool on-load method
extern "C" PUBLIC_API bool OnLoad(HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
                       const char* const* failed_tool_names) {
  timer = new hsa_rt_utils::Timer(table->core_->hsa_system_get_info_fn);

  // API traces switches
  const char* trace_domain = getenv("ROCTRACER_DOMAIN");
  trace_hsa = (trace_domain == NULL) || (strncmp(trace_domain, "hsa", 3) == 0);
  trace_hip = (trace_domain == NULL) || (strncmp(trace_domain, "hip", 3) == 0);

  // API trace vector
  std::vector<std::string> hsa_api_vec;

  // XML input
  const char* xml_name = getenv("ROCP_INPUT");
  if (xml_name != NULL) {
    printf("ROCTracer: input from \"%s\"\n", xml_name);
    xml::Xml* xml = xml::Xml::Create(xml_name);
    if (xml == NULL) {
      fprintf(stderr, "ROCTracer: Input file not found '%s'\n", xml_name);
      abort();
    }
  
    for (const auto* entry : xml->GetNodes("top.trace")) {
      auto it = entry->opts.find("name");
      if (it == entry->opts.end()) fatal("ROCTracer: trace name is missing");
      const std::string& name = it->second;

      std::vector<std::string> api_vec;
      for (const auto* node : entry->nodes) {
        if (node->tag != "parameters") fatal("ROCProfiler: trace node is not supported '" + name + ":" + node->tag + "'");
        get_xml_array(node, "api", ",", &api_vec);
        break;
      }

      if (name == "HSA") {
        trace_hsa |= true;
        hsa_api_vec = api_vec;
      }
      if (name == "HIP") {
        trace_hip |= true;
      }
    }
  }

  // Enable HSA API callbacks
  if (trace_hsa) {
    printf("    HSA-trace");
    if (hsa_api_vec.size() != 0) {
      printf("(");
      for (unsigned i = 0; i < hsa_api_vec.size(); ++i) {
        uint32_t cid = HSA_API_ID_NUMBER;
        const char* api = hsa_api_vec[i].c_str();
        ROCTRACER_CALL(roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api, &cid));
        ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_API, cid, hsa_api_callback, NULL));
        printf(" %s", api);
      }
      printf(" )");
    } else {
      ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API, hsa_api_callback, NULL));
    }
    printf("\n");
  }

  // Enable HIP API callbacks/activity
  if (trace_hip) {
    printf("    HIP-trace\n");
    // Allocating tracing pool
    roctracer_properties_t properties{};
    properties.buffer_size = 12;
    properties.buffer_callback_fun = activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_enable_domain_activity(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, hip_api_callback, NULL));
  }

  return true;
}

// HSA-runtime tool on-unload method
extern "C" PUBLIC_API void OnUnload() {
  if (trace_hsa) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));
  }
  if (trace_hip) {
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HIP_API));
    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HCC_OPS));
    ROCTRACER_CALL(roctracer_close_pool());
  }
}
