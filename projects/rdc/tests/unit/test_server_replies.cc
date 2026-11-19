/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

// rdcd's gRPC service (rdc_api_service.cc) against a fake RDC library: the
// rdc_* C API below replaces librdc_bootstrap. Replies carry no uninitialized
// data, counts within the arrays they count and strings within their buffers;
// requests whose job IDs or names do not fit the C API fail before the library
// sees them.

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"
#include "rdc/rdc_api_service.h"

namespace {

// What the fake library writes into the outputs of a call that succeeds.
enum class Fill {
  kNothing,   // nothing: the outputs keep what the caller left there
  kOverfull,  // 'A' into every byte but enums and bools: unterminated strings,
              // counts past every array
};

Fill g_fill = Fill::kNothing;
int g_calls = 0;                // calls that reached the fake library
std::string g_job_id;           // the job ID of the last job call
std::string g_name;             // the name of the last group or field group created
std::string g_field_str;        // the string value of a field, when set
bool g_negative_links = false;  // the link status has a negative GPU count

// Valid values for the enums and bools of an output filled with 'A': the checks
// are on counts and strings, and UBSan reports a load of an enum or bool that
// holds no valid value.
void make_valid(void*) {}
void make_valid(rdc_field_group_info_t* info) {
  for (rdc_field_t& field_id : info->field_ids) {
    field_id = RDC_FI_GPU_CLOCK;
  }
}
void make_valid(rdc_field_value* value) {
  value->field_id = RDC_FI_GPU_CLOCK;
  value->type = STRING;  // a string of 'A's with no NUL
}
void make_valid(rdc_diag_test_result_t* result) {
  result->status = RDC_DIAG_RESULT_PASS;
  result->test_case = RDC_DIAG_COMPUTE_PROCESS;
}
void make_valid(rdc_diag_response_t* response) {
  for (rdc_diag_test_result_t& result : response->diag_info) {
    make_valid(&result);
  }
}
void make_valid(rdc_policy_t* policy) {
  policy->condition.type = RDC_POLICY_COND_THERMAL;
  policy->action = RDC_POLICY_ACTION_NONE;
}
void make_valid(rdc_health_response_t* response) {
  response->overall_health = RDC_HEALTH_RESULT_PASS;
  for (rdc_health_incidents_t& incident : response->incidents) {
    incident.component = RDC_HEALTH_WATCH_PCIE;
    incident.health = RDC_HEALTH_RESULT_PASS;
  }
}
void make_valid(rdc_device_topology_t* topology) {
  for (rdc_topology_link_info_t& link : topology->link_infos) {
    link.link_type = RDC_IOLINK_TYPE_PCIEXPRESS;
    link.is_p2p_accessible = true;
  }
}
void make_valid(rdc_link_status_t* status) {
  for (rdc_gpu_link_status_t& gpu : status->gpus) {
    gpu.link_types = RDC_IOLINK_TYPE_XGMI;
    for (rdc_link_state_t& state : gpu.link_states) {
      state = RDC_LINK_STATE_UP;
    }
  }
}

template <typename T>
rdc_status_t fill(T* out) {
  ++g_calls;
  if (g_fill == Fill::kOverfull) {
    std::memset(static_cast<void*>(out), 'A', sizeof(*out));
    make_valid(out);
  }
  return RDC_ST_OK;
}

rdc_status_t fill_field(rdc_field_value* value) {
  fill(value);
  if (!g_field_str.empty()) {
    value->type = STRING;
    std::memset(value->value.str, 0, sizeof(value->value.str));
    g_field_str.copy(value->value.str, sizeof(value->value.str) - 1);
  }
  return RDC_ST_OK;
}

rdc_status_t job_call(const char* job_id) {
  ++g_calls;
  g_job_id = job_id;
  return RDC_ST_OK;
}

}  // namespace

extern "C" {

rdc_status_t rdc_init(uint64_t) { return RDC_ST_OK; }
rdc_status_t rdc_shutdown() { return RDC_ST_OK; }
rdc_status_t rdc_start_embedded(rdc_operation_mode_t, rdc_handle_t* handle) {
  *handle = nullptr;
  return RDC_ST_OK;
}
rdc_status_t rdc_stop_embedded(rdc_handle_t) { return RDC_ST_OK; }
rdc_status_t rdc_job_start_stats(rdc_handle_t, rdc_gpu_group_t, const char job_id[64], uint64_t) {
  return job_call(job_id);
}
rdc_status_t rdc_job_get_stats(rdc_handle_t, const char job_id[64], rdc_job_info_t* info) {
  g_job_id = job_id;
  return fill(info);
}
rdc_status_t rdc_job_stop_stats(rdc_handle_t, const char job_id[64]) { return job_call(job_id); }
rdc_status_t rdc_job_remove(rdc_handle_t, const char job_id[64]) { return job_call(job_id); }
rdc_status_t rdc_job_remove_all(rdc_handle_t) { return RDC_ST_OK; }
rdc_status_t rdc_field_update_all(rdc_handle_t, uint32_t) { return RDC_ST_OK; }
rdc_status_t rdc_device_get_all(rdc_handle_t, uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES],
                                uint32_t* count) {
  ++g_calls;
  if (g_fill == Fill::kOverfull) {
    std::memset(gpu_index_list, 'A', sizeof(uint32_t) * RDC_MAX_NUM_DEVICES);
    std::memset(count, 'A', sizeof(*count));
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_device_get_attributes(rdc_handle_t, uint32_t, rdc_device_attributes_t* attr) {
  return fill(attr);
}
rdc_status_t rdc_device_get_component_version(rdc_handle_t, rdc_component_t,
                                              rdc_component_version_t* version) {
  return fill(version);
}
rdc_status_t rdc_group_gpu_create(rdc_handle_t, rdc_group_type_t, const char* group_name,
                                  rdc_gpu_group_t* group_id) {
  g_name = group_name;
  *group_id = 1;
  ++g_calls;
  return RDC_ST_OK;
}
rdc_status_t rdc_group_gpu_add(rdc_handle_t, rdc_gpu_group_t, uint32_t) { return RDC_ST_OK; }
rdc_status_t rdc_group_gpu_get_info(rdc_handle_t, rdc_gpu_group_t, rdc_group_info_t* info) {
  return fill(info);
}
rdc_status_t rdc_group_get_all_ids(rdc_handle_t, rdc_gpu_group_t group_id_list[], uint32_t* count) {
  ++g_calls;
  if (g_fill == Fill::kOverfull) {
    std::memset(group_id_list, 'A', sizeof(rdc_gpu_group_t) * RDC_MAX_NUM_GROUPS);
    std::memset(count, 'A', sizeof(*count));
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_group_gpu_destroy(rdc_handle_t, rdc_gpu_group_t) { return RDC_ST_OK; }
rdc_status_t rdc_group_field_create(rdc_handle_t, uint32_t, rdc_field_t*,
                                    const char* field_group_name, rdc_field_grp_t* field_group_id) {
  g_name = field_group_name;
  *field_group_id = 1;
  ++g_calls;
  return RDC_ST_OK;
}
rdc_status_t rdc_group_field_add_field(rdc_handle_t, rdc_field_grp_t, rdc_field_t) {
  return RDC_ST_OK;
}
rdc_status_t rdc_group_field_get_info(rdc_handle_t, rdc_field_grp_t, rdc_field_group_info_t* info) {
  return fill(info);
}
rdc_status_t rdc_group_field_get_all_ids(rdc_handle_t, rdc_field_grp_t field_group_id_list[],
                                         uint32_t* count) {
  ++g_calls;
  if (g_fill == Fill::kOverfull) {
    std::memset(field_group_id_list, 'A', sizeof(rdc_field_grp_t) * RDC_MAX_NUM_FIELD_GROUPS);
    std::memset(count, 'A', sizeof(*count));
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_group_field_destroy(rdc_handle_t, rdc_field_grp_t) { return RDC_ST_OK; }
rdc_status_t rdc_field_watch(rdc_handle_t, rdc_gpu_group_t, rdc_field_grp_t, uint64_t, double,
                             uint32_t) {
  return RDC_ST_OK;
}
rdc_status_t rdc_field_get_latest_value(rdc_handle_t, uint32_t, rdc_field_t,
                                        rdc_field_value* value) {
  return fill_field(value);
}
rdc_status_t rdc_field_get_value_since(rdc_handle_t, uint32_t, rdc_field_t, uint64_t,
                                       uint64_t* next_since_time_stamp, rdc_field_value* value) {
  if (g_fill == Fill::kOverfull) {
    std::memset(next_since_time_stamp, 'A', sizeof(*next_since_time_stamp));
  }
  return fill_field(value);
}
rdc_status_t rdc_field_unwatch(rdc_handle_t, rdc_gpu_group_t, rdc_field_grp_t) { return RDC_ST_OK; }
rdc_status_t rdc_diagnostic_run(rdc_handle_t, rdc_gpu_group_t, rdc_diag_level_t, const char*,
                                size_t, rdc_diag_response_t* response, rdc_diag_callback_t*) {
  fill(response);
  if (g_fill == Fill::kOverfull) {
    response->results_count = 1;  // a count rdcd rejects otherwise
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_test_case_run(rdc_handle_t, rdc_gpu_group_t, rdc_diag_test_cases_t, const char*,
                               size_t, rdc_diag_test_result_t* result, rdc_diag_callback_t*) {
  return fill(result);
}
const char* rdc_status_string(rdc_status_t) { return "fake"; }
rdc_status_t rdc_policy_set(rdc_handle_t, rdc_gpu_group_t, rdc_policy_t) { return RDC_ST_OK; }
rdc_status_t rdc_policy_get(rdc_handle_t, rdc_gpu_group_t, uint32_t* count,
                            rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) {
  ++g_calls;
  if (g_fill == Fill::kOverfull) {
    std::memset(static_cast<void*>(policies), 'A', sizeof(rdc_policy_t) * RDC_MAX_POLICY_SETTINGS);
    for (uint32_t i = 0; i < RDC_MAX_POLICY_SETTINGS; ++i) {
      make_valid(&policies[i]);
    }
    std::memset(count, 'A', sizeof(*count));
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_policy_delete(rdc_handle_t, rdc_gpu_group_t, rdc_policy_condition_type_t) {
  return RDC_ST_OK;
}
rdc_status_t rdc_policy_register(rdc_handle_t, rdc_gpu_group_t, rdc_policy_register_callback) {
  return RDC_ST_OK;
}
rdc_status_t rdc_policy_unregister(rdc_handle_t, rdc_gpu_group_t) { return RDC_ST_OK; }
rdc_status_t rdc_health_set(rdc_handle_t, rdc_gpu_group_t, unsigned int) { return RDC_ST_OK; }
rdc_status_t rdc_health_get(rdc_handle_t, rdc_gpu_group_t, unsigned int* components) {
  return fill(components);
}
rdc_status_t rdc_health_check(rdc_handle_t, rdc_gpu_group_t, rdc_health_response_t* response) {
  return fill(response);
}
rdc_status_t rdc_health_clear(rdc_handle_t, rdc_gpu_group_t) { return RDC_ST_OK; }
rdc_status_t rdc_device_topology_get(rdc_handle_t, uint32_t, rdc_device_topology_t* results) {
  return fill(results);
}
rdc_status_t rdc_link_status_get(rdc_handle_t, rdc_link_status_t* results) {
  fill(results);
  if (g_negative_links) {
    results->num_of_gpus = -1;
  }
  return RDC_ST_OK;
}
rdc_status_t rdc_config_set(rdc_handle_t, rdc_gpu_group_t, rdc_config_setting_t) {
  return RDC_ST_OK;
}
rdc_status_t rdc_config_get(rdc_handle_t, rdc_gpu_group_t, rdc_config_setting_list_t* settings) {
  return fill(settings);
}
rdc_status_t rdc_config_clear(rdc_handle_t, rdc_gpu_group_t) { return RDC_ST_OK; }
rdc_status_t rdc_instance_profile_get(rdc_handle_t, uint32_t, rdc_instance_resource_type_t,
                                      rdc_resource_profile_t* profile) {
  return fill(profile);
}
rdc_status_t rdc_get_num_partition(rdc_handle_t, uint32_t, uint16_t* num_partition) {
  return fill(num_partition);
}

}  // extern "C"

namespace {

// Larger than the output of every handler checked after dirty_stack(), the
// largest being CheckHealth's rdc_health_response_t (about 257 KB). The
// diagnostic handlers, whose rdc_diag_response_t is about 7 MB, are checked
// through a server instead.
constexpr size_t kDirt = 1 << 20;
static_assert(std::max({sizeof(uint32_t) * RDC_MAX_NUM_DEVICES, sizeof(rdc_device_attributes_t),
                        sizeof(rdc_component_version_t), sizeof(rdc_group_info_t),
                        sizeof(rdc_gpu_group_t) * RDC_MAX_NUM_GROUPS,
                        sizeof(rdc_field_grp_t) * RDC_MAX_NUM_FIELD_GROUPS,
                        sizeof(rdc_field_group_info_t), sizeof(rdc_field_value) + sizeof(uint64_t),
                        sizeof(rdc_job_info_t), sizeof(rdc_policy_t) * RDC_MAX_POLICY_SETTINGS,
                        sizeof(unsigned int), sizeof(rdc_health_response_t),
                        sizeof(rdc_device_topology_t), sizeof(rdc_link_status_t)}) < kDirt,
              "dirty_stack() covers the outputs of the handlers it precedes");

// Fills the stack below the caller's frame with 0xA5, where the frame of the
// caller's next call goes. Not instrumented by AddressSanitizer, which would put
// a redzone between the array and the top of the frame, and so leave the top of
// the next frame clean.
__attribute__((noinline, no_sanitize_address)) void dirty_stack() {
  unsigned char dirt[kDirt];
  std::memset(dirt, 0xA5, sizeof(dirt));
  asm volatile("" : : "r"(dirt) : "memory");
}

// Whether an uninitialized local, in a frame instrumented like the handlers',
// shows the 0xA5 of dirty_stack(): if not, this build places frames
// differently, and a check after dirty_stack() proves nothing.
__attribute__((noinline)) bool shows_dirt() {
  unsigned char local[4096];
  asm volatile("" : : "r"(local) : "memory");
  return local[0] == 0xA5 && local[sizeof(local) - 1] == 0xA5;
}

// Runs a check in a child process: before the fix, a reply could read past an
// array and crash. The child fails on its own failures only, not on those the
// test had before.
void in_child(const std::function<void()>& check) {
  const ::testing::TestResult& result =
      *::testing::UnitTest::GetInstance()->current_test_info()->result();
  const int parts_before = result.total_part_count();
  fflush(nullptr);
  const pid_t pid = fork();
  ASSERT_NE(pid, -1);
  if (pid == 0) {
    check();
    bool failed = false;
    for (int i = parts_before; i < result.total_part_count(); ++i) {
      failed = failed || result.GetTestPartResult(i).failed();
    }
    fflush(nullptr);
    _exit(failed ? 1 : 0);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0)
      << (WIFSIGNALED(status) ? "killed by signal " + std::to_string(WTERMSIG(status))
                              : "exit status " + std::to_string(WEXITSTATUS(status)));
}

// A handler called directly, the way gRPC calls it (with a context of no call).
template <typename Request, typename Reply>
using Handler = ::grpc::Status (amd::rdc::RdcAPIServiceImpl::*)(::grpc::ServerContext*,
                                                                const Request*, Reply*);

// Calls a handler on a dirty stack while the library writes nothing; then
// checks the reply.
template <typename Request, typename Reply>
void unfilled(Handler<Request, Reply> handler, const Request& request,
              const std::function<void(const Reply&)>& check) {
  in_child([&] {
    g_fill = Fill::kNothing;
    amd::rdc::RdcAPIServiceImpl service;
    ::grpc::ServerContext context;
    Reply reply;
    dirty_stack();
    ASSERT_TRUE(shows_dirt()) << "an uninitialized local does not show the stack's dirt";
    dirty_stack();
    ASSERT_TRUE((service.*handler)(&context, &request, &reply).ok());
    EXPECT_EQ(reply.status(), RDC_ST_OK);
    check(reply);
  });
}

// Calls a handler on a dirty stack while the library fills every byte of the
// outputs with 'A' (so a string read past its buffer runs on into the dirt);
// then checks the reply.
template <typename Request, typename Reply>
void overfull(Handler<Request, Reply> handler, const Request& request,
              const std::function<void(const Reply&)>& check) {
  in_child([&] {
    g_fill = Fill::kOverfull;
    amd::rdc::RdcAPIServiceImpl service;
    ::grpc::ServerContext context;
    Reply reply;
    dirty_stack();
    ASSERT_TRUE((service.*handler)(&context, &request, &reply).ok());
    EXPECT_EQ(reply.status(), RDC_ST_OK);
    check(reply);
  });
}

// A request for a field rdcd knows, so that a handler that checks the field ID before it calls
// the library still calls it.
template <typename Request>
Request a_field_request() {
  Request request;
  request.set_field_id(RDC_FI_GPU_CLOCK);
  return request;
}

std::string all_a(size_t length) { return std::string(length, 'A'); }

using amd::rdc::RdcAPIServiceImpl;

TEST(ServerReplies, AnUninitializedLocalShowsTheStacksDirt) {
  // The positive control of the unfilled checks below.
  dirty_stack();
  EXPECT_TRUE(shows_dirt());
}

TEST(ServerReplies, DevicesStayWithinTheirList) {
  overfull<::rdc::Empty, ::rdc::GetAllDevicesResponse>(
      &RdcAPIServiceImpl::GetAllDevices, {},
      [](const auto& reply) { EXPECT_EQ(reply.gpus_size(), RDC_MAX_NUM_DEVICES); });
}

TEST(ServerReplies, DeviceNameAndVersionStayWithinTheirBuffers) {
  overfull<::rdc::GetDeviceAttributesRequest, ::rdc::GetDeviceAttributesResponse>(
      &RdcAPIServiceImpl::GetDeviceAttributes, {}, [](const auto& reply) {
        EXPECT_EQ(reply.attributes().device_name(),
                  all_a(sizeof(rdc_device_attributes_t::device_name)));
      });
  overfull<::rdc::GetComponentVersionRequest, ::rdc::GetComponentVersionResponse>(
      &RdcAPIServiceImpl::GetComponentVersion, {},
      [](const auto& reply) { EXPECT_EQ(reply.version(), all_a(RDC_MAX_VERSION_STR_LENGTH)); });
}

TEST(ServerReplies, GpuGroupInfoIsEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetGpuGroupInfoRequest, ::rdc::GetGpuGroupInfoResponse>(
      &RdcAPIServiceImpl::GetGpuGroupInfo, {}, [](const auto& reply) {
        EXPECT_EQ(reply.group_name(), "");
        EXPECT_EQ(reply.entity_ids_size(), 0);
      });
}

TEST(ServerReplies, GpuGroupInfoStaysWithinItsArrays) {
  overfull<::rdc::GetGpuGroupInfoRequest, ::rdc::GetGpuGroupInfoResponse>(
      &RdcAPIServiceImpl::GetGpuGroupInfo, {}, [](const auto& reply) {
        EXPECT_EQ(reply.group_name(), all_a(RDC_MAX_STR_LENGTH));
        EXPECT_EQ(reply.entity_ids_size(), RDC_GROUP_MAX_ENTITIES);
      });
}

TEST(ServerReplies, GroupListsStayWithinTheirArrays) {
  overfull<::rdc::Empty, ::rdc::GetGroupAllIdsResponse>(
      &RdcAPIServiceImpl::GetGroupAllIds, {},
      [](const auto& reply) { EXPECT_EQ(reply.group_ids_size(), RDC_MAX_NUM_GROUPS); });
  overfull<::rdc::Empty, ::rdc::GetFieldGroupAllIdsResponse>(
      &RdcAPIServiceImpl::GetFieldGroupAllIds, {},
      [](const auto& reply) { EXPECT_EQ(reply.field_group_ids_size(), RDC_MAX_NUM_FIELD_GROUPS); });
}

TEST(ServerReplies, FieldGroupInfoIsEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetFieldGroupInfoRequest, ::rdc::GetFieldGroupInfoResponse>(
      &RdcAPIServiceImpl::GetFieldGroupInfo, {}, [](const auto& reply) {
        EXPECT_EQ(reply.filed_group_name(), "");
        EXPECT_EQ(reply.field_ids_size(), 0);
      });
}

TEST(ServerReplies, FieldGroupInfoStaysWithinItsArrays) {
  overfull<::rdc::GetFieldGroupInfoRequest, ::rdc::GetFieldGroupInfoResponse>(
      &RdcAPIServiceImpl::GetFieldGroupInfo, {}, [](const auto& reply) {
        EXPECT_EQ(reply.filed_group_name(), all_a(RDC_MAX_STR_LENGTH));
        EXPECT_EQ(reply.field_ids_size(), RDC_MAX_FIELD_IDS_PER_FIELD_GROUP);
      });
}

TEST(ServerReplies, FieldValuesAreEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetLatestFieldValueRequest, ::rdc::GetLatestFieldValueResponse>(
      &RdcAPIServiceImpl::GetLatestFieldValue, a_field_request<::rdc::GetLatestFieldValueRequest>(),
      [](const auto& reply) {
        EXPECT_EQ(reply.rdc_status(), RDC_ST_OK);
        EXPECT_EQ(reply.ts(), 0u);
        EXPECT_EQ(reply.field_id(), 0u);
        EXPECT_EQ(reply.l_int(), 0);
      });
  unfilled<::rdc::GetFieldSinceRequest, ::rdc::GetFieldSinceResponse>(
      &RdcAPIServiceImpl::GetFieldSince, a_field_request<::rdc::GetFieldSinceRequest>(),
      [](const auto& reply) {
        EXPECT_EQ(reply.rdc_status(), RDC_ST_OK);
        EXPECT_EQ(reply.next_since_time_stamp(), 0u);
        EXPECT_EQ(reply.ts(), 0u);
        EXPECT_EQ(reply.l_int(), 0);
      });
}

TEST(ServerReplies, FieldStringsStayWithinTheirBuffers) {
  overfull<::rdc::GetLatestFieldValueRequest, ::rdc::GetLatestFieldValueResponse>(
      &RdcAPIServiceImpl::GetLatestFieldValue, a_field_request<::rdc::GetLatestFieldValueRequest>(),
      [](const auto& reply) { EXPECT_EQ(reply.str(), all_a(RDC_MAX_STR_LENGTH)); });
  overfull<::rdc::GetFieldSinceRequest, ::rdc::GetFieldSinceResponse>(
      &RdcAPIServiceImpl::GetFieldSince, a_field_request<::rdc::GetFieldSinceRequest>(),
      [](const auto& reply) { EXPECT_EQ(reply.str(), all_a(RDC_MAX_STR_LENGTH)); });
}

TEST(ServerReplies, FieldStringsSinceATimeLoseTheirTrailingSpaces) {
  const std::pair<std::string, std::string> cases[] = {
      {"abc   ", "abc"}, {"a b", "a b"}, {"    ", ""}, {"x", "x"}};
  for (const auto& c : cases) {
    const std::string& value = c.first;
    const std::string& sent = c.second;
    g_field_str = value;
    in_child([&] {
      g_fill = Fill::kNothing;
      RdcAPIServiceImpl service;
      ::grpc::ServerContext context;
      const auto request = a_field_request<::rdc::GetFieldSinceRequest>();
      ::rdc::GetFieldSinceResponse reply;
      ASSERT_TRUE(service.GetFieldSince(&context, &request, &reply).ok());
      EXPECT_EQ(reply.str(), sent) << "for \"" << value << "\"";
      EXPECT_EQ(reply.str().find('\0'), std::string::npos);
    });
  }
  g_field_str.clear();
}

TEST(ServerReplies, JobStatsAreEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetJobStatsRequest, ::rdc::GetJobStatsResponse>(
      &RdcAPIServiceImpl::GetJobStats, {}, [](const auto& reply) {
        EXPECT_EQ(reply.num_gpus(), 0u);
        EXPECT_EQ(reply.num_processes(), 0u);
        EXPECT_EQ(reply.summary().start_time(), 0u);
      });
}

TEST(ServerReplies, ProcessNamesStayWithinTheirBuffers) {
  overfull<::rdc::GetJobStatsRequest, ::rdc::GetJobStatsResponse>(
      &RdcAPIServiceImpl::GetJobStats, {}, [](const auto& reply) {
        ASSERT_EQ(reply.processes_size(), RDC_MAX_NUM_PROCESSES_STATUS);
        EXPECT_EQ(reply.processes(0).process_name(), all_a(MAX_PROCESS_NAME));
      });
}

TEST(ServerReplies, PoliciesStayWithinTheirArray) {
  overfull<::rdc::GetPolicyRequest, ::rdc::GetPolicyResponse>(
      &RdcAPIServiceImpl::GetPolicy, {}, [](const auto& reply) {
        EXPECT_EQ(reply.response().count(), static_cast<uint32_t>(RDC_MAX_POLICY_SETTINGS));
        EXPECT_EQ(reply.response().policies_size(), RDC_MAX_POLICY_SETTINGS);
      });
}

TEST(ServerReplies, HealthIsEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetHealthRequest, ::rdc::GetHealthResponse>(
      &RdcAPIServiceImpl::GetHealth, {},
      [](const auto& reply) { EXPECT_EQ(reply.components(), 0u); });
  unfilled<::rdc::CheckHealthRequest, ::rdc::CheckHealthResponse>(
      &RdcAPIServiceImpl::CheckHealth, {}, [](const auto& reply) {
        EXPECT_EQ(reply.response().overall_health(), 0u);
        EXPECT_EQ(reply.response().incidents_count(), 0u);
        EXPECT_EQ(reply.response().incidents_size(), 0);
      });
}

TEST(ServerReplies, HealthIncidentsStayWithinTheirArrays) {
  overfull<::rdc::CheckHealthRequest, ::rdc::CheckHealthResponse>(
      &RdcAPIServiceImpl::CheckHealth, {}, [](const auto& reply) {
        EXPECT_EQ(reply.response().incidents_count(),
                  static_cast<uint32_t>(HEALTH_MAX_ERROR_ITEMS));
        ASSERT_EQ(reply.response().incidents_size(), HEALTH_MAX_ERROR_ITEMS);
        EXPECT_EQ(reply.response().incidents(0).error().msg(), all_a(MAX_HEALTH_MSG_LENGTH));
      });
}

TEST(ServerReplies, TopologyAndLinksAreEmptyWhenTheLibraryWritesNothing) {
  unfilled<::rdc::GetTopologyRequest, ::rdc::GetTopologyResponse>(
      &RdcAPIServiceImpl::GetTopology, {}, [](const auto& reply) {
        EXPECT_EQ(reply.toppology().num_of_gpus(), 0u);
        EXPECT_EQ(reply.toppology().link_infos_size(), 0);
      });
  unfilled<::rdc::Empty, ::rdc::GetLinkStatusResponse>(
      &RdcAPIServiceImpl::GetLinkStatus, {}, [](const auto& reply) {
        EXPECT_EQ(reply.linkstatus().num_of_gpus(), 0u);
        EXPECT_EQ(reply.linkstatus().gpus_size(), 0);
      });
}

TEST(ServerReplies, TopologyAndLinksStayWithinTheirArrays) {
  overfull<::rdc::GetTopologyRequest, ::rdc::GetTopologyResponse>(
      &RdcAPIServiceImpl::GetTopology, {}, [](const auto& reply) {
        EXPECT_EQ(reply.toppology().num_of_gpus(), static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
        EXPECT_EQ(reply.toppology().link_infos_size(), RDC_MAX_NUM_DEVICES);
      });
  overfull<::rdc::Empty, ::rdc::GetLinkStatusResponse>(
      &RdcAPIServiceImpl::GetLinkStatus, {}, [](const auto& reply) {
        const auto& links = reply.linkstatus();
        EXPECT_EQ(links.num_of_gpus(), static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
        ASSERT_EQ(links.gpus_size(), RDC_MAX_NUM_DEVICES);
        EXPECT_EQ(links.gpus(0).num_of_links(), static_cast<uint32_t>(RDC_MAX_NUM_OF_LINKS));
        EXPECT_EQ(links.gpus(0).link_states_size(), RDC_MAX_NUM_OF_LINKS);
      });
}

TEST(ServerReplies, LinksOfANegativeGpuCountAreNone) {
  // rdc_link_status_t::num_of_gpus is an int32_t.
  g_negative_links = true;
  overfull<::rdc::Empty, ::rdc::GetLinkStatusResponse>(
      &RdcAPIServiceImpl::GetLinkStatus, {}, [](const auto& reply) {
        EXPECT_EQ(reply.linkstatus().num_of_gpus(), 0u);
        EXPECT_EQ(reply.linkstatus().gpus_size(), 0);
      });
  g_negative_links = false;
}

// Job IDs the C API takes (char job_id[64]) and names that fit RDC's buffers
// reach the library; others fail with RDC_ST_BAD_PARAMETER before it.
TEST(ServerReplies, JobCallsTakeOnlyJobIdsThatFitTheCApi) {
  const std::string longest(63, 'j');
  const std::string rejected[] = {std::string(64, 'j'), "a\nb", "tab\there",
                                  std::string("nul\0in", 6), "del\x7f"};
  auto run = [&](auto handler, auto request, auto reply) {
    for (const std::string& job_id : rejected) {
      in_child([&] {
        RdcAPIServiceImpl service;
        ::grpc::ServerContext context;
        auto req = request;
        auto rep = reply;
        req.set_job_id(job_id);
        ASSERT_TRUE((service.*handler)(&context, &req, &rep).ok());
        EXPECT_EQ(rep.status(), RDC_ST_BAD_PARAMETER);
        EXPECT_EQ(g_calls, 0) << "the library got a job ID of " << job_id.size() << " bytes";
      });
    }
    in_child([&] {
      RdcAPIServiceImpl service;
      ::grpc::ServerContext context;
      auto req = request;
      auto rep = reply;
      req.set_job_id(longest);
      ASSERT_TRUE((service.*handler)(&context, &req, &rep).ok());
      EXPECT_EQ(rep.status(), RDC_ST_OK);
      EXPECT_EQ(g_job_id, longest);
    });
  };
  run(&RdcAPIServiceImpl::StartJobStats, ::rdc::StartJobStatsRequest{},
      ::rdc::StartJobStatsResponse{});
  run(&RdcAPIServiceImpl::GetJobStats, ::rdc::GetJobStatsRequest{}, ::rdc::GetJobStatsResponse{});
  run(&RdcAPIServiceImpl::StopJobStats, ::rdc::StopJobStatsRequest{},
      ::rdc::StopJobStatsResponse{});
  run(&RdcAPIServiceImpl::RemoveJob, ::rdc::RemoveJobRequest{}, ::rdc::RemoveJobResponse{});
}

TEST(ServerReplies, GroupsTakeOnlyNamesThatFitRdcsBuffers) {
  const std::string longest(RDC_MAX_STR_LENGTH - 1, 'g');
  const std::string rejected[] = {std::string(RDC_MAX_STR_LENGTH, 'g'), std::string("nul\0in", 6)};
  for (const std::string& name : rejected) {
    in_child([&] {
      RdcAPIServiceImpl service;
      ::grpc::ServerContext context;
      ::rdc::CreateGpuGroupRequest gpu_request;
      ::rdc::CreateGpuGroupResponse gpu_reply;
      gpu_request.set_group_name(name);
      ASSERT_TRUE(service.CreateGpuGroup(&context, &gpu_request, &gpu_reply).ok());
      EXPECT_EQ(gpu_reply.status(), RDC_ST_BAD_PARAMETER);
      ::rdc::CreateFieldGroupRequest field_request;
      ::rdc::CreateFieldGroupResponse field_reply;
      field_request.set_field_group_name(name);
      field_request.add_field_ids(RDC_FI_GPU_CLOCK);
      ASSERT_TRUE(service.CreateFieldGroup(&context, &field_request, &field_reply).ok());
      EXPECT_EQ(field_reply.status(), RDC_ST_BAD_PARAMETER);
      EXPECT_EQ(g_calls, 0) << "the library got a name of " << name.size() << " bytes";
    });
  }
  in_child([&] {
    RdcAPIServiceImpl service;
    ::grpc::ServerContext context;
    ::rdc::CreateGpuGroupRequest gpu_request;
    ::rdc::CreateGpuGroupResponse gpu_reply;
    gpu_request.set_group_name(longest);
    ASSERT_TRUE(service.CreateGpuGroup(&context, &gpu_request, &gpu_reply).ok());
    EXPECT_EQ(gpu_reply.status(), RDC_ST_OK);
    EXPECT_EQ(g_name, longest);
    ::rdc::CreateFieldGroupRequest field_request;
    ::rdc::CreateFieldGroupResponse field_reply;
    field_request.set_field_group_name(longest);
    field_request.add_field_ids(RDC_FI_GPU_CLOCK);
    ASSERT_TRUE(service.CreateFieldGroup(&context, &field_request, &field_reply).ok());
    EXPECT_EQ(field_reply.status(), RDC_ST_OK);
    EXPECT_EQ(g_name, longest);
  });
}

// The diagnostic replies are streams, which only a server can write: the
// service runs on an abstract Unix socket in the child, named for the process
// and a random number (abstract names are shared by a network namespace).
class Server {
 public:
  Server() {
    const std::string address = "unix-abstract:rdc_server_replies_" + std::to_string(getpid()) +
                                "_" + std::to_string(std::random_device{}());
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    stub_ = ::rdc::RdcAPI::NewStub(
        ::grpc::CreateChannel(address, ::grpc::InsecureChannelCredentials()));
  }
  ~Server() {
    if (server_) {
      server_->Shutdown();
    }
  }
  bool started() const { return server_ != nullptr; }
  ::rdc::RdcAPI::Stub& stub() { return *stub_; }

 private:
  RdcAPIServiceImpl service_;
  std::unique_ptr<::grpc::Server> server_;
  std::unique_ptr<::rdc::RdcAPI::Stub> stub_;
};

// A call that fails rather than waits when the server does not answer.
void set_deadline(::grpc::ClientContext* context) {
  context->set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(60));
}

void expect_bounded(const ::rdc::DiagnosticTestResult& result) {
  EXPECT_EQ(result.details().msg(), all_a(MAX_DIAG_MSG_LENGTH));
  EXPECT_EQ(result.info(), all_a(MAX_DIAG_MSG_LENGTH));
  EXPECT_EQ(result.per_gpu_result_count(), static_cast<uint32_t>(RDC_MAX_NUM_DEVICES));
  ASSERT_EQ(result.gpu_results_size(), RDC_MAX_NUM_DEVICES);
  EXPECT_EQ(result.gpu_results(0).gpu_result().msg(), all_a(MAX_DIAG_MSG_LENGTH));
}

TEST(ServerReplies, DiagnosticResultsStayWithinTheirArrays) {
  in_child([] {
    g_fill = Fill::kOverfull;
    Server server;
    ASSERT_TRUE(server.started());
    {
      ::grpc::ClientContext context;
      set_deadline(&context);
      ::rdc::DiagnosticRunRequest request;
      auto reader = server.stub().DiagnosticRun(&context, request);
      ::rdc::DiagnosticRunResponse reply;
      ASSERT_TRUE(reader->Read(&reply));
      EXPECT_EQ(reply.status(), RDC_ST_OK);
      ASSERT_EQ(reply.response().diag_info_size(), 1);
      expect_bounded(reply.response().diag_info(0));
      while (reader->Read(&reply)) {
      }
      EXPECT_TRUE(reader->Finish().ok());
    }
    {
      ::grpc::ClientContext context;
      set_deadline(&context);
      ::rdc::DiagnosticTestCaseRunRequest request;
      auto reader = server.stub().DiagnosticTestCaseRun(&context, request);
      ::rdc::DiagnosticTestCaseRunResponse reply;
      ASSERT_TRUE(reader->Read(&reply));
      EXPECT_EQ(reply.status(), RDC_ST_OK);
      expect_bounded(reply.result());
      while (reader->Read(&reply)) {
      }
      EXPECT_TRUE(reader->Finish().ok());
    }
  });
}

}  // namespace
