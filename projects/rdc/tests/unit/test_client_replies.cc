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

// The standalone client against a server whose replies state more entries than they
// carry, or values the C API cannot hold: the client must count what the reply carries,
// and reject the rest without writing the caller's output. Also: the deadlines the
// client gives rdcd.

#include <google/protobuf/descriptor.h>
#include <grpcpp/generic/callback_generic_service.h>
#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <array>
#include <chrono>  // NOLINT
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <vector>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"
#include "rdc/rdc_private.h"

namespace {

constexpr uint32_t kGpuId = 7;
constexpr uint32_t kPid = 42;
constexpr uint64_t kValue = 4242;
constexpr char kString[] = "value";
constexpr rdc_field_t kValueField = RDC_FI_GPU_CLOCK;
constexpr uint32_t kValueStatus = RDC_ST_NO_DATA;
constexpr uint64_t kValueTs = 1700000000123;
constexpr uint64_t kNextSince = 1234;

// What the stream of a diagnostic or test case run carries. (Not named Run: in a TEST_F,
// Run is testing::Test::Run().)
enum class RunStream {
  kResults,    // the results
  kNoResults,  // a log line (diagnostic), or a reply with status OK and no result (test case)
  kNothing,    // no reply, as an older rdcd sends for a run that it cannot do
};

// What the service sends. Keep group_ids and processes first: some tests set them by position.
struct Replies {
  int group_ids = 0;               // ids in each group and field group list
  int processes = 1;               // processes of the job
  int devices = 1;                 // GPUs in the device list
  int incidents = 1;               // health incidents
  int topology_gpus = 1;           // GPUs in the topology
  int link_gpus = 1;               // GPUs in the link status
  int link_states = 2;             // link states of each GPU in the link status
  int last_gpu_link_states = -1;   // link states of the last GPU instead, if not -1
  int test_results = 1;            // test results of a diagnostic
  int gpu_results = 1;             // per-GPU results of each diagnostic test result
  int last_test_gpu_results = -1;  // per-GPU results of the last test result instead, if not -1
  RunStream run = RunStream::kResults;  // what each diagnostic and test case run carries
  uint32_t value_type = 0;              // type of each field value (INTEGER)
  uint32_t num_partition = 8;           // partitions of every GPU
  uint32_t value_field = kValueField;   // field of each field value
  std::vector<uint32_t> group_fields = {kValueField};  // fields of the field group
};

// Whether every byte of an output is the 0xA5 the caller filled it with.
template <typename T>
bool untouched(const T& output) {
  const unsigned char* bytes = reinterpret_cast<const unsigned char*>(&output);
  for (size_t i = 0; i < sizeof(output); ++i) {
    if (bytes[i] != 0xA5) return false;
  }
  return true;
}

// A caller's output as an earlier call could have left it.
template <typename T>
std::unique_ptr<T> dirty() {
  auto output = std::make_unique<T>();
  memset(static_cast<void*>(output.get()), 0xA5, sizeof(T));
  return output;
}

// Replies that claim more GPUs, processes and policies than they carry.
class OverstatingService final : public ::rdc::RdcAPI::Service {
 public:
  explicit OverstatingService(Replies replies) : replies_(replies) {}

  ::grpc::Status GetJobStats(::grpc::ServerContext*, const ::rdc::GetJobStatsRequest*,
                             ::rdc::GetJobStatsResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    reply->set_num_gpus(RDC_MAX_NUM_GPUS_PER_JOB);
    reply->add_gpus()->set_gpu_id(kGpuId);
    reply->set_num_processes(RDC_MAX_NUM_PROCESSES_STATUS);
    for (int i = 0; i < replies_.processes; i++) {
      ::rdc::RdcProcessStatsInfo* process = reply->add_processes();
      process->set_pid(kPid + i);
      process->set_process_name("replied");
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetPolicy(::grpc::ServerContext*, const ::rdc::GetPolicyRequest*,
                           ::rdc::GetPolicyResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    ::rdc::PolicyResponse* response = reply->mutable_response();
    response->set_count(RDC_MAX_POLICY_SETTINGS);
    ::rdc::Policy* policy = response->add_policies();
    policy->mutable_condition()->set_type(::rdc::PolicyCondition_Type_COND_THERMAL);
    policy->mutable_condition()->set_value(90);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetGroupAllIds(::grpc::ServerContext*, const ::rdc::Empty*,
                                ::rdc::GetGroupAllIdsResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    for (int i = 0; i < replies_.group_ids; i++) {
      reply->add_group_ids(i);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetFieldGroupAllIds(::grpc::ServerContext*, const ::rdc::Empty*,
                                     ::rdc::GetFieldGroupAllIdsResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    for (int i = 0; i < replies_.group_ids; i++) {
      reply->add_field_group_ids(i);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetAllDevices(::grpc::ServerContext*, const ::rdc::Empty*,
                               ::rdc::GetAllDevicesResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    for (int i = 0; i < replies_.devices; i++) {
      reply->add_gpus(kGpuId);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetTopology(::grpc::ServerContext*, const ::rdc::GetTopologyRequest*,
                             ::rdc::GetTopologyResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    ::rdc::Topology* topology = reply->mutable_toppology();
    topology->set_num_of_gpus(RDC_MAX_NUM_DEVICES);
    for (int i = 0; i < replies_.topology_gpus; i++) {
      topology->add_link_infos()->set_gpu_index(kGpuId);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetLinkStatus(::grpc::ServerContext*, const ::rdc::Empty*,
                               ::rdc::GetLinkStatusResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    ::rdc::LinkStatus* link_status = reply->mutable_linkstatus();
    link_status->set_num_of_gpus(RDC_MAX_NUM_DEVICES);
    for (int g = 0; g < replies_.link_gpus; g++) {
      ::rdc::GpuLinkStatus* gpu = link_status->add_gpus();
      gpu->set_gpu_index(kGpuId);
      gpu->set_num_of_links(RDC_MAX_NUM_OF_LINKS);
      const bool last = g == replies_.link_gpus - 1;
      const int states = last && replies_.last_gpu_link_states != -1 ? replies_.last_gpu_link_states
                                                                     : replies_.link_states;
      for (int i = 0; i < states; i++) {
        // rdcd sends the C API's numbers (rdc_api_service.cc casts them), up and down in turn.
        gpu->add_link_states(static_cast<::rdc::GpuLinkStatus_LinkState>(
            i % 2 == 0 ? RDC_LINK_STATE_UP : RDC_LINK_STATE_DOWN));
      }
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status CheckHealth(::grpc::ServerContext*, const ::rdc::CheckHealthRequest*,
                             ::rdc::CheckHealthResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    ::rdc::HealthResponse* response = reply->mutable_response();
    response->set_incidents_count(HEALTH_MAX_ERROR_ITEMS);
    for (int i = 0; i < replies_.incidents; i++) {
      response->add_incidents()->set_gpu_index(kGpuId);
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status DiagnosticRun(
      ::grpc::ServerContext*, const ::rdc::DiagnosticRunRequest*,
      ::grpc::ServerWriter<::rdc::DiagnosticRunResponse>* writer) override {
    ::rdc::DiagnosticRunResponse reply;
    if (replies_.run == RunStream::kNothing) {
      return ::grpc::Status::OK;
    }
    if (replies_.run == RunStream::kNoResults) {
      reply.set_log("a log line, and no results");
      writer->Write(reply);
      return ::grpc::Status::OK;
    }
    reply.set_status(RDC_ST_OK);
    ::rdc::DiagnosticResponse* response = reply.mutable_response();
    response->set_results_count(MAX_TEST_CASES);
    for (int i = 0; i < replies_.test_results; i++) {
      const bool last = i == replies_.test_results - 1;
      fill(response->add_diag_info(), last && replies_.last_test_gpu_results != -1
                                          ? replies_.last_test_gpu_results
                                          : replies_.gpu_results);
    }
    writer->Write(reply);
    return ::grpc::Status::OK;
  }

  ::grpc::Status DiagnosticTestCaseRun(
      ::grpc::ServerContext*, const ::rdc::DiagnosticTestCaseRunRequest*,
      ::grpc::ServerWriter<::rdc::DiagnosticTestCaseRunResponse>* writer) override {
    ::rdc::DiagnosticTestCaseRunResponse reply;
    if (replies_.run == RunStream::kNothing) {
      return ::grpc::Status::OK;
    }
    reply.set_status(RDC_ST_OK);
    if (replies_.run == RunStream::kResults) {
      fill(reply.mutable_result(), replies_.gpu_results);
    }
    writer->Write(reply);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetFieldGroupInfo(::grpc::ServerContext*, const ::rdc::GetFieldGroupInfoRequest*,
                                   ::rdc::GetFieldGroupInfoResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    reply->set_filed_group_name("fields");
    for (uint32_t field : replies_.group_fields) reply->add_field_ids(field);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetLatestFieldValue(::grpc::ServerContext*,
                                     const ::rdc::GetLatestFieldValueRequest*,
                                     ::rdc::GetLatestFieldValueResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    reply->set_type(static_cast<::rdc::GetLatestFieldValueResponse_FieldType>(replies_.value_type));
    set_value(reply);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetFieldSince(::grpc::ServerContext*, const ::rdc::GetFieldSinceRequest*,
                               ::rdc::GetFieldSinceResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    reply->set_next_since_time_stamp(kNextSince);
    reply->set_type(static_cast<::rdc::GetFieldSinceResponse_FieldType>(replies_.value_type));
    set_value(reply);
    return ::grpc::Status::OK;
  }

  ::grpc::Status GetNumPartition(::grpc::ServerContext*, const ::rdc::GetNumPartitionRequest*,
                                 ::rdc::GetNumPartitionResponse* reply) override {
    reply->set_status(RDC_ST_OK);
    reply->set_num_partition(replies_.num_partition);
    return ::grpc::Status::OK;
  }

 private:
  // Sets the reply's field, status, timestamp and value (a oneof) of the type the replies
  // state.
  template <typename Reply>
  void set_value(Reply* reply) const {
    reply->set_field_id(replies_.value_field);
    reply->set_rdc_status(kValueStatus);
    reply->set_ts(kValueTs);
    if (replies_.value_type == DOUBLE) {
      reply->set_dbl(kValue);
    } else if (replies_.value_type == STRING || replies_.value_type == BLOB) {
      reply->set_str(kString);
    } else {
      reply->set_l_int(kValue);
    }
  }

  // A test result that claims a result for every GPU, and carries gpu_results.
  void fill(::rdc::DiagnosticTestResult* result, int gpu_results) const {
    result->set_per_gpu_result_count(RDC_MAX_NUM_DEVICES);
    for (int i = 0; i < gpu_results; i++) {
      result->add_gpu_results()->set_gpu_index(kGpuId);
    }
  }

  const Replies replies_;
};

// A server for every method of rdcd that records how long the client gave each call,
// and answers UNIMPLEMENTED (the base class's reactor).
class DeadlineRecorder : public ::grpc::CallbackGenericService {
 public:
  ::grpc::ServerGenericBidiReactor* CreateReactor(
      ::grpc::GenericCallbackServerContext* context) override {
    int64_t left = -1;  // no deadline
    if (context->deadline() != std::chrono::system_clock::time_point::max()) {
      left = std::chrono::duration_cast<std::chrono::seconds>(context->deadline() -
                                                              std::chrono::system_clock::now())
                 .count();
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      time_left_[context->method()] = left;
    }
    return ::grpc::CallbackGenericService::CreateReactor(context);
  }

  // Seconds that the client gave each method it called, by method path, -1 for none.
  std::map<std::string, int64_t> time_left() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return time_left_;
  }

 private:
  mutable std::mutex mutex_;
  std::map<std::string, int64_t> time_left_;
};

// An OverstatingService, or the given generic service, on an abstract Unix socket, which
// leaves nothing behind, and an RDC client connected to it.
class Session {
 public:
  explicit Session(Replies replies = {}, ::grpc::CallbackGenericService* generic = nullptr)
      : service_(replies) {
    const std::string address = "unix-abstract:rdc_client_replies_" + std::to_string(getpid());
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, ::grpc::InsecureServerCredentials());
    if (generic != nullptr) {
      builder.RegisterCallbackGenericService(generic);
    } else {
      builder.RegisterService(&service_);
    }
    server_ = builder.BuildAndStart();
    if (!server_ || rdc_init(0) != RDC_ST_OK ||
        rdc_connect(address.c_str(), &handle_, nullptr, nullptr, nullptr) != RDC_ST_OK) {
      ADD_FAILURE() << "cannot start a server on " << address << " and connect to it";
      handle_ = nullptr;
    }
  }

  ~Session() {
    if (handle_ != nullptr) {
      rdc_disconnect(handle_);
      rdc_shutdown();
    }
    if (server_) {
      server_->Shutdown();
    }
  }

  rdc_handle_t handle() const { return handle_; }

 private:
  OverstatingService service_;
  std::unique_ptr<::grpc::Server> server_;
  rdc_handle_t handle_ = nullptr;
};

// Reading past a repeated field can crash, so each check runs in a child process that
// starts its own server. The session ends before _exit(), which skips destructors.
class RdcClientRepliesTest : public ::testing::Test {
 protected:
  void SetUp() override { GTEST_FLAG_SET(death_test_style, "threadsafe"); }
};

TEST_F(RdcClientRepliesTest, JobStatsCountWhatTheReplyCarries) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session;
          // The caller's job info, as left by an earlier call.
          auto info = std::make_unique<rdc_job_info_t>();
          memset(info.get(), 0xA5, sizeof(*info));
          rdc_status_t status = rdc_job_get_stats(session.handle(), "job", info.get());
          ok = status == RDC_ST_OK && info->num_gpus == 1 && info->gpus[0].gpu_id == kGpuId &&
               info->gpus[1].gpu_id == 0 && info->num_processes == 1 &&
               info->processes[0].pid == kPid && info->processes[1].pid == 0;
          if (!ok) {
            fprintf(stderr, "status %d, %u GPUs, %u processes\n", status, info->num_gpus,
                    info->num_processes);
          }
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(RdcClientRepliesTest, PoliciesCountWhatTheReplyCarries) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session;
          rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS] = {};
          uint32_t count = 0;
          rdc_status_t status = rdc_policy_get(session.handle(), 1, &count, policies);
          ok = status == RDC_ST_OK && count == 1 &&
               policies[0].condition.type == RDC_POLICY_COND_THERMAL &&
               policies[0].condition.value == 90;
          if (!ok) {
            fprintf(stderr, "status %d, %u policies\n", status, count);
          }
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

// RDC holds as many groups as field groups; the service lists that many of each.
static_assert(RDC_MAX_NUM_GROUPS == RDC_MAX_NUM_FIELD_GROUPS);

TEST_F(RdcClientRepliesTest, ListsAsManyGroupsAsRdcHolds) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session({RDC_MAX_NUM_GROUPS});
          rdc_gpu_group_t groups[RDC_MAX_NUM_GROUPS] = {};
          rdc_field_grp_t field_groups[RDC_MAX_NUM_FIELD_GROUPS] = {};
          uint32_t count = 0;
          uint32_t field_count = 0;
          rdc_status_t status = rdc_group_get_all_ids(session.handle(), groups, &count);
          rdc_status_t field_status =
              rdc_group_field_get_all_ids(session.handle(), field_groups, &field_count);
          ok = status == RDC_ST_OK && count == RDC_MAX_NUM_GROUPS &&
               groups[RDC_MAX_NUM_GROUPS - 1] == RDC_MAX_NUM_GROUPS - 1 &&
               field_status == RDC_ST_OK && field_count == RDC_MAX_NUM_FIELD_GROUPS;
          if (!ok) {
            fprintf(stderr, "status %d, %u groups; status %d, %u field groups\n", status, count,
                    field_status, field_count);
          }
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(RdcClientRepliesTest, RejectsMoreGroupsThanRdcHolds) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session({RDC_MAX_NUM_GROUPS + 1});
          rdc_gpu_group_t groups[RDC_MAX_NUM_GROUPS] = {};
          rdc_field_grp_t field_groups[RDC_MAX_NUM_FIELD_GROUPS] = {};
          constexpr uint32_t kUnset = 12345;  // the counts must stay as they are
          uint32_t count = kUnset;
          uint32_t field_count = kUnset;
          rdc_status_t status = rdc_group_get_all_ids(session.handle(), groups, &count);
          rdc_status_t field_status =
              rdc_group_field_get_all_ids(session.handle(), field_groups, &field_count);
          ok = status == RDC_ST_MAX_LIMIT && count == kUnset && field_status == RDC_ST_MAX_LIMIT &&
               field_count == kUnset;
          if (!ok) {
            fprintf(stderr, "status %d, count %u; status %d, count %u\n", status, count,
                    field_status, field_count);
          }
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(RdcClientRepliesTest, RejectsMoreProcessesThanAJobHolds) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session({0, RDC_MAX_NUM_PROCESSES_STATUS + 1});
          auto info = std::make_unique<rdc_job_info_t>();
          rdc_status_t status = rdc_job_get_stats(session.handle(), "job", info.get());
          ok = status == RDC_ST_BAD_PARAMETER;
          if (!ok) {
            fprintf(stderr, "status %d\n", status);
          }
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

// Runs check in a child process with a session of these replies, and passes if it
// returns true; it prints what it saw otherwise.
void expect_in_child(const Replies& replies, const std::function<bool(const Session&)>& check) {
  EXPECT_EXIT(
      {
        bool ok = false;
        {
          Session session(replies);
          ok = check(session);
        }
        _exit(ok ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

TEST_F(RdcClientRepliesTest, DeviceListRejectsMoreGpusThanItHolds) {
  Replies replies;
  replies.devices = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto gpus = dirty<std::array<uint32_t, RDC_MAX_NUM_DEVICES>>();
    auto count = dirty<uint32_t>();
    rdc_status_t status = rdc_device_get_all(session.handle(), gpus->data(), count.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*gpus) && untouched(*count);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, TopologyCountsWhatTheReplyCarries) {
  expect_in_child(Replies{}, [](const Session& session) {
    auto topology = dirty<rdc_device_topology_t>();
    rdc_status_t status = rdc_device_topology_get(session.handle(), 0, topology.get());
    bool ok = status == RDC_ST_OK && topology->num_of_gpus == 1 &&
              topology->link_infos[0].gpu_index == kGpuId && untouched(topology->link_infos[1]);
    if (!ok) fprintf(stderr, "status %d, %u GPUs\n", status, topology->num_of_gpus);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, TopologyRejectsMoreGpusThanItHolds) {
  Replies replies;
  replies.topology_gpus = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto topology = dirty<rdc_device_topology_t>();
    rdc_status_t status = rdc_device_topology_get(session.handle(), 0, topology.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*topology);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, LinkStatusCountsWhatTheReplyCarries) {
  expect_in_child(Replies{}, [](const Session& session) {
    auto links = dirty<rdc_link_status_t>();
    rdc_status_t status = rdc_link_status_get(session.handle(), links.get());
    bool ok = status == RDC_ST_OK && links->num_of_gpus == 1 &&
              links->gpus[0].gpu_index == kGpuId && links->gpus[0].num_of_links == 2 &&
              links->gpus[0].link_states[0] == RDC_LINK_STATE_UP &&
              links->gpus[0].link_states[1] == RDC_LINK_STATE_DOWN &&
              untouched(links->gpus[0].link_states[2]) && untouched(links->gpus[1]);
    if (!ok) fprintf(stderr, "status %d, %d GPUs\n", status, links->num_of_gpus);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, LinkStatusRejectsMoreLinksThanItHolds) {
  Replies replies;
  replies.link_states = RDC_MAX_NUM_OF_LINKS + 1;
  expect_in_child(replies, [](const Session& session) {
    auto links = dirty<rdc_link_status_t>();
    rdc_status_t status = rdc_link_status_get(session.handle(), links.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*links);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, LinkStatusChecksTheLinksOfEveryGpu) {
  Replies replies;
  replies.link_gpus = 2;
  replies.last_gpu_link_states = RDC_MAX_NUM_OF_LINKS + 1;
  expect_in_child(replies, [](const Session& session) {
    auto links = dirty<rdc_link_status_t>();
    rdc_status_t status = rdc_link_status_get(session.handle(), links.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*links);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, LinkStatusRejectsMoreGpusThanItHolds) {
  Replies replies;
  replies.link_gpus = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto links = dirty<rdc_link_status_t>();
    rdc_status_t status = rdc_link_status_get(session.handle(), links.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*links);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, HealthIncidentsCountWhatTheReplyCarries) {
  expect_in_child(Replies{}, [](const Session& session) {
    auto response = dirty<rdc_health_response_t>();
    rdc_status_t status = rdc_health_check(session.handle(), 1, response.get());
    bool ok = status == RDC_ST_OK && response->incidents_count == 1 &&
              response->incidents[0].gpu_index == kGpuId && untouched(response->incidents[1]);
    if (!ok) fprintf(stderr, "status %d, %u incidents\n", status, response->incidents_count);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, HealthRejectsMoreIncidentsThanItHolds) {
  Replies replies;
  replies.incidents = HEALTH_MAX_ERROR_ITEMS + 1;
  expect_in_child(replies, [](const Session& session) {
    auto response = dirty<rdc_health_response_t>();
    rdc_status_t status = rdc_health_check(session.handle(), 1, response.get());
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*response);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, DiagnosticResultsCountWhatTheReplyCarries) {
  expect_in_child(Replies{}, [](const Session& session) {
    auto response = dirty<rdc_diag_response_t>();
    rdc_status_t status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT, nullptr, 0,
                                             response.get(), nullptr);
    const rdc_diag_test_result_t& result = response->diag_info[0];
    bool ok = status == RDC_ST_OK && response->results_count == 1 &&
              result.per_gpu_result_count == 1 && result.gpu_results[0].gpu_index == kGpuId &&
              untouched(result.gpu_results[1]) && untouched(response->diag_info[1]);
    if (!ok) {
      fprintf(stderr, "status %d, %u results, %u GPU results\n", status, response->results_count,
              result.per_gpu_result_count);
    }
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, DiagnosticRejectsMoreTestResultsThanItHolds) {
  Replies replies;
  replies.test_results = MAX_TEST_CASES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto response = dirty<rdc_diag_response_t>();
    rdc_status_t status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT, nullptr, 0,
                                             response.get(), nullptr);
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*response);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, DiagnosticRejectsMoreGpuResultsThanItHolds) {
  Replies replies;
  replies.gpu_results = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto response = dirty<rdc_diag_response_t>();
    rdc_status_t status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT, nullptr, 0,
                                             response.get(), nullptr);
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*response);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, DiagnosticChecksTheGpuResultsOfEveryTest) {
  Replies replies;
  replies.test_results = 2;
  replies.last_test_gpu_results = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto response = dirty<rdc_diag_response_t>();
    rdc_status_t status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT, nullptr, 0,
                                             response.get(), nullptr);
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*response);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, TestCaseResultsCountWhatTheReplyCarries) {
  expect_in_child(Replies{}, [](const Session& session) {
    auto result = dirty<rdc_diag_test_result_t>();
    rdc_status_t status = rdc_test_case_run(session.handle(), 1, RDC_DIAG_COMPUTE_PROCESS, nullptr,
                                            0, result.get(), nullptr);
    bool ok = status == RDC_ST_OK && result->per_gpu_result_count == 1 &&
              result->gpu_results[0].gpu_index == kGpuId && untouched(result->gpu_results[1]);
    if (!ok) fprintf(stderr, "status %d, %u GPU results\n", status, result->per_gpu_result_count);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, TestCaseRejectsMoreGpuResultsThanItHolds) {
  Replies replies;
  replies.gpu_results = RDC_MAX_NUM_DEVICES + 1;
  expect_in_child(replies, [](const Session& session) {
    auto result = dirty<rdc_diag_test_result_t>();
    rdc_status_t status = rdc_test_case_run(session.handle(), 1, RDC_DIAG_COMPUTE_PROCESS, nullptr,
                                            0, result.get(), nullptr);
    bool ok = status == RDC_ST_BAD_PARAMETER && untouched(*result);
    if (!ok) fprintf(stderr, "status %d\n", status);
    return ok;
  });
}

// Whether a field value is the one the service sends, as the given type.
bool is_sent_value(const rdc_field_value& value, rdc_field_type_t type) {
  if (value.type != type || value.field_id != kValueField || value.status != kValueStatus ||
      value.ts != kValueTs) {
    return false;
  }
  if (type == INTEGER) return value.value.l_int == static_cast<int64_t>(kValue);
  if (type == DOUBLE) return value.value.dbl == kValue;
  return strcmp(value.value.str, kString) == 0;
}

TEST_F(RdcClientRepliesTest, FieldValuesOfAKnownTypeAreCopied) {
  for (rdc_field_type_t type : {INTEGER, DOUBLE, STRING, BLOB}) {
    SCOPED_TRACE(type);
    Replies replies;
    replies.value_type = type;
    expect_in_child(replies, [type](const Session& session) {
      rdc_field_value value = {};
      uint64_t next_since = 0;
      rdc_status_t latest =
          rdc_field_get_latest_value(session.handle(), 0, RDC_FI_GPU_CLOCK, &value);
      bool ok = latest == RDC_ST_OK && is_sent_value(value, type);
      value = {};
      rdc_status_t since =
          rdc_field_get_value_since(session.handle(), 0, RDC_FI_GPU_CLOCK, 0, &next_since, &value);
      ok = ok && since == RDC_ST_OK && is_sent_value(value, type) && next_since == kNextSince;
      if (!ok) fprintf(stderr, "type %d: status %d and %d\n", type, latest, since);
      return ok;
    });
  }
}

TEST_F(RdcClientRepliesTest, FieldValuesOfAnUnknownTypeAreAClientError) {
  Replies replies;
  replies.value_type = 99;
  expect_in_child(replies, [](const Session& session) {
    auto value = dirty<rdc_field_value>();
    auto next_since = dirty<uint64_t>();
    rdc_status_t latest =
        rdc_field_get_latest_value(session.handle(), 0, RDC_FI_GPU_CLOCK, value.get());
    bool ok = latest == RDC_ST_CLIENT_ERROR && untouched(*value);
    rdc_status_t since = rdc_field_get_value_since(session.handle(), 0, RDC_FI_GPU_CLOCK, 0,
                                                   next_since.get(), value.get());
    ok = ok && since == RDC_ST_CLIENT_ERROR && untouched(*value) && untouched(*next_since);
    if (!ok) fprintf(stderr, "status %d and %d\n", latest, since);
    return ok;
  });
}

// A value of another field than the one asked for is a client error: rdcd mixed up its
// replies, and the field ID could be one that rdc_field_t does not define.
TEST_F(RdcClientRepliesTest, FieldValuesOfAnotherFieldAreAClientError) {
  for (uint32_t field : {static_cast<uint32_t>(RDC_FI_MEM_CLOCK), 99999u, 0xffffffffu}) {
    SCOPED_TRACE(field);
    Replies replies;
    replies.value_field = field;
    expect_in_child(replies, [](const Session& session) {
      auto value = dirty<rdc_field_value>();
      auto next_since = dirty<uint64_t>();
      rdc_status_t latest =
          rdc_field_get_latest_value(session.handle(), 0, RDC_FI_GPU_CLOCK, value.get());
      bool ok = latest == RDC_ST_CLIENT_ERROR && untouched(*value);
      rdc_status_t since = rdc_field_get_value_since(session.handle(), 0, RDC_FI_GPU_CLOCK, 0,
                                                     next_since.get(), value.get());
      ok = ok && since == RDC_ST_CLIENT_ERROR && untouched(*value) && untouched(*next_since);
      if (!ok) fprintf(stderr, "status %d and %d\n", latest, since);
      return ok;
    });
  }
}

// A field group can hold fields that this client does not know, such as fields that a
// newer rdcd added: they read as RDC_FI_INVALID, and the other fields as they are.
TEST_F(RdcClientRepliesTest, FieldsTheClientDoesNotKnowReadAsInvalid) {
  Replies replies;
  replies.group_fields = {RDC_FI_GPU_CLOCK, 99999u, RDC_FI_MEM_CLOCK, 0xffffffffu};
  expect_in_child(replies, [](const Session& session) {
    auto info = dirty<rdc_field_group_info_t>();
    rdc_status_t status = rdc_group_field_get_info(session.handle(), 1, info.get());
    // Read as numbers: an rdc_field_t must not be read with a value that it does not define.
    static_assert(sizeof(rdc_field_t) == sizeof(uint32_t), "rdc_field_t is 32 bits");
    uint32_t fields[4] = {};
    bool ok = status == RDC_ST_OK && info->count == 4;
    if (ok) memcpy(fields, info->field_ids, sizeof(fields));
    ok = ok && fields[0] == RDC_FI_GPU_CLOCK && fields[1] == RDC_FI_INVALID &&
         fields[2] == RDC_FI_MEM_CLOCK && fields[3] == RDC_FI_INVALID;
    if (!ok) {
      fprintf(stderr, "status %d, count %u, fields %u %u %u %u\n", status, info->count, fields[0],
              fields[1], fields[2], fields[3]);
    }
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, PartitionCountsThatDoNotFitAreAClientError) {
  expect_in_child(Replies{}, [](const Session& session) {
    uint16_t count = 0;
    rdc_status_t status = rdc_get_num_partition(session.handle(), 0, &count);
    bool ok = status == RDC_ST_OK && count == 8;
    if (!ok) fprintf(stderr, "status %d, %u partitions\n", status, count);
    return ok;
  });
  Replies replies;
  replies.num_partition = UINT16_MAX + 1;
  expect_in_child(replies, [](const Session& session) {
    uint16_t count = 0xA5A5;
    rdc_status_t status = rdc_get_num_partition(session.handle(), 0, &count);
    bool ok = status == RDC_ST_CLIENT_ERROR && count == 0xA5A5;
    if (!ok) fprintf(stderr, "status %d, %u partitions\n", status, count);
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, RepliesThatFillTheCApiAreAccepted) {
  Replies replies;
  replies.topology_gpus = RDC_MAX_NUM_DEVICES;
  replies.link_gpus = RDC_MAX_NUM_DEVICES;
  replies.link_states = RDC_MAX_NUM_OF_LINKS;
  replies.incidents = HEALTH_MAX_ERROR_ITEMS;
  replies.test_results = MAX_TEST_CASES;
  replies.gpu_results = RDC_MAX_NUM_DEVICES;
  replies.num_partition = UINT16_MAX;
  expect_in_child(replies, [](const Session& session) {
    auto topology = dirty<rdc_device_topology_t>();
    rdc_status_t topology_status = rdc_device_topology_get(session.handle(), 0, topology.get());
    auto links = dirty<rdc_link_status_t>();
    rdc_status_t links_status = rdc_link_status_get(session.handle(), links.get());
    auto health = dirty<rdc_health_response_t>();
    rdc_status_t health_status = rdc_health_check(session.handle(), 1, health.get());
    auto response = dirty<rdc_diag_response_t>();
    rdc_status_t diag_status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT, nullptr,
                                                  0, response.get(), nullptr);
    auto result = dirty<rdc_diag_test_result_t>();
    rdc_status_t test_status = rdc_test_case_run(session.handle(), 1, RDC_DIAG_COMPUTE_PROCESS,
                                                 nullptr, 0, result.get(), nullptr);
    uint16_t partitions = 0;
    rdc_status_t partition_status = rdc_get_num_partition(session.handle(), 0, &partitions);
    bool ok = topology_status == RDC_ST_OK && topology->num_of_gpus == RDC_MAX_NUM_DEVICES &&
              links_status == RDC_ST_OK && links->num_of_gpus == RDC_MAX_NUM_DEVICES &&
              links->gpus[RDC_MAX_NUM_DEVICES - 1].num_of_links == RDC_MAX_NUM_OF_LINKS &&
              health_status == RDC_ST_OK && health->incidents_count == HEALTH_MAX_ERROR_ITEMS &&
              diag_status == RDC_ST_OK && response->results_count == MAX_TEST_CASES &&
              response->diag_info[MAX_TEST_CASES - 1].per_gpu_result_count == RDC_MAX_NUM_DEVICES &&
              test_status == RDC_ST_OK && result->per_gpu_result_count == RDC_MAX_NUM_DEVICES &&
              partition_status == RDC_ST_OK && partitions == UINT16_MAX;
    if (!ok) {
      fprintf(stderr, "status %d, %d, %d, %d, %d and %d\n", topology_status, links_status,
              health_status, diag_status, test_status, partition_status);
    }
    return ok;
  });
}

TEST_F(RdcClientRepliesTest, RunsWithoutResultsAreAClientError) {
  for (RunStream run : {RunStream::kNoResults, RunStream::kNothing}) {
    SCOPED_TRACE(static_cast<int>(run));
    Replies replies;
    replies.run = run;
    expect_in_child(replies, [](const Session& session) {
      auto response = dirty<rdc_diag_response_t>();
      rdc_status_t diag_status = rdc_diagnostic_run(session.handle(), 1, RDC_DIAG_LVL_SHORT,
                                                    nullptr, 0, response.get(), nullptr);
      auto result = dirty<rdc_diag_test_result_t>();
      rdc_status_t test_status = rdc_test_case_run(session.handle(), 1, RDC_DIAG_COMPUTE_PROCESS,
                                                   nullptr, 0, result.get(), nullptr);
      bool ok = diag_status == RDC_ST_CLIENT_ERROR && untouched(*response) &&
                test_status == RDC_ST_CLIENT_ERROR && untouched(*result);
      if (!ok) fprintf(stderr, "status %d and %d\n", diag_status, test_status);
      return ok;
    });
  }
}

// rdcd gets 120 s to answer a call, and 30 minutes to run a diagnostic.
int no_callback(rdc_policy_callback_response_t*) { return 0; }

// Calls every RDC API function that sends rdcd a request.
void call_every_method(rdc_handle_t handle) {
  uint32_t gpus[RDC_MAX_NUM_DEVICES] = {};
  uint32_t count = 0;
  rdc_device_get_all(handle, gpus, &count);
  rdc_device_attributes_t attributes = {};
  rdc_device_get_attributes(handle, 0, &attributes);
  rdc_component_version_t version = {};
  rdc_device_get_component_version(handle, RDC_AMDSMI_COMPONENT, &version);
  mixed_component_version_t rdcd_version = {};
  get_mixed_component_version(handle, RDCD_COMPONENT, &rdcd_version);

  rdc_gpu_group_t group = 1;
  rdc_group_gpu_create(handle, RDC_GROUP_EMPTY, "group", &group);
  rdc_group_gpu_add(handle, 1, 0);
  rdc_group_info_t group_info = {};
  rdc_group_gpu_get_info(handle, 1, &group_info);
  rdc_gpu_group_t group_ids[RDC_MAX_NUM_GROUPS] = {};
  rdc_group_get_all_ids(handle, group_ids, &count);
  rdc_field_t fields[] = {RDC_FI_GPU_CLOCK};
  rdc_field_grp_t field_group = 1;
  rdc_group_field_create(handle, 1, fields, "fields", &field_group);
  rdc_group_field_add_field(handle, 1, RDC_FI_POWER_USAGE);
  rdc_field_group_info_t field_group_info = {};
  rdc_group_field_get_info(handle, 1, &field_group_info);
  rdc_field_grp_t field_group_ids[RDC_MAX_NUM_FIELD_GROUPS] = {};
  rdc_group_field_get_all_ids(handle, field_group_ids, &count);

  rdc_field_watch(handle, 1, 1, 1000000, 60, 10);
  rdc_field_update_all(handle, 0);
  rdc_field_value value = {};
  rdc_field_get_latest_value(handle, 0, RDC_FI_GPU_CLOCK, &value);
  uint64_t next_since = 0;
  rdc_field_get_value_since(handle, 0, RDC_FI_GPU_CLOCK, 0, &next_since, &value);
  rdc_field_unwatch(handle, 1, 1);

  rdc_job_start_stats(handle, 1, "job", 1000000);
  auto job = std::make_unique<rdc_job_info_t>();
  rdc_job_get_stats(handle, "job", job.get());
  rdc_job_stop_stats(handle, "job");
  rdc_job_remove(handle, "job");
  rdc_job_remove_all(handle);

  auto response = std::make_unique<rdc_diag_response_t>();
  rdc_diagnostic_run(handle, 1, RDC_DIAG_LVL_SHORT, nullptr, 0, response.get(), nullptr);
  auto result = std::make_unique<rdc_diag_test_result_t>();
  rdc_test_case_run(handle, 1, RDC_DIAG_RVS_GST_TEST, nullptr, 0, result.get(), nullptr);

  rdc_policy_t policy = {};
  policy.condition = {RDC_POLICY_COND_THERMAL, 90};
  policy.action = RDC_POLICY_ACTION_NONE;
  rdc_policy_set(handle, 1, policy);
  rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS] = {};
  rdc_policy_get(handle, 1, &count, policies);
  rdc_policy_delete(handle, 1, RDC_POLICY_COND_THERMAL);
  rdc_policy_register(handle, 1, no_callback);
  rdc_policy_unregister(handle, 1);

  rdc_health_set(handle, 1, RDC_HEALTH_WATCH_PCIE);
  unsigned int components = 0;
  rdc_health_get(handle, 1, &components);
  auto health = std::make_unique<rdc_health_response_t>();
  rdc_health_check(handle, 1, health.get());
  rdc_health_clear(handle, 1);

  auto topology = std::make_unique<rdc_device_topology_t>();
  rdc_device_topology_get(handle, 0, topology.get());
  auto links = std::make_unique<rdc_link_status_t>();
  rdc_link_status_get(handle, links.get());
  uint16_t partitions = 0;
  rdc_get_num_partition(handle, 0, &partitions);
  rdc_resource_profile_t profile = {};
  rdc_instance_profile_get(handle, 0, RDC_ACCELERATOR_XCC, &profile);

  rdc_config_setting_t setting = {};
  setting.type = RDC_CFG_POWER_LIMIT;
  setting.target_value = 200;
  rdc_config_set(handle, 1, setting);
  rdc_config_setting_list_t settings = {};
  rdc_config_get(handle, 1, &settings);
  rdc_config_clear(handle, 1);

  rdc_group_field_destroy(handle, 1);
  rdc_group_gpu_destroy(handle, 1);
}

// Whether the client gave each method of rdcd's RdcAPI service the deadline it should.
bool every_call_has_its_deadline() {
  DeadlineRecorder recorder;
  {
    Session session({}, &recorder);
    call_every_method(session.handle());
  }
  const std::map<std::string, int64_t> time_left = recorder.time_left();
  // The methods of the service, from the descriptor of its messages.
  const ::google::protobuf::ServiceDescriptor* api =
      ::rdc::Empty::descriptor()->file()->FindServiceByName("RdcAPI");
  if (api == nullptr || api->method_count() == 0) {
    fprintf(stderr, "no RdcAPI service\n");
    return false;
  }
  bool ok = true;
  for (int i = 0; i < api->method_count(); ++i) {
    const std::string name(api->method(i)->name());
    const auto it = time_left.find("/rdc.RdcAPI/" + name);
    if (it == time_left.end()) {
      fprintf(stderr, "%s was not called\n", name.c_str());
      ok = false;
      continue;
    }
    // gRPC rounds the time left up to the next millisecond, and then to a unit it can
    // send: 120 s go out as 120 s ("2M") or 121 s ("121S"), 30 minutes as 1800 s ("30M")
    // or 1810 s (181 ten-second units). The server sees them less the time the call took
    // to arrive, in whole seconds: 119 or 120, and 1799 or 1809 (measured: 119 or 120, and
    // 1809). The lower bounds leave room for a slow machine.
    bool expected = false;
    if (name == "DiagnosticTestCaseRun" || name == "RegisterPolicy") {
      expected = it->second == -1;
    } else if (name == "DiagnosticRun") {
      expected = it->second > 30 * 60 - 10 && it->second <= 30 * 60 + 10;
    } else {
      expected = it->second > 100 && it->second <= 120;
    }
    if (!expected) {
      fprintf(stderr, "%s: %lld s\n", name.c_str(), static_cast<long long>(it->second));
      ok = false;
    }
  }
  return ok;
}

TEST_F(RdcClientRepliesTest, EveryCallHasItsDeadline) {
  EXPECT_EXIT(_exit(every_call_has_its_deadline() ? 0 : 1), ::testing::ExitedWithCode(0), "");
}

}  // namespace
