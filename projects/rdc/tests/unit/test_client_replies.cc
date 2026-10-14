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
// carry: the client must count what the reply carries.

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <cstdint>
#include <memory>
#include <string>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"

namespace {

constexpr uint32_t kGpuId = 7;
constexpr uint32_t kPid = 42;

// What the service sends.
struct Replies {
  int group_ids = 0;  // ids in each group and field group list
  int processes = 1;  // processes of the job
};

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

 private:
  const Replies replies_;
};

// An OverstatingService on an abstract Unix socket, which leaves nothing behind, and an
// RDC client connected to it.
class Session {
 public:
  explicit Session(Replies replies = {}) : service_(replies) {
    const std::string address = "unix-abstract:rdc_client_replies_" + std::to_string(getpid());
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
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

}  // namespace
