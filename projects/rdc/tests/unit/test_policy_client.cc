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

// rdc_policy_register() and rdc_policy_unregister() through the standalone
// client, against a fake rdcd that admits, rejects or never answers a
// RegisterPolicy stream, or answers like an rdcd that predates admission
// (initial metadata only with the first notification).

#include <grpcpp/grpcpp.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace {

using std::chrono::milliseconds;
using std::chrono::seconds;
using std::chrono::steady_clock;

// What the fake rdcd does with a group's RegisterPolicy stream.
enum class Answer {
  kAdmit,              // admission metadata at once, then notifications
  kSilent,             // nothing until the client cancels
  kOldServer,          // a notification at once, its metadata without the admission key
  kReject,             // ends the stream with `code` at once
  kRejectLater,        // ends the stream with `code` after `delay`
  kAdmitLater,         // admission metadata after `delay`, then notifications
  kAdmitOnUnregister,  // admission metadata once the client unregisters the group, then the end
};

struct Plan {
  Answer answer = Answer::kAdmit;
  ::grpc::StatusCode code = ::grpc::StatusCode::OK;
  milliseconds delay{0};
};

class FakePolicyService final : public ::rdc::RdcAPI::Service {
 public:
  void plan(uint32_t group, Plan plan) {
    std::lock_guard<std::mutex> lock(mutex_);
    plans_[group] = plan;
  }

  // Sends one notification on every open stream of the group.
  void notify(uint32_t group) {
    std::lock_guard<std::mutex> lock(mutex_);
    notifications_[group]++;
    cv_.notify_all();
  }

  // Waits until `count` RegisterPolicy calls for the group have arrived.
  bool wait_for_calls(uint32_t group, int count) {
    std::unique_lock<std::mutex> lock(mutex_);
    return cv_.wait_for(lock, seconds(30), [&] { return calls_[group] >= count; });
  }

  ::grpc::Status RegisterPolicy(
      ::grpc::ServerContext* context, const ::rdc::RegisterPolicyRequest* request,
      ::grpc::ServerWriter<::rdc::RegisterPolicyResponse>* writer) override {
    const uint32_t group = request->group_id();
    Plan plan;
    int sent = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      plan = plans_[group];
      calls_[group]++;
      sent = notifications_[group];  // only later notifications go to this stream
      cv_.notify_all();
    }
    const auto start = steady_clock::now();
    switch (plan.answer) {
      case Answer::kReject:
        return ::grpc::Status(plan.code, "rejected");
      case Answer::kRejectLater:
        while (steady_clock::now() - start < plan.delay) {
          if (context->IsCancelled()) {
            return ::grpc::Status::CANCELLED;
          }
          std::this_thread::sleep_for(milliseconds(10));
        }
        return ::grpc::Status(plan.code, "rejected late");
      case Answer::kSilent:
        while (!context->IsCancelled()) {
          std::this_thread::sleep_for(milliseconds(10));
        }
        return ::grpc::Status::CANCELLED;
      case Answer::kOldServer:
        if (!writer->Write(notification(group))) {
          return ::grpc::Status::OK;
        }
        break;
      case Answer::kAdmitLater:
        while (steady_clock::now() - start < plan.delay) {
          if (context->IsCancelled()) {
            return ::grpc::Status::CANCELLED;
          }
          std::this_thread::sleep_for(milliseconds(10));
        }
        context->AddInitialMetadata(kRdcPolicyAdmissionKey, "1");
        writer->SendInitialMetadata();
        break;
      case Answer::kAdmitOnUnregister: {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!context->IsCancelled() && !unregistered_[group]) {
          cv_.wait_for(lock, milliseconds(10));
        }
        if (context->IsCancelled()) {
          return ::grpc::Status::CANCELLED;
        }
        lock.unlock();
        context->AddInitialMetadata(kRdcPolicyAdmissionKey, "1");
        writer->SendInitialMetadata();
        lock.lock();
        admitted_on_unregister_[group] = true;
        cv_.notify_all();
        return ::grpc::Status::OK;
      }
      case Answer::kAdmit:
        context->AddInitialMetadata(kRdcPolicyAdmissionKey, "1");
        writer->SendInitialMetadata();
        break;
    }
    // Deliver notifications until the client goes away or unregisters.
    std::unique_lock<std::mutex> lock(mutex_);
    while (!context->IsCancelled() && !unregistered_[group]) {
      if (notifications_[group] > sent) {
        sent++;
        lock.unlock();
        bool written = writer->Write(notification(group));
        lock.lock();
        if (!written) {
          break;
        }
        continue;
      }
      cv_.wait_for(lock, milliseconds(10));
    }
    return ::grpc::Status::OK;
  }

  ::grpc::Status UnRegisterPolicy(::grpc::ServerContext*,
                                  const ::rdc::UnRegisterPolicyRequest* request,
                                  ::rdc::UnRegisterPolicyResponse* reply) override {
    const uint32_t group = request->group_id();
    std::unique_lock<std::mutex> lock(mutex_);
    unregistered_[group] = true;
    cv_.notify_all();
    if (plans_[group].answer == Answer::kAdmitOnUnregister) {
      // Answer after the admission has gone out, and had time to reach the client.
      cv_.wait_for(lock, seconds(10), [&] { return admitted_on_unregister_[group]; });
      lock.unlock();
      std::this_thread::sleep_for(seconds(1));
    }
    reply->set_status(RDC_ST_OK);
    return ::grpc::Status::OK;
  }

  void allow_again(uint32_t group) {
    std::lock_guard<std::mutex> lock(mutex_);
    unregistered_[group] = false;
    admitted_on_unregister_[group] = false;
  }

 private:
  static ::rdc::RegisterPolicyResponse notification(uint32_t group) {
    ::rdc::RegisterPolicyResponse reply;
    reply.set_status(RDC_ST_OK);
    reply.set_version(1);
    reply.set_group_id(group);
    reply.set_value(100);
    reply.mutable_condition()->set_type(::rdc::PolicyCondition_Type_COND_THERMAL);
    reply.mutable_condition()->set_value(1);
    return reply;
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::map<uint32_t, Plan> plans_;
  std::map<uint32_t, int> calls_;
  std::map<uint32_t, int> notifications_;
  std::map<uint32_t, bool> unregistered_;
  std::map<uint32_t, bool> admitted_on_unregister_;
};

// Notifications the client delivered, by group.
std::mutex g_callback_mutex;
std::condition_variable g_callback_cv;
std::map<uint32_t, int> g_callbacks;

int callback(rdc_policy_callback_response_t* response) {
  std::lock_guard<std::mutex> lock(g_callback_mutex);
  g_callbacks[response->group_id]++;
  g_callback_cv.notify_all();
  return 0;
}

bool wait_for_callbacks(uint32_t group, int count) {
  std::unique_lock<std::mutex> lock(g_callback_mutex);
  return g_callback_cv.wait_for(lock, seconds(30), [&] { return g_callbacks[group] >= count; });
}

// unregister_callback unregisters g_unregister_group of g_unregister_handle
// once, then counts the notification.
rdc_handle_t g_unregister_handle = nullptr;
std::atomic<uint32_t> g_unregister_group{0};
std::atomic<int> g_unregister_status{-1};

int unregister_callback(rdc_policy_callback_response_t* response) {
  const uint32_t group = g_unregister_group.exchange(0);
  if (group != 0) {
    g_unregister_status = rdc_policy_unregister(g_unregister_handle, group);
  }
  return callback(response);
}

double seconds_since(steady_clock::time_point start) {
  return std::chrono::duration<double>(steady_clock::now() - start).count();
}

// A FakePolicyService on an abstract Unix socket and an RDC client connected to it.
class PolicyClientTest : public ::testing::Test {
 protected:
  void SetUp() override {
    {
      std::lock_guard<std::mutex> lock(g_callback_mutex);
      g_callbacks.clear();
    }
    g_unregister_handle = nullptr;
    g_unregister_group = 0;
    g_unregister_status = -1;
    static int session = 0;
    address_ = "unix-abstract:rdc_policy_client_" + std::to_string(getpid()) + "_" +
               std::to_string(session++);
    ::grpc::ServerBuilder builder;
    builder.AddListeningPort(address_, ::grpc::InsecureServerCredentials());
    builder.RegisterService(&service_);
    server_ = builder.BuildAndStart();
    ASSERT_TRUE(server_) << "cannot start a server on " << address_;
    ASSERT_EQ(rdc_init(0), RDC_ST_OK);
    ASSERT_EQ(connect(), RDC_ST_OK);
  }

  void TearDown() override {
    if (handle_ != nullptr) {
      EXPECT_EQ(rdc_disconnect(handle_), RDC_ST_OK);
    }
    rdc_shutdown();
    if (server_) {
      server_->Shutdown(std::chrono::system_clock::now() + seconds(5));
    }
  }

  rdc_status_t connect() {
    return rdc_connect(address_.c_str(), &handle_, nullptr, nullptr, nullptr);
  }
  rdc_status_t reg(uint32_t group) { return rdc_policy_register(handle_, group, callback); }

  std::string address_;
  FakePolicyService service_;
  std::unique_ptr<::grpc::Server> server_;
  rdc_handle_t handle_ = nullptr;
};

}  // namespace

TEST_F(PolicyClientTest, AdmittedRegistrationDeliversNotifications) {
  auto start = steady_clock::now();
  ASSERT_EQ(reg(1), RDC_ST_OK);
  EXPECT_LT(seconds_since(start), 4.0) << "an admitted registration waited for the timeout";
  EXPECT_EQ(reg(1), RDC_ST_CONFLICT);
  service_.notify(1);
  EXPECT_TRUE(wait_for_callbacks(1, 1));
  EXPECT_EQ(rdc_policy_unregister(handle_, 1), RDC_ST_OK);
  service_.allow_again(1);
  ASSERT_EQ(reg(1), RDC_ST_OK);
  service_.notify(1);
  EXPECT_TRUE(wait_for_callbacks(1, 2));
}

TEST_F(PolicyClientTest, RejectionReturnsItsStatus) {
  service_.plan(1, {Answer::kReject, ::grpc::StatusCode::NOT_FOUND});
  service_.plan(2, {Answer::kReject, ::grpc::StatusCode::RESOURCE_EXHAUSTED});
  service_.plan(3, {Answer::kReject, ::grpc::StatusCode::UNAVAILABLE});
  EXPECT_EQ(reg(1), RDC_ST_GROUP_NOT_FOUND);
  EXPECT_EQ(reg(2), RDC_ST_MAX_LIMIT);
  EXPECT_EQ(reg(3), RDC_ST_CLIENT_ERROR);
  // A rejected registration leaves nothing behind.
  service_.plan(1, {Answer::kAdmit});
  ASSERT_EQ(reg(1), RDC_ST_OK);
  service_.notify(1);
  EXPECT_TRUE(wait_for_callbacks(1, 1));
}

TEST_F(PolicyClientTest, SilentServerIsPresumedOld) {
  service_.plan(1, {Answer::kSilent});
  service_.plan(2, {Answer::kSilent});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(1), RDC_ST_OK);
  EXPECT_GE(seconds_since(start), 4.5) << "returned before the admission wait ended";
  start = steady_clock::now();
  EXPECT_EQ(reg(2), RDC_ST_OK);
  EXPECT_LT(seconds_since(start), 2.0) << "the second registration waited again";
}

TEST_F(PolicyClientTest, OldServerIsNotWaitedFor) {
  service_.plan(1, {Answer::kOldServer});
  service_.plan(2, {Answer::kSilent});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(1), RDC_ST_OK);
  EXPECT_LT(seconds_since(start), 4.0) << "waited although a notification arrived";
  EXPECT_TRUE(wait_for_callbacks(1, 1));
  start = steady_clock::now();
  EXPECT_EQ(reg(2), RDC_ST_OK);
  EXPECT_LT(seconds_since(start), 2.0) << "waited for an rdcd known to predate admission";
}

TEST_F(PolicyClientTest, LateRejectionIsDropped) {
  // Rejected only after the client stopped waiting: register returns OK, and
  // once the stream has ended the group can be registered again. (The handle
  // now presumes an rdcd that predates admission, so that register does not
  // wait.)
  service_.plan(1, {Answer::kRejectLater, ::grpc::StatusCode::NOT_FOUND, milliseconds(6000)});
  EXPECT_EQ(reg(1), RDC_ST_OK);
  std::this_thread::sleep_for(seconds(2));  // the rejection arrives about now
  service_.plan(1, {Answer::kAdmit});
  auto deadline = steady_clock::now() + seconds(30);
  rdc_status_t status = RDC_ST_CONFLICT;
  while (status == RDC_ST_CONFLICT && steady_clock::now() < deadline) {
    status = reg(1);
    if (status == RDC_ST_CONFLICT) {
      std::this_thread::sleep_for(milliseconds(100));
    }
  }
  ASSERT_EQ(status, RDC_ST_OK);
  ASSERT_TRUE(service_.wait_for_calls(1, 2));
  service_.notify(1);
  EXPECT_TRUE(wait_for_callbacks(1, 1));
}

TEST_F(PolicyClientTest, LateAdmissionEndsThePresumption) {
  // rdcd admits only after the client stopped waiting: the handle presumes an
  // rdcd that predates admission until it sees the admission, then waits again
  // and returns rdcd's rejection.
  service_.plan(1, {Answer::kAdmitLater, ::grpc::StatusCode::OK, milliseconds(6000)});
  service_.plan(2, {Answer::kReject, ::grpc::StatusCode::NOT_FOUND});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(1), RDC_ST_OK);
  EXPECT_GE(seconds_since(start), 4.5) << "returned before the admission wait ended";
  std::this_thread::sleep_for(seconds(2));  // admitted about now
  service_.notify(1);
  ASSERT_TRUE(wait_for_callbacks(1, 1));
  EXPECT_EQ(rdc_policy_unregister(handle_, 1), RDC_ST_OK);
  EXPECT_EQ(reg(2), RDC_ST_GROUP_NOT_FOUND);
}

TEST_F(PolicyClientTest, AdmissionDuringUnregisterEndsThePresumption) {
  // rdcd admits the registration only while the client unregisters it, after
  // the client presumed an old rdcd: the unregister still sees the admission.
  service_.plan(1, {Answer::kAdmitOnUnregister});
  service_.plan(2, {Answer::kReject, ::grpc::StatusCode::NOT_FOUND});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(1), RDC_ST_OK);
  EXPECT_GE(seconds_since(start), 4.5) << "returned before the admission wait ended";
  EXPECT_EQ(rdc_policy_unregister(handle_, 1), RDC_ST_OK);
  EXPECT_EQ(reg(2), RDC_ST_GROUP_NOT_FOUND);
}

TEST_F(PolicyClientTest, AdmissionAfterACallbackUnregisterEndsThePresumption) {
  // Like AdmissionDuringUnregisterEndsThePresumption, with the unregister from
  // a callback of another group: the stream it gives up is joined later, and
  // its admission still counts.
  service_.plan(1, {Answer::kOldServer});
  service_.plan(2, {Answer::kAdmitOnUnregister});
  service_.plan(3, {Answer::kReject, ::grpc::StatusCode::NOT_FOUND});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(2), RDC_ST_OK);
  EXPECT_GE(seconds_since(start), 4.5) << "returned before the admission wait ended";
  g_unregister_handle = handle_;
  g_unregister_group = 2;
  EXPECT_EQ(rdc_policy_register(handle_, 1, unregister_callback), RDC_ST_OK);
  ASSERT_TRUE(wait_for_callbacks(1, 1));
  EXPECT_EQ(g_unregister_status, RDC_ST_OK);
  std::this_thread::sleep_for(seconds(1));  // the reader of group 2 ends
  EXPECT_EQ(reg(3), RDC_ST_GROUP_NOT_FOUND);
}

TEST_F(PolicyClientTest, LateAdmissionSeenByTheNextRegister) {
  // Like LateAdmissionEndsThePresumption, with the late-admitted registration
  // still in place: the next register sees it and waits for rdcd's answer.
  service_.plan(1, {Answer::kAdmitLater, ::grpc::StatusCode::OK, milliseconds(6000)});
  service_.plan(2, {Answer::kReject, ::grpc::StatusCode::NOT_FOUND});
  auto start = steady_clock::now();
  EXPECT_EQ(reg(1), RDC_ST_OK);
  EXPECT_GE(seconds_since(start), 4.5) << "returned before the admission wait ended";
  std::this_thread::sleep_for(seconds(2));
  service_.notify(1);
  ASSERT_TRUE(wait_for_callbacks(1, 1));
  EXPECT_EQ(reg(2), RDC_ST_GROUP_NOT_FOUND);
}

TEST_F(PolicyClientTest, UnregisterEndsAWaitingRegistration) {
  service_.plan(1, {Answer::kSilent});
  std::atomic<rdc_status_t> registered{RDC_ST_UNKNOWN_ERROR};
  std::atomic<bool> returned{false};
  std::thread waiter([&] {
    registered = reg(1);
    returned = true;
  });
  ASSERT_TRUE(service_.wait_for_calls(1, 1));
  std::this_thread::sleep_for(milliseconds(200));  // the waiter is parked
  EXPECT_EQ(rdc_policy_unregister(handle_, 1), RDC_ST_OK);
  waiter.join();
  EXPECT_TRUE(returned);
  EXPECT_EQ(registered, RDC_ST_OK);
}

// rdc_disconnect() while a registration waits for admission: the waiter
// returns, and the handle is freed only after it left (AddressSanitizer would
// report a use after free otherwise). Repeated under RDC_TEST_STRESS=1.
TEST_F(PolicyClientTest, DisconnectEndsAWaitingRegistration) {
  const int rounds = std::getenv("RDC_TEST_STRESS") != nullptr ? 20 : 1;
  for (int round = 0; round < rounds; round++) {
    SCOPED_TRACE(round);
    if (round > 0) {
      ASSERT_EQ(connect(), RDC_ST_OK);
    }
    service_.plan(1, {Answer::kSilent});
    std::atomic<rdc_status_t> registered{RDC_ST_UNKNOWN_ERROR};
    std::atomic<bool> returned{false};
    std::thread waiter([&] {
      registered = reg(1);
      returned = true;
    });
    ASSERT_TRUE(service_.wait_for_calls(1, round + 1));
    std::this_thread::sleep_for(milliseconds(200));  // the waiter is parked
    EXPECT_EQ(rdc_disconnect(handle_), RDC_ST_OK);
    handle_ = nullptr;
    waiter.join();
    EXPECT_TRUE(returned);
    EXPECT_EQ(registered, RDC_ST_CLIENT_ERROR);
  }
}
