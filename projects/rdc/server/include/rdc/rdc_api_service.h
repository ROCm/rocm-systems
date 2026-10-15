/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_
#define SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_

#include <grpcpp/server_context.h>

#include <condition_variable>  // NOLINT
#include <map>
#include <memory>
#include <mutex>  // NOLINT
#include <vector>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

class RdcAPIServiceImpl final : public ::rdc::RdcAPI::Service {
 public:
  RdcAPIServiceImpl();
  ~RdcAPIServiceImpl();

  rdc_status_t Initialize(uint64_t rdcd_init_flags = 0);
  void Shutdown();

  ::grpc::Status GetAllDevices(::grpc::ServerContext* context, const ::rdc::Empty* request,
                               ::rdc::GetAllDevicesResponse* reply) override;

  ::grpc::Status GetDeviceAttributes(::grpc::ServerContext* context,
                                     const ::rdc::GetDeviceAttributesRequest* request,
                                     ::rdc::GetDeviceAttributesResponse* reply) override;

  ::grpc::Status GetComponentVersion(::grpc::ServerContext* context,
                                     const ::rdc::GetComponentVersionRequest* request,
                                     ::rdc::GetComponentVersionResponse* reply) override;

  ::grpc::Status CreateGpuGroup(::grpc::ServerContext* context,
                                const ::rdc::CreateGpuGroupRequest* request,
                                ::rdc::CreateGpuGroupResponse* reply) override;

  ::grpc::Status AddToGpuGroup(::grpc::ServerContext* context,
                               const ::rdc::AddToGpuGroupRequest* request,
                               ::rdc::AddToGpuGroupResponse* reply) override;

  ::grpc::Status GetGpuGroupInfo(::grpc::ServerContext* context,
                                 const ::rdc::GetGpuGroupInfoRequest* request,
                                 ::rdc::GetGpuGroupInfoResponse* reply) override;

  ::grpc::Status GetGroupAllIds(::grpc::ServerContext* context, const ::rdc::Empty* request,
                                ::rdc::GetGroupAllIdsResponse* reply) override;

  ::grpc::Status DestroyGpuGroup(::grpc::ServerContext* context,
                                 const ::rdc::DestroyGpuGroupRequest* request,
                                 ::rdc::DestroyGpuGroupResponse* reply) override;

  ::grpc::Status CreateFieldGroup(::grpc::ServerContext* context,
                                  const ::rdc::CreateFieldGroupRequest* request,
                                  ::rdc::CreateFieldGroupResponse* reply) override;

  ::grpc::Status AddFieldToFieldGroup(::grpc::ServerContext* context,
                                      const ::rdc::AddFieldToFieldGroupRequest* request,
                                      ::rdc::AddFieldToFieldGroupResponse* reply) override;

  ::grpc::Status GetFieldGroupInfo(::grpc::ServerContext* context,
                                   const ::rdc::GetFieldGroupInfoRequest* request,
                                   ::rdc::GetFieldGroupInfoResponse* reply) override;

  ::grpc::Status GetFieldGroupAllIds(::grpc::ServerContext* context, const ::rdc::Empty* request,
                                     ::rdc::GetFieldGroupAllIdsResponse* reply) override;

  ::grpc::Status DestroyFieldGroup(::grpc::ServerContext* context,
                                   const ::rdc::DestroyFieldGroupRequest* request,
                                   ::rdc::DestroyFieldGroupResponse* reply) override;

  ::grpc::Status WatchFields(::grpc::ServerContext* context,
                             const ::rdc::WatchFieldsRequest* request,
                             ::rdc::WatchFieldsResponse* reply) override;

  ::grpc::Status GetLatestFieldValue(::grpc::ServerContext* context,
                                     const ::rdc::GetLatestFieldValueRequest* request,
                                     ::rdc::GetLatestFieldValueResponse* reply) override;

  ::grpc::Status GetFieldSince(::grpc::ServerContext* context,
                               const ::rdc::GetFieldSinceRequest* request,
                               ::rdc::GetFieldSinceResponse* reply) override;

  ::grpc::Status UnWatchFields(::grpc::ServerContext* context,
                               const ::rdc::UnWatchFieldsRequest* request,
                               ::rdc::UnWatchFieldsResponse* reply) override;

  ::grpc::Status UpdateAllFields(::grpc::ServerContext* context,
                                 const ::rdc::UpdateAllFieldsRequest* request,
                                 ::rdc::UpdateAllFieldsResponse* reply) override;

  ::grpc::Status StartJobStats(::grpc::ServerContext* context,
                               const ::rdc::StartJobStatsRequest* request,
                               ::rdc::StartJobStatsResponse* reply) override;

  ::grpc::Status GetJobStats(::grpc::ServerContext* context,
                             const ::rdc::GetJobStatsRequest* request,
                             ::rdc::GetJobStatsResponse* reply) override;

  ::grpc::Status StopJobStats(::grpc::ServerContext* context,
                              const ::rdc::StopJobStatsRequest* request,
                              ::rdc::StopJobStatsResponse* reply) override;

  ::grpc::Status RemoveJob(::grpc::ServerContext* context, const ::rdc::RemoveJobRequest* request,
                           ::rdc::RemoveJobResponse* reply) override;

  ::grpc::Status RemoveAllJob(::grpc::ServerContext* context, const ::rdc::Empty* request,
                              ::rdc::RemoveAllJobResponse* reply) override;

  ::grpc::Status DiagnosticRun(::grpc::ServerContext* context,
                               const ::rdc::DiagnosticRunRequest* request,
                               ::grpc::ServerWriter<::rdc::DiagnosticRunResponse>* writer) override;

  ::grpc::Status DiagnosticTestCaseRun(
      ::grpc::ServerContext* context, const ::rdc::DiagnosticTestCaseRunRequest* request,
      ::grpc::ServerWriter<::rdc::DiagnosticTestCaseRunResponse>* writer) override;

  ::grpc::Status GetMixedComponentVersion(::grpc::ServerContext* context,
                                          const ::rdc::GetMixedComponentVersionRequest* request,
                                          ::rdc::GetMixedComponentVersionResponse* reply) override;

  ::grpc::Status SetPolicy(::grpc::ServerContext* context, const ::rdc::SetPolicyRequest* request,
                           ::rdc::SetPolicyResponse* reply) override;

  ::grpc::Status GetPolicy(::grpc::ServerContext* context, const ::rdc::GetPolicyRequest* request,
                           ::rdc::GetPolicyResponse* reply) override;

  ::grpc::Status DeletePolicy(::grpc::ServerContext* context,
                              const ::rdc::DeletePolicyRequest* request,
                              ::rdc::DeletePolicyResponse* reply) override;

  ::grpc::Status RegisterPolicy(
      ::grpc::ServerContext* context, const ::rdc::RegisterPolicyRequest* request,
      ::grpc::ServerWriter<::rdc::RegisterPolicyResponse>* stream) override;

  ::grpc::Status UnRegisterPolicy(::grpc::ServerContext* context,
                                  const ::rdc::UnRegisterPolicyRequest* request,
                                  ::rdc::UnRegisterPolicyResponse* reply) override;

  ::grpc::Status GetTopology(::grpc::ServerContext* context,
                             const ::rdc::GetTopologyRequest* request,
                             ::rdc::GetTopologyResponse* reply) override;

  ::grpc::Status GetLinkStatus(::grpc::ServerContext* context, const ::rdc::Empty* request,
                               ::rdc::GetLinkStatusResponse* reply) override;

  ::grpc::Status SetHealth(::grpc::ServerContext* context, const ::rdc::SetHealthRequest* request,
                           ::rdc::SetHealthResponse* reply) override;

  ::grpc::Status GetHealth(::grpc::ServerContext* context, const ::rdc::GetHealthRequest* request,
                           ::rdc::GetHealthResponse* reply) override;

  ::grpc::Status CheckHealth(::grpc::ServerContext* context,
                             const ::rdc::CheckHealthRequest* request,
                             ::rdc::CheckHealthResponse* reply) override;

  ::grpc::Status ClearHealth(::grpc::ServerContext* context,
                             const ::rdc::ClearHealthRequest* request,
                             ::rdc::ClearHealthResponse* reply) override;

  ::grpc::Status SetConfig(::grpc::ServerContext* context, const ::rdc::SetConfigRequest* request,
                           ::rdc::SetConfigResponse* reply) override;

  ::grpc::Status GetConfig(::grpc::ServerContext* context, const ::rdc::GetConfigRequest* request,
                           ::rdc::GetConfigResponse* reply) override;

  ::grpc::Status ClearConfig(::grpc::ServerContext* context,
                             const ::rdc::ClearConfigRequest* request,
                             ::rdc::ClearConfigResponse* reply) override;

  ::grpc::Status GetNumPartition(::grpc::ServerContext* context,
                                 const ::rdc::GetNumPartitionRequest* request,
                                 ::rdc::GetNumPartitionResponse* reply) override;

  ::grpc::Status GetInstanceProfile(::grpc::ServerContext* context,
                                    const ::rdc::GetInstanceProfileRequest* request,
                                    ::rdc::GetInstanceProfileResponse* reply) override;

 private:
  bool copy_gpu_usage_info(const rdc_gpu_usage_info_t& src, ::rdc::GpuUsageInfo* target);
  rdc_handle_t rdc_handle_;

  // The most RegisterPolicy streams rdcd serves at once, over all clients and groups.
  static constexpr size_t kMaxPolicyStreams = 64;

  // One RegisterPolicy stream. PolicyCallback() leaves the latest notification in it; the
  // stream's own handler thread writes it to the client.
  struct PolicyStream {
    std::mutex mutex;  // guards the fields below
    std::condition_variable cv;
    rdc_policy_callback_response_t response;
    bool pending = false;   // response is not written yet
    bool stop = false;      // UnRegisterPolicy or shutdown ends the stream
    bool shutdown = false;  // shutdown ended it: the client learns that rdcd stops
  };
  static void stop_policy_stream(PolicyStream* stream);
  // Removes a stream that ended; the last stream of a group drops its library registration.
  void end_policy_stream(uint32_t group_id, const std::shared_ptr<PolicyStream>& stream);
  static int PolicyCallback(rdc_policy_callback_response_t* userData);

  // Guards the policy state below, and RDC's policy register and unregister calls. RDC never
  // calls PolicyCallback() while it holds its own policy lock, so PolicyCallback() can take it.
  // The state is static because PolicyCallback() is; rdcd has one RdcAPIServiceImpl.
  static std::mutex policy_mutex_;
  // The streams of each group with a library registration.
  static std::map<uint32_t, std::vector<std::shared_ptr<PolicyStream>>> policy_streams_;
  static size_t policy_stream_count_;  // streams over all groups
  static bool policy_shutdown_;        // set by Shutdown(); no stream starts after it
};

}  // namespace rdc
}  // namespace amd

#endif  // SERVER_INCLUDE_RDC_RDC_API_SERVICE_H_
