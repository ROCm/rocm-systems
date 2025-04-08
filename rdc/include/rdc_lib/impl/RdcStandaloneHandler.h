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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
#define INCLUDE_RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
#include <grpcpp/grpcpp.h>

#include <future>
#include <memory>
#include <thread>

#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc.h"
#include "rdc_lib/RdcHandler.h"

namespace amd {
namespace rdc {

class RdcStandaloneHandler : public RdcHandler {
 public:
  // Job RdcAPI
  rdc_status_t rdc_job_start_stats(rdc_gpu_group_t groupId, const char job_id[64],
                                   uint64_t update_freq) override;
  rdc_status_t rdc_job_get_stats(const char jobId[64], rdc_job_info_t* p_job_info) override;
  rdc_status_t rdc_job_stop_stats(const char job_id[64]) override;
  rdc_status_t rdc_job_remove(const char job_id[64]) override;
  rdc_status_t rdc_job_remove_all() override;

  // Discovery RdcAPI
  rdc_status_t rdc_device_get_all(uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES],
                                  uint32_t* count) override;
  rdc_status_t rdc_device_get_attributes(uint32_t gpu_index,
                                         rdc_device_attributes_t* p_rdc_attr) override;
  rdc_status_t rdc_device_get_component_version(rdc_component_t component,
                                                rdc_component_version_t* p_rdc_compv) override;

  // Group RdcAPI
  rdc_status_t rdc_group_gpu_create(rdc_group_type_t type, const char* group_name,
                                    rdc_gpu_group_t* p_rdc_group_id) override;
  rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t groupId, uint32_t gpu_index) override;
  rdc_status_t rdc_group_field_create(uint32_t num_field_ids, rdc_field_t* field_ids,
                                      const char* field_group_name,
                                      rdc_field_grp_t* rdc_field_group_id) override;
  rdc_status_t rdc_group_field_get_info(rdc_field_grp_t rdc_field_group_id,
                                        rdc_field_group_info_t* field_group_info) override;
  rdc_status_t rdc_group_gpu_get_info(rdc_gpu_group_t p_rdc_group_id,
                                      rdc_group_info_t* p_rdc_group_info) override;
  rdc_status_t rdc_group_get_all_ids(rdc_gpu_group_t group_id_list[], uint32_t* count) override;
  rdc_status_t rdc_group_field_get_all_ids(rdc_field_grp_t field_group_id_list[],
                                           uint32_t* count) override;
  rdc_status_t rdc_group_gpu_destroy(rdc_gpu_group_t p_rdc_group_id) override;
  rdc_status_t rdc_group_field_destroy(rdc_field_grp_t rdc_field_group_id) override;

  // Field RdcAPI
  rdc_status_t rdc_field_watch(rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id,
                               uint64_t update_freq, double max_keep_age,
                               uint32_t max_keep_samples) override;
  rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index, rdc_field_t field,
                                          rdc_field_value* value) override;
  rdc_status_t rdc_field_get_value_since(uint32_t gpu_index, rdc_field_t field,
                                         uint64_t since_time_stamp, uint64_t* next_since_time_stamp,
                                         rdc_field_value* value) override;
  rdc_status_t rdc_field_unwatch(rdc_gpu_group_t group_id, rdc_field_grp_t field_group_id) override;
  // Diagnostic API
  rdc_status_t rdc_diagnostic_run(rdc_gpu_group_t group_id, rdc_diag_level_t level,
                                  const char* config, size_t config_size,
                                  rdc_diag_response_t* response,
                                  rdc_diag_callback_t* callback) override;
  rdc_status_t rdc_test_case_run(rdc_gpu_group_t group_id, rdc_diag_test_cases_t test_case,
                                 const char* config, size_t config_size,
                                 rdc_diag_test_result_t* result,
                                 rdc_diag_callback_t* callback) override;

  // Control RdcAPI
  rdc_status_t rdc_field_update_all(uint32_t wait_for_update) override;

  // Set one configure
  rdc_status_t rdc_config_set(rdc_gpu_group_t group_id, rdc_config_setting_t setting) override;

  // Get the setting
  rdc_status_t rdc_config_get(rdc_gpu_group_t group_id,
                              rdc_config_setting_list_t* settings) override;

  // Clear the setting
  rdc_status_t rdc_config_clear(rdc_gpu_group_t group_id) override;

  // It is just a client interface under the GRPC framework and is not used as an RDC API.
  // Pure virtual functions need to be overridden
  rdc_status_t get_mixed_component_version(mixed_component_t component,
                                           mixed_component_version_t* p_mixed_compv) override;
  // Policy API
  rdc_status_t rdc_policy_set(rdc_gpu_group_t group_id, rdc_policy_t policy) override;

  rdc_status_t rdc_policy_get(rdc_gpu_group_t group_id, uint32_t* count,
                              rdc_policy_t policies[RDC_MAX_POLICY_SETTINGS]) override;

  rdc_status_t rdc_policy_delete(rdc_gpu_group_t group_id,
                                 rdc_policy_condition_type_t condition_type) override;

  rdc_status_t rdc_policy_register(rdc_gpu_group_t group_id,
                                   rdc_policy_register_callback callback) override;

  rdc_status_t rdc_policy_unregister(rdc_gpu_group_t group_id) override;

  // Health API
  rdc_status_t rdc_health_set(rdc_gpu_group_t group_id, unsigned int components) override;
  rdc_status_t rdc_health_get(rdc_gpu_group_t group_id, unsigned int* components) override;
  rdc_status_t rdc_health_check(rdc_gpu_group_t group_id, rdc_health_response_t* response) override;
  rdc_status_t rdc_health_clear(rdc_gpu_group_t group_id) override;
  rdc_status_t rdc_device_topology_get(uint32_t gpu_index, rdc_device_topology_t* results) override;

  rdc_status_t rdc_link_status_get(rdc_link_status_t* results) override;

  rdc_status_t rdc_get_num_partition(uint32_t index, uint16_t* num_partition) override;

  rdc_status_t rdc_instance_profile_get(uint32_t entity_index,
                                        rdc_instance_resource_type_t resource_type,
                                        rdc_resource_profile_t* profile) override;

  explicit RdcStandaloneHandler(const char* ip_and_port, const char* root_ca,
                                const char* client_cert, const char* client_key);

 private:
  // Helper function to handle the error
  rdc_status_t error_handle(::grpc::Status status, uint32_t rdc_status);

  bool copy_gpu_usage_info(const ::rdc::GpuUsageInfo& src, rdc_gpu_usage_info_t* target);

  std::unique_ptr<::rdc::RdcAPI::Stub> stub_;
  // thread for policy callback

  struct policy_thread_context {
    bool start;
    std::thread* t;
  };

  std::map<uint32_t, struct policy_thread_context> policy_threads_;
};

}  // namespace rdc
}  // namespace amd

extern "C" {
amd::rdc::RdcHandler* make_handler(const char* ip_port, const char* root_ca,
                                   const char* client_cert, const char* client_key);
}

#endif  // INCLUDE_RDC_LIB_IMPL_RDCSTANDALONEHANDLER_H_
