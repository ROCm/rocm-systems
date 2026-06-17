// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#define ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE 1
#include "lib/aqlprofile/aql_profile_v2.h"
#undef ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE

int
aql_profile_v2_c_compatibility_test(void)
{
    aqlprofile_att_buffer_status_t status    = {0};
    aqlprofile_spm_parameter_t     parameter = {AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE, 0};
    aqlprofile_spm_decode_query_t  query     = AQLPROFILE_SPM_DECODE_QUERY_LAST;

    aqlprofile_spm_available_configuration_t     spm_config = {0};
    aqlprofile_spm_available_configurations_cb_t spm_cb     = NULL;
    typedef hsa_status_t (*spm_query_fn_t)(
        aqlprofile_agent_handle_t, aqlprofile_spm_available_configurations_cb_t, void*);
    spm_query_fn_t spm_query_fn = &aqlprofile_spm_query_agent_configurations;

    (void) status;
    (void) parameter;
    (void) query;
    (void) spm_config;
    (void) spm_cb;
    (void) spm_query_fn;
    return 0;
}
