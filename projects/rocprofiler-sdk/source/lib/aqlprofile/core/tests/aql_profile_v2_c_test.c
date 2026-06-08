// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#define ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE 1
#include "lib/aqlprofile/aql_profile_v2.h"
#undef ROCPROFILER_INTERNAL_AQLPROFILE_INCLUDE

/* C-side ABI lock for the WGP-harvesting types added to V2. Mirrors the
 * static_assert in aql_profile_v2.h so the size guarantee is verified from the
 * C side too, not just from C++ consumers. */
_Static_assert(sizeof(aqlprofile_cu_bitmap_t) == AQLPROFILE_DRM_CU_BITMAP_NUM_SE *
                                                     AQLPROFILE_DRM_CU_BITMAP_NUM_SA_PER_SE *
                                                     sizeof(uint32_t),
               "aqlprofile_cu_bitmap_t size mismatch on C side; "
               "V2 ABI broken or AQLPROFILE_DRM_CU_BITMAP_NUM_* macros out of sync");

int
aql_profile_v2_c_compatibility_test(void)
{
    aqlprofile_att_buffer_status_t status    = {0};
    aqlprofile_spm_parameter_t     parameter = {AQLPROFILE_SPM_PARAMETER_TYPE_BUFFER_SIZE, 0};
    aqlprofile_spm_decode_query_t  query     = AQLPROFILE_SPM_DECODE_QUERY_LAST;

    /* Confirm the WGP-harvesting V2 types are visible and constructible from C. */
    aqlprofile_cu_bitmap_t     bitmap  = {{{0}}};
    aqlprofile_agent_info_v2_t info_v2 = {0};

    aqlprofile_spm_available_configuration_t     spm_config = {0};
    aqlprofile_spm_available_configurations_cb_t spm_cb     = NULL;
    typedef hsa_status_t (*spm_query_fn_t)(
        aqlprofile_agent_handle_t, aqlprofile_spm_available_configurations_cb_t, void*);
    spm_query_fn_t spm_query_fn = &aqlprofile_spm_query_agent_configurations;

    (void) status;
    (void) parameter;
    (void) query;
    (void) bitmap;
    (void) info_v2;
    (void) spm_config;
    (void) spm_cb;
    (void) spm_query_fn;
    return 0;
}
