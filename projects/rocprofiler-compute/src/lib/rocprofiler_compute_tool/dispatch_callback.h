// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                       rocprofiler_counter_config_id_t*             config,
                       rocprofiler_user_data_t* /*user_data*/,
                       void* callback_data_args);