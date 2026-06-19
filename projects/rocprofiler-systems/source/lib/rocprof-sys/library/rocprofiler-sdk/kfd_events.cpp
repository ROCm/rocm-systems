// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// All kfd_events<Backend> template implementations are defined in kfd_events.hpp.
// This explicit instantiation stamps out the production specialisation.

#include "library/rocprofiler-sdk/kfd_events.hpp"

#if ROCPROFILER_VERSION >= 10000

namespace rocprofsys::rocprofiler_sdk
{

template void
kfd_event_metadata_initialize<backend>(const client_data<backend>*);
template void
tool_kfd_page_fault_callback<backend>(const client_data<backend>*,
                                      const backend::kfd_page_fault_record*);
template void
tool_kfd_page_migrate_callback<backend>(const client_data<backend>*,
                                        const backend::kfd_page_migrate_record*);
template void
tool_kfd_queue_callback<backend>(const client_data<backend>*,
                                 const backend::kfd_queue_record*);
template void
tool_kfd_event_queue_callback<backend>(const client_data<backend>*,
                                       const backend::kfd_event_queue_record*);
template void
tool_kfd_event_unmap_from_gpu_callback<backend>(const client_data<backend>*,
                                                const backend::kfd_event_unmap_record*);
template void
tool_kfd_event_dropped_events_callback<backend>(const client_data<backend>*,
                                                const backend::kfd_event_dropped_record*);

}  // namespace rocprofsys::rocprofiler_sdk

#endif
