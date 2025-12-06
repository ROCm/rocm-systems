// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdint.h>

// Enums extracted from rocprofvis_controller_enums.h that are needed by the model layer

typedef enum rocprofvis_event_data_category_enum_t
{
    // Internal information, user should not see it
    kRocProfVisEventEssentialDataInternal,
    // Uncategorized information
    kRocProfVisEventEssentialDataUncategorized,
    // Event Id
    kRocProfVisEventEssentialDataId,
    // Event category
    kRocProfVisEventEssentialDataCategory,
    // Event name
    kRocProfVisEventEssentialDataName,
    // Event start
    kRocProfVisEventEssentialDataStart,
    // Event end
    kRocProfVisEventEssentialDataEnd,
    // Event duration
    kRocProfVisEventEssentialDataDuration,
    // Event node id
    kRocProfVisEventEssentialDataNode,
    // Event PID
    kRocProfVisEventEssentialDataProcess,
    // Event TID
    kRocProfVisEventEssentialDataThread,
    // Event Agent type - GPU/CPU
    kRocProfVisEventEssentialDataAgentType,
    // Event agent index
    kRocProfVisEventEssentialDataAgentIndex,
    // Event queue
    kRocProfVisEventEssentialDataQueue,
    // Event stream
    kRocProfVisEventEssentialDataStream,
    // Event track
    kRocProfVisEventEssentialDataTrack,
    // Event stream track
    kRocProfVisEventEssentialDataStreamTrack,
    // Event track level
    kRocProfVisEventEssentialDataLevel,
    // Event stream track level
    kRocProfVisEventEssentialDataStreamLevel,

} rocprofvis_event_data_category_enum_t;
