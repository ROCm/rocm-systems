// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"

// FlowRecord  is a data storage class for flow trace parameters
class FlowRecord 
{
    public:
        // FlowRecord class constructor. Object stores data flow endpoint parameters
        // @param timestamp - endpoint timestamp
        // @param event_id - endpoint 60-bit event id and 4-bit operation type
        // @param track_id - endpoint track id
        FlowRecord(const rocprofvis_dm_timestamp_t start_timestamp,
                   const rocprofvis_dm_timestamp_t end_timestamp,
                   const rocprofvis_dm_event_id_t  event_id,
                   const rocprofvis_dm_track_id_t  track_id,
                   const rocprofvis_dm_index_t     category_id,
                   const rocprofvis_dm_index_t     symbol_id,
                   const rocprofvis_dm_event_level_t level)
        :
            m_start_timestamp(start_timestamp), m_end_timestamp(end_timestamp), m_event_id(event_id), m_track_id(track_id), m_category_id(category_id),m_symbol_id(symbol_id),m_level(level) {};
        // Returns endpoint timestamp
        rocprofvis_dm_timestamp_t       Timestamp() {return m_start_timestamp;}
        // Returns endpoint end timestamp
        rocprofvis_dm_timestamp_t       EndTimestamp() { return m_end_timestamp; }
        // Returns endpoint event id (60-bit event id and 4-bit operation type)
        rocprofvis_dm_event_id_t        EventId() {return m_event_id;}
        // Returns endpoint track id
        rocprofvis_dm_track_id_t        TrackId() {return m_track_id;}
        // Returns endpoint category id
        rocprofvis_dm_index_t           CategoryId() { return m_category_id; }
        // Returns endpoint category id
        rocprofvis_dm_index_t           SymbolId() { return m_symbol_id; }
        // Returns endpoint level in track
        rocprofvis_dm_event_level_t     Level() { return m_level; }
    private:
        // endpoint timestamp
        rocprofvis_dm_timestamp_t           m_start_timestamp;
        // endpoint end timestamp
        rocprofvis_dm_timestamp_t           m_end_timestamp;
        // endpoint 60-bit event id and 4-bit operation type
        rocprofvis_dm_event_id_t            m_event_id;
        // endpoint track id
        rocprofvis_dm_track_id_t            m_track_id;
        // category id
        rocprofvis_dm_index_t               m_category_id;
        // name id
        rocprofvis_dm_index_t               m_symbol_id;
        // endpoint level in track
        rocprofvis_dm_event_level_t         m_level;
};

