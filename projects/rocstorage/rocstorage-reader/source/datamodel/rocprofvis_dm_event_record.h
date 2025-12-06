// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"

namespace RocProfVis
{
namespace DataModel
{

/*  RpvEventRecord is a data storage class for Event record data for any types of event
**  It's POD (plain old data) class for memory usage optimization.
**  Event represents an object with timeline location and duration as primary properties.
**  Event has event category and event description indexes pointing to string and 
**  symbol tables stored in Trace class. Strings are reindexed when loaded
**  resulting in possibility of using 32-bit indexes
*/ 
class EventRecord 
{
    public:
        // EventRecord class constructor
        // @param event_id - 60-bit event id and 4-bit operation type 
        // @param timestamp - event start time
        // @param duration - event duration
        // @param category_index - index of category name in string array
        // @param symbol_index - index of symbol name in string array
        EventRecord(
                const rocprofvis_dm_event_id_t event_id, 
                const rocprofvis_dm_timestamp_t timestamp, 
                const rocprofvis_dm_duration_t duration, 
                const rocprofvis_dm_index_t category_index, 
                const rocprofvis_dm_index_t symbol_index, 
                const rocprofvis_dm_event_level_t event_level 
                ):
                m_event_id(event_id), 
                m_timestamp(timestamp), 
                m_duration(duration), 
                m_category_index(category_index), 
                m_symbol_index(symbol_index),
                m_event_level(event_level){};

        // Event ID getter
        // @return 60-bit event id
        const rocprofvis_dm_id_t                EventId() {return m_event_id.bitfield.event_id;}
        // Event ID + operation getter
        const rocprofvis_dm_event_id_t          EventIdFull() { return m_event_id; }
        // Timestamp getter
        // @return 64-bit timestamp value
        const rocprofvis_dm_timestamp_t         Timestamp() {return m_timestamp;}
        // Duration getter
        // @return 64-bit duration value
        const rocprofvis_dm_duration_t          Duration() {return m_duration;}
        // Category index getter
        // @return 32-bit index of string array
        const rocprofvis_dm_index_t             CategoryIndex() {return m_category_index;}
        // Symbol index getter
        // @return 32-bit index of string array
        const rocprofvis_dm_index_t             SymbolIndex() {return m_symbol_index;}
        // Operation getter
        // @return 4-bit operation as rocprofvis_dm_event_operation_t enumeration value
        const rocprofvis_dm_event_operation_t   Operation() {return (rocprofvis_dm_event_operation_t)m_event_id.bitfield.event_op;}
        // @return 8-bit event level
        const rocprofvis_dm_id_t                EventLevel() { return m_event_level; }

    private:
        // 60-bit event id and 4-bit operation type , used as a search key for stacktrace, flowtrace and no-essential information
        rocprofvis_dm_event_id_t        m_event_id;      
        // 64-bit timestamp                
        rocprofvis_dm_timestamp_t       m_timestamp;  
        // signed 64-bit duration. Negative number should be invalidated by controller.                 
        rocprofvis_dm_duration_t        m_duration;    
        // 32-bit category index of array of strings                  
        rocprofvis_dm_index_t           m_category_index; 
        // 32-bit symbol index of array of strings               
        rocprofvis_dm_index_t           m_symbol_index;    
        // 8-bit event level on a graph
        rocprofvis_dm_event_level_t     m_event_level;
};

}  // namespace DataModel
}  // namespace RocProfVis