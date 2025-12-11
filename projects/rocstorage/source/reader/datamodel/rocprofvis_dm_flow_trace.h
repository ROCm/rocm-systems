// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "rocprofvis_dm_base.h"
#include "rocprofvis_dm_flow_record.h"
#include "vector"
#include "memory"

namespace RocProfVis
{
namespace DataModel
{

class Trace;

// FlowTrace is class of container object for flow trace records (FlowRecord)
class FlowTrace : public DmBase {
    public:
        // FlowTrace class constructor
        // @param  ctx - Trace object context
        // @param event_id - 60-bit event id and 4-bit operation type 
        FlowTrace(    Trace* ctx, 
                            rocprofvis_dm_event_id_t event_id) : 
                            m_ctx(ctx), 
                            m_event_id(event_id){}; 
        // FlowTrace class destructor, not required unless declared as virtual
        ~FlowTrace(){};
        // Method to add flow trace record
        // @param data - reference to a structure with the record data
        // @return status of operation
        rocprofvis_dm_result_t  AddRecord( rocprofvis_db_flow_data_t & data);
        // Method to get number of flow trace records
        // @return number of records
        rocprofvis_dm_size_t    GetNumberOfRecords() {return m_flows.size();};
        // Method to get amount of memory used by FlowTrace class object
        // @return used memory size
        rocprofvis_dm_size_t    GetMemoryFootprint();

        // Return trace context pointer
        // @return context pointer
        Trace*                     Ctx() {return m_ctx;};
        // Returns event id
        // @return 60-bit event id and 4-bit operation type
        rocprofvis_dm_event_id_t        EventId() {return m_event_id;}
        // Returns class mutex
        std::shared_mutex* Mutex() override { return &m_lock; }

        // Method to read FlowTrace object property as uint64
        // @param property - property enumeration rocprofvis_dm_flowtrace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation        
        rocprofvis_dm_result_t          GetPropertyAsUint64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            uint64_t* value) override;
        // Method to read FlowTrace object property as char*
        // @param property - property enumeration rocprofvis_dm_flowtrace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation           
        rocprofvis_dm_result_t          GetPropertyAsCharPtr(
                                                            rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index,
                                                            char** value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_flowtrace_property_t
        // @return pointer to property symbol string 
        const char*                     GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif

    private:

        // 60-bit event id and 4-bit operation type
        rocprofvis_dm_event_id_t                     m_event_id;
        // pointer to Trace context object
        Trace*                                  m_ctx;
        // vector array of flow trace records
        std::vector<FlowRecord>                 m_flows;
        // object mutex, for shared access
        mutable std::shared_mutex               m_lock;

        // Method to get flow trace record timestamp at provided record index
        // @param index - record index
        // @category  - reference to timestamp
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp);
        // Method to get flow trace record end timestamp at provided record index
        // @param index - record index
        // @category  - reference to timestamp
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordEndTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp);
        // Method to get flow trace record event at provided record index
        // @param index - record index
        // @category  - reference to event id (60-bit event id and 4-bit operation type)
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_id_t & id);
        // Method to get flow trace record track id at provided record index
        // @param index - record index
        // @track_id  - reference to track id
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordTrackIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_track_id_t & track_id);
        // Method to get flow trace record category string at provided record index
        // @param index - record index
        // @category_charptr  - reference category string
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr);
        // Method to get flow trace record symbol string at provided record index
        // @param index - record index
        // @symbol_charptr  - reference to symbol string
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol_charptr);
        // Method to get flow trace record level at provided record index
        // @param index - record index
        // @level  - reference to level
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t & level);
};

}  // namespace DataModel
}  // namespace RocProfVis