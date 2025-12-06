// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "rocprofvis_dm_base.h"
#include "rocprofvis_dm_stack_record.h"
#include "vector"
#include "memory"

namespace RocProfVis
{
namespace DataModel
{

class Trace;

// StackTrace is class of container object for stack trace records (StackRecord)
class StackTrace : public DmBase {
    public:
        // StackTrace class constructor
        // @param  ctx - Trace object context
        // @param event_id - 60-bit event id and 4-bit operation type 
        StackTrace(    Trace* ctx, 
                            rocprofvis_dm_event_id_t event_id) : 
                            m_ctx(ctx), 
                            m_event_id(event_id){}; 
        // StackTrace class destructor, not required unless declared as virtual
        ~StackTrace(){}
        // Method to add stack trace record
        // @param data - reference to a structure with the record data
        // @return status of operation
        rocprofvis_dm_result_t          AddRecord( rocprofvis_db_stack_data_t & data);
        // Method to get number of stack trace records
        // @return number of records
        rocprofvis_dm_size_t            GetNumberOfRecords() {return m_stack_frames.size();};
        // Method to get amount of memory used by StackTrace class object
        // @return used memory size
        rocprofvis_dm_size_t            GetMemoryFootprint();

        // Return trace context pointer
        // @return context pointer
        Trace*                     Ctx() {return m_ctx;};
        // Returns event id
        // @return 60-bit event id and 4-bit operation type
        rocprofvis_dm_event_id_t        EventId() {return m_event_id;}
        // Returns class mutex
        std::shared_mutex* Mutex() override { return &m_lock; }

        // Method to read StackTrace object property as uint64
        // @param property - property enumeration rocprofvis_dm_stacktrace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation        
        rocprofvis_dm_result_t          GetPropertyAsUint64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            uint64_t* value) override;
        // Method to read StackTrace object property as char*
        // @param property - property enumeration rocprofvis_dm_stacktrace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to char* return value
        // @return status of operation 
        rocprofvis_dm_result_t          GetPropertyAsCharPtr(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            char** value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_stacktrace_property_t
        // @return pointer to property symbol string 
        const char*                     GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif
    private:
        // 60-bit event id and 4-bit operation type
        rocprofvis_dm_event_id_t                     m_event_id;
        // pointer to Trace context object
        Trace*                                  m_ctx;
        // vector array of stack trace records
        std::vector<StackFrameRecord>           m_stack_frames;
        // object mutex, for shared access
        mutable std::shared_mutex               m_lock;

        // Method to get pointer to stack frame symbol string at provided index
        // @param index - record index
        // @param category - reference to symbol string
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordSymbolAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol);
        // Method to get pointer to stack frame arguments string at provided index
        // @param index - record index
        // @param args - reference to arguments string
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordArgsAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t &args);
        // Method to get pointer to stack frame code line string at provided index
        // @param index - record index
        // @param code_line - reference to code line string
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordCodeLineAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & code_line);
        // Method to get stack frame depth at provided index
        // @param index - record index
        // @param depth - reference to stack depth value
        // @return status of operation
        rocprofvis_dm_result_t          GetRecordDepthAt(const rocprofvis_dm_property_index_t index, uint32_t & depth);
};

}  // namespace DataModel
}  // namespace RocProfVis