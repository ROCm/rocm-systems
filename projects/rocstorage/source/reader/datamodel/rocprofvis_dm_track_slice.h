// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "internal_types.h"
#include "rocprofvis_dm_base.h"
#include <bitset>
#include <map>

namespace RocProfVis
{
namespace DataModel
{

class Track;

constexpr uint32_t kMemPoolBitSetSize = 1024;


struct MemoryPool
{
    void*                           m_base;
    size_t                          m_size;
    uint32_t                        m_pos;

    MemoryPool(size_t size)
    : m_size(size)
    , m_pos(0)
    {
        m_base = ::operator new(size * kMemPoolBitSetSize);
    }

    ~MemoryPool() { 
        ::operator delete(m_base); 
    }
};


// Base class for PMC and Event time slices
// Defines set of virtual methods to access an array of POD storage objects 
class TrackSlice : public DmBase {
    public:
        // TrackSlice class constructor
        // @param start - time slice start timestamp
        // @param start - time slice end timestamp
        TrackSlice(    Track* ctx, 
                            rocprofvis_dm_timestamp_t start, 
                            rocprofvis_dm_timestamp_t end) : 
                            m_ctx(ctx), 
                            m_start_timestamp(start), 
                            m_end_timestamp(end), 
                            m_current_pool(nullptr),
                            m_complete(false){}; 
        // TrackSlice class destructor, required in order to free resources of derived classes
        virtual ~TrackSlice() { Cleanup(); }
        // Pure virtual method to add a record to a time slice
        // @param data - reference to a structure with record data
        // @return status of operation       
        virtual rocprofvis_dm_result_t  AddRecord( rocprofvis_db_record_data_t & data) = 0;
        // Pure virtual method to get amount of memory used by the class object
        // @return used memory size
        virtual rocprofvis_dm_size_t    GetMemoryFootprint() = 0;
        // Pure virtual method to get time slice number of event records
        // @return number of records
        virtual rocprofvis_dm_size_t    GetNumberOfRecords() = 0;
        // Pure virtual method to convert a timestamp to index in event record array 
        // @param timestamp - timestamp to convert
        // @param index - reference to index
        // @return status of operation
        virtual rocprofvis_dm_result_t  ConvertTimestampToIndex(const rocprofvis_dm_timestamp_t timestamp, 
                                                                    rocprofvis_dm_index_t & index)=0;

        // Returns context pointer
        Track*                     Ctx() {return m_ctx;};
        // Returns start timestamp
        rocprofvis_dm_timestamp_t       StartTime() {return m_start_timestamp;}
        // Returns end timestamp
        rocprofvis_dm_timestamp_t       EndTime() {return m_end_timestamp;}
        // Returns class mutex
        std::shared_mutex*              Mutex() override { return &m_lock; }

        void SetComplete();
        void WaitComplete();

        // Pure virtual method to get event timestamp value by provided index of record
        // @param index - index of the record
        // @param timestamp - reference to timestamp value
        // @return status of operation
        virtual rocprofvis_dm_result_t  GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp) = 0;
        // Pure virtual method to get event ID value by provided index of record
        // @param index - index of the record
        // @param id - reference to ID value
        // @return status of operation    
        virtual rocprofvis_dm_result_t  GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_id_t & id);
        // Pure virtual method to get event operation value by provided index of record
        // @param index - index of the record
        // @param op - reference to operation value
        // @return status of operation        
        virtual rocprofvis_dm_result_t  GetRecordOperationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_op_t & op);
        // Pure virtual method to get event operation string by provided index of record
        // @param index - index of the record
        // @param op - reference to operation string
        // @return status of operation   
        virtual rocprofvis_dm_result_t  GetRecordOperationStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & op);
        // Pure virtual method to get PMC value by provided index of record
        // @param index - index of the record
        // @param value - reference to PMC value
        // @return status of operation        
        virtual rocprofvis_dm_result_t  GetRecordValueAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_value_t & value);
        // Pure virtual method to get event duration value by provided index of record
        // @param index - index of the record
        // @param duration - reference to duration value
        // @return status of operation
        virtual rocprofvis_dm_result_t  GetRecordDurationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_duration_t & duration);
        // Pure virtual method to get event category string index by provided index of record
        // @param index - index of the record
        // @param category_index - reference to category index
        // @return status of operation  
        virtual rocprofvis_dm_result_t  GetRecordCategoryIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & category_index);
        // Pure virtual method to get event symbol string index by provided index of record
        // @param index - index of the record
        // @param symbol_index - reference to symbol index
        // @return status of operation 
        virtual rocprofvis_dm_result_t  GetRecordSymbolIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & symbol_index);
        // Pure virtual method to get event category string by provided index of record
        // @param index - index of the record
        // @param category_charptr - reference to category string
        // @return status of operation   
        virtual rocprofvis_dm_result_t  GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr);
        // Pure virtual method to get event symbol string by provided index of record
        // @param index - index of the record
        // @param symbol_charptr - reference to symbol string
        // @return status of operation  
        virtual rocprofvis_dm_result_t  GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol_charptr);
        // Method to get event level value by provided index of record
        // @param index - index of the record
        // @param level - graph level for the event
        // @return status of operation
        virtual rocprofvis_dm_result_t GetRecordGraphLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t & level);

        // Method to read TrackSlice object property as uint64
        // @param property - property enumeration rocprofvis_dm_slice_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation  
        rocprofvis_dm_result_t          GetPropertyAsUint64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            uint64_t* value) override;
        // Method to read TrackSlice object property as int64
        // @param property - property enumeration rocprofvis_dm_slice_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to int64_t return value
        // @return status of operation  
        rocprofvis_dm_result_t          GetPropertyAsInt64(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            int64_t* value) override;
        // Method to read TrackSlice object property as char*
        // @param property - property enumeration rocprofvis_dm_slice_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to char* return value
        // @return status of operation  
        rocprofvis_dm_result_t          GetPropertyAsCharPtr(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            char** value) override;
        // Method to read TrackSlice object property as double
        // @param property - property enumeration rocprofvis_dm_slice_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to double return value
        // @return status of operation  
        rocprofvis_dm_result_t          GetPropertyAsDouble(rocprofvis_dm_property_t property, 
                                                            rocprofvis_dm_property_index_t index, 
                                                            double* value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_slice_property_t
        // @return pointer to property symbol string 
        const char*                     GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif

    private:
        // time slice start 
        rocprofvis_dm_timestamp_t                    m_start_timestamp;
        // time slice end
        rocprofvis_dm_timestamp_t                    m_end_timestamp;
        // context pointer
        Track*                                      m_ctx;
        // object mutex, for shared access
        mutable std::shared_mutex                   m_lock;
        std::map<void*, MemoryPool*>                m_object_pools;
        MemoryPool*                                 m_current_pool;
        bool                                        m_complete;
        std::condition_variable_any                 m_cv;

        void  Cleanup();

    protected:
        void* Allocate(size_t rec_size);
};

}  // namespace DataModel
}  // namespace RocProfVis