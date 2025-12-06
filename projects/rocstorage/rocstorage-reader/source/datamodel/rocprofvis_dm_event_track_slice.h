// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_dm_track_slice.h"
#include "rocprofvis_dm_event_record.h"
#include <vector>
#include <memory>

namespace RocProfVis
{
namespace DataModel
{

/* Class for Event time slices, based on pvDmTrackSlice
** Overrides set of methods for accessing array of Event records 
*/
class EventTrackSlice : public TrackSlice {
    public:
        // EventTrackSlice class constructor
        // @param ctx - Track object context
        // @param start - start time of the slice
        // @param end - end time of the slice
        EventTrackSlice(   Track* ctx, 
                                rocprofvis_dm_timestamp_t start, 
                                rocprofvis_dm_timestamp_t end); 
        // EventTrackSlice class destructor, not required unless declared as virtual
        ~EventTrackSlice(){}
        // Method to add event record to the time slice
        // @param data - reference to a structure with record data
        // @return status of operation
        rocprofvis_dm_result_t  AddRecord( rocprofvis_db_record_data_t & data) override;
        // Method to get amount of memory used by the class object
        // @return used memory size
        rocprofvis_dm_size_t    GetMemoryFootprint() override;
        // Method to get time slice number of event records
        // @return number of records
        rocprofvis_dm_size_t    GetNumberOfRecords() override;
        // Method to convert a timestamp to index in event record array 
        // @param timestamp - timestamp to convert
        // @param index - reference to index
        // @return status of operation
        rocprofvis_dm_result_t  ConvertTimestampToIndex(const rocprofvis_dm_timestamp_t timestamp, rocprofvis_dm_index_t & index) override;
        // Method to get event timestamp value by provided index of record
        // @param index - index of the record
        // @param timestamp - reference to timestamp value
        // @return status of operation
        rocprofvis_dm_result_t  GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp) override;
        // Method to get event ID value by provided index of record
        // @param index - index of the record
        // @param id - reference to ID value
        // @return status of operation
        rocprofvis_dm_result_t  GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_id_t & id) override;
        // Method to get event operation value by provided index of record
        // @param index - index of the record
        // @param op - reference to operation value
        // @return status of operation
        rocprofvis_dm_result_t  GetRecordOperationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_op_t & op) override;
        // Method to get event operation string by provided index of record
        // @param index - index of the record
        // @param op - reference to operation string
        // @return status of operation        
        rocprofvis_dm_result_t  GetRecordOperationStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & op) override;
        // Method to get event duration value by provided index of record
        // @param index - index of the record
        // @param duration - reference to duration value
        // @return status of operation
        rocprofvis_dm_result_t  GetRecordDurationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_duration_t & duration) override;
        // Method to get event category string index by provided index of record
        // @param index - index of the record
        // @param category_index - reference to category index
        // @return status of operation       
        rocprofvis_dm_result_t  GetRecordCategoryIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & category_index) override;
        // Method to get event symbol string index by provided index of record
        // @param index - index of the record
        // @param symbol_index - reference to symbol index
        // @return status of operation 
        rocprofvis_dm_result_t  GetRecordSymbolIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & symbol_index) override;
        // Method to get event category string by provided index of record
        // @param index - index of the record
        // @param category_charptr - reference to category string
        // @return status of operation         
        rocprofvis_dm_result_t  GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr) override;
        // Method to get event symbol string by provided index of record
        // @param index - index of the record
        // @param symbol_charptr - reference to symbol string
        // @return status of operation  
        rocprofvis_dm_result_t  GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol_charptr) override;
        // Method to get event level value by provided index of record
        // @param index - index of the record
        // @param level - graph level for the event
        // @return status of operation
        rocprofvis_dm_result_t GetRecordGraphLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t & level) override;

    private:

        // vector array of event records
        std::vector<EventRecord*>    m_samples;

};

}  // namespace DataModel
}  // namespace RocProfVis