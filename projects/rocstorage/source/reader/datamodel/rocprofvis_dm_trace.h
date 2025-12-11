// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocprofvis_dm_flow_trace.h"
#include "rocprofvis_dm_stack_trace.h"
#include "rocprofvis_dm_track_slice.h"
#include "rocprofvis_dm_ext_data.h"
#include "rocprofvis_dm_track.h"
#include "rocprofvis_dm_table.h"
#include <vector> 
#include <memory> 
#include <map>
#include <mutex>

namespace RocProfVis
{
namespace DataModel
{

class RpvDatabase;

typedef std::map<uint64_t, uint32_t> event_level_map_t;

// Trace class is main data model class. It serves as a container of all other data model objects
// and dispatcher for most of the database component calls and external interface calls
class Trace : public DmBase{
    public:
        // class constructor
        Trace();
        // Trace class destructor, not required unless declared as virtual
        ~Trace(){}
        // Returns trace start time, as uint64_t timestamp
        rocprofvis_dm_timestamp_t                       StartTime() {return m_parameters.start_time;}
        // Returns trace end time, as uint64_t timestamp
        rocprofvis_dm_timestamp_t                       EndTime() {return m_parameters.end_time;}
        // Return number of track objects
        rocprofvis_dm_size_t                            NumberOfTracks() {return m_tracks.size();}
        // Returns number table objects
        rocprofvis_dm_size_t                            NumberOfTables() {return m_tables.size();}
        // Returns pointer to database object
        rocprofvis_dm_database_t                        Database() { return m_db; }
        // Returns pointer to database binding structure 
        rocprofvis_dm_db_bind_struct*                   BindingInfo() {return &m_binding_info;}
        // Returns class mutex
        std::shared_mutex*                              Mutex() override { return &m_lock; }

        std::shared_mutex*                              EventPropertyMutex(rocprofvis_dm_event_property_type_t type){ return &m_event_property_lock[type];}

        rocprofvis_dm_size_t                            NumberOfHistogramBuckets() {return m_parameters.histogram_bucket_count;}
        rocprofvis_dm_size_t                            HistogramBucketsSize() {return m_parameters.histogram_bucket_size;}

        // Method to bind database object
        // @param db - pointer to database
        // @param bind_data - reference to pointer to bind data structure
        // @return status of operation 
        rocprofvis_dm_result_t                          BindDatabase(rocprofvis_dm_database_t db, rocprofvis_dm_db_bind_struct* &bind_data);
        // Method to delete a time slice with provide start and stop timestamps
        // @param start - time slice start timestamp
        // @param stop - time slice stop timestamp
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteSliceAtTimeRange(rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end);
        // Method to delete a time slice with provided handle
        // param track - track id to delete slice from
        // @param slice - handle
        // @return status of operation
        rocprofvis_dm_result_t DeleteSliceByHandle(rocprofvis_dm_track_id_t track, rocprofvis_dm_handle_t   slice);
        // Method to delete all time slices
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteAllSlices();
        // Method to delete flowtrace, stacktrace and extended data property objects for event
        // @param type - property type (kEventFlowTrace, kEventStackTrace or kEventExtData)
        // @param event_id - 60-bit event id and 4-bit operation type
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteEventPropertyFor(rocprofvis_dm_event_property_type_t type, rocprofvis_dm_event_id_t event_id);
        // Method to delete flowtrace, stacktrace and extended data property objects for all events
        // @param type - property type (kEventFlowTrace, kEventStackTrace or kEventExtData)
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteAllEventPropertiesFor(rocprofvis_dm_event_property_type_t type);
        // Method to delete table object at provided index
        // @param index of table
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteTableAt(rocprofvis_dm_table_id_t id);
        // Method to delete all table objects
        // @return status of operation 
        rocprofvis_dm_result_t                          DeleteAllTables();

        // Method to get pointer to a string at specified index. Used for event records category and symbol
        // @return pointer to a string 
        rocprofvis_dm_charptr_t                         GetStringAt(rocprofvis_dm_index_t index);

        // Method to get the string ids of strings inside string table that contain all of the passed in substrings
        // @param num - number of substrings strings to search for
        // @param substrings - array of substrings to search for
        // @param indices - output array of string table indices 
        // @return status of operation 
        rocprofvis_dm_result_t                          GetStringIndicesWithSubstring(rocprofvis_dm_num_string_table_filters_t num, rocprofvis_dm_string_table_filters_t substrings, std::vector<rocprofvis_dm_index_t>& indices);

        // Method to read Trace object property as uint64
        // @param property - property enumeration rocprofvis_dm_trace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to uint64_t return value
        // @return status of operation  
        rocprofvis_dm_result_t                          GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value) override;
        // Method to read Trace object property as handle rocprofvis_dm_handle_t (aka void*)
        // @param property - property enumeration rocprofvis_dm_trace_property_t
        // @param index - index of any indexed property
        // @param value - pointer reference to rocprofvis_dm_handle_t return value
        // @return status of operation 
        rocprofvis_dm_result_t                          GetPropertyAsHandle(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t* value) override;
#ifdef TEST
        // Method to get property symbol for testing/debugging
        // @param property - property enumeration rocprofvis_dm_trace_property_t
        // @return pointer to property symbol string 
        const char*                                     GetPropertySymbol(rocprofvis_dm_property_t property) override;
#endif

    private:
        // Method to get handle to a Track object at provided index
        // @param index - index of track object
        // @param track - reference to track handle
        // @return status of operation 
        rocprofvis_dm_result_t                          GetTrackAtIndex(rocprofvis_dm_property_index_t index, rocprofvis_dm_track_t & track);
        // Method to get flow trace handle for specified event id  
        // @param event_id - 60-bit event id and 4-bit operation type 
        // @param flowtrace - reference to flowtrace handle
        // @return status of operation         
        rocprofvis_dm_result_t                          GetFlowTraceHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_flowtrace_t & flowtrace);
        // Method to get stack trace handle for specified event id  
        // @param event_id - 60-bit event id and 4-bit operation type 
        // @param stacktrace - reference to stacktrace handle
        // @return status of operation  
        rocprofvis_dm_result_t                          GetStackTraceHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_stacktrace_t & stacktrace);
        // Method to get extended info handle for specified event id  
        // @param event_id - 60-bit event id and 4-bit operation type 
        // @param extinfo - reference to extended info handle
        // @return status of operation  
        rocprofvis_dm_result_t                          GetExtInfoHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_extdata_t & extinfo);
        // Method to get queried table handle at specified index  
        // @param id - id of queried table
        // @param extinfo - reference to table handle
        // @return status of operation  
        rocprofvis_dm_result_t                          GetTableHandle(rocprofvis_dm_table_id_t id, rocprofvis_dm_table_t & table);
        // Method to get amount of memory used by Trace object, includes memory footprint of all other data model objects
        // @return used memory size        
        rocprofvis_dm_size_t                            GetMemoryFootprint();
        // Static method to add track object. Used by database component via binding interface
        // @param object - trace object handle to add new track to
        // @param params - pointer to track parameters structure (shared with database component) 
        // @return status of operation 
        static rocprofvis_dm_result_t                   AddTrack(const rocprofvis_dm_trace_t object, rocprofvis_dm_track_params_t * params);
        // Static method to create and add slice object to a track. Used by database component via binding interface
        // @param object - trace object handle to add new track to
        // @param track_id - id of track the slice to be added to
        // @param start - start timestamp of the slice
        // @param end - end timestamp of the track 
        // @return status of operation 
        static rocprofvis_dm_slice_t                    AddSlice(const rocprofvis_dm_trace_t object, const rocprofvis_dm_track_id_t track_id, const rocprofvis_dm_timestamp_t start, const rocprofvis_dm_timestamp_t end);  
        // Static method to  add new record to a slice object. Used by database component via binding interface
        // @param object - slice object handle to add new record to
        // @param data - reference to a new record parameters structure
        // @return status of operation         
        static rocprofvis_dm_result_t                   AddRecord(const rocprofvis_dm_slice_t object, rocprofvis_db_record_data_t & data);      
        // Static method to  add new string to list of strings. Used by database component via binding interface
        // @param object - trace object handle to add new string to
        // @param string_value - pointer to string
        // @return status of operation
        static rocprofvis_dm_index_t                    AddString(const rocprofvis_dm_trace_t object,  const char* stringValue);
        // Static method to add event flow trace. Used by database component via binding interface
        // @param object - trace object handle to add flowtrace object to.
        // @param event_id - 60-bit event id and 4-bit operation type
        // @return status of operation
        static rocprofvis_dm_flowtrace_t                AddFlowTrace(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id);
        // Static method to add flow record to event flow trace. Used by database component via binding interface
        // @param object - flowtrace object handle to add new flow record to.
        // @param data - reference to a new record parameters structure
        // @return status of operation
        static rocprofvis_dm_result_t                   AddFlow(const rocprofvis_dm_flowtrace_t object, rocprofvis_db_flow_data_t & data);
        // Static method to add event stack trace. Used by database component via binding interface
        // @param object - trace object handle to add stacktrace object to.
        // @param event_id - 60-bit event id and 4-bit operation type
        // @return status of operation
        static rocprofvis_dm_stacktrace_t               AddStackTrace(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id);
        // Static method to add stack frame to event stack trace. Used by database component via binding interface
        // @param object - stacktrace object handle to add new stack frame record to.
        // @param data - reference to a new record parameters structure
        // @return status of operation       
        static rocprofvis_dm_result_t                   AddStackFrame(const rocprofvis_dm_stacktrace_t object, rocprofvis_db_stack_data_t & data);
        // Static method to add event extended data. Used by database component via binding interface
        // @param object - trace object handle to add extended data object to.
        // @param event_id - 60-bit event id and 4-bit operation type
        // @return status of operation        
        static rocprofvis_dm_flowtrace_t                AddExtData(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id);
        // Static method to add new record to extended data object. Used by database component via binding interface
        // @param object - extended data object handle to add new record to.
        // @param data - reference to a new record parameters structure
        // @return status of operation      
        static rocprofvis_dm_result_t                   AddExtDataRecord(const rocprofvis_dm_extdata_t object, rocprofvis_db_ext_data_t & data);
        // Static method to add new table. Used by database component via binding interface
        // @param object - trace object handle to add table object to.
        // @param query - pointer to table query string
        // @param description - pointer to table description string
        // @return status of operation           
        static rocprofvis_dm_table_t                    AddTable(const rocprofvis_dm_trace_t object, rocprofvis_dm_charptr_t query, rocprofvis_dm_charptr_t description);
        // Static method to add new empty row to table object. Used by database component via binding interface
        // @param object - table object handle to add new row to.
        // @return status of operation  
        static rocprofvis_dm_table_row_t                AddTableRow(const rocprofvis_dm_table_t object);
        // Static method to add new column name to table object. Used by database component via binding interface
        // @param object - table object handle to add new column to.
        // @param column_name - name of new column
        // @return status of operation  
        static rocprofvis_dm_result_t                   AddTableColumn(const rocprofvis_dm_table_t object, rocprofvis_dm_charptr_t column_name);
        // Static method to add new cell to table row object. Used by database component via binding interface
        // @param object - table row object handle to add new cell to.
        // @param cell_value - pointer to table value string
        // @return status of operation  
        static rocprofvis_dm_result_t                   AddTableRowCell(const rocprofvis_dm_table_row_t object, rocprofvis_dm_charptr_t cell_value);
        // Static method to  add event level to a map array. Used by database component via binding interface
        // @param object - trace object handle to add new string to
        // @param event_id - 60-bit event id and 4-bit operation type
        // @param level - event level in graph
        // @return status of operation
        static rocprofvis_dm_result_t                   AddEventLevel(const rocprofvis_dm_trace_t object,  const rocprofvis_dm_event_id_t event_id, rocprofvis_dm_event_level_t level);

        static rocprofvis_dm_result_t                   CheckSliceExists(const rocprofvis_dm_trace_t object, const rocprofvis_dm_timestamp_t start, const rocprofvis_dm_timestamp_t end, const rocprofvis_db_num_of_tracks_t num, const rocprofvis_db_track_selection_t tracks);
        static rocprofvis_dm_result_t                   CheckEventPropertyExists(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_property_type_t type, const rocprofvis_dm_event_id_t event_id);
        static rocprofvis_dm_result_t                   CheckTableExists(const rocprofvis_dm_trace_t object, const rocprofvis_dm_table_id_t table_id);
        static rocprofvis_dm_result_t                   CompleteSlice(const rocprofvis_dm_slice_t object);
        static rocprofvis_dm_result_t                   RemoveSlice(const rocprofvis_dm_trace_t trace, const rocprofvis_dm_track_id_t track_id, const rocprofvis_dm_slice_t object);

        // trace parameters structure
        rocprofvis_dm_trace_params_t                    m_parameters;
        // binding info structure
        rocprofvis_dm_db_bind_struct                    m_binding_info;
        // pointer to database object
        rocprofvis_dm_database_t                        m_db;
        // vector array of Track objects
        std::vector<std::unique_ptr<Track>>             m_tracks;
        // vector array of Flow trace objects
        std::vector<std::shared_ptr<FlowTrace>>         m_flow_traces;
        // vector array of Stack trace objects
        std::vector<std::shared_ptr<StackTrace>>        m_stack_traces;
        // vector array of Extended data objects
        std::vector<std::shared_ptr<ExtData>>           m_ext_data;
        // vector array of Table objects
        std::vector<std::shared_ptr<Table>>             m_tables;
        // vector array of strings
        std::vector<std::string>                        m_strings;
        // map of every event level in graph
        event_level_map_t                               m_event_level_map;
        // object mutex, for shared access
        mutable std::shared_mutex                       m_lock;
        // object mutex, for shared access
        mutable std::shared_mutex                       m_event_property_lock[kRPVDMNumEventPropertyTypes];

};


}  // namespace DataModel
}  // namespace RocProfVis
