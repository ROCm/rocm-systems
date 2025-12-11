// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_trace.h"
#include "rocprofvis_dm_table_row.h"

namespace RocProfVis
{
namespace DataModel
{

Trace::Trace()
{
    m_db = nullptr;
    m_parameters.start_time=0;
    m_parameters.end_time=0;
    m_parameters.metadata_loaded=false;
}

rocprofvis_dm_result_t Trace::BindDatabase(rocprofvis_dm_database_t db, rocprofvis_dm_db_bind_struct* & bind_data)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(db, ERROR_DATABASE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
     
    m_binding_info.trace_object = this;
    m_binding_info.trace_properties = &m_parameters;
    m_binding_info.FuncAddTrack = AddTrack;
    m_binding_info.FuncAddRecord = AddRecord;
    m_binding_info.FuncAddSlice = AddSlice;
    m_binding_info.FuncRemoveSlice = RemoveSlice;
    m_binding_info.FuncAddString = AddString;
    m_binding_info.FuncAddFlowTrace = AddFlowTrace;
    m_binding_info.FuncAddFlow = AddFlow;
    m_binding_info.FuncAddStackTrace = AddStackTrace;
    m_binding_info.FuncAddStackFrame = AddStackFrame;
    m_binding_info.FuncAddExtData = AddExtData;
    m_binding_info.FuncAddExtDataRecord = AddExtDataRecord;
    m_binding_info.FuncAddTable = AddTable;
    m_binding_info.FuncAddTableRow = AddTableRow;
    m_binding_info.FuncAddTableColumn = AddTableColumn;
    m_binding_info.FuncAddTableRowCell = AddTableRowCell;
    m_binding_info.FuncAddEventLevel = AddEventLevel;
    m_binding_info.FuncCheckEventPropertyExists = CheckEventPropertyExists;
    m_binding_info.FuncCheckSliceExists = CheckSliceExists;
    m_binding_info.FuncCheckTableExists = CheckTableExists;
    m_binding_info.FuncCompleteSlice  = CompleteSlice;
    bind_data = &m_binding_info;
    m_db = db;
    return kRocProfVisDmResultSuccess;
}
    
rocprofvis_dm_result_t Trace::DeleteSliceAtTimeRange(rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end){
    for (int i=0; i < m_tracks.size(); i++)
    {
        m_tracks[i].get()->DeleteSliceAtTime(start, end);
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t
Trace::DeleteSliceByHandle(rocprofvis_dm_track_id_t track, rocprofvis_dm_handle_t slice)
{  
    return m_tracks[track].get()->DeleteSliceByHandle(slice);
}

rocprofvis_dm_result_t Trace::DeleteAllSlices(){
   
    for(int i = 0; i < m_tracks.size(); i++)
    {
        m_tracks[i].get()->DeleteAllSlices();
    }
    return kRocProfVisDmResultSuccess;
}
 
rocprofvis_dm_result_t  Trace::DeleteEventPropertyFor(     rocprofvis_dm_event_property_type_t type,
                                                                rocprofvis_dm_event_id_t event_id) {
    switch (type)
    {
        case kRPVDMEventFlowTrace:
        {
            // To delete single vector array element thread-safe, we must retain a local copy 
            // The element is protected by its own mutex and will be deleted outside the scope when mutex is unlocked
            std::shared_ptr<FlowTrace> item;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type),__func__, this);
                auto it = std::find_if(
                    m_flow_traces.begin(), m_flow_traces.end(),
                                       [&event_id](std::shared_ptr<FlowTrace>& x) {
                                           return x.get()->EventId().value == event_id.value;
                                       });
                if (it == m_flow_traces.end())
                {
                    return kRocProfVisDmResultNotLoaded;
                }
                item    = *it;
                m_flow_traces.erase(it);
            }
            return kRocProfVisDmResultSuccess;
        }
        break;
        case kRPVDMEventStackTrace:
        {
            // To delete single vector array element thread-safe, we must retain a local copy 
            // The element is protected by its own mutex and will be deleted outside the scope when mutex is unlocked
            std::shared_ptr<StackTrace> item;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type), __func__, this);
                auto             it =
                    std::find_if(m_stack_traces.begin(), m_stack_traces.end(),
                                 [&event_id](std::shared_ptr<StackTrace>& x) {
                                     return x.get()->EventId().value == event_id.value;
                                 });
                if(it == m_stack_traces.end())
                {
                    return kRocProfVisDmResultNotLoaded;
                }
                item = *it;
                m_stack_traces.erase(it);
            }
            return kRocProfVisDmResultSuccess;
        }
        break;
        case kRPVDMEventExtData:
        {
            // To delete single vector array element thread-safe, we must retain a local copy 
            // The element is protected by its own mutex and will be deleted outside the scope when mutex is unlocked
            std::shared_ptr<ExtData> item;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type), __func__, this);
                auto             it =
                    std::find_if(m_ext_data.begin(), m_ext_data.end(),
                                 [&event_id](std::shared_ptr<ExtData>& x) {
                                     return x.get()->EventId().value == event_id.value;
                                 });
                if(it == m_ext_data.end())
                {
                    return kRocProfVisDmResultNotLoaded;
                }
                item = *it;
                m_ext_data.erase(it);
            }
            return kRocProfVisDmResultSuccess;
        }  
        break;   
    }
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_UNSUPPORTED_PROPERTY, kRocProfVisDmResultNotSupported); 
}

rocprofvis_dm_result_t  Trace::DeleteAllEventPropertiesFor(rocprofvis_dm_event_property_type_t type){
    switch(type)
    {
        case kRPVDMEventFlowTrace:
        {
            // To delete all vector array elements thread-safe, we must swap its content with local array while protected by main mutex
            // The elements of local array are protected by their own mutexes and will be deleted outside the scope when mutexes are unlocked 
            std::vector<std::shared_ptr<FlowTrace>> flow_traces;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type), __func__, this);
                flow_traces.swap(m_flow_traces);
            }
            return kRocProfVisDmResultSuccess;
        }
        break;
        case kRPVDMEventStackTrace:
        {
            // To delete all vector array elements thread-safe, we must swap its content with local array while protected by main mutex
            // The elements of local array are protected by their own mutexes and will be deleted outside the scope when mutexes are unlocked 
            std::vector<std::shared_ptr<StackTrace>> stack_traces;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type), __func__,this);
                stack_traces.swap(m_stack_traces);
            }
            return kRocProfVisDmResultSuccess;
        }
        break;
        case kRPVDMEventExtData:
        {
            // To delete all vector array elements thread-safe, we must swap its content with local array while protected by main mutex
            // The elements of local array are protected by their own mutexes and will be deleted outside the scope when mutexes are unlocked 
            std::vector<std::shared_ptr<ExtData>> ext_data;
            {
                TimedLock<std::unique_lock<std::shared_mutex>> lock(*EventPropertyMutex(type), __func__, this);
                ext_data.swap(m_ext_data);
            }
            return kRocProfVisDmResultSuccess;
        }
        break;  
    }
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_UNSUPPORTED_PROPERTY, kRocProfVisDmResultNotSupported); 
}

rocprofvis_dm_result_t Trace::DeleteTableAt(rocprofvis_dm_table_id_t id){
    // To delete single vector array element thread-safe, we must retain a local copy 
    // The element is protected by its own mutex and will be deleted outside the scope when mutex is unlocked
    std::shared_ptr<Table> item = nullptr;
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
        auto             it = std::find_if(
            m_tables.begin(), m_tables.end(),
            [&id](std::shared_ptr<Table>& x) { return x.get()->Id() == id; });
        if(it == m_tables.end())
        {
            return kRocProfVisDmResultNotLoaded;
        }
        item = *it;
        m_tables.erase(it);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Trace::DeleteAllTables(){
    // To delete all vector array elements thread-safe, we must swap its content with local array while protected by main mutex
    // The elements of local array are protected by their own mutexes and will be deleted outside the scope when mutexes are unlocked 
    std::vector<std::shared_ptr<Table>> tables;
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
        tables.swap(m_tables);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Trace::GetTrackAtIndex(rocprofvis_dm_property_index_t index, rocprofvis_dm_track_t & value) {
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_tracks.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultInvalidParameter);
    value = m_tracks[index].get();
    return kRocProfVisDmResultSuccess;
}  

rocprofvis_dm_size_t Trace::GetMemoryFootprint(){
    rocprofvis_dm_size_t size = sizeof(Trace);
    for (int i=0; i < m_tracks.size(); i++)
    {
        size+=m_tracks[i].get()->GetMemoryFootprint();
    }   
    for (int i=0; i < m_flow_traces.size(); i++)
    {
        size+=m_flow_traces[i].get()->GetMemoryFootprint();
    }
    for (int i=0; i < m_stack_traces.size(); i++)
    {
        size+=m_stack_traces[i].get()->GetMemoryFootprint();
    }
    for (int i=0; i < m_ext_data.size(); i++)
    {
        size+=m_ext_data[i].get()->GetMemoryFootprint();
    }
    for (int i=0; i < m_tables.size(); i++)
    {
        size+=m_tables[i].get()->GetMemoryFootprint();
    }
    for (int i=0; i < m_strings.size(); i++)
    {
        size+=m_strings[i].length()+1;
    }
    return size;
}

rocprofvis_dm_result_t Trace::AddTrack(const rocprofvis_dm_trace_t object, rocprofvis_dm_track_params_t * params){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    Trace* trace = (Trace*)object;
    try{
        trace->m_tracks.push_back(std::make_unique<Track>(trace, params));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Failure allocating track", kRocProfVisDmResultAllocFailure);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_slice_t Trace::AddSlice(const rocprofvis_dm_trace_t object, const rocprofvis_dm_track_id_t track_id, const rocprofvis_dm_timestamp_t start, const rocprofvis_dm_timestamp_t end){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, nullptr);
    Trace* trace = (Trace*)object;
    rocprofvis_dm_track_t track = nullptr;
    rocprofvis_dm_result_t result = trace->GetTrackAtIndex(track_id, track);
    if (result == kRocProfVisDmResultSuccess)
    {
        ROCPROFVIS_ASSERT_MSG_RETURN(track, ERROR_TRACK_CANNOT_BE_NULL, nullptr);
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*((Track*)track)->Mutex(), __func__,
                                                            trace);
        return ((Track*)track)->AddSlice(start, end);
    }
    else
        return nullptr;
}  

rocprofvis_dm_result_t Trace::AddRecord(const rocprofvis_dm_slice_t object, rocprofvis_db_record_data_t & data){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_SLICE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    TrackSlice* slice = (TrackSlice*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*slice->Mutex(), __func__, slice);
    return slice->AddRecord(data);
}      

rocprofvis_dm_index_t Trace::AddString(const rocprofvis_dm_trace_t object,  const char* stringValue){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, INVALID_INDEX);
    Trace* trace = (Trace*)object;
    rocprofvis_dm_index_t current_index = (rocprofvis_dm_index_t)trace->m_strings.size();
    try{
        trace->m_strings.push_back(stringValue);
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN( "Error! Failure allocating string memory", INVALID_INDEX);
    }
    return current_index;
}

rocprofvis_dm_result_t Trace::AddFlow(const rocprofvis_dm_flowtrace_t object, rocprofvis_db_flow_data_t & data){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_FLOW_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    FlowTrace* flowtrace = (FlowTrace*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*flowtrace->Mutex(), __func__, flowtrace);
    return flowtrace->AddRecord(data);
}

rocprofvis_dm_flowtrace_t Trace::AddFlowTrace(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, nullptr);
    Trace* trace = (Trace*)object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*trace->EventPropertyMutex(kRPVDMEventFlowTrace), __func__, trace);
    try{
        trace->m_flow_traces.push_back(std::make_shared<FlowTrace>(trace, event_id));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN( "Error! Failure allocating flowtrace object", nullptr);
    }
    return trace->m_flow_traces.back().get();
}  

rocprofvis_dm_result_t Trace::AddStackFrame(const rocprofvis_dm_stacktrace_t object, rocprofvis_db_stack_data_t & data){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_STACK_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    StackTrace* stacktrace = (StackTrace*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*stacktrace->Mutex(), __func__, stacktrace);
    return stacktrace->AddRecord(data);
}

rocprofvis_dm_stacktrace_t Trace::AddStackTrace(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, nullptr);
    Trace* trace = (Trace*)object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*trace->EventPropertyMutex(kRPVDMEventStackTrace), __func__, trace);
    try{
        trace->m_stack_traces.push_back(std::make_shared<StackTrace>(trace, event_id));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN( "Error! Failure allocating stacktrace object", nullptr);
    }
    return trace->m_stack_traces.back().get();
}  

rocprofvis_dm_extdata_t  Trace::AddExtData(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, nullptr);
    Trace* trace = (Trace*)object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*trace->EventPropertyMutex(kRPVDMEventExtData), __func__, trace);
    try{
        trace->m_ext_data.push_back(std::make_shared<ExtData>(trace,event_id));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN( "Error! Failure allocating extended data object", nullptr);
    }
    return trace->m_ext_data.back().get();
}


rocprofvis_dm_result_t Trace::AddEventLevel(const rocprofvis_dm_trace_t object, const rocprofvis_dm_event_id_t event_id, rocprofvis_dm_event_level_t level)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_STACK_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    Trace* trace = (Trace*) object;
    try
    {
        trace->m_event_level_map[*(uint64_t*) &event_id] = level;
    } 
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN("Error! Failure allocating event level array",
                                            kRocProfVisDmResultAllocFailure);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  Trace::AddExtDataRecord(const rocprofvis_dm_extdata_t object, rocprofvis_db_ext_data_t & data){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_EXT_DATA_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ExtData* ext_data = (ExtData*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*ext_data->Mutex(), __func__, ext_data);
    if (!ext_data->HasRecord(data)){
        return ext_data->AddRecord(data);   
    }
    return kRocProfVisDmResultSuccess;
}


rocprofvis_dm_table_t Trace::AddTable(const rocprofvis_dm_trace_t object, rocprofvis_dm_charptr_t query, rocprofvis_dm_charptr_t description){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL, nullptr);
    Trace* trace = (Trace*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*trace->Mutex(), __func__, trace);
    try{
        trace->m_tables.push_back(std::make_shared<Table>(trace,description, query));
    }
    catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN( "Error! Failure allocating table object", nullptr);
    }
    return trace->m_tables.back().get();
}

rocprofvis_dm_table_row_t Trace::AddTableRow(const rocprofvis_dm_table_t object){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TABLE_CANNOT_BE_NULL, nullptr);
    Table* table = (Table*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*table->Mutex(), __func__, table);
    return table->AddRow();
}

rocprofvis_dm_result_t Trace::AddTableColumn(const rocprofvis_dm_table_t object, rocprofvis_dm_charptr_t column_name){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TABLE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    Table* table = (Table*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*table->Mutex(), __func__, table);
    return table->AddColumn(column_name);
}

rocprofvis_dm_result_t Trace::AddTableRowCell(const rocprofvis_dm_table_row_t object, rocprofvis_dm_charptr_t cell_value){
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TABLE_ROW_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    TableRow* table_row = (TableRow*) object;
    TimedLock<std::unique_lock<std::shared_mutex>> lock(*table_row->Mutex(), __func__, table_row);
    return table_row->AddCellValue(cell_value);
}

rocprofvis_dm_result_t Trace::CompleteSlice(const rocprofvis_dm_slice_t object)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_SLICE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    TrackSlice*                                    slice = (TrackSlice*) object;
    {
        TimedLock<std::unique_lock<std::shared_mutex>> lock(*slice->Mutex(), __func__, slice);
        slice->SetComplete();
    }
    return kRocProfVisDmResultSuccess;
}      

rocprofvis_dm_result_t
Trace::RemoveSlice(const rocprofvis_dm_trace_t    trace_object,
                   const rocprofvis_dm_track_id_t track_id,
                   const rocprofvis_dm_slice_t    object)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(trace_object, ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultUnknownError);
    Trace*                 trace  = (Trace*) trace_object; 
    return trace->DeleteSliceByHandle(track_id, object);
}      


rocprofvis_dm_result_t Trace::CheckSliceExists(
                        const rocprofvis_dm_trace_t     object,
                        const rocprofvis_dm_timestamp_t start,
                        const rocprofvis_dm_timestamp_t end,
                        const rocprofvis_db_num_of_tracks_t num,
                        const rocprofvis_db_track_selection_t tracks)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    Trace* trace = (Trace*) object;
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*trace->Mutex(), __func__, trace);
    // TODO: Change the interface to only accept a single track ID instead of array of track IDs.
    // For now do a search for each of the passed in track IDs.
    for(int n = 0; n < num; n++)
    {
        for(int i = 0; i < trace->m_tracks.size(); i++)
        {
            if(trace->m_tracks[i]->TrackId() == tracks[n])
            {
                rocprofvis_dm_slice_t  object = nullptr;
                rocprofvis_dm_result_t result =
                    trace->m_tracks[i].get()->GetSliceAtTime(hash_combine(start,end), object);
                if(result == kRocProfVisDmResultSuccess)
                {
                    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_SLICE_CANNOT_BE_NULL,
                                                    kRocProfVisDmResultUnknownError);
                    TrackSlice* slice = (TrackSlice*) object;
                    lock.unlock();

                    slice->WaitComplete();
                    return kRocProfVisDmResultSuccess;

                }
            }
        }
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Trace::CheckEventPropertyExists(
                        const rocprofvis_dm_trace_t     object,
                        const rocprofvis_dm_event_property_type_t type,
                        const rocprofvis_dm_event_id_t event_id)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    Trace* trace = (Trace*) object;
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*trace->EventPropertyMutex(type), __func__, trace);
    switch(type)
    {
        case kRPVDMEventFlowTrace:
        {
            
            auto it = std::find_if(trace->m_flow_traces.begin(), trace->m_flow_traces.end(),
                                      [&event_id](std::shared_ptr<FlowTrace>& x) {
                                       return x.get()->EventId().value == event_id.value;
                                   });
            if(it != trace->m_flow_traces.end())
            {
                return kRocProfVisDmResultSuccess;
            }
        }
        break;
        case kRPVDMEventStackTrace:
        {
            auto it =
                std::find_if(trace->m_stack_traces.begin(), trace->m_stack_traces.end(),
                             [&event_id](std::shared_ptr<StackTrace>& x) {
                                 return x.get()->EventId().value == event_id.value;
                             });

            if(it != trace->m_stack_traces.end())
            {
                return kRocProfVisDmResultSuccess;
            }
        }
        break;
        case kRPVDMEventExtData:
        {
            auto it = std::find_if(trace->m_ext_data.begin(), trace->m_ext_data.end(),
                                   [&event_id](std::shared_ptr<ExtData>& x) {
                                 return x.get()->EventId().value == event_id.value;
                             });

            if(it != trace->m_ext_data.end())
            {
                return kRocProfVisDmResultSuccess;
            }
        }
        break;
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Trace::CheckTableExists(
    const rocprofvis_dm_trace_t object,
    const rocprofvis_dm_table_id_t table_id)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(object, ERROR_TRACE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    Trace* trace = (Trace*) object;
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*trace->Mutex(), __func__, trace);
    auto it =
        std::find_if(trace->m_tables.begin(), trace->m_tables.end(),
        [&table_id](std::shared_ptr<Table>& x) { return x.get()->Id() == table_id; });
    if(it != trace->m_tables.end())
    {
       return kRocProfVisDmResultSuccess;
    }
    
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_charptr_t Trace::GetStringAt(rocprofvis_dm_index_t index){
    ROCPROFVIS_ASSERT_MSG_RETURN(m_parameters.metadata_loaded, ERROR_METADATA_IS_NOT_LOADED, nullptr);
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_strings.size(), ERROR_INDEX_OUT_OF_RANGE, nullptr);
    return m_strings[index].c_str();
}

rocprofvis_dm_result_t Trace::GetStringIndicesWithSubstring(rocprofvis_dm_num_string_table_filters_t num, rocprofvis_dm_string_table_filters_t substrings, std::vector<rocprofvis_dm_index_t>& indices)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(m_parameters.metadata_loaded, ERROR_METADATA_IS_NOT_LOADED, kRocProfVisDmResultNotLoaded);
    for(int i = 0; i < num; i ++)
    {
        ROCPROFVIS_ASSERT_RETURN(substrings[i], kRocProfVisDmResultInvalidParameter);
    }
    for(int i = 0; i < m_strings.size(); i ++)
    {
        std::string_view sv = m_strings[i];
        bool match = true;
        for(int j = 0; j < num; j ++)
        {
            std::string_view sub_sv = substrings[j];
            auto it = std::search(sv.begin(), sv.end(), sub_sv.begin(), sub_sv.end(),
                        [](unsigned char c1, unsigned char c2) {
                            return std::tolower(c1) == std::tolower(c2);
                        });
            if (it == sv.end())
            {
                match = false;
                break;
            }
        }
        if(match)
        {
            indices.push_back(i);
        }
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  Trace::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMStartTimeUInt64:
            *value = StartTime();
            return kRocProfVisDmResultSuccess;
        case kRPVDMEndTimeUInt64:
            *value = EndTime();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfTracksUInt64:
            *value = NumberOfTracks();
            return kRocProfVisDmResultSuccess;
        case kRPVDMNumberOfTablesUInt64:
            *value = NumberOfTables();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTraceMemoryFootprintUInt64:
            *value = GetMemoryFootprint();
            return kRocProfVisDmResultSuccess;
        case kRPVDMHistogramNumBuckets:
            *value = NumberOfHistogramBuckets();
            return kRocProfVisDmResultSuccess;
        case kRPVDMHistogramBucketSize:
            *value = HistogramBucketsSize();
            return kRocProfVisDmResultSuccess;
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}


rocprofvis_dm_result_t    Trace::GetPropertyAsHandle(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, rocprofvis_dm_handle_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMTrackHandleIndexed:
            return GetTrackAtIndex(index, *value);
        case kRPVDMDatabaseHandle:
            *value = Database();
            return kRocProfVisDmResultSuccess;
        case kRPVDMFlowTraceHandleByEventID:
            return GetFlowTraceHandle(*(rocprofvis_dm_event_id_t*)&index, *value);
        case kRPVDMStackTraceHandleByEventID:
            return GetStackTraceHandle(*(rocprofvis_dm_event_id_t*)&index, *value);
        case kRPVDMExtInfoHandleByEventID:
            return GetExtInfoHandle(*(rocprofvis_dm_event_id_t*)&index, *value);
        case kRPVDMTableHandleByID:
            return GetTableHandle(*(rocprofvis_dm_table_id_t*) &index, *value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

#ifdef TEST
const char*  Trace::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMStartTimeUInt64:
            return "kRPVDMStartTimeUInt64";        
        case kRPVDMEndTimeUInt64:
            return "kRPVDMEndTimeUInt64";
        case kRPVDMNumberOfTracksUInt64:
            return "kRPVDMNumberOfTracksUInt64";
        case kRPVDMNumberOfTablesUInt64:
            return "kRPVDMNumberOfTablesUInt64";
        case kRPVDMTraceMemoryFootprintUInt64:
            return "kRPVDMTraceMemoryFootprintUInt64";
        case kRPVDMTrackHandleIndexed:
            return "kRPVDMTrackHandleIndexed";
        case kRPVDMDatabaseHandle:
            return "kRPVDMDatabaseHandle";
        case kRPVDMFlowTraceHandleByEventID:
            return "kRPVDMFlowTraceHandleByEventID";
        case kRPVDMStackTraceHandleByEventID:
            return "kRPVDMStackTraceHandleByEventID";
        case kRPVDMExtInfoHandleByEventID:
            return "kRPVDMExtInfoHandleByEventID";
        case kRPVDMTableHandleIndexed:
            return "kRPVDMTableHandleIndexed";
        case kRPVDMHistogramNumBuckets:
            return "kRPVDMHistogramNumBuckets";
        case kRPVDMHistogramBucketSize:
            return "kRPVDMHistogramBucketSize";
        default:
            return "Unknown property";
    }   
}
#endif


rocprofvis_dm_result_t Trace::GetExtInfoHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_extdata_t & extinfo){
    
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*EventPropertyMutex(kRPVDMEventExtData), __func__, this);
    auto it =
        find_if(m_ext_data.begin(), m_ext_data.end(), 
                        [&](std::shared_ptr<ExtData>& x) {
        return x.get()->EventId().value == event_id.value;
    });
    if (it != m_ext_data.end())
    {
        extinfo = it->get();
        return kRocProfVisDmResultSuccess;
    } 
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Trace::GetFlowTraceHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_flowtrace_t & flowtrace){
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*EventPropertyMutex(kRPVDMEventFlowTrace), __func__, this);
    auto it = find_if(m_flow_traces.begin(), m_flow_traces.end(),
                      [&](std::shared_ptr<FlowTrace>& x) {
        return x.get()->EventId().value == event_id.value;
    });
    if (it != m_flow_traces.end())
    {
        flowtrace = it->get();
        return kRocProfVisDmResultSuccess;
    } 
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Trace::GetStackTraceHandle(rocprofvis_dm_event_id_t event_id, rocprofvis_dm_stacktrace_t & stacktrace){
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*EventPropertyMutex(kRPVDMEventStackTrace), __func__, this);
    auto it = find_if(m_stack_traces.begin(), m_stack_traces.end(),
                      [&](std::shared_ptr<StackTrace>& x) {
        return x.get()->EventId().value == event_id.value;
        });
    if (it != m_stack_traces.end())
    {
        stacktrace = it->get();
        return kRocProfVisDmResultSuccess;
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t Trace::GetTableHandle(rocprofvis_dm_table_id_t id, rocprofvis_dm_table_t & table){
    TimedLock<std::shared_lock<std::shared_mutex>> lock(*Mutex(), __func__, this);
    auto  it = find_if(m_tables.begin(), m_tables.end(),
                      [&](std::shared_ptr<Table>& x) {
                          return x.get()->Id() == id;
                      });
    if(it != m_tables.end())
    {
        table = it->get();
        return kRocProfVisDmResultSuccess;
    }
    return kRocProfVisDmResultNotLoaded;

}

}  // namespace DataModel
}  // namespace RocProfVis