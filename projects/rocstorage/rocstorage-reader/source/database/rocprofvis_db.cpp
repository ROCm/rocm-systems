// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db.h"
#include "rocprofvis_db_profile.h"
#include <spdlog/spdlog.h>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cfloat>

namespace RocProfVis
{
namespace DataModel
{

bool Database::IsNumber(const std::string& s) {
    std::istringstream iss(s);
    uint64_t d;
    return iss >> std::noskipws >> d && iss.eof();
}

rocprofvis_db_type_t Database::Autodetect(
                                                    rocprofvis_db_filename_t filename){
    rocprofvis_db_type_t db_type = ProfileDatabase::Detect(filename);
    if (db_type!=rocprofvis_db_type_t::kAutodetect)
            return db_type;
    return rocprofvis_db_type_t::kAutodetect;
}


rocprofvis_dm_result_t Database::AddTrackProperties(
                                                    rocprofvis_dm_track_params_t& props) {
    try {
        m_track_properties.push_back(std::make_unique<rocprofvis_dm_track_params_t>(props));
    }
    catch (std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_MEMORY_ALLOCATION_FAILURE, kRocProfVisDmResultAllocFailure);
    }
    return kRocProfVisDmResultSuccess;
}

void  Database::ShowProgress(
                                                    double step, 
                                                    rocprofvis_dm_charptr_t action, 
                                                    rocprofvis_db_status_t status, 
                                                    Future* future){
    future->ShowProgress(Path(), step, action, status);
}

rocprofvis_dm_result_t Database::BindTrace(
                                                    rocprofvis_dm_db_bind_struct * binding_info){
    m_binding_info = binding_info;
    m_binding_info->FuncFindCachedTableValue = FindCachedTableValue;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  Database::ReadTraceMetadataAsync(
                                                    rocprofvis_db_future_t object){
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED, kRocProfVisDmResultResourceBusy);
    try {
        future->SetWorker(std::move(std::thread(Database::ReadTraceMetadataStatic, this, future)));
    }
    catch (std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  Database::ReadTraceSliceAsync(
                                                    rocprofvis_dm_timestamp_t start,
                                                    rocprofvis_dm_timestamp_t end,
                                                    rocprofvis_db_num_of_tracks_t num,
                                                    rocprofvis_db_track_selection_t tracks,
                                                    rocprofvis_db_future_t object){
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED, kRocProfVisDmResultResourceBusy);
    rocprofvis_dm_result_t result = BindObject()->FuncCheckSliceExists(BindObject()->trace_object, start, end, num, tracks); 
    if(result != kRocProfVisDmResultNotLoaded)
    {
        spdlog::debug("Slice ({},{}) exists!", start, end);
        return future->SetPromise(result);
    }
    try {
        future->SetWorker(std::move(std::thread(Database::ReadTraceSliceStatic, this, start, end, num, tracks, future)));
    }
    catch (std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t   Database::ReadEventPropertyAsync(
                                                    rocprofvis_dm_event_property_type_t type,
                                                    rocprofvis_dm_event_id_t event_id,
                                                    rocprofvis_db_future_t object){
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED, kRocProfVisDmResultResourceBusy);
    rocprofvis_dm_result_t result =  BindObject()->FuncCheckEventPropertyExists(BindObject()->trace_object, type, event_id);
    if(result != kRocProfVisDmResultNotLoaded)
    {
        return future->SetPromise(result);
    }
    try {
        future->SetWorker(std::move(std::thread(ReadEventPropertyStatic, this, type, event_id, future)));
    }
    catch (std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Database::ExportTableCSVAsync(rocprofvis_dm_string_t query,
                                                     rocprofvis_dm_string_t file_path,
                                                     rocprofvis_db_future_t object)
{
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(!file_path.empty(), "Output path cannot be empty.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED,
                                 kRocProfVisDmResultResourceBusy);
    try
    {
        future->SetWorker(std::move(std::thread(&ExportTableCSVStatic, this, query, file_path, future)));
    } catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Database::ExportTableCSVStatic(Database* db,
                                                      rocprofvis_dm_string_t query,
                                                      rocprofvis_dm_string_t file_path,
                                                      Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(!file_path.empty(), "New DB path cannot be empty.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);    
    return db->ExportTableCSV(query.c_str(), file_path.c_str(), future);
}

rocprofvis_dm_result_t Database::ExportTableCSV(rocprofvis_dm_charptr_t query,
                                                rocprofvis_dm_charptr_t file_path, 
                                                Future* future)
{
    (void) query;
    (void) file_path;
    (void) future;
    return kRocProfVisDmResultNotSupported;
}

rocprofvis_dm_result_t
Database::SaveTrimmedDataAsync(rocprofvis_dm_timestamp_t start,
                               rocprofvis_dm_timestamp_t end,
                               rocprofvis_dm_string_t new_db_path,
                               rocprofvis_db_future_t object)
{
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(!new_db_path.empty(), "New DB path cannot be empty.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED,
                                 kRocProfVisDmResultResourceBusy);
    rocprofvis_dm_result_t result = kRocProfVisDmResultUnknownError;
    try
    {
        future->SetWorker(std::move(std::thread(&SaveTrimmedDataStatic, this, start, end, new_db_path, future)));
    } catch(std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t Database::SaveTrimmedDataStatic(Database* db, rocprofvis_dm_timestamp_t start,
    rocprofvis_dm_timestamp_t end, rocprofvis_dm_string_t new_db_path, Future* future)
{
    ROCPROFVIS_ASSERT_MSG_RETURN(!new_db_path.empty(), "New DB path cannot be empty.",
                                 kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL,
                                 kRocProfVisDmResultInvalidParameter);


    //check if a db file exists and if it does delete it (we will overwrite it)
    std::ifstream file(new_db_path);
    if(file.good())
    {
        file.close();
        int remove_result = std::remove(new_db_path.c_str());
        if(remove_result != 0)
        {
            spdlog::error("Failed to overwrite existing file: {}, code: {}", new_db_path,
                          remove_result);
            
            db->ShowProgress(0, "Failed to trim track! Could not overwrite existing file.", kRPVDbError, future);
            future->SetPromise(kRocProfVisDmResultDbAccessFailed);
            return kRocProfVisDmResultDbAccessFailed;
        }
    }

    return db->SaveTrimmedData(start, end, new_db_path.c_str(), future);
}


rocprofvis_dm_result_t  Database::ExecuteQueryAsync(
                                                    rocprofvis_dm_charptr_t query,
                                                    rocprofvis_dm_charptr_t description,
                                                    rocprofvis_db_future_t object, 
                                                    rocprofvis_dm_table_id_t* id)
{
    Future* future = (Future*) object;
    ROCPROFVIS_ASSERT_MSG_RETURN(future, ERROR_FUTURE_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    ROCPROFVIS_ASSERT_MSG_RETURN(!future->IsWorking(), ERROR_FUTURE_CANNOT_BE_USED, kRocProfVisDmResultResourceBusy);
    *id = std::hash<std::string>{}(query);
    rocprofvis_dm_result_t   result = BindObject()->FuncCheckTableExists(BindObject()->trace_object, *id);
    if(result != kRocProfVisDmResultNotLoaded)
    {
        return future->SetPromise(result);
    }
    try {
        future->SetWorker(std::move(std::thread(ExecuteQueryStatic, this, query, description, future)));
    }
    catch (std::exception ex)
    {
        ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ex.what(), kRocProfVisDmResultUnknownError);
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  Database::ReadTraceMetadataStatic(
                                                    Database* db, 
                                                    Future* object){
    return db->ReadTraceMetadata(object);
}

rocprofvis_dm_result_t  Database::ReadTraceSliceStatic(
                                                    Database* db,
                                                    rocprofvis_dm_timestamp_t start,
                                                    rocprofvis_dm_timestamp_t end,
                                                    rocprofvis_db_num_of_tracks_t num,
                                                    rocprofvis_db_track_selection_t tracks,
                                                    Future* object){
    return db->ReadTraceSlice(start, end, num, tracks, object);
}


rocprofvis_dm_result_t   Database::ReadEventPropertyStatic(
                                                    Database* db, 
                                                    rocprofvis_dm_event_property_type_t type,
                                                    rocprofvis_dm_event_id_t event_id,
                                                    Future* object){
    switch (type) {
        case kRPVDMEventFlowTrace:
            return db->ReadFlowTraceInfo(event_id,object);
        case kRPVDMEventStackTrace:
            return db->ReadStackTraceInfo(event_id,object);
        case kRPVDMEventExtData:
            return db->ReadExtEventInfo(event_id,object);           
    }  
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_UNSUPPORTED_PROPERTY, kRocProfVisDmResultNotSupported); 
}



rocprofvis_dm_result_t   Database::ExecuteQueryStatic(
                                                    Database* db,
                                                    rocprofvis_dm_charptr_t query,
                                                    rocprofvis_dm_charptr_t description,
                                                    Future* object){
    return db->ExecuteQuery(query,description,object);
}

const char* Database::ProcessNameSuffixFor(rocprofvis_dm_track_category_t category){
    switch(category){
        case kRocProfVisDmPmcTrack:
        case kRocProfVisDmKernelDispatchTrack:
        case kRocProfVisDmMemoryAllocationTrack:
        case kRocProfVisDmMemoryCopyTrack:
            return "GPU:";
        case kRocProfVisDmRegionSampleTrack: 
            return "Sample PID:";
        case kRocProfVisDmRegionTrack:
        case kRocProfVisDmRegionMainTrack:
            return "Thread PID:";
        case kRocProfVisDmStreamTrack: 
            return "STREAM:";
    }
    return "";
}

const char* Database::SubProcessNameSuffixFor(rocprofvis_dm_track_category_t category){
    switch(category){
        case kRocProfVisDmPmcTrack:
        case kRocProfVisDmKernelDispatchTrack:
        case kRocProfVisDmMemoryAllocationTrack:
        case kRocProfVisDmMemoryCopyTrack:
            return "Queue:";
        case kRocProfVisDmRegionTrack:
        case kRocProfVisDmRegionMainTrack:
        case kRocProfVisDmRegionSampleTrack:
            return "TID:";
    }
    return "";
}

rocprofvis_dm_result_t DatabaseCache::PopulateTrackExtendedDataTemplate(Database * db, const char* table_name, uint64_t instance_id ){
    rocprofvis_dm_track_params_t* track_properties = db->TrackPropertiesLast();
    auto m = references[table_name][instance_id];
    std::string str_id = std::to_string(instance_id);
    for(std::map<std::string,std::string>::iterator it = m.begin(); it != m.end(); ++it) {
        rocprofvis_db_ext_data_t record;
        record.name = it->first.c_str();
        record.data     = str_id.c_str();
        record.category = table_name;
        record.type  = kRPVDataTypeString;
        rocprofvis_dm_result_t result = db->BindObject()->FuncAddExtDataRecord(track_properties->extdata, record);
        if (result != kRocProfVisDmResultSuccess) return result;
    }
    return kRocProfVisDmResultSuccess;
}


rocprofvis_dm_result_t   Database::FindCachedTableValue(  
                                                        const rocprofvis_dm_database_t object, 
                                                        rocprofvis_dm_charptr_t table, 
                                                        const rocprofvis_dm_id_t id, 
                                                        rocprofvis_dm_charptr_t column,
                                                        rocprofvis_dm_charptr_t* value){
    Database* db = (Database*) object;
    *value = db->CachedTables()->GetTableCell(table, id, column); 
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_size_t DatabaseCache::GetMemoryFootprint()
{
    size_t size = 0;
    for (std::map<std::string, table_map_t>::iterator it = references.begin(); it != references.end(); ++it) {
        size+=sizeof(std::string);
        size+=sizeof(table_map_t);
        size+=3 * sizeof(void*);
        size+=it->first.length();
        for (std::map<uint64_t, table_dict_t>::iterator it1 = it->second.begin(); it1 != it->second.end(); ++it1) {
            size += sizeof(uint64_t);
            size += sizeof(table_dict_t);
            size += 3 * sizeof(void*);
            for (std::map<std::string, std::string>::iterator it2 = it1->second.begin(); it2 != it1->second.end(); ++it2) {
                size += sizeof(std::string) * 2;
                size += 3 * sizeof(void*);
                size += it2->first.length();
                size += it2->second.length();
            }
        }
    }
    return size;
}

rocprofvis_dm_size_t Database::GetMemoryFootprint()
{
    rocprofvis_dm_size_t size = m_cached_tables.GetMemoryFootprint();
    size+=NumTracks()*(sizeof(rocprofvis_dm_track_params_t)+sizeof(std::unique_ptr<rocprofvis_dm_track_params_t>));
    size+=strlen(Path());
    return size;
}


rocprofvis_dm_track_params_it
Database::FindTrack(rocprofvis_dm_process_identifiers_t& process)
{
        return std::find_if(
            TrackPropertiesBegin(), TrackPropertiesEnd(),
            [process](std::unique_ptr<rocprofvis_dm_track_params_t>& params) {
                if(params.get()->process.category == process.category)
                {
                    for(int i = 0; i < NUMBER_OF_TRACK_IDENTIFICATION_PARAMETERS; i++)
                    {
                        if(process.is_numeric[i])
                        {
                            if(params.get()->process.id[i] != process.id[i])
                                return false;
                        }
                        else
                        {
                            if(params.get()->process.name[i] != process.name[i])
                                return false;
                        }
                    }
                    return true;
                }
                return false;
            });
}

void
Database::UpdateQueryForTrack(  rocprofvis_dm_track_params_it it, 
                                rocprofvis_dm_track_params_t& newprops,
                                rocprofvis_dm_charptr_t*      newqueries)
{

    int slice_query_category        = newprops.process.category == kRocProfVisDmStreamTrack
                                          ? kRPVQuerySliceByStream
                                          : kRPVQuerySliceByQueue;
    int slice_source_query_category = newprops.process.category == kRocProfVisDmStreamTrack
                                          ? kRPVSourceQuerySliceByStream
                                          : kRPVSourceQuerySliceByQueue;
    if (it != TrackPropertiesEnd()) {
            std::vector<rocprofvis_dm_string_t>::iterator s = 
            std::find_if(it->get()->query[slice_query_category].begin(), 
                            it->get()->query[slice_query_category].end(),
                [newqueries, slice_source_query_category](rocprofvis_dm_string_t& str) {
                                 return str == newqueries[slice_source_query_category];
                         });
            if(s == it->get()->query[slice_query_category].end())
            {
                it->get()->query[slice_query_category].push_back(
                    newqueries[slice_source_query_category]);
            }
            
            s = std::find_if(it->get()->query[kRPVQueryTable].begin(),
                             it->get()->query[kRPVQueryTable].end(),
                             [newqueries](rocprofvis_dm_string_t& str) {
                                 return str == newqueries[kRPVSourceQueryTable];
                             });
            if(s == it->get()->query[kRPVQueryTable].end())
            {
                it->get()->query[kRPVQueryTable].push_back(newqueries[kRPVSourceQueryTable]);
            }
            s = std::find_if(it->get()->query[kRPVQueryLevel].begin(),
                             it->get()->query[kRPVQueryLevel].end(),
                             [newqueries](rocprofvis_dm_string_t& str) {
                                 return str == newqueries[kRPVSourceQueryLevel];
                             });
            if(s == it->get()->query[kRPVQueryLevel].end())
            {
                it->get()->query[kRPVQueryLevel].push_back(newqueries[kRPVSourceQueryLevel]);
            }
        return;
    } 
    newprops.query[slice_query_category].push_back(newqueries[slice_source_query_category]);
    newprops.query[kRPVQueryTable].push_back(newqueries[kRPVSourceQueryTable]);
    newprops.query[kRPVQueryLevel].push_back(newqueries[kRPVSourceQueryLevel]); 
    newprops.max_ts = 0;
    newprops.min_ts = UINT64_MAX;
    newprops.max_value = 0;
    newprops.min_value = DBL_MAX;
}

}  // namespace DataModel
}  // namespace RocProfVis