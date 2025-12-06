// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_flow_trace.h"
#include "rocprofvis_dm_trace.h"

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_size_t  FlowTrace::GetMemoryFootprint(){
    return sizeof(FlowTrace) + (sizeof(FlowRecord) * m_flows.size());
}

rocprofvis_dm_result_t  FlowTrace::AddRecord( rocprofvis_db_flow_data_t & data){
    try{
        m_flows.push_back(FlowRecord(data.time, data.end_time, data.id, data.track_id, data.category_id, data.symbol_id, data.level));
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t FlowTrace::GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    timestamp = m_flows[index].Timestamp();
    return kRocProfVisDmResultSuccess;
}
rocprofvis_dm_result_t FlowTrace::GetRecordEndTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    timestamp = m_flows[index].EndTimestamp();
    return kRocProfVisDmResultSuccess;
}
rocprofvis_dm_result_t FlowTrace::GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_id_t & event_id){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    event_id = m_flows[index].EventId();
    return kRocProfVisDmResultSuccess;
}
rocprofvis_dm_result_t FlowTrace::GetRecordTrackIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_track_id_t & track_id){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    track_id = m_flows[index].TrackId();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  FlowTrace::GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    category_charptr = Ctx()->GetStringAt(m_flows[index].CategoryId());
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  FlowTrace::GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol_charptr){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    symbol_charptr = Ctx()->GetStringAt(m_flows[index].SymbolId());
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  FlowTrace::GetRecordLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t & level)
{    
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_flows.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    level = m_flows[index].Level();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t FlowTrace::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMNumberOfEndpointsUInt64:
            *value = GetNumberOfRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMEndpointTimestampUInt64Indexed:
            return GetRecordTimestampAt(index, *value);
        case kRPVDMEndpointEndTimestampUInt64Indexed:
            return GetRecordEndTimestampAt(index, *value);
        case kRPVDMEndpointIDUInt64Indexed:
            return GetRecordIdAt(index, *(rocprofvis_dm_event_id_t*)value);
        case kRPVDMEndpointTrackIDUInt64Indexed:
            return GetRecordTrackIdAt(index, *(rocprofvis_dm_track_id_t*)value);
        case kRPVDMEndpointLevelUInt64Indexed:
            return GetRecordLevelAt(index, *(rocprofvis_dm_event_level_t*) value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}

 rocprofvis_dm_result_t   FlowTrace::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMEndpointSymbolCharPtrIndexed:
            return GetRecordSymbolStringAt(index, *(rocprofvis_dm_charptr_t*) value);
        case kRPVDMEndpointCategoryCharPtrIndexed:
            return GetRecordCategoryStringAt(index, *(rocprofvis_dm_charptr_t*) value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}


#ifdef TEST
const char*  FlowTrace::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMNumberOfEndpointsUInt64:
            return "kRPVDMNumberOfEndpointsUInt64";        
        case kRPVDMEndpointTimestampUInt64Indexed:
            return "kRPVDMEndpointTimestampUInt64Indexed";
        case kRPVDMEndpointIDUInt64Indexed:
            return "kRPVDMEndpointIDUInt64Indexed";
        case kRPVDMEndpointTrackIDUInt64Indexed:
            return "kRPVDMEndpointTrackIDUInt64Indexed";
        case kRPVDMEndpointSymbolCharPtrIndexed:
            return "kRPVDMEndpointSymbolCharPtrIndexed";
        case kRPVDMEndpointCategoryCharPtrIndexed:
            return "kRPVDMEndpointCategoryCharPtrIndexed";
        default:
            return "Unknown property";
    }   
}
#endif

}  // namespace DataModel
}  // namespace RocProfVis