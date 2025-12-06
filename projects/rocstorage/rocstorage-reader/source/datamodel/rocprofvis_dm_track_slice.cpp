// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_track_slice.h"

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_result_t TrackSlice::GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_id_t & id){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordOperationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_op_t & op){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordOperationStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & op){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordValueAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_value_t & value){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordDurationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_duration_t & duration){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordCategoryIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & category_index){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordSymbolIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & symbol_index){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t, rocprofvis_dm_charptr_t & symbol_charptr){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}
rocprofvis_dm_result_t TrackSlice::GetRecordGraphLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t & level){
    ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_VIRTUAL_METHOD_CALL, kRocProfVisDmResultNotLoaded);
}


rocprofvis_dm_result_t    TrackSlice::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMRecordIndexByTimestampUInt64:
            *value = 0;
            return ConvertTimestampToIndex(index, *(rocprofvis_dm_index_t*)value);
        case kRPVDMNumberOfRecordsUInt64:
            *value = GetNumberOfRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMTimestampUInt64Indexed:
            return GetRecordTimestampAt(index, *value);
        case kRPVDMEventIdUInt64Indexed:
            return GetRecordIdAt(index, *value);
        case kRPVDMEventIdOperationEnumIndexed:
            *value = 0;
            return GetRecordOperationAt(index, *(rocprofvis_dm_op_t*)value);
        case kRPVDMSliceMemoryFootprintUInt64:
            *value = GetMemoryFootprint();
            return kRocProfVisDmResultSuccess;
        case kRPVDMEventLevelUInt64Indexed:
            *value = 0;
            return GetRecordGraphLevelAt(index, *(rocprofvis_dm_event_level_t*) value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}

#ifdef TEST
const char*  TrackSlice::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMRecordIndexByTimestampUInt64:
            return "kRPVDMRecordIndexByTimestampUInt64";        
        case kRPVDMNumberOfRecordsUInt64:
            return "kRPVDMNumberOfRecordsUInt64";
        case kRPVDMTimestampUInt64Indexed:
            return "kRPVDMTimestampUInt64Indexed";
        case kRPVDMEventIdUInt64Indexed:
            return "kRPVDMEventIdUInt64Indexed";
        case kRPVDMEventIdOperationEnumIndexed:
            return "kRPVDMEventIdOperationEnumIndexed";
        case kRPVDMSliceMemoryFootprintUInt64:
            return "kRPVDMSliceMemoryFootprintUInt64";
        case kRPVDMEventDurationInt64Indexed:
            return "kRPVDMEventDurationInt64Indexed";
        case kRPVDMEventTypeStringCharPtrIndexed:
            return "kRPVDMEventTypeStringCharPtrIndexed";
        case kRPVDMEventSymbolStringCharPtrIndexed:
            return "kRPVDMEventSymbolStringCharPtrIndexed";
        case kRPVDMEventIdOperationCharPtrIndexed:
            return "kRPVDMEventIdOperationCharPtrIndexed";
        case kRPVDMPmcValueDoubleIndexed:
            return "kRPVDMPmcValueDoubleIndexed";
        case kRPVDMEventLevelUInt64Indexed: 
            return "kRPVDMEventLevelUInt64Indexed";
        default:
            return "Unknown property";
    }   
}
#endif
rocprofvis_dm_result_t    TrackSlice::GetPropertyAsInt64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, int64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMEventDurationInt64Indexed:
            return GetRecordDurationAt(index, *value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}
 rocprofvis_dm_result_t   TrackSlice::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMEventTypeStringCharPtrIndexed:
            return GetRecordCategoryStringAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMEventSymbolStringCharPtrIndexed:
            return GetRecordSymbolStringAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMEventIdOperationCharPtrIndexed:
            return GetRecordOperationStringAt(index, *(rocprofvis_dm_charptr_t*)value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}
rocprofvis_dm_result_t   TrackSlice::GetPropertyAsDouble(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, double* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMPmcValueDoubleIndexed:
            return GetRecordValueAt(index, *value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}

void*
TrackSlice::Allocate(size_t rec_size)
{
    void*       ptr          = nullptr;
    MemoryPool* current_pool = m_current_pool;
    if(current_pool == nullptr)
    {
        current_pool  = new MemoryPool(rec_size);
        m_object_pools[current_pool->m_base] = current_pool;
        m_current_pool = current_pool;
    }
    ptr = static_cast<char*>(current_pool->m_base) + (current_pool->m_pos * rec_size);
    if(++current_pool->m_pos == kMemPoolBitSetSize)
    {
        m_current_pool = nullptr;
    }
    return ptr;
}

void
TrackSlice::Cleanup()
{
    for(auto it = m_object_pools.begin(); it != m_object_pools.end(); ++it)
    {
        delete it->second;
    }
}

void
TrackSlice::SetComplete()
{

    m_complete = true;
    m_cv.notify_one();
}

void
TrackSlice::WaitComplete()
{
    std::unique_lock<std::shared_mutex> lock(m_lock);
    m_cv.wait(lock, [&] { return m_complete; });
}

}  // namespace DataModel
}  // namespace RocProfVis
