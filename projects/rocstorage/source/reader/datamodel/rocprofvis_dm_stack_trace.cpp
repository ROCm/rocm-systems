// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_stack_trace.h"

namespace RocProfVis
{
namespace DataModel
{

rocprofvis_dm_size_t  StackTrace::GetMemoryFootprint(){
    return sizeof(StackTrace) + (sizeof(StackFrameRecord) * m_stack_frames.size());
}

rocprofvis_dm_result_t  StackTrace::AddRecord( rocprofvis_db_stack_data_t & data){
    try{
        m_stack_frames.push_back(StackFrameRecord(data.symbol, data.args, data.line, data.depth));
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t StackTrace::GetRecordSymbolAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_stack_frames.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    symbol = m_stack_frames[index].Symbol();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t StackTrace::GetRecordArgsAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t &args){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_stack_frames.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    args = m_stack_frames[index].Args();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t StackTrace::GetRecordCodeLineAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & code_line){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_stack_frames.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    code_line = m_stack_frames[index].CodeLine();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t StackTrace::GetRecordDepthAt(const rocprofvis_dm_property_index_t index, uint32_t & depth){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_stack_frames.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    depth = m_stack_frames[index].Depth();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t StackTrace::GetPropertyAsUint64(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, uint64_t* value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMNumberOfFramesUInt64:
            *value = GetNumberOfRecords();
            return kRocProfVisDmResultSuccess;
        case kRPVDMFrameDepthUInt64Indexed:
            *value = 0;
            return GetRecordDepthAt(index, *(uint32_t*)value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }

}

 rocprofvis_dm_result_t   StackTrace::GetPropertyAsCharPtr(rocprofvis_dm_property_t property, rocprofvis_dm_property_index_t index, char** value){
    ROCPROFVIS_ASSERT_MSG_RETURN(value, ERROR_REFERENCE_POINTER_CANNOT_BE_NULL, kRocProfVisDmResultInvalidParameter);
    switch(property)
    {
        case kRPVDMFrameSymbolCharPtrIndexed:
            return GetRecordSymbolAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMFrameArgsCharPtrIndexed:
            return GetRecordArgsAt(index, *(rocprofvis_dm_charptr_t*)value);
        case kRPVDMFrameCodeLineCharPtrIndexed:
            return GetRecordCodeLineAt(index, *(rocprofvis_dm_charptr_t*)value);
        default:
            ROCPROFVIS_ASSERT_ALWAYS_MSG_RETURN(ERROR_INVALID_PROPERTY_GETTER, kRocProfVisDmResultInvalidProperty);
    }
}

#ifdef TEST
const char*  StackTrace::GetPropertySymbol(rocprofvis_dm_property_t property) {
    switch(property)
    {
        case kRPVDMNumberOfFramesUInt64:
            return "kRPVDMNumberOfFramesUInt64";        
        case kRPVDMFrameDepthUInt64Indexed:
            return "kRPVDMFrameDepthUInt64Indexed";
        case kRPVDMFrameSymbolCharPtrIndexed:
            return "kRPVDMFrameSymbolCharPtrIndexed";
        case kRPVDMFrameArgsCharPtrIndexed:
            return "kRPVDMFrameArgsCharPtrIndexed";
        case kRPVDMFrameCodeLineCharPtrIndexed:
            return "kRPVDMFrameCodeLineCharPtrIndexed";
        default:
            return "Unknown property";
    }   
}
#endif

}  // namespace DataModel
}  // namespace RocProfVis