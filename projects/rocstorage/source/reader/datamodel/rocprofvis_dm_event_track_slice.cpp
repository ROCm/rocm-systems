// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_event_track_slice.h"
#include "rocprofvis_dm_track.h"
#include "rocprofvis_dm_trace.h"

namespace RocProfVis
{
namespace DataModel
{

EventTrackSlice::EventTrackSlice(Track* ctx, rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end)
: TrackSlice(ctx, start, end)
{
    m_samples.reserve(ctx->NumRecords());
}; 

rocprofvis_dm_result_t EventTrackSlice::AddRecord(rocprofvis_db_record_data_t & data){
    try {
        void*        record_memory = Allocate(sizeof(EventRecord));
        EventRecord* record        = new(record_memory)
            EventRecord(data.event.id, data.event.timestamp, data.event.duration,
                        data.event.category, data.event.symbol, data.event.level);
        m_samples.emplace_back(record);
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_size_t EventTrackSlice::GetMemoryFootprint(){
    return sizeof(EventTrackSlice) + ((sizeof(EventRecord)+sizeof(std::unique_ptr<EventRecord>)) * m_samples.size());
}

rocprofvis_dm_size_t EventTrackSlice::GetNumberOfRecords(){
    return m_samples.size();
}

rocprofvis_dm_result_t EventTrackSlice::ConvertTimestampToIndex(const rocprofvis_dm_timestamp_t timestamp, rocprofvis_dm_index_t & index){
    std::vector<EventRecord*>::iterator it = std::find_if( m_samples.begin(), m_samples.end(),
            [&timestamp](EventRecord* & x) { return x->Timestamp() >= timestamp;});
    if (it != m_samples.end())
    {
        index = (uint32_t)(it - m_samples.begin());
        return kRocProfVisDmResultSuccess;
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t EventTrackSlice::GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    timestamp = m_samples[index]->Timestamp();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t EventTrackSlice::GetRecordIdAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_id_t & id){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    id = m_samples[index]->EventIdFull().value;
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t EventTrackSlice::GetRecordOperationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_op_t & op){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    op = m_samples[index]->Operation();
    return kRocProfVisDmResultSuccess;
}


rocprofvis_dm_result_t EventTrackSlice::GetRecordOperationStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & op){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    op = "Invalid";
    switch (m_samples[index]->Operation()){
        case kRocProfVisDmOperationLaunch: 
            op = "Launch";
            break;
        case kRocProfVisDmOperationLaunchSample: 
            op = "LaunchSample";
            break;
        case kRocProfVisDmOperationDispatch: 
            op = "Dispatch";
            break;
        case kRocProfVisDmOperationMemoryAllocate: 
            op = "MemAlloc";
            break;
        case kRocProfVisDmOperationMemoryCopy: 
            op = "MemCopy";
            break;
    }
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  EventTrackSlice::GetRecordDurationAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_duration_t & duration){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    duration = m_samples[index]->Duration();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  EventTrackSlice::GetRecordCategoryIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & category_index){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    category_index = m_samples[index]->CategoryIndex();
    return kRocProfVisDmResultSuccess; 
}

rocprofvis_dm_result_t  EventTrackSlice::GetRecordSymbolIndexAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_index_t & symbol_index){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    symbol_index = m_samples[index]->SymbolIndex();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  EventTrackSlice::GetRecordCategoryStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & category_charptr){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx()->Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    category_charptr = Ctx()->Ctx()->GetStringAt(m_samples[index]->CategoryIndex());
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t  EventTrackSlice::GetRecordSymbolStringAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_charptr_t & symbol_charptr){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx()->Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    symbol_charptr = Ctx()->Ctx()->GetStringAt(m_samples[index]->SymbolIndex());
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t EventTrackSlice::GetRecordGraphLevelAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_event_level_t& level) {
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx(), ERROR_TRACK_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    ROCPROFVIS_ASSERT_MSG_RETURN(Ctx()->Ctx(), ERROR_TRACE_CANNOT_BE_NULL, kRocProfVisDmResultNotLoaded);
    level = m_samples[index]->EventLevel();
    return kRocProfVisDmResultSuccess;
}

}  // namespace DataModel
}  // namespace RocProfVis