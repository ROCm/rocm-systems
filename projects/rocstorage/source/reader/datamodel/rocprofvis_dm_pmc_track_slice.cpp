// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_dm_pmc_track_slice.h"
#include "rocprofvis_dm_track.h"

namespace RocProfVis
{
namespace DataModel
{

PmcTrackSlice::PmcTrackSlice(Track* ctx, rocprofvis_dm_timestamp_t start, rocprofvis_dm_timestamp_t end)
: TrackSlice(ctx, start, end)
{
    m_samples.reserve(ctx->NumRecords());
}; 

//AddRecord method adds an object to a m_samples
rocprofvis_dm_result_t PmcTrackSlice::AddRecord(rocprofvis_db_record_data_t & data){
    try{
        void*        record_memory = Allocate(sizeof(PmcRecord));
        PmcRecord* record        = new(record_memory) PmcRecord(data.pmc.timestamp, data.pmc.value);
        m_samples.emplace_back(record);
    }
    catch(std::exception ex)
    {
        return kRocProfVisDmResultAllocFailure;
    }
    return kRocProfVisDmResultSuccess;
}

size_t PmcTrackSlice::GetMemoryFootprint(){
    return sizeof(std::vector<PmcRecord>) + ((sizeof(PmcRecord)+sizeof(std::unique_ptr<PmcRecord>)) * m_samples.size());
}

size_t PmcTrackSlice::GetNumberOfRecords(){
    return m_samples.size();
}

rocprofvis_dm_result_t PmcTrackSlice::ConvertTimestampToIndex(const rocprofvis_dm_timestamp_t timestamp, rocprofvis_dm_index_t & index){
    std::vector<PmcRecord*>::iterator it = std::find_if( m_samples.begin(), m_samples.end(),
            [&timestamp](PmcRecord* & x) { return x->Timestamp() >= timestamp;});
    if (it != m_samples.end())
    {
        index = (rocprofvis_dm_index_t)(it - m_samples.begin());
        return kRocProfVisDmResultSuccess;
    }
    return kRocProfVisDmResultNotLoaded;
}

rocprofvis_dm_result_t PmcTrackSlice::GetRecordTimestampAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_timestamp_t & timestamp){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    timestamp = m_samples[index]->Timestamp();
    return kRocProfVisDmResultSuccess;
}

rocprofvis_dm_result_t PmcTrackSlice::GetRecordValueAt(const rocprofvis_dm_property_index_t index, rocprofvis_dm_value_t & value){
    ROCPROFVIS_ASSERT_MSG_RETURN(index < m_samples.size(), ERROR_INDEX_OUT_OF_RANGE, kRocProfVisDmResultNotLoaded);
    value = m_samples[index]->Value();
    return kRocProfVisDmResultSuccess;
}

}  // namespace DataModel
}  // namespace RocProfVis