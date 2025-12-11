// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprofvis_db_query_factory.h"
#include "rocprofvis_db_sqlite.h"
#include <sstream>

namespace RocProfVis
{
namespace DataModel
{
    QueryFactory::QueryFactory(SqliteDatabase* db):m_db(db) {
    }

    void QueryFactory::SetVersion(const char* version) {
        m_db_version = ConvertVersionStringToInt(version);
    }

    std::vector<uint32_t>
        QueryFactory::ConvertVersionStringToInt(const char* version)
    {
        std::vector<uint32_t> version_array;
        std::stringstream ss(version);
        std::string       token;
        while(std::getline(ss, token, '.'))
        {
            version_array.push_back(std::stoi(token));
        }
        return version_array;
    }

    bool QueryFactory::IsVersionEqual(const char* version)
    {
        std::vector<uint32_t> db_version = ConvertVersionStringToInt(version);

        for (int i = 0; i < db_version.size(); i++)
        {
            uint32_t token = (m_db_version.size() > i) ? m_db_version[i] : 0;
            if(db_version[i] != token)
            {
                return false;
            }
        }
        return true;
    }

    bool QueryFactory::IsVersionGreaterOrEqual(const char* version) 
    {
        std::vector<uint32_t> db_version = ConvertVersionStringToInt(version);

        for(int i = 0; i < db_version.size(); i++)
        {
            uint32_t token = (m_db_version.size() > i) ? m_db_version[i] : 0;
            if(token > db_version[i])
            {
                return true;
            } else if (token < db_version[i])
            {
                return false;
            }
        }
        return true;
    }

/*************************************************************************************************
*                               Track build queries for CPU/GPU tracks
**************************************************************************************************/

    std::string QueryFactory::GetRocprofRegionTrackQuery(bool is_sample_track) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_region_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParamCategory(is_sample_track ? kRocProfVisDmRegionSampleTrack : kRocProfVisDmRegionMainTrack) },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid")},
                { is_sample_track ? Builder::Blank() : Builder::Where("SAMPLE.id", " IS ", "NULL") },
                }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_region_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParamCategory(is_sample_track ? kRocProfVisDmRegionSampleTrack : kRocProfVisDmRegionMainTrack) },
                { Builder::From("rocpd_region", "R"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid")},
                { is_sample_track ? Builder::Blank() : Builder::Where("SAMPLE.id", " IS ", "NULL") },
                }));
        }
    }

    std::string QueryFactory::GetRocprofKernelDispatchTrackQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmKernelDispatchTrack) },
                { Builder::From("rocpd_kernel_dispatch","K"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmKernelDispatchTrack) },
                { Builder::From("rocpd_kernel_dispatch") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocTrackQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmMemoryAllocationTrack) },
                { Builder::From("rocpd_memory_allocate","M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmMemoryAllocationTrack) },
                { Builder::From("rocpd_memory_allocate") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopyTrackQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmMemoryCopyTrack) },
                { Builder::From("rocpd_memory_copy","M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid",          Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id",     Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmMemoryCopyTrack) },
                { Builder::From("rocpd_memory_copy") } }));
        }
    }

    std::string QueryFactory::GetRocprofPerformanceCountersTrackQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmPmcTrack) },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid")
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmPmcTrack) },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid")
                } }));
        }

    }

    std::string QueryFactory::GetRocprofSMIPerformanceCountersTrackQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmPmcTrack) },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = S.track_id AND T.guid = S.guid")
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("PMC_I.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParamCategory(kRocProfVisDmPmcTrack) },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                } }));
        }

    }

/*************************************************************************************************
*                               Track build queries for stream tracks
**************************************************************************************************/

    std::string QueryFactory::GetRocprofKernelDispatchTrackQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_kernel_dispatch") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocTrackQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_memory_allocate") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopyTrackQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_track_query_format(
                { { Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParamCategory(kRocProfVisDmStreamTrack) },
                { Builder::From("rocpd_memory_copy") } }));
        }
    }

/*************************************************************************************************
*                               Event level calculation queries
**************************************************************************************************/

    std::string QueryFactory::GetRocprofRegionLevelQuery(bool is_sample_track) {
        if (IsVersionGreaterOrEqual("4"))
        {
           return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("R.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::SpaceSaver(0) },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.start_id AND TS.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R.end_id AND TE.guid = R.guid"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid")
                } }));
        }
        else
        {
           return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("start", Builder::START_SERVICE_NAME),
                Builder::QParam("end", Builder::END_SERVICE_NAME),
                Builder::QParam("R.id"),
                Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::SpaceSaver(0) },
                { Builder::From("rocpd_region", "R"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid")
                } }));
        }
    }

    std::string QueryFactory::GetRocprofKernelDispatchLevelQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("K.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_kernel_dispatch","K"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid")} }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("start", Builder::START_SERVICE_NAME), 
                Builder::QParam("end", Builder::END_SERVICE_NAME),
                Builder::QParam("id"),
                Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_kernel_dispatch") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocLevelQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_allocate","M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid")} }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("start", Builder::START_SERVICE_NAME), 
                Builder::QParam("end", Builder::END_SERVICE_NAME),
                Builder::QParam("id"),
                Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_allocate") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopyLevelQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_copy","M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid")} }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { {    Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("start", Builder::START_SERVICE_NAME), 
                Builder::QParam("end", Builder::END_SERVICE_NAME), 
                Builder::QParam("id"),
                Builder::QParam("nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("stream_id", Builder::STREAM_ID_SERVICE_NAME)},
                { Builder::From("rocpd_memory_copy") } }));
        }
    }

    std::string QueryFactory::GetRocprofPerformanceCountersLevelQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::SpaceSaver(0)
                },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME),
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::SpaceSaver(0)
                },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid")
                } }));
        }
    }

    std::string QueryFactory::GetRocprofSMIPerformanceCountersLevelQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TS.value", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::SpaceSaver(0)
                },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = S.timestamp AND TS.guid = S.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = S.track_id AND T.guid = S.guid"),
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_level_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("S.timestamp", Builder::START_SERVICE_NAME), 
                Builder::QParam("S.timestamp", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0), 
                Builder::QParam("PMC_I.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::SpaceSaver(0)
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                } }));
        }
    }

/*************************************************************************************************
*                               Time slice queries for CPU/GPU tracks
**************************************************************************************************/
    std::string QueryFactory::GetRocprofRegionSliceQuery(bool is_sample_track) {
        if (IsVersionGreaterOrEqual("4"))
        {
           return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"),
                Builder::QParam("R.name_id"),
                Builder::QParam("R.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = R.event_id AND E.guid = R.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.start_id AND TS.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R.end_id AND TE.guid = R.guid"),
                Builder::LeftJoin(Builder::LevelTable(is_sample_track ? "launch_sample":"launch"), "L", "R.id = L.eid"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid")
                } }));
        }
        else
        {
           return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("R.start", Builder::START_SERVICE_NAME),
                Builder::QParam("R.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"),
                Builder::QParam("R.name_id"),
                Builder::QParam("R.id"),
                Builder::QParam("R.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = R.event_id AND E.guid = R.guid"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid"),
                Builder::LeftJoin(Builder::LevelTable(is_sample_track ? "launch_sample":"launch"), "L", "R.id = L.eid") } }));
        }
    }

    std::string QueryFactory::GetRocprofKernelDispatchSliceQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("K.kernel_id"),
                Builder::QParam("K.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") } }));
        }
        else
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("K.kernel_id"),
                Builder::QParam("K.id"),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("K.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocSliceQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("E.category_id"),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("E.category_id"),
                Builder::QParam("M.id"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopySliceQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("M.name_id"),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("M.name_id"),
                Builder::QParam("M.id"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("L.level") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid")
                } }));
        }
    }

    std::string QueryFactory::GetRocprofPerformanceCountersSliceQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::SpaceSaver(0),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", "level"),                        
                },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                } }));
        }
        else
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME),
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::SpaceSaver(0),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", "level"),                        
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                } }));
        }
    }


    std::string QueryFactory::GetRocprofSMIPerformanceCountersSliceQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME),
                Builder::QParam("TS.value", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::SpaceSaver(0),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.pmc_id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", "level"),                        
                },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "T.id = S.timestamp_id AND T.guid = S.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = S.track_id AND T.guid = S.guid"),
                } }));
        }
        else
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("S.timestamp", Builder::START_SERVICE_NAME), 
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME),
                Builder::QParam("S.timestamp", Builder::END_SERVICE_NAME),
                Builder::SpaceSaver(0),
                Builder::SpaceSaver(0),
                Builder::QParam("PMC_I.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", "level"),                        
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                } }));
        }
    }


/*************************************************************************************************
*                               Time slice queries for stream tracks
**************************************************************************************************/

    std::string QueryFactory::GetRocprofKernelDispatchSliceQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("K.kernel_id"),
                Builder::QParam("K.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("K.kernel_id"),
                Builder::QParam("K.id"),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocSliceQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("E.category_id"),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("E.category_id"),
                Builder::QParam("M.id"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopySliceQueryForStream() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("M.name_id"),
                Builder::QParam("M.id"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_slice_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("E.category_id"), 
                Builder::QParam("M.name_id"),
                Builder::QParam("M.id"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME),
                Builder::SpaceSaver(-1),
                Builder::QParam("L.level_for_stream", "level") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") } }));
        }
    }

/*************************************************************************************************
*                               Table view queries
**************************************************************************************************/

    std::string QueryFactory::GetRocprofRegionTableQuery(bool is_sample_track) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("R.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("S.string", "name"),
                Builder::QParamBlank("stream"),
                Builder::QParamBlank("queue"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("TH.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParamBlank("device_index"),
                Builder::QParamBlank("device"),
                Builder::QParamBlank("device_name"),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("(TE.value-TS.value)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParamBlank("size"),
                Builder::QParamBlank("address"),
                Builder::QParamBlank("level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("-1", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = R.event_id AND E.guid = R.guid"),
                Builder::InnerJoin("rocpd_string", "S", "S.id = R.name_id AND S.guid = R.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.start_id AND TS.guid = R.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R.end_id AND TE.guid = R.guid"),
                Builder::InnerJoin("rocpd_info_process", "P", "P.id = T.pid AND P.guid = R.guid"),
                Builder::InnerJoin("rocpd_info_thread", "TH", "TH.id = T.tid AND TH.guid = R.guid"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid"),
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(is_sample_track ? kRocProfVisDmOperationLaunchSample : kRocProfVisDmOperationLaunch),
                Builder::QParam("R.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("S.string", "name"),
                Builder::QParamBlank("stream"),
                Builder::QParamBlank("queue"),
                Builder::QParam("R.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("P.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParamBlank("device_index"),
                Builder::QParamBlank("device"),
                Builder::QParamBlank("device_name"),
                Builder::QParam("R.start", Builder::START_SERVICE_NAME),
                Builder::QParam("R.end", Builder::END_SERVICE_NAME),
                Builder::QParam("(R.end-R.start)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParamBlank("size"),
                Builder::QParamBlank("address"),
                Builder::QParamBlank("level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("-1", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_region", "R"),
                is_sample_track ? Builder::InnerJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid") : 
                                  Builder::LeftJoin("rocpd_sample", "SAMPLE", "SAMPLE.event_id = R.event_id AND SAMPLE.guid = R.guid"),
                Builder::InnerJoin("rocpd_string", "S", "S.id = R.name_id AND S.guid = R.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = R.event_id AND E.guid = R.guid"),
                Builder::InnerJoin("rocpd_info_process", "P", "P.id = R.pid AND P.guid = R.guid"),
                Builder::InnerJoin("rocpd_info_thread", "T", "T.id = R.tid AND T.guid = R.guid") } }));
        }
    }

    std::string QueryFactory::GetRocprofKernelDispatchTableQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("K.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("S.display_name", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("AG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"AG.type","AG.type_index"}), "device"),
                Builder::QParam("AG.name", "device_name"),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("(TE.value-TS.value)", "duration"),

                Builder::QParam("K.grid_size_x", "GridSizeX"),
                Builder::QParam("K.grid_size_y", "GridSizeY"),
                Builder::QParam("K.grid_size_z", "GridSizeZ"),

                Builder::QParam("K.workgroup_size_x", "WGSizeX"),
                Builder::QParam("K.workgroup_size_y", "WGSizeY"),
                Builder::QParam("K.workgroup_size_z", "WGSizeZ"),

                Builder::QParam("K.group_segment_size", "LDSSize"),
                Builder::QParam("K.private_segment_size", "ScratchSize"),

                Builder::QParam("S.group_segment_size", "StaticLDSSize"),
                Builder::QParam("S.private_segment_size", "StaticScratchSize"),

                Builder::QParamBlank("size"),
                Builder::QParamBlank("address"),
                Builder::QParamBlank("level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::InnerJoin("rocpd_info_agent", "AG", "AG.id = T.agent_id AND AG.guid = K.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::InnerJoin("rocpd_info_kernel_symbol", "S", "S.id = K.kernel_id AND S.guid = K.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = T.queue_id AND Q.guid = K.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = K.stream_id AND ST.guid = K.guid")
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("K.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("S.display_name", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("K.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("AG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"AG.type","AG.type_index"}), "device"),
                Builder::QParam("AG.name", "device_name"),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME),
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::QParam("(K.end-K.start)", "duration"),

                Builder::QParam("K.grid_size_x", "GridSizeX"),
                Builder::QParam("K.grid_size_y", "GridSizeY"),
                Builder::QParam("K.grid_size_z", "GridSizeZ"),

                Builder::QParam("K.workgroup_size_x", "WGSizeX"),
                Builder::QParam("K.workgroup_size_y", "WGSizeY"),
                Builder::QParam("K.workgroup_size_z", "WGSizeZ"),

                Builder::QParam("K.group_segment_size", "LDSSize"),
                Builder::QParam("K.private_segment_size", "ScratchSize"),

                Builder::QParam("S.group_segment_size", "StaticLDSSize"),
                Builder::QParam("S.private_segment_size", "StaticScratchSize"),

                Builder::QParamBlank("size"),
                Builder::QParamBlank("address"),
                Builder::QParamBlank("level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("K.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("K.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_info_agent", "AG", "AG.id = K.agent_id AND AG.guid = K.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = K.event_id AND E.guid = K.guid"),
                Builder::InnerJoin("rocpd_info_kernel_symbol", "S", "S.id = K.kernel_id AND S.guid = K.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = K.queue_id AND Q.guid = K.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = K.stream_id AND ST.guid = K.guid")
                } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryAllocTableQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("M.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("M.type", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("AG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"AG.type","AG.type_index"}), "device"),
                Builder::QParam("AG.name", "device_name"),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("(TE.value-TS.value)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParam("M.size", "size"),
                Builder::QParam("M.address", "address"),
                Builder::QParam("M.level", "level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::InnerJoin("rocpd_info_agent", "AG", "AG.id = T.agent_id AND AG.guid = M.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = T.queue_id AND Q.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = M.stream_id AND ST.guid = M.guid")
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("M.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("M.type", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("M.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("AG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"AG.type","AG.type_index"}), "device"),
                Builder::QParam("AG.name", "device_name"),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME),
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("(M.end-M.start)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParam("M.size", "size"),
                Builder::QParam("M.address", "address"),
                Builder::QParam("M.level", "level"),

                Builder::QParamBlank("SrcIndex"),
                Builder::QParamBlank("SrcDevice"),
                Builder::QParamBlank("SrcName"),
                Builder::QParamBlank("SrcAddr"),

                Builder::QParam("M.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::LeftJoin("rocpd_info_agent", "AG", "AG.id = M.agent_id AND AG.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = M.queue_id AND Q.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = M.stream_id AND ST.guid = M.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid") } }));
        }
    }

    std::string QueryFactory::GetRocprofMemoryCopyTableQuery() {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("M.id"),
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE "
                    "RS.id = E.category_id AND RS.guid = E.guid)",
                    "category"),
                Builder::QParam("M.type", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("DSTAG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"DSTAG.type","DSTAG.type_index"}), "device"),
                Builder::QParam("DSTAG.name", "device_name"),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME),
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("(TE.value-TS.value)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParam("M.size", "size"),
                Builder::QParam("M.dst_address", "address"),
                Builder::QParamBlank("level"),

                Builder::QParam("SRCAG.absolute_index", "SrcIndex"),
                Builder::QParam(Builder::Concat({"SRCAG.type","SRCAG.type_index"}), "SrcDevice"),
                Builder::QParam("SRCAG.name", "SrcName"),
                Builder::QParam("M.src_address", "SrcAddr"),

                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME) },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                Builder::InnerJoin("rocpd_info_agent", "DSTAG", "DSTAG.id = M.dst_agent_id AND DSTAG.guid = M.guid"),
                Builder::InnerJoin("rocpd_info_agent", "SRCAG", "SRCAG.id = M.src_agent_id AND SRCAG.guid = M.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = T.queue_id AND Q.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = M.stream_id AND ST.guid = M.guid")
                } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("M.id"), 
                Builder::QParam("( SELECT string FROM `rocpd_string` RS WHERE RS.id = E.category_id AND RS.guid = E.guid)", "category"),
                Builder::QParam("S.string", "name"),
                Builder::QParam("ST.name", "stream"),
                Builder::QParam("Q.name", "queue"),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.pid", Builder::PROCESS_ID_PUBLIC_NAME),
                Builder::QParam("M.tid", Builder::THREAD_ID_PUBLIC_NAME),
                Builder::QParam("DSTAG.absolute_index", "device_index"),
                Builder::QParam(Builder::Concat({"DSTAG.type","DSTAG.type_index"}), "device"),
                Builder::QParam("DSTAG.name", "device_name"),
                Builder::QParam("M.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("M.end", Builder::END_SERVICE_NAME),
                Builder::QParam("(M.end-M.start)", "duration"),

                Builder::QParamBlank("GridSizeX"),
                Builder::QParamBlank("GridSizeY"),
                Builder::QParamBlank("GridSizeZ"),

                Builder::QParamBlank("WGSizeX"),
                Builder::QParamBlank("WGSizeY"),
                Builder::QParamBlank("WGSizeZ"),

                Builder::QParamBlank("LDSSize"),
                Builder::QParamBlank("ScratchSize"),

                Builder::QParamBlank("StaticLDSSize"),
                Builder::QParamBlank("StaticScratchSize"),

                Builder::QParam("M.size", "size"),
                Builder::QParam("M.dst_address", "address"),
                Builder::QParamBlank("level"),

                Builder::QParam("SRCAG.absolute_index", "SrcIndex"),
                Builder::QParam(Builder::Concat({"SRCAG.type","SRCAG.type_index"}), "SrcDevice"),
                Builder::QParam("SRCAG.name", "SrcName"),
                Builder::QParam("M.src_address", "SrcAddr"),

                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME)},
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_string", "S", "S.id = M.name_id AND S.guid = M.guid"),
                Builder::InnerJoin("rocpd_info_agent", "DSTAG", "DSTAG.id = M.dst_agent_id AND DSTAG.guid = M.guid"),
                Builder::InnerJoin("rocpd_info_agent", "SRCAG", "SRCAG.id = M.src_agent_id AND SRCAG.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_queue", "Q", "Q.id = M.queue_id AND Q.guid = M.guid"),
                Builder::LeftJoin("rocpd_info_stream", "ST", "ST.id = M.stream_id AND ST.guid = M.guid"),
                Builder::InnerJoin("rocpd_event", "E", "E.id = M.event_id AND E.guid = M.guid")
                } }));
        }
    }

    std::string QueryFactory::GetRocprofPerformanceCountersTableQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_sample_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TE.value", Builder::END_SERVICE_NAME),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME)                        
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                } }));
        }
        else
        {
            return  Builder::Select(rocprofvis_db_sqlite_sample_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("K.start", Builder::START_SERVICE_NAME), 
                Builder::QParam("K.end", Builder::END_SERVICE_NAME),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME)
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = PMC_E.event_id AND K.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                } }));
        }
    }



    std::string QueryFactory::GetRocprofSMIPerformanceCountersTableQuery() {

        if (IsVersionGreaterOrEqual("4"))
        {
            return  Builder::Select(rocprofvis_db_sqlite_sample_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("TS.value", Builder::START_SERVICE_NAME), 
                Builder::QParam("TS.value", Builder::END_SERVICE_NAME),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME)                        
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = S.timestamp_id AND TS.guid = S.guid"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = S.track_id AND T.guid = S.guid"),
                } }));
        }
        else
        {
            return  Builder::Select(rocprofvis_db_sqlite_sample_table_query_format(
                { m_db,
                { Builder::QParamOperation(kRocProfVisDmOperationNoOp),
                Builder::QParam("S.timestamp", Builder::START_SERVICE_NAME), 
                Builder::QParam("S.timestamp", Builder::END_SERVICE_NAME),
                Builder::QParam("PMC_I.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("PMC_I.id", Builder::COUNTER_ID_SERVICE_NAME),
                Builder::QParam("PMC_E.value", Builder::COUNTER_VALUE_SERVICE_NAME)
                    },
                { Builder::From("rocpd_pmc_event", "PMC_E"),
                Builder::InnerJoin("rocpd_info_pmc", "PMC_I", "PMC_I.id = PMC_E.pmc_id AND PMC_I.guid = PMC_E.guid"),
                Builder::InnerJoin("rocpd_sample", "S", "S.event_id = PMC_E.event_id AND S.guid = PMC_E.guid"),
                } }));
        }
    }

    std::string QueryFactory::GetRocprofDataFlowQueryForRegionEvent(uint64_t event_id)
    {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("R2.id"),
                        Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                        Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                        Builder::QParam("TS.value","start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("R2.name_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("TE.value", "end"),

                        },
                    { 
                        Builder::From("rocpd_region", "R1"),
                        Builder::InnerJoin("rocpd_event", "E1", "R1.event_id = E1.id AND E1.stack_id != 0 AND R1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R2", "R2.event_id = E2.id AND R2.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = R2.track_id AND T.guid = R2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R2.start_id AND TS.guid = R2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R2.end_id AND TE.guid = R2.guid"), 
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L","R2.id = L.eid") },
                    { Builder::Where(
                        "R1.id", "==", std::to_string(event_id))
                    } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("K.id"),
                        Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("TS.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("K.kernel_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("TE.end"),

                        },
                    { 
                        Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1","R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = E2.id AND K.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K.start_id AND TS.guid = K.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = K.end_id AND TE.guid = K.guid"),
                        Builder::InnerJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid")
                    },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("M.id"),
                        Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("TS.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("M.name_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("TE.end"),

                        },
                    { Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1", "R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_copy", "M", "M.event_id = E2.id AND M.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"), 
                        Builder::InnerJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("M.id"),
                        Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("TS.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("E2.category_id"), // This should be name_id, but Alloc table does not have column
                        Builder::QParam("L.level"),
                        Builder::QParam("TE.end"),

                        },
                    { 
                        Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1", "R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_allocate", "M", "M.event_id = E2.id AND M.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M.start_id AND TS.guid = M.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = M.end_id AND TE.guid = M.guid"),
                        Builder::InnerJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })));
        }
        else
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("R2.id"),
                        Builder::QParam("R2.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("R2.pid", Builder::PROCESS_ID_SERVICE_NAME),
                        Builder::QParam("R2.tid", Builder::THREAD_ID_SERVICE_NAME),
                        Builder::QParam("R2.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("R2.name_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("R2.end"),
                        },
                    { 
                        Builder::From("rocpd_region", "R1"),
                        Builder::InnerJoin("rocpd_event", "E1", "R1.event_id = E1.id AND E1.stack_id != 0 AND R1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R2", "R2.event_id = E2.id AND R2.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L","R2.id = L.eid") },
                    { Builder::Where(
                        "R1.id", "==", std::to_string(event_id))
                    } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("K.id"),
                        Builder::QParam("K.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("K.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("K.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("K.kernel_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("K.end"),

                        },
                    { 
                        Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1","R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_kernel_dispatch", "K", "K.event_id = E2.id AND K.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid")
                    },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("M.id"),
                        Builder::QParam("M.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("M.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("M.name_id"),
                        Builder::QParam("L.level"),
                        Builder::QParam("M.end"),

                        },
                    { Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1", "R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_copy", "M", "M.event_id = E2.id AND M.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })) +
                Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                        Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                        Builder::QParam("E2.id", "id"),
                        Builder::QParam("M.id"),
                        Builder::QParam("M.nid", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("M.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                        Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                        Builder::QParam("M.start"),
                        Builder::QParam("E2.category_id"),
                        Builder::QParam("E2.category_id"), // This should be name_id, but Alloc table does not have column
                        Builder::QParam("L.level"),
                        Builder::QParam("M.end"),

                        },
                    { 
                        Builder::From("rocpd_region", "R"),
                        Builder::InnerJoin("rocpd_event", "E1", "R.event_id = E1.id AND E1.stack_id != 0 AND R.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_allocate", "M", "M.event_id = E2.id AND M.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") },
                    { Builder::Where(
                        "R.id", "==", std::to_string(event_id)) } })));
        }
    }

    std::string QueryFactory::GetRocprofDataFlowQueryForKernelDispatchEvent(uint64_t event_id)
    {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("TS.value", "start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("TE.value", "end"),

                        },
                    {
                        Builder::From("rocpd_kernel_dispatch", "K"),
                        Builder::InnerJoin("rocpd_event", "E1", "K.event_id = E1.id AND E1.stack_id != 0 AND K.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.start_id AND TS.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE","TE.id = R.end_id AND TE.guid = R.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { Builder::Where("K.id", "==", std::to_string(event_id)) } })) +
                    Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("K2.id"),
                            Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                            Builder::QParam("TS.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("K2.kernel_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("TE.end"),

                        },
                    {
                        Builder::From("rocpd_kernel_dispatch", "K1"),
                        Builder::InnerJoin("rocpd_event", "E1", "K1.event_id = E1.id AND E1.stack_id != 0 AND K1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_kernel_dispatch", "K2", "K2.event_id = E2.id AND K2.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = K2.track_id AND T.guid = K2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = K2.start_id AND TS.guid = K2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE","TE.id = K2.end_id AND TE.guid = K2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("dispatch"), "L", "K2.id = L.eid")
                    },
                    { Builder::Where(
                        "K1.id", "==", std::to_string(event_id)) } })));
        }
        else
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("R.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("R.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("R.end"),

                        },
                    {
                        Builder::From("rocpd_kernel_dispatch", "K"),
                        Builder::InnerJoin("rocpd_event", "E1", "K.event_id = E1.id AND E1.stack_id != 0 AND K.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { Builder::Where("K.id", "==", std::to_string(event_id)) } })) +
                    Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("K2.id"),
                            Builder::QParam("K2.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("K2.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("K2.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                            Builder::QParam("K2.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("K2.kernel_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("K2.end"),

                        },
                    {
                        Builder::From("rocpd_kernel_dispatch", "K1"),
                        Builder::InnerJoin("rocpd_event", "E1", "K1.event_id = E1.id AND E1.stack_id != 0 AND K1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_kernel_dispatch", "K2", "K2.event_id = E2.id AND K2.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("dispatch"), "L", "K2.id = L.eid")
                    },
                    { Builder::Where(
                        "K1.id", "==", std::to_string(event_id)) } }))
            );
        }
    }

    std::string QueryFactory::GetRocprofDataFlowQueryForMemoryCopyEvent(uint64_t event_id)
    {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("TS.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("TE.end"),

                        },
                    { 
                        Builder::From("rocpd_memory_copy", "M"),
                        Builder::InnerJoin("rocpd_event", "E1", "M.event_id = E1.id AND E1.stack_id != 0 AND M.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.end_id AND TS.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R.end_id AND TE.guid = R.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { 
                        Builder::Where("M.id", "==", std::to_string(event_id)) } })) +
                        Builder::Union() +
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("M2.id"),
                            Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("M2.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                            Builder::QParam("TS.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("M2.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("TE.end"),

                        },
                    { 
                        Builder::From("rocpd_memory_copy", "M1"),
                        Builder::InnerJoin("rocpd_event", "E1", "M1.event_id = E1.id AND E1.stack_id != 0 AND M1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_copy", "M2", "M2.event_id = E2.id AND M2.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = M2.track_id AND T.guid = M2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = M2.start_id AND TS.guid = M2.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE","TE.id = M2.end_id AND TE.guid = M2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("mem_copy"), "L", "M2.id = L.eid") 
                    },
                    { Builder::Where(
                        "M1.id", "==", std::to_string(event_id)) } })));
        }
        else
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("R.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("R.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("R.end"),

                    },
                    { 
                        Builder::From("rocpd_memory_copy", "M"),
                        Builder::InnerJoin("rocpd_event", "E1", "M.event_id = E1.id AND E1.stack_id != 0 AND M.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { 
                        Builder::Where("M.id", "==", std::to_string(event_id)) } })) +
                        Builder::Union() +
                        Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("M2.id"),
                            Builder::QParam("M2.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("M2.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("M2.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                            Builder::QParam("M2.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("M2.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("M2.end"),

                    },
                    { 
                        Builder::From("rocpd_memory_copy", "M1"),
                        Builder::InnerJoin("rocpd_event", "E1", "M1.event_id = E1.id AND E1.stack_id != 0 AND M1.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_memory_copy", "M2", "M2.event_id = E2.id AND M2.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("mem_copy"), "L", "M2.id = L.eid") 
                    },
                    { Builder::Where(
                        "M1.id", "==", std::to_string(event_id)) } })));
        }
    }

    std::string QueryFactory::GetRocprofDataFlowQueryForMemoryAllocEvent(uint64_t event_id)
    {

        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("T.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("TS.value","start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("TE.value", "end"),

                        },
                    { 
                        Builder::From("rocpd_memory_allocate", "M"),
                        Builder::InnerJoin("rocpd_event", "E1", "M.event_id = E1.id AND E1.stack_id != 0 AND M.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TS", "TS.id = R.start_id AND TS.guid = R.guid"),
                        Builder::InnerJoin("rocpd_timestamp", "TE", "TE.id = R.end_id AND TE.guid = R.guid"),

                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { Builder::Where("M.id", "==", std::to_string(event_id)) } })));
        }
        else
        {
            return Builder::SelectAll(
                Builder::Select(rocprofvis_db_sqlite_dataflow_query_format(
                    { {
                            Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                            Builder::QParam("E2.id", "id"),
                            Builder::QParam("R.id"),
                            Builder::QParam("R.nid", Builder::AGENT_ID_SERVICE_NAME),
                            Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                            Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                            Builder::QParam("R.start"),
                            Builder::QParam("E2.category_id"),
                            Builder::QParam("R.name_id"),
                            Builder::QParam("L.level"),
                            Builder::QParam("R.end"),

                        },
                    { 
                        Builder::From("rocpd_memory_allocate", "M"),
                        Builder::InnerJoin("rocpd_event", "E1", "M.event_id = E1.id AND E1.stack_id != 0 AND M.guid = E1.guid"),
                        Builder::InnerJoin("rocpd_event", "E2", "E1.stack_id = E2.stack_id AND E1.id != E2.id AND E1.guid = E2.guid"),
                        Builder::InnerJoin("rocpd_region", "R", "R.event_id = E2.id AND R.guid = E2.guid"),
                        Builder::InnerJoin(Builder::LevelTable("launch"), "L", "R.id = L.eid") },
                    { Builder::Where("M.id", "==", std::to_string(event_id)) } })));
        }
    }


    std::string QueryFactory::GetRocprofEssentialInfoQueryForRegionEvent(uint64_t event_id, bool is_sample_track) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("T.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("-1", Builder::STREAM_ID_SERVICE_NAME),
                Builder::QParam("L.level"),
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_region", "R"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = R.track_id AND T.guid = R.guid"),
                Builder::LeftJoin(Builder::LevelTable(is_sample_track? "launch_sample" : "launch"), "L", "R.id = L.eid") },
                { Builder::Where("R.id", "==", std::to_string(event_id)) } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationLaunch),
                Builder::QParam("R.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("R.pid", Builder::PROCESS_ID_SERVICE_NAME),
                Builder::QParam("R.tid", Builder::THREAD_ID_SERVICE_NAME),
                Builder::QParam("-1", Builder::STREAM_ID_SERVICE_NAME),
                Builder::QParam("L.level"),
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_region", "R"),
                Builder::LeftJoin(Builder::LevelTable(is_sample_track? "launch_sample" : "launch"), "L", "R.id = L.eid") },
                { Builder::Where("R.id", "==", std::to_string(event_id)) } }));
        }
    }
    std::string QueryFactory::GetRocprofEssentialInfoQueryForKernelDispatchEvent(uint64_t event_id) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"), 
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = K.track_id AND T.guid = K.guid"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") },
                { Builder::Where("K.id", "==", std::to_string(event_id)) } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationDispatch),
                Builder::QParam("K.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("K.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("K.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("K.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"), 
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_kernel_dispatch", "K"),
                Builder::LeftJoin(Builder::LevelTable("dispatch"), "L", "K.id = L.eid") },
                { Builder::Where("K.id", "==", std::to_string(event_id)) } }));
        }

    }
    std::string QueryFactory::GetRocprofEssentialInfoQueryForMemoryAllocEvent(uint64_t event_id) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("T.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"),
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") },
                { Builder::Where("M.id", "==", std::to_string(event_id)) } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryAllocate),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"),
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_memory_allocate", "M"),
                Builder::LeftJoin(Builder::LevelTable("mem_alloc"), "L", "M.id = L.eid") },
                { Builder::Where("M.id", "==", std::to_string(event_id)) } }));
        }
    }
    std::string QueryFactory::GetRocprofEssentialInfoQueryForMemoryCopyEvent(uint64_t event_id) {
        if (IsVersionGreaterOrEqual("4"))
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("T.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("T.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("T.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"), 
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::InnerJoin("rocpd_track", "T", "T.id = M.track_id AND T.guid = M.guid"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") },
                { Builder::Where("M.id", "==", std::to_string(event_id)) } }));
        }
        else
        {
            return Builder::Select(rocprofvis_db_sqlite_essential_data_query_format(
                { { Builder::QParamOperation(kRocProfVisDmOperationMemoryCopy),
                Builder::QParam("M.nid", Builder::NODE_ID_SERVICE_NAME),
                Builder::QParam("M.dst_agent_id", Builder::AGENT_ID_SERVICE_NAME),
                Builder::QParam("M.queue_id", Builder::QUEUE_ID_SERVICE_NAME),
                Builder::QParam("M.stream_id", Builder::STREAM_ID_SERVICE_NAME), 
                Builder::QParam("L.level"), 
                Builder::QParam("L.level_for_stream") },
                { Builder::From("rocpd_memory_copy", "M"),
                Builder::LeftJoin(Builder::LevelTable("mem_copy"), "L", "M.id = L.eid") },
                { Builder::Where("M.id", "==", std::to_string(event_id)) } }));
        }
    }

}  // namespace DataModel
}  // namespace RocProfVis