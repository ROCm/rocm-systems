//
//
//

#include "lib/aqlprofile/core/spm_common.hpp"

#include <assert.h>
#include <stdlib.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <map>
#include <atomic>
#include <future>
#include <fstream>
#include <cstring>

#ifdef _WIN32
#    define PUBLIC_API
#else
#    define PUBLIC_API __attribute__((visibility("default")))
#endif

PUBLIC_API hsa_status_t
aqlprofile_spm_decode_query(aqlprofile_spm_buffer_desc_t  desc_bin,
                            aqlprofile_spm_decode_query_t query,
                            uint64_t*                     param_out)
{
    SpmBufferDesc* desc = (SpmBufferDesc*) desc_bin.data;

    if(query == AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE)
        *param_out = (desc->global_num_line + desc->se_num_line * desc->num_se) * 32;
    else if(query == AQLPROFILE_SPM_DECODE_QUERY_NUM_XCC)
        *param_out = desc->num_xcc;
    else if(query == AQLPROFILE_SPM_DECODE_QUERY_EVENT_COUNT)
        *param_out = desc->num_events;
    else if(query == AQLPROFILE_SPM_DECODE_QUERY_COUNTER_MAP_BYTE_OFFSET)
        *param_out = size_t(desc->get_counter_map()) - size_t(desc);
    else
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    return HSA_STATUS_SUCCESS;
}

static inline int
encode_spm_shader_engine(uint32_t se_index, uint32_t sa_index = 0, uint32_t wgp_index = 0)
{
    return int((wgp_index << 24) | (sa_index << 16) | se_index);
}

PUBLIC_API hsa_status_t
aqlprofile_spm_decode_stream_v1(aqlprofile_spm_buffer_desc_t        desc_bin,
                                aqlprofile_spm_decode_callback_v1_t decode_cb,
                                void*                               _data,
                                size_t                              _size,
                                void*                               userdata)
{
    SpmBufferDesc* desc = (SpmBufferDesc*) desc_bin.data;

    if(desc->version != 1) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    size_t seg_elem = 0;
    aqlprofile_spm_decode_query(desc_bin, AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE, &seg_elem);
    seg_elem /= 2;

    uint16_t*       datain   = (uint16_t*) _data;
    size_t          datasize = _size / sizeof(uint16_t);
    uint16_t* const data_end = datain + datasize;

    auto decode_bufvalue = [](uint16_t lo, uint16_t hi, bool is_32bit) {
        if(is_32bit)
        {
            if((lo == 0xFFFF) && (hi == 0xFFFF)) return uint64_t(-1);
            return uint64_t(lo) | (uint64_t(hi) << 16);
        }

        if(lo == 0xFFFF) return uint64_t(-1);
        return uint64_t(lo);
    };

    while(datain < data_end)
    {
        if(datain + seg_elem > data_end) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

        uint64_t timestamp = *(uint64_t*) datain;
        size_t   i_exp     = desc->num_events;

        for(int i = 0; i < desc->num_events; i++)
        {
            uint16_t index     = desc->get_counter_map()[i];
            bool     is_global = (index & SPM_COUNTER_MAP_GLOBAL_FLAG) ? true : false;
            bool     is_sa     = (index & SPM_COUNTER_MAP_SA_FLAG) ? true : false;
            bool     is_32bit  = (index & SPM_COUNTER_MAP_32BIT_FLAG) ? true : false;
            bool     is_wgp    = (index & SPM_COUNTER_MAP_WGP_FLAG) ? true : false;
            index &= SPM_COUNTER_MAP_INDEX_MASK;
            uint32_t sa_count       = is_sa ? desc->num_sa : 1;
            uint32_t wgp_count      = is_wgp ? desc->num_wgp : 1;
            size_t   expanded_count = size_t(sa_count) * size_t(wgp_count) - 1;

            if(is_global)
            {
                auto bufvalue = decode_bufvalue(datain[index], is_32bit ? datain[index + 16] : 0,
                                               is_32bit);
                decode_cb(timestamp, bufvalue, i, -1, userdata);
            }
            else
            {
                uint16_t se_base = desc->global_num_line * 16;
                uint16_t se_step = desc->se_num_line * 16;
                size_t   event_exp_start = i_exp;
                for(int j = 0; j < desc->num_se; j++)
                {
                    auto bufvalue = decode_bufvalue(datain[index + se_base + se_step * j],
                                                   is_32bit ? datain[index + 16 + se_base +
                                                                          se_step * j]
                                                            : 0,
                                                   is_32bit);
                    decode_cb(timestamp, bufvalue, i, j, userdata);

                    size_t event_i_exp = event_exp_start;
                    for(uint32_t sa = 0; sa < sa_count; ++sa)
                        for(uint32_t wgp = 0; wgp < wgp_count; ++wgp)
                        {
                            if((sa == 0) && (wgp == 0)) continue;

                            uint16_t expanded_index =
                                desc->get_counter_map()[event_i_exp++] & SPM_COUNTER_MAP_INDEX_MASK;
                            auto bufvalue =
                                decode_bufvalue(datain[expanded_index + se_base + se_step * j],
                                                is_32bit ? datain[expanded_index + 16 + se_base +
                                                                      se_step * j]
                                                         : 0,
                                                is_32bit);
                            decode_cb(timestamp,
                                      bufvalue,
                                      i,
                                      encode_spm_shader_engine(j, sa, wgp),
                                      userdata);
                        }
                }
            }

            i_exp += expanded_count;
        }

        datain += seg_elem;
    }

    return HSA_STATUS_SUCCESS;
}
