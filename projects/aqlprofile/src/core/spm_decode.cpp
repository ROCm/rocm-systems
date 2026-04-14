#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
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
#include "src/core/include/spm_common.hpp"

#define PUBLIC_API __attribute__((visibility("default")))

PUBLIC_API hsa_status_t aqlprofile_spm_decode_query(
    aqlprofile_spm_buffer_desc_t desc_bin,
	aqlprofile_spm_decode_query_t query,
    uint64_t* param_out
) {
	SpmBufferDesc* desc = (SpmBufferDesc*)desc_bin.data;

	if (query == AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE)
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

PUBLIC_API hsa_status_t
aqlprofile_spm_decode_stream_v1(
    aqlprofile_spm_buffer_desc_t        desc_bin,
    aqlprofile_spm_decode_callback_v1_t decode_cb,
    void*                               _data,
    size_t                              _size,
    void*                               userdata
) {
	SpmBufferDesc* desc = (SpmBufferDesc*)desc_bin.data;

	if (desc->version != 1) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

	size_t seg_elem = 0;
	aqlprofile_spm_decode_query(desc_bin, AQLPROFILE_SPM_DECODE_QUERY_SEG_SIZE, &seg_elem);
	seg_elem /= 2;

	uint16_t* datain = (uint16_t*)_data;
	size_t datasize  = _size / sizeof(uint16_t);
	uint16_t* const data_end = datain + datasize;

	while (datain < data_end)
	{
		if (datain + seg_elem > data_end) return HSA_STATUS_ERROR_INVALID_ARGUMENT;

		uint64_t timestamp = *(uint64_t*)datain;

		for (int i = 0; i < desc->num_events; i++)
		{
			uint64_t counter_value = 0;

			uint16_t index = desc->get_counter_map()[i];
			bool is_global = (index & 0x8000) ? true : false;
			index &= 0x7FFF;

			if (is_global)
			{
				auto bufvalue = datain[index];
				decode_cb(timestamp, bufvalue, i, -1, userdata);
			}
			else
			{
				uint16_t se_base = desc->global_num_line * 16;
				uint16_t se_step = desc->se_num_line * 16;
				for (int j = 0; j < desc->num_se; j++)
				{
					auto bufvalue = datain[index + se_base + se_step * j];
					decode_cb(timestamp, bufvalue, i, j, userdata);
				}
			}
		}

		datain += seg_elem;
	}

	return HSA_STATUS_SUCCESS;
}