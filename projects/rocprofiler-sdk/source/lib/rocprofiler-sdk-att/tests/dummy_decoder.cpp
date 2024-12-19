// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "../att_decoder.h"

#include <array>
#include <vector>

__attribute__((visibility("default"))) rocprofiler_att_decoder_status_t
rocprofiler_att_decoder_parse_data(rocprofiler_att_decoder_se_data_callback_t se_data_callback,
                                   rocprofiler_att_decoder_trace_callback_t   trace_callback,
                                   rocprofiler_att_decoder_isa_callback_t     isa_callback,
                                   void*                                      userdata)
{
    const int gfxip = 9;

    trace_callback(ROCPROFILER_ATT_DECODER_TYPE_GFXIP, 0, (void*) gfxip, 0, userdata);
    {
        std::vector<rocprofiler_att_decoder_info_t> infos{};
        for(size_t i = 1; i < ROCPROFILER_ATT_DECODER_INFO_LAST; i++)
            infos.emplace_back(static_cast<rocprofiler_att_decoder_info_t>(i));

        trace_callback(ROCPROFILER_ATT_DECODER_TYPE_INFO, 0, infos.data(), infos.size(), userdata);
    }
    {
        uint64_t             memory_size = 0, size = 16;
        std::array<char, 16> inst;

        isa_callback(inst.data(), &memory_size, &size, pcinfo_t{0, 0}, userdata);
    }
    {
        int      se_id       = 0;
        uint8_t* buffer      = nullptr;
        size_t   buffer_size = 0;

        while(se_data_callback(&se_id, &buffer, &buffer_size, userdata))
        {};
    }

    {
        std::vector<att_occupancy_info_v2_t> vec;
        att_occupancy_info_v2_t              occ{};
        occ.cu = occ.se = occ.simd = occ.slot = 1;
        occ.pc.marker_id                      = 0;
        occ.pc.addr                           = 0;

        occ.time  = 0;
        occ.start = 1;
        vec.push_back(occ);
        occ.simd = 0;
        vec.push_back(occ);

        occ.time  = 1024;
        occ.start = 0;
        vec.push_back(occ);
        occ.simd = 1;
        vec.push_back(occ);

        trace_callback(ROCPROFILER_ATT_DECODER_TYPE_OCCUPANCY, 0, vec.data(), vec.size(), userdata);
    }

    {
        std::vector<att_wave_data_t> waves{};

        att_wave_data_t wave{};
        wave.cu = wave.simd = wave.wave_id = wave.traceID = 1;

        wave.begin_time = 0;
        wave.end_time   = 1024;

        std::vector<att_wave_state_t> states;
        for(int j = 0; j < 2; j++)
            for(int i = 1; i < ATT_WAVE_STATE_LAST; i++)
                states.emplace_back(att_wave_state_t{i, 128});

        std::vector<att_wave_instruction_t> insts;
        for(int i = 1; i < ATT_INST_LAST; i++)
        {
            att_wave_instruction_t inst{};
            inst.category     = i;
            inst.duration     = 48;
            inst.time         = i * 64 - 32;
            inst.pc.marker_id = 1;
            inst.pc.addr      = 8 * i;
            insts.emplace_back(std::move(inst));
        }

        wave.instructions_array = insts.data();
        wave.instructions_size  = insts.size();
        wave.timeline_array     = states.data();
        wave.timeline_size      = states.size();

        waves.push_back(wave);
        wave.simd = 2;
        waves.push_back(wave);

        trace_callback(ROCPROFILER_ATT_DECODER_TYPE_WAVE, 0, waves.data(), waves.size(), userdata);
    }

    return ROCPROFILER_ATT_DECODER_STATUS_SUCCESS;
}

__attribute__((visibility("default"))) const char*
rocprofiler_att_decoder_get_info_string(rocprofiler_att_decoder_info_t info)
{
    return std::vector<const char*>{"ROCPROFILER_ATT_DECODER_INFO_NONE",
                                    "ROCPROFILER_ATT_DECODER_INFO_DATA_LOST",
                                    "ROCPROFILER_ATT_DECODER_INFO_STITCH_INCOMPLETE",
                                    "ROCPROFILER_ATT_DECODER_INFO_LAST"}
        .at((size_t) info);
}

__attribute__((visibility("default"))) const char*
rocprofiler_att_decoder_get_status_string(rocprofiler_att_decoder_status_t status)
{
    return std::vector<const char*>{"ROCPROFILER_ATT_DECODER_STATUS_SUCCESS",
                                    "ROCPROFILER_ATT_DECODER_STATUS_ERROR",
                                    "ROCPROFILER_ATT_DECODER_STATUS_ERROR_OUT_OF_RESOURCES",
                                    "ROCPROFILER_ATT_DECODER_STATUS_ERROR_INVALID_ARGUMENT",
                                    "ROCPROFILER_ATT_DECODER_STATUS_ERROR_INVALID_SHADER_DATA",
                                    "ROCPROFILER_ATT_DECODER_STATUS_LAST"}
        .at((size_t) status);
}
