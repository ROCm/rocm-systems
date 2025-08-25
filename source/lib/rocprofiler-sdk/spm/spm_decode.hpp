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

#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include <rocprofiler-sdk/fwd.h>
namespace rocprofiler
{
namespace SPM
{
struct values_vec_t
{
    std::vector<uint64_t> timestamps;
    std::vector<uint64_t> values;
};

struct instances_t
{
    std::vector<values_vec_t> shaders;
    bool                      is_global = false;
};

typedef struct
{
    void*  data;
    size_t size;
    size_t seg_size;
    size_t buffer_num;
} spm_descriptor_t;

typedef std::vector<instances_t> counter_vec;
void
decode_cb(uint64_t timestamp, uint64_t value, uint64_t index, int shader_engine, void* userdata);
void
aql_data_callback(size_t, void*, size_t, int, void*);
struct spm_counter_instance_t
{
    rocprofiler_counter_id_t id;
    uint64_t                 instance;
};

// event_map = (char*)this + this->size
// aql_desc = event_map + num_events * aql_desc_size;
struct spm_desc_v0_t
{
    uint64_t version{0};
    uint64_t struct_size{sizeof(spm_desc_v0_t)};
    uint64_t aql_desc_size{0};
    uint64_t num_events{0};
    uint64_t event_elem_size{sizeof(spm_counter_instance_t)};
    uint64_t reserved{0};

    bool valid()
    {
        return version == 0 && aql_desc_size != 0 && struct_size == sizeof(spm_desc_v0_t) &&
               event_elem_size == sizeof(spm_counter_instance_t);
    }

    spm_counter_instance_t* events() { return reinterpret_cast<spm_counter_instance_t*>(this + 1); }

    void* aqlprofile_desc() { return events() + num_events; }
};

static_assert((sizeof(spm_desc_v0_t) % sizeof(spm_counter_instance_t)) == 0 &&
              "invalid descriptor and counter combination");

}  // namespace SPM
}  // namespace rocprofiler