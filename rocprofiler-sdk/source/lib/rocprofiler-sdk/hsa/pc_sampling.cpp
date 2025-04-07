// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/pc_sampling.hpp"

#if ROCPROFILER_SDK_HSA_PC_SAMPLING > 0

#    include "lib/common/logging.hpp"
#    include "lib/rocprofiler-sdk/hsa/defines.hpp"
#    include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#    include <rocprofiler-sdk/agent.h>
#    include <rocprofiler-sdk/fwd.h>
#    include <rocprofiler-sdk/hsa/api_id.h>
#    include <rocprofiler-sdk/hsa/table_id.h>

#    include <glog/logging.h>
#    include <hsa/amd_hsa_signal.h>
#    include <hsa/hsa.h>
#    include <hsa/hsa_api_trace.h>
#    include <hsa/hsa_ven_amd_pc_sampling.h>

#    include <cstddef>
#    include <cstdint>
#    include <cstdlib>
#    include <type_traits>
#    include <utility>

HSA_API_TABLE_LOOKUP_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                                ::PcSamplingExtTable,
                                pc_sampling_ext)

namespace rocprofiler
{
namespace hsa
{
namespace
{
enum pc_sampling_event_kind
{
    hsa_ven_amd_pcs_id_none = 0,
    hsa_ven_amd_pcs_id_iterate_configuration,
    hsa_ven_amd_pcs_id_create,
    hsa_ven_amd_pcs_id_create_from_id,
    hsa_ven_amd_pcs_id_destroy,
    hsa_ven_amd_pcs_id_start,
    hsa_ven_amd_pcs_id_stop,
    hsa_ven_amd_pcs_id_flush,
    hsa_ven_amd_pcs_id_last,
};
}  // namespace
}  // namespace hsa
}  // namespace rocprofiler

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_iterate_configuration,
                        hsa_ven_amd_pcs_iterate_configuration,
                        hsa_ven_amd_pcs_iterate_configuration_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_create,
                        hsa_ven_amd_pcs_create,
                        hsa_ven_amd_pcs_create_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_create_from_id,
                        hsa_ven_amd_pcs_create_from_id,
                        hsa_ven_amd_pcs_create_from_id_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_destroy,
                        hsa_ven_amd_pcs_destroy,
                        hsa_ven_amd_pcs_destroy_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_start,
                        hsa_ven_amd_pcs_start,
                        hsa_ven_amd_pcs_start_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_stop,
                        hsa_ven_amd_pcs_stop,
                        hsa_ven_amd_pcs_stop_fn);

HSA_API_META_DEFINITION(ROCPROFILER_HSA_TABLE_ID_PcSamplingExt,
                        hsa_ven_amd_pcs_id_flush,
                        hsa_ven_amd_pcs_flush,
                        hsa_ven_amd_pcs_flush_fn);

namespace rocprofiler
{
namespace hsa
{
namespace pc_sampling
{
namespace
{
template <size_t TableIdx, typename LookupT = internal_table, size_t OpIdx>
void
copy_table(hsa_pc_sampling_ext_table_t* _orig, uint64_t _tbl_instance)
{
    using table_type = typename hsa_table_lookup<TableIdx>::type;

    static_assert(std::is_same<hsa_pc_sampling_ext_table_t, table_type>::value);

    if constexpr(OpIdx > hsa_ven_amd_pcs_id_none)
    {
        auto _info = hsa_api_meta<TableIdx, OpIdx>{};

        auto& _orig_table = _info.get_table(_orig);
        auto& _orig_func  = _info.get_table_func(_orig_table);

        if(_info.offset() >= _orig->version.minor_id) return;

        auto& _copy_table = _info.get_table(hsa_table_lookup<TableIdx>{}(LookupT{}));
        auto& _copy_func  = _info.get_table_func(_copy_table);

        ROCP_FATAL_IF(_copy_func && _tbl_instance == 0)
            << _info.name << " has non-null function pointer " << _copy_func
            << " despite this being the first instance of the library being copies";

        if(!_copy_func)
        {
            ROCP_TRACE << "copying table entry for " << _info.name;
            _copy_func = _orig_func;
        }
        else
        {
            ROCP_TRACE << "skipping copying table entry for " << _info.name
                       << " from table instance " << _tbl_instance;
        }
    }
}

template <size_t TableIdx, typename LookupT = internal_table, size_t... OpIdx>
void
copy_table(hsa_pc_sampling_ext_table_t* _orig,
           uint64_t                     _tbl_instance,
           std::index_sequence<OpIdx...>)
{
    static_assert(
        std::is_same<hsa_pc_sampling_ext_table_t, typename hsa_table_lookup<TableIdx>::type>::value,
        "unexpected type");

    (copy_table<TableIdx, LookupT, OpIdx>(_orig, _tbl_instance), ...);
}

}  // namespace

void
copy_table(hsa_pc_sampling_ext_table_t* _orig, uint64_t _tbl_instance)
{
    if(_orig)
        copy_table<ROCPROFILER_HSA_TABLE_ID_PcSamplingExt, internal_table>(
            _orig, _tbl_instance, std::make_index_sequence<hsa_ven_amd_pcs_id_last>{});
}

void
update_table(hsa_pc_sampling_ext_table_t* /*_orig*/, uint64_t /*_tbl_instance*/)
{}
}  // namespace pc_sampling
}  // namespace hsa
}  // namespace rocprofiler

#else

namespace rocprofiler
{
namespace hsa
{
namespace pc_sampling
{
const char* name_by_id(uint32_t) { return nullptr; }

std::vector<uint32_t>
get_ids()
{
    return std::vector<uint32_t>{};
}
}  // namespace pc_sampling
}  // namespace hsa
}  // namespace rocprofiler

#endif
