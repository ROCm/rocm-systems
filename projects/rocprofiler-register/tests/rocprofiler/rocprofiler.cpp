
#include <rocprofiler-register/rocprofiler-register.h>
#include <amdhip/amdhip.hpp>
#include <hsa-runtime/hsa-runtime.hpp>
#include <rccl/rccl.hpp>
#include <rocdecode/rocdecode.hpp>
#include <rocjpeg/rocjpeg.hpp>
#include <roctx/roctx.hpp>

#include <dlfcn.h>
#include <pthread.h>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <vector>

#ifndef ROCP_REG_FILE_NAME
#    define ROCP_REG_FILE_NAME                                                           \
        ::std::string{ __FILE__ }                                                        \
            .substr(::std::string_view{ __FILE__ }.find_last_of('/') + 1)                \
            .c_str()
#endif

namespace rocprofiler
{
void
hip_init()
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
}

void
hsa_init()
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
}

ncclResult_t
ncclGetVersion(int*)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
    return {};
}

rocDecStatus
rocDecCreateDecoder(rocDecDecoderHandle*, RocDecoderCreateInfo*)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
    return {};
}

RocJpegStatus
rocJpegStreamCreate(RocJpegStreamHandle* jpeg_stream_handle)
{
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
    return {};
}

void
roctx_range_push(const char* name)
{
    printf("[%s][push] %s\n", ROCP_REG_FILE_NAME, name);
}

void
roctx_range_pop(const char* name)
{
    printf("[%s][pop] %s\n", ROCP_REG_FILE_NAME, name);
}

using reginfo_vec_t = std::vector<rocprofiler_register_registration_info_t>;

bool
check_registration_info(const char*          name,
                        uint64_t             lib_version,
                        uint64_t             num_tables,
                        const reginfo_vec_t& infovec)
{
    for(const auto& itr : infovec)
    {
        if(std::string_view{ name } == std::string_view{ itr.common_name })
        {
            return std::tie(lib_version, num_tables) ==
                   std::tie(itr.lib_version, itr.api_table_length);
        }
    }

    return false;
}
}  // namespace rocprofiler

extern "C" {
int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
    __attribute__((visibility("default")));

int
rocprofiler_set_api_table(const char* name,
                          uint64_t    lib_version,
                          uint64_t    lib_instance,
                          void**      tables,
                          uint64_t    num_tables)
{
    printf("[%s] %s :: %lu :: %lu :: %lu\n",
           ROCP_REG_FILE_NAME,
           name,
           lib_version,
           lib_instance,
           num_tables);

    auto* _tool_libs = std::getenv("ROCP_TOOL_LIBRARIES");
    if(_tool_libs)
    {
        auto* _handle = dlopen(_tool_libs, RTLD_GLOBAL | RTLD_LAZY);
        if(!_handle)
            throw std::runtime_error{ std::string{ "error opening tool library " } +
                                      _tool_libs };
        auto* _sym = dlsym(_handle, "rocprofiler_configure");
        if(!_sym)
            throw std::runtime_error{ std::string{ "tool library " } +
                                      std::string{ _tool_libs } +
                                      " did not contain rocprofiler_configure symbol" };
    }

    auto registration_info = ::rocprofiler::reginfo_vec_t{};
    {
        auto* _handle =
            dlopen("librocprofiler-register.so", RTLD_LAZY | RTLD_LOCAL | RTLD_NOLOAD);
        if(!_handle)
            throw std::runtime_error{
                "error opening librocprofiler-register.so library "
            };
        auto* _sym = dlsym(_handle, "rocprofiler_register_iterate_registration_info");
        if(!_sym)
            throw std::runtime_error{
                "librocprofiler-register.so did not contain "
                "rocprofiler_register_iterate_registration_info symbol"
            };

        auto _func = [](rocprofiler_register_registration_info_t* _info,
                        void*                                     _vdata) -> int {
            auto* _vec = static_cast<::rocprofiler::reginfo_vec_t*>(_vdata);
            _vec->emplace_back(*_info);
            return 0;
        };

        auto iterate_registration_info =
            reinterpret_cast<decltype(&rocprofiler_register_iterate_registration_info)>(
                _sym);
        iterate_registration_info(_func, &registration_info);
    }

    using hip_table_t       = hip::HipApiTable;
    using hsa_table_t       = hsa::HsaApiTable;
    using roctx_table_t     = roctx::ROCTxApiTable;
    using rccl_table_t      = rccl::rcclApiFuncTable;
    using rocdecode_table_t = rocdecode::rocdecodeApiFuncTable;
    using rocjpeg_table_t   = rocjpeg::rocjpegApiFuncTable;

    auto* _wrap_v = std::getenv("ROCP_REG_TEST_WRAP");
    bool  _wrap   = (_wrap_v != nullptr && std::stoi(_wrap_v) != 0);

    if(_wrap)
    {
        if(num_tables != 1)
            throw std::runtime_error{ std::string{ "unexpected number of tables: " } +
                                      std::to_string(num_tables) };

        if(tables == nullptr) throw std::runtime_error{ "nullptr to tables" };
        if(tables[0] == nullptr) throw std::runtime_error{ "nullptr to tables[0]" };

        if(std::string_view{ name } == "hip")
        {
            hip_table_t* _table = static_cast<hip_table_t*>(tables[0]);
            _table->hip_init_fn = &::rocprofiler::hip_init;
        }
        else if(std::string_view{ name } == "hsa")
        {
            hsa_table_t* _table = static_cast<hsa_table_t*>(tables[0]);
            _table->hsa_init_fn = &::rocprofiler::hsa_init;
        }
        else if(std::string_view{ name } == "roctx")
        {
            roctx_table_t* _table     = static_cast<roctx_table_t*>(tables[0]);
            _table->roctxRangePush_fn = &::rocprofiler::roctx_range_push;
            _table->roctxRangePop_fn  = &::rocprofiler::roctx_range_pop;
        }
        else if(std::string_view{ name } == "rccl")
        {
            rccl_table_t* _table      = static_cast<rccl_table_t*>(tables[0]);
            _table->ncclGetVersion_fn = &::rocprofiler::ncclGetVersion;
        }
        else if(std::string_view{ name } == "rocdecode")
        {
            rocdecode_table_t* _table      = static_cast<rocdecode_table_t*>(tables[0]);
            _table->rocDecCreateDecoder_fn = &rocprofiler::rocDecCreateDecoder;
        }
        else if(std::string_view{ name } == "rocjpeg")
        {
            rocjpeg_table_t* _table        = static_cast<rocjpeg_table_t*>(tables[0]);
            _table->rocJpegStreamCreate_fn = &rocprofiler::rocJpegStreamCreate;
        }
    }

    if(!::rocprofiler::check_registration_info(
           name, lib_version, num_tables, registration_info))
    {
        auto ss = std::stringstream{};
        ss << "no matching registration info for " << name << " "
           << " version " << lib_version << " (# tables = " << num_tables << ")";
        throw std::runtime_error{ ss.str() };
    }

    return 0;
}
}

void
     rocp_ctor(void) __attribute__((constructor));
bool rocprofiler_test_lib_link = false;

void
rocp_ctor()
{
    rocprofiler_test_lib_link = true;
    printf("[%s] %s\n", ROCP_REG_FILE_NAME, __FUNCTION__);
}
