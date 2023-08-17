
#include <amdhip/amdhip.hpp>
#include <hsa-runtime/hsa-runtime.hpp>
#include <roctx/roctx.hpp>

#include <pthread.h>
#include <stdexcept>
#include <string_view>

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
}  // namespace rocprofiler

extern "C" {
struct rocprofiler_configure_result_t
{
    int (*initialize)(void*) = nullptr;
    int (*finalize)(void*)   = nullptr;
    void* data               = nullptr;
};

rocprofiler_configure_result_t*
rocprofiler_configure(uint32_t, const char*, uint32_t, uint32_t)
    __attribute__((visibility("default")));

int
rocprofiler_set_api_table(const char*, uint64_t, uint64_t, void**, uint64_t)
    __attribute__((visibility("default")));

rocprofiler_configure_result_t*
rocprofiler_configure(uint32_t    version,
                      const char* runtime_version,
                      uint32_t    priority,
                      uint32_t    tool_id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;
    (void) tool_id;

    return nullptr;
}

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

    using hip_table_t   = hip::HipApiTable;
    using hsa_table_t   = hsa::HsaApiTable;
    using roctx_table_t = roctx::ROCTxApiTable;

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
            _table->hip_init_fn = &rocprofiler::hip_init;
        }
        else if(std::string_view{ name } == "hsa")
        {
            hsa_table_t* _table = static_cast<hsa_table_t*>(tables[0]);
            _table->hsa_init_fn = &rocprofiler::hsa_init;
        }
        else if(std::string_view{ name } == "roctx")
        {
            roctx_table_t* _table     = static_cast<roctx_table_t*>(tables[0]);
            _table->roctxRangePush_fn = &rocprofiler::roctx_range_push;
            _table->roctxRangePop_fn  = &rocprofiler::roctx_range_pop;
        }
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
