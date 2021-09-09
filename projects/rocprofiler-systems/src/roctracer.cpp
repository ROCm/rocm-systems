
#include "roctracer.hpp"
#include "library.hpp"

#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hcc.h>
#include <roctracer_hip.h>
#include <roctracer_kfd.h>

#define AMD_INTERNAL_BUILD 1
#include <ext/hsa_rt_utils.hpp>
#include <roctracer_hsa.h>

#include <atomic>

// Macro to check ROC-tracer calls status
#define ROCTRACER_CALL(call)                                                             \
    do                                                                                   \
    {                                                                                    \
        int err = call;                                                                  \
        if(err != 0)                                                                     \
        {                                                                                \
            std::cerr << roctracer_error_string() << " in: " << #call << std::flush;     \
        }                                                                                \
    } while(0)

using roctracer_bundle_t = tim::component_tuple<comp::roctracer_data>;

namespace units = tim::units;

namespace
{
auto&
get_roctracer_kernels()
{
    static auto _v = std::unordered_set<uint64_t>{};
    return _v;
}

auto&
get_roctracer_hip_data()
{
    static auto _v = std::unordered_map<uint64_t, roctracer_bundle_t>{};
    return _v;
}

auto&
get_roctracer_key_data()
{
    static auto _v = std::unordered_map<uint64_t, const char*>{};
    return _v;
}

using data_type_mutex_t = std::decay_t<decltype(get_roctracer_hip_data())>;
using hsa_timer_t       = hsa_rt_utils::Timer;
using timestamp_t       = hsa_timer_t::timestamp_t;

auto&
get_hsa_timer()
{
    static auto _v = std::unique_ptr<hsa_timer_t>{};
    return _v;
}

// HSA API callback function
void
hsa_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    (void) arg;
    static auto           _scope = scope::config{} + scope::flat{};
    const hsa_api_data_t* data   = reinterpret_cast<const hsa_api_data_t*>(callback_data);
    HOSTTRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
                    roctracer_op_string(domain, cid, 0), cid, data->correlation_id,
                    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    static thread_local timestamp_t hsa_begin_timestamp = 0;
    static auto&                    timer               = get_hsa_timer();

    if(!timer)
        return;

    switch(cid)
    {
        case HSA_API_ID_hsa_init:
        case HSA_API_ID_hsa_shut_down:
        case HSA_API_ID_hsa_agent_get_exception_policies:
        case HSA_API_ID_hsa_agent_get_info:
        case HSA_API_ID_hsa_amd_agent_iterate_memory_pools:
        case HSA_API_ID_hsa_amd_agent_memory_pool_get_info:
        case HSA_API_ID_hsa_amd_coherency_get_type:
        case HSA_API_ID_hsa_amd_memory_pool_get_info:
        case HSA_API_ID_hsa_amd_pointer_info:
        case HSA_API_ID_hsa_amd_pointer_info_set_userdata:
        case HSA_API_ID_hsa_amd_profiling_async_copy_enable:
        case HSA_API_ID_hsa_amd_profiling_get_async_copy_time:
        case HSA_API_ID_hsa_amd_profiling_get_dispatch_time:
        case HSA_API_ID_hsa_amd_profiling_set_profiler_enabled:
        case HSA_API_ID_hsa_cache_get_info:
        case HSA_API_ID_hsa_code_object_get_info:
        case HSA_API_ID_hsa_code_object_get_symbol:
        case HSA_API_ID_hsa_code_object_get_symbol_from_name:
        case HSA_API_ID_hsa_code_object_reader_create_from_memory:
        case HSA_API_ID_hsa_code_symbol_get_info:
        case HSA_API_ID_hsa_executable_create_alt:
        case HSA_API_ID_hsa_executable_freeze:
        case HSA_API_ID_hsa_executable_get_info:
        case HSA_API_ID_hsa_executable_get_symbol:
        case HSA_API_ID_hsa_executable_get_symbol_by_name:
        case HSA_API_ID_hsa_executable_symbol_get_info:
        case HSA_API_ID_hsa_extension_get_name:
        case HSA_API_ID_hsa_ext_image_data_get_info:
        case HSA_API_ID_hsa_ext_image_data_get_info_with_layout:
        case HSA_API_ID_hsa_ext_image_get_capability:
        case HSA_API_ID_hsa_ext_image_get_capability_with_layout:
        case HSA_API_ID_hsa_isa_get_exception_policies:
        case HSA_API_ID_hsa_isa_get_info:
        case HSA_API_ID_hsa_isa_get_info_alt:
        case HSA_API_ID_hsa_isa_get_round_method:
        case HSA_API_ID_hsa_region_get_info:
        case HSA_API_ID_hsa_system_extension_supported:
        case HSA_API_ID_hsa_system_get_extension_table:
        case HSA_API_ID_hsa_system_get_info:
        case HSA_API_ID_hsa_system_get_major_extension_table:
        case HSA_API_ID_hsa_wavefront_get_info: break;
        default: {
            if(data->phase == ACTIVITY_API_PHASE_ENTER)
            {
                hsa_begin_timestamp = timer->timestamp_fn_ns();
            }
            else
            {
                auto              _name         = roctracer_op_string(domain, cid, 0);
                const timestamp_t end_timestamp = (cid == HSA_API_ID_hsa_shut_down)
                                                      ? hsa_begin_timestamp
                                                      : timer->timestamp_fn_ns();

                if(get_use_perfetto())
                {
                    TRACE_EVENT_BEGIN("device", perfetto::StaticString{ _name },
                                      hsa_begin_timestamp);
                    TRACE_EVENT_END("device", end_timestamp);
                }

                // timemory is disabled in this callback because collecting data in this
                // thread causes strange segmentation faults
            }
        }
    }
}

void
hsa_activity_callback(uint32_t op, activity_record_t* record, void* arg)
{
    static const char* copy_op_name     = "hsa_async_copy";
    static const char* dispatch_op_name = "hsa_dispatch";
    static const char* barrier_op_name  = "hsa_barrier";
    const char**       _name            = nullptr;

    switch(op)
    {
        case HSA_OP_ID_DISPATCH: _name = &dispatch_op_name; break;
        case HSA_OP_ID_COPY: _name = &copy_op_name; break;
        case HSA_OP_ID_BARRIER: _name = &barrier_op_name; break;
        default: break;
    }

    if(!_name)
        return;

    if(get_use_perfetto())
    {
        TRACE_EVENT_BEGIN("device", perfetto::StaticString{ *_name }, record->begin_ns);
        TRACE_EVENT_END("device", record->end_ns);
    }

    // timemory is disabled in this callback because collecting data in this thread
    // causes strange segmentation faults
    tim::consume_parameters(arg);
}

// HIP API callback function
void
hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(get_state() != State::Active || !trait::runtime_enabled<comp::roctracer>::get())
        return;

    static auto           _scope = scope::flat() + scope::timeline();
    const hip_api_data_t* data   = reinterpret_cast<const hip_api_data_t*>(callback_data);
    HOSTTRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
                    roctracer_op_string(domain, cid, 0), cid, data->correlation_id,
                    (data->phase == ACTIVITY_API_PHASE_ENTER) ? "on-enter" : "on-exit");

    if(data->phase == ACTIVITY_API_PHASE_ENTER)
    {
        switch(cid)
        {
            case HIP_API_ID___hipPushCallConfiguration:
            case HIP_API_ID___hipPopCallConfiguration: break;
            case HIP_API_ID_hipLaunchKernel: {
                const char* _name =
                    hipKernelNameRefByPtr(data->args.hipLaunchKernel.function_address,
                                          data->args.hipLaunchKernel.stream);
                tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
                get_roctracer_kernels().emplace(data->correlation_id);
                if(get_use_perfetto())
                {
                    get_roctracer_key_data().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_hip_data().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }(), _scope });
                }
                break;
            }
            case HIP_API_ID_hipModuleLaunchKernel: {
                const char* _name = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
                tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
                get_roctracer_kernels().emplace(data->correlation_id);
                if(get_use_perfetto())
                {
                    get_roctracer_key_data().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_hip_data().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }(), _scope });
                }
                break;
            }
            default: {
                tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
                const char*      _name = roctracer_op_string(domain, cid, 0);
                if(get_use_perfetto())
                {
                    get_roctracer_key_data().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_hip_data().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }(), _scope });
                }
                break;
            }
        }
    }
    else if(data->phase == ACTIVITY_API_PHASE_EXIT)
    {}
    tim::consume_parameters(domain, arg);
}

// Activity tracing callback
void
hip_activity_callback(const char* begin, const char* end, void*)
{
    if(!trait::runtime_enabled<comp::roctracer>::get())
        return;
    static auto _kernel_names        = std::unordered_map<const char*, std::string>{};
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);
    std::unordered_set<uint64_t> _indexes{};

    tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
    auto&            _data    = get_roctracer_hip_data();
    auto&            _keys    = get_roctracer_key_data();
    auto&            _kernels = get_roctracer_kernels();

    HOSTTRACE_DEBUG("Activity records:\n");
    while(record < end_record)
    {
        HOSTTRACE_DEBUG("\t%-30s\tcorrelation_id(%lu) time_ns(%lu:%lu) device_id(%d) "
                        "stream_id(%lu)\n",
                        roctracer_op_string(record->domain, record->correlation_id, 0),
                        record->correlation_id, record->begin_ns, record->end_ns,
                        record->device_id, record->queue_id);

        auto _is_kernel = _kernels.find(record->correlation_id) != _kernels.end();
        if(_is_kernel && record->device_id != 0 && record->queue_id != 0)
        {
            // these are overheads associated with the kernel launch, not kernel runtime
            ROCTRACER_CALL(roctracer_next_record(record, &record));
            continue;
        }

        auto kitr =
            (get_use_perfetto()) ? _keys.find(record->correlation_id) : _keys.end();
        if(kitr != _keys.end())
        {
            if(_kernel_names.find(kitr->second) == _kernel_names.end())
                _kernel_names.emplace(kitr->second, tim::demangle(kitr->second));
            TRACE_EVENT_BEGIN(
                "device",
                perfetto::StaticString{ _kernel_names.at(kitr->second).c_str() },
                record->begin_ns);
            TRACE_EVENT_END("device", record->end_ns);
            _indexes.emplace(kitr->first);
        }

        auto itr =
            (get_use_timemory()) ? _data.find(record->correlation_id) : _data.end();
        if(itr != _data.end())
        {
            itr->second.start()
                .store(std::plus<double>{},
                       static_cast<double>(record->end_ns - record->begin_ns))
                .stop();
            _indexes.emplace(itr->first);
        }
        // code
        ROCTRACER_CALL(roctracer_next_record(record, &record));
    }

    if(get_use_perfetto())
    {
        for(auto& itr : _indexes)
            _keys.erase(itr);
    }

    if(get_use_timemory())
    {
        for(auto& itr : _indexes)
            _data.erase(itr);
    }

    HOSTTRACE_DEBUG("[%s] recorded %lu phases\n", __FUNCTION__,
                    (unsigned long) _indexes.size());
}

bool&
roctracer_is_setup()
{
    static bool _v = false;
    return _v;
}
}  // namespace

#if !defined(HOSTTRACE_ROCTRACER_LIBKFDWRAPPER)
#    define HOSTTRACE_ROCTRACER_LIBKFDWRAPPER "/opt/rocm/roctracer/lib/libkfdwrapper64.so"
#endif

struct dynamic_library
{
    dynamic_library()                           = delete;
    dynamic_library(const dynamic_library&)     = delete;
    dynamic_library(dynamic_library&&) noexcept = default;
    dynamic_library& operator=(const dynamic_library&) = delete;
    dynamic_library& operator=(dynamic_library&&) noexcept = default;

    dynamic_library(const char* _env, const char* _fname,
                    int _flags = (RTLD_NOW | RTLD_GLOBAL), bool _store = false)
    : envname{ _env }
    , filename{ tim::get_env<std::string>(_env, _fname, _store) }
    , flags{ _flags }
    {
        handle = dlopen(filename.c_str(), flags);
        if(!handle)
            fprintf(stderr, "%s\n", dlerror());
        dlerror();  // Clear any existing error
    }

    ~dynamic_library()
    {
        if(handle)
            dlclose(handle);
    }

    std::string envname  = {};
    std::string filename = {};
    int         flags    = 0;
    void*       handle   = nullptr;
};

namespace tim
{
namespace component
{
void
roctracer::preinit()
{
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);
    roctracer_data::label()       = "roctracer";
    roctracer_data::description() = "ROCm tracer (activity API)";
}

void
roctracer::setup()
{
    if(!get_use_timemory() && !get_use_perfetto())
        return;

    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(roctracer_is_setup())
        return;
    roctracer_is_setup() = true;
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    tim::set_env("HSA_TOOLS_LIB", "libhosttrace.so", 0);

    auto _kfdwrapper = dynamic_library{ "HOSTTRACE_ROCTRACER_LIBKFDWRAPPER",
                                        HOSTTRACE_ROCTRACER_LIBKFDWRAPPER };

    // Allocating tracing pool
    roctracer_properties_t properties{};
    properties.buffer_size         = 0x1000;
    properties.buffer_callback_fun = hip_activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));
    ROCTRACER_CALL(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));

    // Enable API callbacks, all domains
    ROCTRACER_CALL(roctracer_enable_callback(hip_api_callback, nullptr));
    // Enable activity tracing, all domains
    ROCTRACER_CALL(roctracer_enable_activity());
}

void
roctracer::tear_down()
{
    auto_lock_t _lk{ type_mutex<roctracer>() };
    if(!roctracer_is_setup())
        return;
    roctracer_is_setup() = false;
    HOSTTRACE_DEBUG("[%s]\n", __FUNCTION__);

    // flush all the activity
    ROCTRACER_CALL(roctracer_flush_activity());

    // flush all buffers
    roctracer_flush_buf();

    ROCTRACER_CALL(roctracer_disable_domain_callback(ACTIVITY_DOMAIN_HSA_API));

    ROCTRACER_CALL(
        roctracer_disable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));

    // Disable tracing and closing the pool
    ROCTRACER_CALL(roctracer_disable_callback());
    ROCTRACER_CALL(roctracer_disable_activity());

    // closing the pool with HSA enabled causes segfaults
    // ROCTRACER_CALL(roctracer_close_pool());
}

void
roctracer::start()
{
    if(tracker_type::start() == 0)
        setup();
}

void
roctracer::stop()
{
    // flush all the activity
    ROCTRACER_CALL(roctracer_flush_activity());
    if(tracker_type::stop() == 0)
        tear_down();
}
}  // namespace component
}  // namespace tim

TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_INSTANTIATE_EXTERN_COMPONENT(roctracer_data, true, double)

// HSA-runtime tool on-load method
extern "C" TIMEMORY_VISIBILITY("default") bool OnLoad(
    HsaApiTable* table, uint64_t runtime_version, uint64_t failed_tool_count,
    const char* const* failed_tool_names)
{
    puts(__FUNCTION__);
    tim::consume_parameters(table, runtime_version, failed_tool_count, failed_tool_names);

    // ONLOAD_TRACE_BEG();
    // on_exit(exit_handler, nullptr);

    get_hsa_timer() = std::make_unique<hsa_timer_t>(table->core_->hsa_system_get_info_fn);

    // const char* output_prefix = getenv("ROCP_OUTPUT_DIR");
    const char* output_prefix = nullptr;

    // App begin timestamp begin_ts_file.txt
    // begin_ts_file_handle = open_output_file(output_prefix, "begin_ts_file.txt");
    // const timestamp_t app_start_time = timer->timestamp_fn_ns();
    // fprintf(begin_ts_file_handle, "%lu\n", app_start_time);

    bool trace_hsa_api = tim::get_env("HOSTTRACE_ROCTRACER_HSA_API", true);
    std::vector<std::string> hsa_api_vec =
        tim::delimit(tim::get_env<std::string>("HOSTTRACE_ROCTRACER_HSA_API_TYPES", ""));

    // Enable HSA API callbacks/activity
    if(trace_hsa_api)
    {
        // hsa_api_file_handle = open_output_file(output_prefix, "hsa_api_trace.txt");

        // initialize HSA tracing
        roctracer_set_properties(ACTIVITY_DOMAIN_HSA_API, (void*) table);

        fprintf(stdout, "    HSA-trace(");
        fflush(stdout);
        if(!hsa_api_vec.empty())
        {
            for(unsigned i = 0; i < hsa_api_vec.size(); ++i)
            {
                uint32_t    cid = HSA_API_ID_NUMBER;
                const char* api = hsa_api_vec[i].c_str();
                ROCTRACER_CALL(
                    roctracer_op_code(ACTIVITY_DOMAIN_HSA_API, api, &cid, nullptr));
                ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HSA_API, cid,
                                                            hsa_api_callback, nullptr));
                printf(" %s", api);
            }
        }
        else
        {
            ROCTRACER_CALL(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HSA_API,
                                                            hsa_api_callback, nullptr));
        }
        printf(")\n");
    }

    bool trace_hsa_activity = tim::get_env("HOSTTRACE_ROCTRACER_HSA_ACTIVITY", true);
    // Enable HSA GPU activity
    if(trace_hsa_activity)
    {
        // initialize HSA tracing
        roctracer::hsa_ops_properties_t ops_properties{
            table, reinterpret_cast<activity_async_callback_t>(hsa_activity_callback),
            nullptr, output_prefix
        };
        roctracer_set_properties(ACTIVITY_DOMAIN_HSA_OPS, &ops_properties);

        fprintf(stdout, "    HSA-activity-trace()\n");
        fflush(stdout);
        ROCTRACER_CALL(
            roctracer_enable_op_activity(ACTIVITY_DOMAIN_HSA_OPS, HSA_OP_ID_COPY));
    }

    return true;
}

// HSA-runtime on-unload method
extern "C" TIMEMORY_VISIBILITY("default") void OnUnload()
{
    puts(__FUNCTION__);
    // ONLOAD_TRACE("");
}
