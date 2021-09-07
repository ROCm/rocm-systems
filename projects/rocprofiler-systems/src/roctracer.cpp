
#include "roctracer.hpp"
#include "library.hpp"

#include <roctracer.h>
#include <roctracer_hcc.h>
#include <roctracer_hip.h>

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
static auto&
get_roctracer_kernels()
{
    static auto _v = std::unordered_set<uint64_t>{};
    return _v;
}

static auto&
get_roctracer_data_map()
{
    static auto _v = std::unordered_map<uint64_t, roctracer_bundle_t>{};
    return _v;
}

static auto&
get_roctracer_key_map()
{
    static auto _v = std::unordered_map<uint64_t, const char*>{};
    return _v;
}

using data_type_mutex_t = std::decay_t<decltype(get_roctracer_data_map())>;

// HIP API callback function
void
hip_api_callback(uint32_t domain, uint32_t cid, const void* callback_data, void* arg)
{
    if(!trait::runtime_enabled<comp::roctracer>::get())
        return;
    const hip_api_data_t* data = reinterpret_cast<const hip_api_data_t*>(callback_data);
    HOSTTRACE_DEBUG("<%-30s id(%u)\tcorrelation_id(%lu) %s>\n",
                    roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0), cid,
                    data->correlation_id,
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
                    get_roctracer_key_map().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_data_map().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }() });
                }
                break;
            }
            case HIP_API_ID_hipModuleLaunchKernel: {
                const char* _name = hipKernelNameRef(data->args.hipModuleLaunchKernel.f);
                tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
                get_roctracer_kernels().emplace(data->correlation_id);
                if(get_use_perfetto())
                {
                    get_roctracer_key_map().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_data_map().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }(),
                                            tim::scope::get_default() });
                }
                break;
            }
            default: {
                tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
                const char* _name = roctracer_op_string(ACTIVITY_DOMAIN_HIP_API, cid, 0);
                if(get_use_perfetto())
                {
                    get_roctracer_key_map().emplace(data->correlation_id, _name);
                }
                if(get_use_timemory())
                {
                    get_roctracer_data_map().emplace(
                        data->correlation_id,
                        roctracer_bundle_t{ tim::static_string{ _name }(),
                                            tim::scope::get_default() });
                }
                break;
            }
        }
    }
    tim::consume_parameters(domain, arg);
}

// Activity tracing callback
void
activity_callback(const char* begin, const char* end, void*)
{
    if(!trait::runtime_enabled<comp::roctracer>::get())
        return;
    static auto _kernel_names        = std::unordered_map<const char*, std::string>{};
    const roctracer_record_t* record = reinterpret_cast<const roctracer_record_t*>(begin);
    const roctracer_record_t* end_record =
        reinterpret_cast<const roctracer_record_t*>(end);
    std::unordered_set<uint64_t> _indexes{};

    tim::auto_lock_t _lk{ tim::type_mutex<data_type_mutex_t>() };
    auto&            _data    = get_roctracer_data_map();
    auto&            _keys    = get_roctracer_key_map();
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
            _indexes.emplace(kitr->first);
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
#    define HOSTTRACE_ROCTRACER_LIBKFDWRAPPER "libkfdwrapper64.so"
#endif

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

    auto libpath   = tim::get_env<std::string>("HOSTTRACE_ROCTRACER_LIBKFDWRAPPER",
                                             HOSTTRACE_ROCTRACER_LIBKFDWRAPPER, false);
    auto libhandle = dlopen(libpath.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if(!libhandle)
        fprintf(stderr, "%s\n", dlerror());
    dlerror();  // Clear any existing error

    // Allocating tracing pool
    roctracer_properties_t properties{};
    properties.buffer_size         = 0x1000;
    properties.buffer_callback_fun = activity_callback;
    ROCTRACER_CALL(roctracer_open_pool(&properties));
    ROCTRACER_CALL(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));

    // Enable API callbacks, all domains
    ROCTRACER_CALL(roctracer_enable_callback(hip_api_callback, nullptr));
    /*ROCTRACER_CALL(roctracer_enable_op_callback(ACTIVITY_DOMAIN_HIP_API,
                                                HIP_API_ID_hipModuleLaunchKernel,
                                                hip_api_callback, nullptr));
    ROCTRACER_CALL(roctracer_enable_op_callback(
        ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipLaunchKernel, hip_api_callback, nullptr));
    ROCTRACER_CALL(roctracer_enable_op_callback(
        ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMalloc, hip_api_callback, nullptr));
    ROCTRACER_CALL(roctracer_enable_op_callback(
        ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpy, hip_api_callback, nullptr));
    ROCTRACER_CALL(roctracer_enable_op_callback(
        ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipFree, hip_api_callback, nullptr));*/
    // Enable activity tracing, all domains
    ROCTRACER_CALL(roctracer_enable_activity());

    if(libhandle)
        dlclose(libhandle);
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
    // Disable tracing and closing the pool
    ROCTRACER_CALL(roctracer_disable_callback());
    /*ROCTRACER_CALL(roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API,
                                                 HIP_API_ID_hipModuleLaunchKernel));
    ROCTRACER_CALL(roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API,
                                                 HIP_API_ID_hipLaunchKernel));
    ROCTRACER_CALL(
        roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMemcpy));
    ROCTRACER_CALL(
        roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipMalloc));
    ROCTRACER_CALL(
        roctracer_disable_op_callback(ACTIVITY_DOMAIN_HIP_API, HIP_API_ID_hipFree));
        */
    ROCTRACER_CALL(roctracer_disable_activity());
    ROCTRACER_CALL(roctracer_close_pool());
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
