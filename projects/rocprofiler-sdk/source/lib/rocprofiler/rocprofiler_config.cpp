
#include <rocprofiler/rocprofiler.h>
#include <rocprofiler/config.h>

#include "config_helpers.hpp"
#include "config_internal.hpp"

#include <atomic>
#include <cstddef>
#include <roctracer/roctx.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <mutex>
#include <iostream>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>
#include <hsa/hsa_api_trace.h>

typedef enum
{
    ACTIVITY_API_PHASE_ENTER = 0,
    ACTIVITY_API_PHASE_EXIT  = 1
} activity_api_phase_t;

typedef struct roctx_api_data_s
{
    union
    {
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        };
        struct
        {
            const char* message;
        } roctxMarkA;
        struct
        {
            const char* message;
        } roctxRangePushA;
        struct
        {
            const char* message;
        } roctxRangePop;
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        } roctxRangeStartA;
        struct
        {
            const char*      message;
            roctx_range_id_t id;
        } roctxRangeStop;
    } args;
} roctx_api_data_t;

// helper macros ensuring C and C++ structs adhere to specific naming convention
#define ROCP_PUBLIC_CONFIG(TYPE)  ::rocprofiler_##TYPE
#define ROCP_PRIVATE_CONFIG(TYPE) ::rocprofiler::internal::TYPE

// Below asserts at compile time that the external C object has the same size as internal
// C++ object, e.g.,
//      sizeof(rocprofiler_domain_config) == sizeof(rocprofiler::internal::domain_config)
#define ROCP_ASSERT_CONFIG_ABI(TYPE)                                                               \
    static_assert(sizeof(ROCP_PUBLIC_CONFIG(TYPE)) == sizeof(ROCP_PRIVATE_CONFIG(TYPE)),           \
                  "Error! rocprofiler_" #TYPE " ABI error");

// Below asserts at compile time that the external C struct members has the same offset as
// internal C++ struct members
#define ROCP_ASSERT_CONFIG_OFFSET_ABI(TYPE, PUB_FIELD, PRIV_FIELD)                                 \
    static_assert(offsetof(ROCP_PUBLIC_CONFIG(TYPE), PUB_FIELD) ==                                 \
                      offsetof(ROCP_PRIVATE_CONFIG(TYPE), PRIV_FIELD),                             \
                  "Error! rocprofiler_" #TYPE "." #PUB_FIELD " ABI offset error");                 \
    static_assert(sizeof(ROCP_PUBLIC_CONFIG(TYPE)::PUB_FIELD) ==                                   \
                      sizeof(ROCP_PRIVATE_CONFIG(TYPE)::PRIV_FIELD),                               \
                  "Error! rocprofiler_" #TYPE "." #PUB_FIELD " ABI size error");

// this defines a template specialization for ensuring that the reinterpret_cast is only
// applied between public C structs and private C++ structs which are compatible.
#define ROCP_DEFINE_API_CAST_IMPL(INPUT_TYPE, OUTPUT_TYPE)                                         \
    namespace traits                                                                               \
    {                                                                                              \
    template <>                                                                                    \
    struct api_cast<INPUT_TYPE>                                                                    \
    {                                                                                              \
        using input_type  = INPUT_TYPE;                                                            \
        using output_type = OUTPUT_TYPE;                                                           \
                                                                                                   \
        output_type* operator()(input_type* _v) const                                              \
        {                                                                                          \
            return reinterpret_cast<output_type*>(_v);                                             \
        }                                                                                          \
                                                                                                   \
        const output_type* operator()(const input_type* _v) const                                  \
        {                                                                                          \
            return reinterpret_cast<const output_type*>(_v);                                       \
        }                                                                                          \
    };                                                                                             \
    }

// define C -> C++ and C++ -> C casting rules
#define ROCP_DEFINE_API_CAST_D(TYPE)                                                               \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PUBLIC_CONFIG(TYPE), ROCP_PRIVATE_CONFIG(TYPE))                 \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PRIVATE_CONFIG(TYPE), ROCP_PUBLIC_CONFIG(TYPE))

// use only when C++ struct is just an alias for C struct
#define ROCP_DEFINE_API_CAST_S(TYPE)                                                               \
    ROCP_DEFINE_API_CAST_IMPL(ROCP_PUBLIC_CONFIG(TYPE), ROCP_PRIVATE_CONFIG(TYPE))

namespace
{
namespace traits
{
// left undefined to ensure template specialization
template <typename PublicT>
struct api_cast;

// ensure api_cast<decltype(a)> where decltype(a) is const Tp equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<const PublicT> : api_cast<PublicT>
{};

// ensure api_cast<decltype(a)> where decltype(a) is Tp& equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<PublicT&> : api_cast<PublicT>
{};

// ensure api_cast<decltype(a)> where decltype(a) is Tp* equates to api_cast<Tp>
template <typename PublicT>
struct api_cast<PublicT*> : api_cast<PublicT>
{};
}  // namespace traits

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//                          SEE BELOW! VERY IMPORTANT!
//
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//  EVERY NEW CONFIG AND ALL OF ITS MEMBER FIELDS NEED TO HAVE THESE COMPILE TIME CHECKS!
//
//  these checks verify the two structs have the same size and that each
//  member field has the same size and offset into the struct
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

ROCP_ASSERT_CONFIG_ABI(config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, size, size)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, compat_version, compat_version)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, api_version, api_version)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, reserved0, context_idx)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, user_data, user_data)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, buffer, buffer)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, domain, domain)
ROCP_ASSERT_CONFIG_OFFSET_ABI(config, filter, filter)

ROCP_ASSERT_CONFIG_ABI(domain_config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, callback, user_sync_callback)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, reserved0, domains)
ROCP_ASSERT_CONFIG_OFFSET_ABI(domain_config, reserved1, opcodes)

ROCP_ASSERT_CONFIG_ABI(buffer_config)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, callback, callback)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, buffer_size, buffer_size)
// ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, reserved0, buffer)
ROCP_ASSERT_CONFIG_OFFSET_ABI(buffer_config, reserved1, buffer_idx)

ROCP_DEFINE_API_CAST_D(config)
ROCP_DEFINE_API_CAST_D(domain_config)
ROCP_DEFINE_API_CAST_D(buffer_config)
ROCP_DEFINE_API_CAST_S(filter_config)

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
//
//
//                          SEE ABOVE! VERY IMPORTANT!
//
//
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

/// use this to ensure that reinterpret_cast from public C struct to internal C++ struct
/// is valid, e.g. guard against accidentally casting to wrong type
template <typename Tp>
auto
rocp_cast(Tp* _val)
{
    return traits::api_cast<Tp>{}(_val);
}

/// helper function for making copies of the fields in rocprofiler_config. If the config
/// field needs to be copied in some special way, use a template specialization of the
/// "construct" function in the allocator to handle this, e.g.:
///
///     using special_config = ::rocprofiler::internal::special_config;
///
///     template <>
///     void
///     allocator<special_config, 8>::construct(special_config* const _p,
///                                             const special_config& _v) const
///     {
///         auto _tmp = special_config{};
///         // ... special copy of fields from _v into _tmp
///
///         // placement new of _tmp into _p
///         _p = new(_p) special_config{ _tmp };
///     }
///
///     template <>
///     void
///     allocator<special_config, 8>::construct(special_config* const _p,
///                                             special_config&& _v) const
///     {
///         auto _tmp = std::move(_v);
///         // ... perform special needs
///
///         // placement new of _tmp into _p
///         _p = new(_p) special_config{ std::move(_tmp) };
///     }
///
template <typename Tp, typename Up>
Tp*&
copy_config_field(Tp*& _dst, Up* _src_v)
{
    static auto _allocator = allocator<Tp>{};

    if constexpr(!std::is_same<Tp, Up>::value)
    {
        using PrivateT = typename traits::api_cast<Up>::output_type;
        static_assert(std::is_same<PrivateT, Tp>::value, "Error incorrect field copy");

        auto _src = rocp_cast(_src_v);
        if(_src)
        {
            _dst = _allocator.allocate(1);
            _allocator.construct(_dst, *_src);
        }
        return _dst;
    }
    else
    {
        if(_src_v)
        {
            _dst = _allocator.allocate(1);
            _allocator.construct(_dst, *_src_v);
        }
        return _dst;
    }
}

auto&
get_configs_buffer()
{
    static char
        _v[::rocprofiler::internal::max_configs_count * sizeof(rocprofiler::internal::config)];
    return _v;
}

auto&
get_configs_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

inline uint32_t
get_tid()
{
    return syscall(__NR_gettid);
}

constexpr auto rocp_max_configs = ::rocprofiler::internal::max_configs_count;
}  // namespace

namespace rocprofiler
{
namespace internal
{
std::array<rocprofiler::internal::config*, max_configs_count>&
get_registered_configs()
{
    static auto _v = std::array<rocprofiler::internal::config*, max_configs_count>{};
    return _v;
}

std::array<std::atomic<rocprofiler::internal::config*>, max_configs_count>&
get_active_configs()
{
    static auto _v = std::array<std::atomic<rocprofiler::internal::config*>, max_configs_count>{};
    return _v;
}
}  // namespace internal
}  // namespace rocprofiler

extern "C" {

rocprofiler_status_t
rocprofiler_allocate_config(rocprofiler_config* _inp_cfg)
{
    // perform checks that rocprofiler can be activated

    ::memset(_inp_cfg, 0, sizeof(rocprofiler_config));

    auto* _cfg = rocp_cast(_inp_cfg);

    _cfg->size           = sizeof(::rocprofiler_config);
    _cfg->compat_version = 0;
    _cfg->api_version    = ROCPROFILER_API_VERSION_ID;
    _cfg->context_idx    = std::numeric_limits<decltype(_cfg->context_idx)>::max();

    // initial value checks
    assert(_cfg->size == sizeof(rocprofiler::internal::config));
    assert(_cfg->compat_version == 0);
    assert(_cfg->api_version == ROCPROFILER_API_VERSION_ID);
    assert(_cfg->buffer == nullptr);
    assert(_cfg->domain == nullptr);
    assert(_cfg->filter == nullptr);
    assert(_cfg->context_idx ==
           std::numeric_limits<decltype(rocprofiler::internal::config::context_idx)>::max());

    // ... allocate any internal space needed to handle another config ...
    {
        auto _lk = std::unique_lock<std::mutex>{get_configs_mutex()};
        // ...
    }

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_validate_config(const rocprofiler_config* cfg_v)
{
    const auto* cfg = rocp_cast(cfg_v);

    if(cfg->buffer == nullptr) return ROCPROFILER_STATUS_ERROR_BUFFER_NOT_FOUND;

    if(cfg->filter == nullptr) return ROCPROFILER_STATUS_ERROR_FILTER_NOT_FOUND;

    if(cfg->domain == nullptr || cfg->domain->domains == 0)
        return ROCPROFILER_STATUS_ERROR_INCORRECT_DOMAIN;

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_start_config(rocprofiler_config* cfg_v, rocprofiler_context_id_t* context_id)
{
    if(rocprofiler_validate_config(cfg_v) != ROCPROFILER_STATUS_SUCCESS)
    {
        std::cerr << "rocprofiler_start_config() provided an invalid configuration. tool "
                     "should use rocprofiler_validate_config() to check whether the "
                     "config is valid and adapt accordingly to issues before trying to "
                     "start the configuration."
                  << std::endl;
        abort();
    }

    auto* cfg = rocp_cast(cfg_v);

    uint64_t idx = rocp_max_configs;
    {
        auto _lk = std::unique_lock<std::mutex>{get_configs_mutex()};
        for(size_t i = 0; i < rocp_max_configs; ++i)
        {
            if(rocprofiler::internal::get_registered_configs().at(i) == nullptr)
            {
                idx = i;
                break;
            }
        }
    }

    // too many configs already registered
    if(idx == rocp_max_configs) return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_ACTIVE;

    cfg->context_idx   = idx;
    context_id->handle = idx;

    // using the context id, compute the location in the buffer of configs
    auto* _offset = get_configs_buffer() + (idx * sizeof(rocprofiler::internal::config));

    // placement new into the buffer
    auto* _copy_cfg = new(_offset) rocprofiler::internal::config{*cfg};

    // make copies of non-null config fields
    copy_config_field(_copy_cfg->buffer, cfg->buffer);
    copy_config_field(_copy_cfg->domain, cfg->domain);
    copy_config_field(_copy_cfg->filter, cfg->filter);

    // store until "deallocation"
    rocprofiler::internal::get_registered_configs().at(idx) = _copy_cfg;

    using config_t = rocprofiler::internal::config;
    // atomic swap the pointer into the "active" array used internally
    config_t* _expected = nullptr;
    bool      success = rocprofiler::internal::get_active_configs().at(idx).compare_exchange_strong(
        _expected, rocprofiler::internal::get_registered_configs().at(idx));

    if(!success) return ROCPROFILER_STATUS_ERROR_HAS_ACTIVE_CONTEXT;  // need relevant enum

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_stop_config(rocprofiler_context_id_t idx)
{
    // atomically assign the config pointer to NULL so that it is skipped in future
    // callbacks
    auto* _expected =
        rocprofiler::internal::get_active_configs().at(idx.handle).load(std::memory_order_relaxed);
    bool success = rocprofiler::internal::get_active_configs()
                       .at(idx.handle)
                       .compare_exchange_strong(_expected, nullptr);

    if(!success)
        return ROCPROFILER_STATUS_ERROR_CONTEXT_NOT_FOUND;  // compare exchange strong
                                                            // failed

    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_domain(struct rocprofiler_domain_config*    _inp_cfg,
                              rocprofiler_tracer_activity_domain_t _domain)
{
    auto* _cfg = rocp_cast(_inp_cfg);
    if(_domain <= ROCPROFILER_TRACER_ACTIVITY_DOMAIN_NONE ||
       _domain >= ROCPROFILER_TRACER_ACTIVITY_DOMAIN_LAST)
        return ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID;

    _cfg->domains |= (1 << _domain);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_domains(struct rocprofiler_domain_config*     _inp_cfg,
                               rocprofiler_tracer_activity_domain_t* _domains,
                               size_t                                _ndomains)
{
    for(size_t i = 0; i < _ndomains; ++i)
    {
        auto _status = rocprofiler_domain_add_domain(_inp_cfg, _domains[i]);
        if(_status != ROCPROFILER_STATUS_SUCCESS) return _status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_op(struct rocprofiler_domain_config*    _inp_cfg,
                          rocprofiler_tracer_activity_domain_t _domain,
                          uint32_t                             _op)
{
    auto* _cfg = rocp_cast(_inp_cfg);
    if(_domain <= ROCPROFILER_TRACER_ACTIVITY_DOMAIN_NONE ||
       _domain >= ROCPROFILER_TRACER_ACTIVITY_DOMAIN_LAST)
        return ROCPROFILER_STATUS_ERROR_INVALID_DOMAIN_ID;

    if(_op >= get_domain_max_op(_domain)) return ROCPROFILER_STATUS_ERROR_INVALID_OPERATION_ID;

    auto _offset = (_domain * rocprofiler::internal::domain_ops_offset);
    _cfg->opcodes.set(_offset + _op, true);
    return ROCPROFILER_STATUS_SUCCESS;
}

rocprofiler_status_t
rocprofiler_domain_add_ops(struct rocprofiler_domain_config*    _inp_cfg,
                           rocprofiler_tracer_activity_domain_t _domain,
                           uint32_t*                            _ops,
                           size_t                               _nops)
{
    for(size_t i = 0; i < _nops; ++i)
    {
        auto _status = rocprofiler_domain_add_op(_inp_cfg, _domain, _ops[i]);
        if(_status != ROCPROFILER_STATUS_SUCCESS) return _status;
    }
    return ROCPROFILER_STATUS_SUCCESS;
}

// ------------------------------------------------------------------------------------ //
//
//                  demo of internal implementation
//
// ------------------------------------------------------------------------------------ //

void
api_callback(rocprofiler_tracer_activity_domain_t domain,
             uint32_t                             cid,
             const void* /*callback_data*/,
             void*)
{
    for(const auto& aitr : rocprofiler::internal::get_active_configs())
    {
        auto* itr = aitr.load();
        if(!itr) continue;

        // below should be valid so this might need to raise error
        if(!itr->domain) continue;

        // if the given domain + op is not enabled, skip this config
        if(!(*itr->domain)(domain, cid)) continue;

        if(itr->filter)
        {
            if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_ROCTX)
            {}
            else if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_API)
            {
                if(itr->filter->hsa_function_id && itr->filter->hsa_function_id(cid) == 0) continue;
            }
            else if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_API)
            {
                if(itr->filter->hip_function_id && itr->filter->hip_function_id(cid) == 0) continue;
            }
        }

        auto& _domain      = (*itr->domain);
        auto& _correlation = (*itr->correlation_id);

        auto _correlation_id = rocprofiler::internal::correlation_config::get_unique_record_id();
        if(_correlation.external_id_callback)
            _correlation.external_id =
                _correlation.external_id_callback(domain, cid, _correlation_id);

        auto timestamp_ns = []() -> uint64_t {
            return std::chrono::steady_clock::now().time_since_epoch().count();
        };

        (void) _domain;
        (void) timestamp_ns;
        /*
        auto _header        = rocprofiler_record_header_t{ROCPROFILER_TRACER_RECORD,
                                                   rocprofiler_record_id_t{_correlation_id}};
        auto _op_id         = rocprofiler_tracer_operation_id_t{cid};
        auto _agent_id      = rocprofiler_agent_id_t{0};
        auto _queue_id      = rocprofiler_queue_id_t{0};
        auto _thread_id     = rocprofiler_thread_id_t{get_tid()};
        auto _context       = rocprofiler_context_id_t{itr->context_idx};
        auto _timestamp_raw = rocprofiler_timestamp_t{timestamp_ns()};
        auto _timestamp     = rocprofiler_record_header_timestamp_t{_timestamp_raw, _timestamp_raw};

        if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_ROCTX)
        {
            auto                    _api_data = rocprofiler_tracer_api_data_t{};
            const roctx_api_data_t* _data =
                reinterpret_cast<const roctx_api_data_t*>(callback_data);

            if(itr->filter && itr->filter->name && itr->filter->name(_data->args.message) == 0)
                continue;

            _api_data.roctx = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{ROCPROFILER_PHASE_ENTER};
            _timestamp  = {_timestamp_raw, _timestamp_raw};

            auto _external_cid = rocprofiler_tracer_external_id_t{_data ? _data->args.id : 0};
            auto _activity_cid = rocprofiler_tracer_activity_correlation_id_t{0};
            const char* _name  = _data->args.message;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_ROCTX,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _context);
        }
        else if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_API)
        {
            auto                  _api_data = rocprofiler_tracer_api_data_t{};
            const hsa_api_data_t* _data = reinterpret_cast<const hsa_api_data_t*>(callback_data);
            _api_data.hsa               = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{(_data->phase == ACTIVITY_API_PHASE_ENTER)
                                                              ? ROCPROFILER_PHASE_ENTER
                                                              : ROCPROFILER_PHASE_EXIT};

            if(_phase == ROCPROFILER_PHASE_ENTER)
                _timestamp.begin = _timestamp_raw;
            else
                _timestamp.end = _timestamp_raw;

            auto _external_cid = rocprofiler_tracer_external_id_t{0};
            auto _activity_cid =
                rocprofiler_tracer_activity_correlation_id_t{_data->correlation_id};
            const char* _name = nullptr;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_HSA_API,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _context);
        }
        else if(domain == ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_API)
        {
            auto                  _api_data = rocprofiler_tracer_api_data_t{};
            const hip_api_data_t* _data = reinterpret_cast<const hip_api_data_t*>(callback_data);
            _api_data.hip               = _data;

            auto _phase = rocprofiler_api_tracing_phase_t{(_data->phase == ACTIVITY_API_PHASE_ENTER)
                                                              ? ROCPROFILER_PHASE_ENTER
                                                              : ROCPROFILER_PHASE_EXIT};

            if(_phase == ROCPROFILER_PHASE_ENTER)
                _timestamp.begin = _timestamp_raw;
            else
                _timestamp.end = _timestamp_raw;

            auto _external_cid = rocprofiler_tracer_external_id_t{0};
            auto _activity_cid =
                rocprofiler_tracer_activity_correlation_id_t{_data->correlation_id};
            const char* _name = nullptr;

            _domain.user_sync_callback(rocprofiler_record_tracer_t{_header,
                                                                   _external_cid,
                                                                   ACTIVITY_DOMAIN_HIP_API,
                                                                   _op_id,
                                                                   _api_data,
                                                                   _activity_cid,
                                                                   _timestamp,
                                                                   _agent_id,
                                                                   _queue_id,
                                                                   _thread_id,
                                                                   _phase,
                                                                   _name},
                                       _context);
        }
        */
    }
}

void
InitRoctracer()
{
    for(const auto& itr : rocprofiler::internal::get_registered_configs())
    {
        if(!itr) continue;

        // below should be valid so this might need to raise error
        if(!itr->domain) continue;

        for(auto ditr : {ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_API,
                         ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_API,
                         ROCPROFILER_TRACER_ACTIVITY_DOMAIN_ROCTX})
        {
            if((*itr->domain)(ditr))
            {
                if(itr->domain->user_sync_callback)
                {
                    // ...
                }
                else
                {
                    // ...
                }
            }
        }

        for(auto ditr : {ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HSA_OPS,
                         ROCPROFILER_TRACER_ACTIVITY_DOMAIN_HIP_OPS})
        {
            if((*itr->domain)(ditr))
            {
                if(itr->domain->opcodes.none())
                {
                    // ...
                }
                else
                {
                    for(size_t i = 0; i < itr->domain->opcodes.size(); ++i)
                    {
                        if((*itr->domain)(ditr, i))
                        {
                            // ...
                        }
                    }
                }
            }
        }
    }
}
}
