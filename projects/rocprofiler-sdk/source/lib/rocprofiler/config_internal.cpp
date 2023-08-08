
#include "config_internal.hpp"

namespace rocprofiler
{
namespace internal
{
uint64_t
correlation_config::get_unique_record_id()
{
    static auto _v = std::atomic<uint64_t>{};
    return _v++;
}

bool
domain_config::operator()(rocprofiler_tracer_activity_domain_t _domain) const
{
    return ((1 << _domain) & domains) == (1 << _domain);
}

bool
domain_config::operator()(rocprofiler_tracer_activity_domain_t _domain, uint32_t _op) const
{
    auto _offset = (_domain * rocprofiler::internal::domain_ops_offset);
    return (*this)(_domain) && (opcodes.none() || opcodes.test(_offset + _op));
}
}  // namespace internal
}  // namespace rocprofiler
