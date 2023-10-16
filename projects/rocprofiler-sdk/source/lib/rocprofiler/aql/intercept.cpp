#include "lib/rocprofiler/aql/intercept.hpp"

#include "lib/rocprofiler/hsa/hsa.hpp"

namespace rocprofiler
{
namespace aql
{
std::shared_ptr<const Intercept>
Intercept::create(const std::function<void(HsaApiTable&)>& mod_cb)
{
    return std::make_shared<const Intercept>(mod_cb);
}

Intercept::Intercept(const std::function<void(HsaApiTable&)>& mod_cb)
: _original(rocprofiler::hsa::get_table())
, _modified(rocprofiler::hsa::get_table())
{
    mod_cb(_modified);
};

}  // namespace aql
}  // namespace rocprofiler
