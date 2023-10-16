#pragma once

#include <functional>
#include <memory>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>

namespace rocprofiler
{
namespace aql
{
class Intercept
{
public:
    static std::shared_ptr<const Intercept> create(const std::function<void(HsaApiTable&)>& mod_cb);

    explicit Intercept(const std::function<void(HsaApiTable&)>& mod_cb);

private:
    HsaApiTable  _original;
    HsaApiTable& _modified;
};

}  // namespace aql
}  // namespace rocprofiler
