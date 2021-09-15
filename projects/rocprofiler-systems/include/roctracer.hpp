
#pragma once

#include "timemory/api.hpp"
#include "timemory/components/base.hpp"
#include "timemory/components/data_tracker/components.hpp"
#include "timemory/components/macros.hpp"
#include "timemory/enum.h"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/type_traits.hpp"
#include "timemory/mpl/types.hpp"

TIMEMORY_DECLARE_COMPONENT(roctracer)

#if !defined(HOSTTRACE_USE_ROCTRACER)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_available, component::roctracer, false_type)
#endif

namespace tim
{
namespace component
{
using roctracer_data = data_tracker<double, roctracer>;

struct roctracer
: base<roctracer, void>
, private policy::instance_tracker<roctracer, false>
{
    using value_type   = void;
    using base_type    = base<roctracer, void>;
    using tracker_type = policy::instance_tracker<roctracer, false>;

    TIMEMORY_DEFAULT_OBJECT(roctracer)

    static void preinit();
    static void global_init() { setup(); }
    static void global_finalize() { tear_down(); }

    static bool is_setup();
    static void setup();
    static void tear_down();
    static void add_setup(const std::string&, std::function<void()>&&);
    static void add_tear_down(const std::string&, std::function<void()>&&);
    static void remove_setup(const std::string&);
    static void remove_tear_down(const std::string&);

    void start();
    void stop();
    void set_prefix(const char* _v) { m_prefix = _v; }

private:
    const char* m_prefix = nullptr;
};
}  // namespace component
}  // namespace tim

TIMEMORY_SET_COMPONENT_API(component::roctracer_data, project::timemory, category::timing,
                           os::supports_unix)
TIMEMORY_DEFINE_CONCRETE_TRAIT(is_timing_category, component::roctracer_data, true_type)
TIMEMORY_DEFINE_CONCRETE_TRAIT(uses_timing_units, component::roctracer_data, true_type)

#include "timemory/operations.hpp"

TIMEMORY_DECLARE_EXTERN_COMPONENT(roctracer, false, void)
TIMEMORY_DECLARE_EXTERN_COMPONENT(roctracer_data, true, double)
