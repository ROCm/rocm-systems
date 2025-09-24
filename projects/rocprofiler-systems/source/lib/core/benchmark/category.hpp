#pragma once
#include <cstddef>
#include <ostream>
#include <sys/types.h>

namespace rocprofsys
{
namespace benchmark
{
namespace category
{

#define DEFINE_CATEGORY(category_name)                                                   \
    struct impl_##category_name                                                          \
    {                                                                                    \
        static constexpr size_t      index = __COUNTER__;                                \
        static constexpr const char* name  = #category_name;                             \
    };                                                                                   \
    inline std::ostream& operator<<(std::ostream& os, const impl_##category_name&)       \
    {                                                                                    \
        return os << "Name: " << impl_##category_name::name                              \
                  << ", index: " << impl_##category_name::index;                         \
    }                                                                                    \
    static impl_##category_name category_name;

DEFINE_CATEGORY(kernel_dispatch)
DEFINE_CATEGORY(memory_copy)
DEFINE_CATEGORY(memory_allocate)
DEFINE_CATEGORY(hip_kernel_dispatch)
DEFINE_CATEGORY(amd_smi_sample)

DEFINE_CATEGORY(category_end_marker)
constexpr size_t total_count = category_end_marker.index;

constexpr const char*
to_string(const size_t index)
{
    switch(index)
    {
        case impl_kernel_dispatch::index: return impl_kernel_dispatch::name;
        case impl_memory_copy::index: return impl_memory_copy::name;
        case impl_memory_allocate::index: return impl_memory_allocate::name;
        case impl_hip_kernel_dispatch::index: return impl_hip_kernel_dispatch::name;
        case impl_amd_smi_sample::index: return impl_amd_smi_sample::name;
        default: return "Unknown category";
    }
}

}  // namespace category
}  // namespace benchmark
}  // namespace rocprofsys
