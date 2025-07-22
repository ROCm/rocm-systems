
#include <dlfcn.h>
#include <cstdlib>
#include <string>

#include "common/fwd.hpp"

std::string
run(const std::string& name)
{
    resolve_symbols<ROCP_REG_TEST_HSA | ROCP_REG_TEST_HIP>();

    if(hsa_init_fn)
    {
        hsa_init_fn();
    }

    if(hip_init_fn)
    {
        hip_init_fn();
    }

    if(roctxRangePush_fn)
    {
        roctxRangePush_fn(name.c_str());
    }

    if(roctxRangePop_fn)
    {
        roctxRangePop_fn(name.c_str());
    }

    return name;
}

auto run_name = run("thread-ctor");

int
main()
{
    return (run_name == "thread-ctor") ? EXIT_SUCCESS : EXIT_FAILURE;
}
