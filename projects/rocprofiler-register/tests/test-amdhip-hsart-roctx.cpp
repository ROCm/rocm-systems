
#include <dlfcn.h>
#include <string>

#include "common/fwd.hpp"

void
run(const std::string& name)
{
    if(hip_init_fn)
    {
        hip_init_fn();
    }

    if(hsa_init_fn)
    {
        hsa_init_fn();
    }

    if(roctxRangePush_fn)
    {
        roctxRangePush_fn(name.c_str());
    }

    if(roctxRangePop_fn)
    {
        roctxRangePop_fn(name.c_str());
    }
}

int
main(int argc, char** argv)
{
    unsigned long n = 1;
    if(argc > 1) n = std::stoul(argv[1]);

    resolve_symbols<ROCP_REG_TEST_HSA | ROCP_REG_TEST_HIP | ROCP_REG_TEST_ROCTX>();

    for(unsigned long i = 0; i < n; ++i)
        run("thread-main");

    return 0;
}
