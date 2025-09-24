
#include "benchmark.hpp"

namespace rocprofsys
{
namespace benchmark
{
namespace
{
impl::benchmark bc;
}

impl::benchmark&
get_benchmark()
{
    return bc;
}

void
show_results()
{
    bc.show_results();
}

}  // namespace benchmark
}  // namespace rocprofsys
