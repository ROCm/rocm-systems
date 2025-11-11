#include "library/amd_smi.hpp"

namespace rocprofsys
{
namespace amd_ainic
{

void
setup()
{
    fprintf(stderr, "aleks: amd_ainic::setup\n");
}


void
config()
{
    fprintf(stderr, "aleks: enter amd_ainic::config\n");
}

void
sample()
{
    fprintf(stderr, "aleks: amd_ainic::sample\n");
}

void
shutdown()
{
    fprintf(stderr, "aleks: amd_ainic::shutdown\n");
}

void
post_process()
{
}

}  // namespace amd_ainic
}  // namespace rocprofsys
