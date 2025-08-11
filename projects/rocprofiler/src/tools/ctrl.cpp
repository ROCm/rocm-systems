#include <hsa/hsa.h>
#include "rocprofiler.h"

#include <cstdio>

static int info_callback(const rocprofiler_counter_info_t info, const char* gpu_name,
                         uint32_t gpu_index) {
  fprintf(stdout, "\n  %s:%u : %s : %s\n", gpu_name, gpu_index, info.name, info.description);
  if (info.expression != nullptr) {
    fprintf(stdout, "      %s = %s\n", info.name, info.expression);
  } else {
    if (info.instances_count > 1) fprintf(stdout, "[0-%u]", info.instances_count - 1);
    fprintf(stdout, " : %s\n", info.description);
    fprintf(stdout, "      block %s can only handle %u counters at a time\n", info.block_name,
            info.block_counters);
  }
  fflush(stdout);
  return 1;
}

int main(int argc, char** argv) {
  hsa_init();
  rocprofiler_iterate_counters(info_callback);
}