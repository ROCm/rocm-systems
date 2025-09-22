
#include "common.hpp"
#include "rocm_async.hpp"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>

void RocmAsync::RunIOBenchmark(async_trans_t& trans) {

  std::cout << "Unsupported Request - Read / Write" << std::endl;
  exit(1);
}
