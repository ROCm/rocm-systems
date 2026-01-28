/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "LL_MoE.hpp"
#include <unistd.h>

using dtype = short;

/******************************************************************************
 * This example intentionally MIMICS the communication pattern used by
 * DeepEP low-latency (LL) mode MoE kernels.
 *
 * In particular, it mirrors the structure of DeepEP's:
 *   - dispatch kernel  (token -> expert / rank)
 *   - combine kernel   (expert -> token / rank)
 *
 * NOTE: It is NOT a performance model; it is a communication-pattern model.
 *****************************************************************************/

int main (int argc, char **argv)
{
  int rank, num_ranks;

  // DeepEP_LL parameters with default values
  int num_tokens  = 128;
  int hidden      = 7168;
  int num_topk    = 8;
  int num_experts = 288;

  int num_iterations = 10;

  // parse command line arguments
  int opt;
  while ((opt = getopt(argc, argv, "n:h:k:e:i:")) != -1) {
    switch (opt) {
      case 'n':
        num_tokens = atoi(optarg);
        break;
      case 'h':
        hidden = atoi(optarg);
        break;
      case 'k':
        num_topk = atoi(optarg);
        break;
      case 'e':
        num_experts = atoi(optarg);
        break;
      case 'i':
        num_iterations = atoi(optarg);
        break;
      case '?':
        if (optopt == 'n' || optopt == 'h' || optopt == 'k' ||
            optopt == 'e' || optopt == 'i') {
          std::cerr << "Option -" << static_cast<char>(optopt)
                    << " requires an argument." << std::endl;
        } else {
          std::cerr << "Unknown option -"
                  << static_cast<char>(optopt) << std::endl;
        }
        std::cerr << "Usage: " << argv[0]
                  << " [-n num_tokens] [-h hidden] [-k num_topk]"
                  << " [-e num_experts] [-i num_iterations]"
                  << std::endl;
        return EXIT_FAILURE;
    }
  }

  LLMoE<dtype> ll_moe(num_tokens, hidden, num_topk, num_experts);

  rank = ll_moe.get_rank();
  num_ranks = ll_moe.get_num_ranks();

  std::cout << "rank: " << rank << ", n_RANKs: " << num_ranks << std::endl;

  // Run dispatch and combine multiple times
  for (int iter = 0; iter < num_iterations; iter++) {
      ll_moe.ll_dispatch();
      ll_moe.ll_combine();
  }

  // Print all the iterations were successful
  if (rank == 0) {
      std::cout << "All " << num_iterations
                << " iterations completed successfully!" << std::endl;
  }

  return 0;

}