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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_TEAM_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_TEAM_HPP_

#include "team.hpp"

namespace rocshmem {

class Backend;

class ROTeam : public Team {
 public:
  ROTeam(Backend* handle, TeamInfo* team_info_wrt_parent,
         TeamInfo* team_info_wrt_world, int num_pes, int my_pe,
         MPI_Comm team_comm);

  virtual ~ROTeam();

  void* ata_buffer;
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_RO_NET_TEAM_HPP_
