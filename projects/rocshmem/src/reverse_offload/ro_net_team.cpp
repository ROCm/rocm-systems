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

#include "ro_net_team.hpp"

#include "backend_type.hpp"
#include "backend_ro.hpp"

namespace rocshmem {

ROTeam::ROTeam(Backend* backend, TeamInfo* team_info_wrt_parent,
               TeamInfo* team_info_wrt_world, int num_pes, int my_pe,
               MPI_Comm mpi_comm)
    : Team(backend, team_info_wrt_parent, team_info_wrt_world, num_pes, my_pe,
           mpi_comm) {
  type = BackendType::RO_BACKEND;

  // Disable allocating ata_buffer for now. It is not
  // used at the moment, but might come back in future versions.
  ata_buffer = nullptr;
}

ROTeam::~ROTeam() {
  if (ata_buffer != nullptr) {
    free(ata_buffer);
    ata_buffer = nullptr;
  }
}

}  // namespace rocshmem
