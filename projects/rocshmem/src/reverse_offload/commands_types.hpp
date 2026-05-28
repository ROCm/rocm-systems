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

#ifndef LIBRARY_SRC_REVERSE_OFFLOAD_COMMANDS_TYPES_HPP_
#define LIBRARY_SRC_REVERSE_OFFLOAD_COMMANDS_TYPES_HPP_


namespace rocshmem {

enum ro_net_cmds {
  RO_NET_PUT,
  RO_NET_P,
  RO_NET_GET,
  RO_NET_PUT_NBI,
  RO_NET_GET_NBI,
  RO_NET_AMO_FOP,
  RO_NET_AMO_FCAS,
  RO_NET_FENCE,
  RO_NET_QUIET,
  RO_NET_FINALIZE,
  RO_NET_TEAM_REDUCE,
  RO_NET_SYNC,
  RO_NET_BARRIER,
  RO_NET_TEAM_BROADCAST,
  RO_NET_ALLTOALL,
  RO_NET_FCOLLECT,
};

enum ro_net_types {
  RO_NET_FLOAT,
  RO_NET_CHAR,
  RO_NET_SIGNED_CHAR,
  RO_NET_UNSIGNED_CHAR,
  RO_NET_DOUBLE,
  RO_NET_INT,
  RO_NET_LONG,
  RO_NET_UNSIGNED_LONG,
  RO_NET_LONG_LONG,
  RO_NET_SHORT,
  RO_NET_LONG_DOUBLE
};

}  // namespace rocshmem

#endif  // LIBRARY_SRC_REVERSE_OFFLOAD_COMMANDS_TYPES_HPP_
