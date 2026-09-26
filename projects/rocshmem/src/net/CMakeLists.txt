###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################

###############################################################################
# BACKEND-NEUTRAL VERBS NETWORKING LAYER
#
# Transport-neutral building blocks (dlopen'd libibverbs, RC control plane, MR
# registry, addressing/completion helpers) shared by any verbs consumer -- the
# reverse-offload device proxy, the host-initiated path, and (future) GDA host
# reuse. No backend-specific code lives here. libibverbs is loaded via dlopen
# (the target already links `dl`); only the <infiniband/verbs.h> header is a
# build-time dependency.
###############################################################################
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    ibv.cpp
    ib_ctrl.cpp
    mr_registry.cpp
)
