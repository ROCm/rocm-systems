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

#ifndef LIBRARY_SRC_CONTEXT_INCL_HPP_
#define LIBRARY_SRC_CONTEXT_INCL_HPP_

#include "context.hpp"
#include "context_tmpl_device.hpp"
#include "context_tmpl_host.hpp"
#if defined(USE_GDA)
#include "gda/context_gda_device.hpp"
#include "gda/context_gda_host.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/context_ro_device.hpp"
#include "reverse_offload/context_ro_host.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/context_ipc_device.hpp"
#include "ipc/context_ipc_host.hpp"
#endif

#endif  // LIBRARY_SRC_CONTEXT_INCL_HPP_
