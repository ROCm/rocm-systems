
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include "rdc/rdc_admin_service.h"

#include <assert.h>
#include <grpcpp/grpcpp.h>
#include <unistd.h>

#include <csignal>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>

#include "rdc.grpc.pb.h"  // NOLINT

namespace amd {
namespace rdc {

RDCAdminServiceImpl::RDCAdminServiceImpl() {}

RDCAdminServiceImpl::~RDCAdminServiceImpl() {}
::grpc::Status RDCAdminServiceImpl::VerifyConnection(::grpc::ServerContext* context,
                                                     const ::rdc::VerifyConnectionRequest* request,
                                                     ::rdc::VerifyConnectionResponse* reply) {
  (void)context;  // Quiet warning for now

  reply->set_echo_magic_num(request->magic_num());
  return ::grpc::Status::OK;
}

}  // namespace rdc
}  // namespace amd
