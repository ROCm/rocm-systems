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
#ifndef SERVER_INCLUDE_RDC_RDC_ADMIN_SERVICE_H_
#define SERVER_INCLUDE_RDC_RDC_ADMIN_SERVICE_H_

#include "amd_smi/amdsmi.h"
#include "rdc.grpc.pb.h"  // NOLINT
#include "rdc/rdc_admin_service.h"

namespace amd {
namespace rdc {

class RDCAdminServiceImpl final : public ::rdc::RdcAdmin::Service {
 public:
  RDCAdminServiceImpl();
  ~RDCAdminServiceImpl();
  ::grpc::Status VerifyConnection(::grpc::ServerContext* context,
                                  const ::rdc::VerifyConnectionRequest* request,
                                  ::rdc::VerifyConnectionResponse* reply) override;

 private:
};

}  // namespace rdc
}  // namespace amd

#endif  // SERVER_INCLUDE_RDC_RDC_ADMIN_SERVICE_H_
