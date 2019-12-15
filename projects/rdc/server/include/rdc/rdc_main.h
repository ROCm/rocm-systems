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
#ifndef SERVER_INCLUDE_RDC_RDC_MAIN_H_
#define SERVER_INCLUDE_RDC_RDC_MAIN_H_

#include <grpcpp/grpcpp.h>

#include <string>
#include <memory>

#include "rdc/rdc_rsmi_service.h"

class RDCServer {
 public:
    RDCServer();
    ~RDCServer();

    void Initialize();

    void Run(void);

    bool start_rsmi_service(void) const {return start_rsmi_service_;}
    void set_start_rsmi_service(bool s) {start_rsmi_service_ = s;}

    void ShutDown(void);

 private:
    void HandleSignal(int sig);
    std::string server_address_;
    bool start_rsmi_service_;
    std::unique_ptr<::grpc::Server> server_;

    RsmiServiceImpl *rsmi_service_;
};

#endif  // SERVER_INCLUDE_RDC_RDC_MAIN_H_
