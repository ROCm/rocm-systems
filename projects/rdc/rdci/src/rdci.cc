/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <string>
#include "rdc_lib/rdc_common.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "RdciDiscoverySubSystem.h"
#include "RdciDmonSubSystem.h"
#include "RdciDiagSubSystem.h"
#include "RdciFieldGroupSubSystem.h"
#include "RdciGroupSubSystem.h"
#include "RdciStatsSubSystem.h"


int main(int argc, char ** argv) {
    const std::string usage_help =
    "Usage:\trdci <subsystem>\nsubsystem: discovery, dmon, group, "
    "fieldgroup, stats, diag\n";

    if (argc <= 1) {
        std::cout << usage_help;
        exit(0);
    }

    amd::rdc::RdciSubSystemPtr subsystem;
    try {
       std::string subsystem_name = argv[1];
       if (subsystem_name == "discovery") {
           subsystem.reset(new amd::rdc::RdciDiscoverySubSystem());
       } else if (subsystem_name == "dmon") {
           subsystem.reset(new amd::rdc::RdciDmonSubSystem());
       } else if (subsystem_name == "diag") {
           subsystem.reset(new amd::rdc::RdciDiagSubSystem());
       } else if (subsystem_name == "group") {
           subsystem.reset(new amd::rdc::RdciGroupSubSystem());
       } else if (subsystem_name == "fieldgroup") {
           subsystem.reset(new amd::rdc::RdciFieldGroupSubSystem());
       } else if (subsystem_name == "stats") {
           subsystem.reset(new amd::rdc::RdciStatsSubSystem());
       } else {
           std::cout << usage_help;
           exit(0);
       }

       subsystem->parse_cmd_opts(argc, argv);

       subsystem->connect();

       subsystem->process();
    } catch (const amd::rdc::RdcException& e) {
        if (subsystem && subsystem->is_json_output()) {
            std::cout << "\"status\": \"error\", \"description\": \""
                << e.what() << '"';
        } else {
            std::cout << "rdci Error: " << e.what()  << std::endl;
        }
        return e.error_code();
    } catch (...) {
        if (subsystem && subsystem->is_json_output()) {
            std::cout << "\"status\": \"error\", \"description\": "
                    << "\"Unhandled exception.\"";
        } else {
            std::cout << "Unhandled exception." << std::endl;
        }        return 1;
    }

    return 0;
}
