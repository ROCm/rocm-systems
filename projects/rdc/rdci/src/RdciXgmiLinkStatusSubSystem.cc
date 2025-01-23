/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.
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
#include "RdciXgmiLinkStatusSubSystem.h"

#include <getopt.h>
#include <signal.h>
#include <unistd.h>

#include "common/rdc_utils.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcException.h"
#include "rdc_lib/rdc_common.h"
namespace amd {
namespace rdc {
RdciXgmiLinkStatusSubSystem::RdciXgmiLinkStatusSubSystem() : link_status_ops_(XMGI_LINK_UNKNOWN) {}

void RdciXgmiLinkStatusSubSystem::parse_cmd_opts(int argc, char** argv) {
  const int HOST_OPTIONS = 1000;
  const struct option long_options[] = {{"host", required_argument, nullptr, HOST_OPTIONS},
                                        {"help", optional_argument, nullptr, 'h'},
                                        {"unauth", optional_argument, nullptr, 'u'},
                                        {"xgmi-link-status", optional_argument, nullptr, 'l'},
                                        {nullptr, 0, nullptr, 0}};
  int option_index = 0;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "hul", long_options, &option_index)) != -1) {
    {
      switch (opt) {
        case HOST_OPTIONS:
          ip_port_ = optarg;
          break;
        case 'h':
          show_help();
          break;
        case 'u':
          use_auth_ = false;
          break;
        case 'l':
          link_status_ops_ = XMGI_LINK_STATUS;
          break;
        default:
          show_help();
          throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command line options");
      }
    }
  }
}
void RdciXgmiLinkStatusSubSystem::show_help() const {
  std::cout << " link -- Used to link Get the link status the link is up or down.\n\n";
  std::cout << "Usage\n";
  std::cout << "    rdci link [--host <IP/FQDN>:port] [--xgmi-link-status] -l \n";
  std::cout << "\nFlags:\n";
  show_common_usage();
}

static const char* xgmi_link_status_to_str(rdc_link_state_t state) {
  if (state == RDC_LINK_STATE_DISABLED) return "X ";
  if (state == RDC_LINK_STATE_DOWN) return "D ";
  if (state == RDC_LINK_STATE_UP) return "U ";
  return "N/A ";
}

void RdciXgmiLinkStatusSubSystem::process() {
  rdc_status_t result = RDC_ST_OK;
  switch (link_status_ops_) {
    case XMGI_LINK_STATUS: {
      rdc_link_status_t link_status;
      result = rdc_link_status_get(rdc_handle_, &link_status);
      if (result == RDC_ST_OK) {
        std::cout << "GPUs:\n";
        for (int32_t i = 0; i < link_status.num_of_gpus; i++) {
          std::cout << "    GPU:" << link_status.gpus[i].gpu_index << "\n";
          std::cout << "        ";
          for (uint32_t n = 0; n < link_status.gpus[i].num_of_links; n++) {
            std::cout << xgmi_link_status_to_str(link_status.gpus[i].link_states[n]);
          }
          std::cout << "\n";
        }
          std::cout << "\n";
          std::cout << "* U:Up D:Down X:Disabled\n";
      }
      break;
    }
    default:
      throw RdcException(RDC_ST_BAD_PARAMETER, "Unknown command");
      break;
  }
}
}  // namespace rdc
}  // namespace amd