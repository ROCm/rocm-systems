/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <cassert>
#include <cstdint>
#include <unistd.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include "amd_smi/amdsmi.h"
#include <cstring>
#include <cmath>

#define SHOWLINESZ 256

#define CHK_AMDSMI_RET(RET)                                                    \
    {                                                                          \
        if (RET != AMDSMI_STATUS_SUCCESS) {                                    \
            const char *err_str;                                               \
            amdsmi_status_t status;                                            \
            status = amdsmi_get_esmi_err_msg(RET, &err_str);                   \
            std::cout << "AMDSMI call returned " << status << " at line "      \
                      << __LINE__ << std::endl;                                \
            std::cout << err_str << std::endl;                                 \
            return RET;                                                        \
        }                                                                      \
    }

using std::cin;
using std::cout;
using std::endl;
using std::fixed;
using std::setprecision;
using std::vector;

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv) {
  amdsmi_status_t ret;
  uint32_t proto_ver;
  amdsmi_smu_fw_version_t smu_fw = {};

  // Initialize esmi for AMD CPUs
  ret = amdsmi_init(AMDSMI_INIT_AMD_CPUS);
  CHK_AMDSMI_RET(ret)

  // Get all sockets
  uint32_t socket_count = 0;

  ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  CHK_AMDSMI_RET(ret)

  // Allocate the memory for the sockets
  vector<amdsmi_socket_handle> sockets(socket_count);

  // Get the sockets of the system
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
  CHK_AMDSMI_RET(ret)

  cout << "Total Socket: " << socket_count << endl;

  // For each socket, get cpus and cores
  for (uint32_t i = 0; i < socket_count; i++) {
    cout << endl << "Socket " << i << endl;
    uint32_t cpu_count = 0;
    uint32_t core_count = 0;

    // Set processor type as AMDSMI_PROCESSOR_TYPE_AMD_CPU
    processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_AMD_CPU;
    ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, nullptr, &cpu_count);
    CHK_AMDSMI_RET(ret)

    // Allocate the memory for the cpus
    vector<amdsmi_processor_handle> plist(cpu_count);

    // Get the cpus for each socket
    ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, &plist[0], &cpu_count);
    CHK_AMDSMI_RET(ret)

    // Set processor type as AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE
    processor_type = AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE;
    ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, nullptr, &core_count);
    CHK_AMDSMI_RET(ret)

    // Allocate the memory for the cpu cores
    vector<amdsmi_processor_handle> core_list(core_count);

    // Get the cpu cores for each socket
    ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, &core_list[0], &core_count);
    CHK_AMDSMI_RET(ret)

    for (uint32_t index = 0; index < plist.size(); index++) {
      ret = amdsmi_get_cpu_hsmp_proto_ver(plist[index], &proto_ver);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get hsmp proto version"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      cout<<"\n------------------------------------------";
      cout<<"\n| HSMP Proto Version  |  "<< proto_ver <<"\t\t |"<< endl;
      cout<<"------------------------------------------\n";

      ret = amdsmi_get_cpu_smu_fw_version(plist[index], &smu_fw);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get smu fw version"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      cout<<"\n------------------------------------------";
      cout<<"\n| SMU FW Version  |  "
              <<(unsigned)smu_fw.major<<"."
              <<(unsigned)smu_fw.minor<<"."
              <<(unsigned)smu_fw.debug
              <<"\t\t |"<<endl;
      cout<<"------------------------------------------\n";

      uint32_t err_bits = 0;

      uint32_t prochot;
      cout<<setprecision(3)<<" CPU "<<index<<"\t|";
      cout<<"\n-------------------------------------------------";
      cout<<"\n| ProchotStatus:\t\t |";

      ret = amdsmi_get_cpu_prochot_status(plist[index], &prochot);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get prochot status"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<setprecision(7)<< (prochot ? "active" : "inactive")<<"\t|";
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err:" <<ret<<"     |";
      }
      cout<<"\n-------------------------------------------------\n";

      size_t len;
      char str[SHOWLINESZ] = {};
      int retVal = 0;
      cout<<setprecision(3)<<" CPU "<<index<<"\t|";
      cout<<"\n-------------------------------------------------";
      cout<<"\n| fclk (Mhz)\t\t\t |";
      retVal = snprintf(str, SHOWLINESZ, "\n| mclk (Mhz)\t\t\t |");

      len = strlen(str);
      uint32_t fclk, mclk;
      err_bits = 0;

      ret = amdsmi_get_cpu_fclk_mclk(plist[index], &fclk, &mclk);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu fclk mclk"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<setprecision(7)<<" "<<fclk<<"\t\t|";
          retVal = snprintf(str + len, SHOWLINESZ - len, " %d\t\t|", mclk);
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err: "<<setprecision(2)<<ret<<"     |";
          retVal = snprintf(str + len, SHOWLINESZ - len, " NA (Err: %-2d)     |", ret);
      }
      if (retVal > 0 && retVal < SHOWLINESZ)
          cout << str;
      else
          cout <<"error writing to buffer" << endl;

      cout<<"\n-------------------------------------------------\n";

      uint32_t socket_power;
      cout<<setprecision(3)<<" CPU "<<index<<"\t|";
      cout<<"\n-------------------------------------------------";
      cout<<"\n| Power (Watts)\t\t\t | ";

      ret = amdsmi_get_cpu_socket_power(plist[index], &socket_power);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu socket power"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<fixed<<setprecision(3)<<static_cast<double>(socket_power)/1000<<"\t|";
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err:" <<ret<<"     |";
      }

      uint32_t power_limit = 0;
      cout<<"\n| PowerLimit (Watts)\t\t | ";

      ret = amdsmi_get_cpu_socket_power_cap(plist[index], &power_limit);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu socket power cap"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<fixed<<setprecision(3)<<static_cast<double>(power_limit)/1000<<"\t|";
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err:" <<ret<<"     |";
      }

      uint32_t power_max = 0;
      cout<<"\n| PowerLimitMax (Watts)\t\t | ";

      ret = amdsmi_get_cpu_socket_power_cap_max(plist[index], &power_max);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu socket power cap max"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<fixed<<setprecision(3)<<static_cast<double>(power_max)/1000<<"\t|";
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err:" <<ret<<"     |";
      }
      cout<<"\n-------------------------------------------------\n";

      uint32_t input_power;
      power_max = 0;
      cout<<"\nEnter the max power to be set:\n";
      cin>>input_power;

      ret = amdsmi_get_cpu_socket_power_cap_max(plist[index], &power_max);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu socket power cap max"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if ((ret == AMDSMI_STATUS_SUCCESS) && (input_power > power_max)) {
          cout<<"Input power is more than max power limit,"
              " limiting to "<<static_cast<double>(power_max)/1000<<"Watts\n";
          input_power = power_max;
      }

      ret = amdsmi_set_cpu_socket_power_cap(plist[index], input_power);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to set cpu socket power cap"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<"CPU ["<<index<<"] power_limit set to "
          <<fixed<<setprecision(3)<<static_cast<double>(input_power)/1000<<" Watts successfully\n";
      }

      power_limit = 0;
      cout<<"\n| PowerLimit (Watts) \t\t | ";

      ret = amdsmi_get_cpu_socket_power_cap(plist[index], &power_limit);
      if(ret != AMDSMI_STATUS_SUCCESS)
          cout<<"Failed to get cpu socket power cap"<<"["<<index<<"] , Err["<<ret<<"] "<< endl;

      if (!ret) {
          cout<<fixed<<setprecision(3)<<static_cast<double>(power_limit)/1000<<"\t|";
      } else {
          err_bits |= 1 << ret;
          cout<<" NA (Err:" <<ret<<"     |";
      }
      cout<<"\n-------------------------------------------------\n";

      double fraction_q10 = 1/pow(2,10);
      double fraction_uq10 = fraction_q10;

      amdsmi_hsmp_metrics_table_t mtbl = {};
      ret = amdsmi_get_hsmp_metrics_table(plist[index], &mtbl);

      if (ret != AMDSMI_STATUS_SUCCESS) {
          cout<<"Failed to get Metrics Table for CPU["<<index<<"], Err["<<ret<<"]" << endl;
      } else {
          cout<<"\n| METRICS TABLE                 \t\t\t\t |\n";

          cout<<"\n| ACCUMULATOR COUNTER                   |  "<<mtbl.accumulation_counter<<"\t\t|";
          cout<<"\n| SOCKET POWER LIMIT                    |  "<<(mtbl.socket_power_limit * fraction_uq10)<<" W\t\t|";
          cout<<"\n| MAX SOCKET POWER LIMIT                |  "<<(mtbl.max_socket_power_limit * fraction_uq10)<<" W\t\t|";
          cout<<"\n| SOCKET POWER                          |  "<<(mtbl.socket_power * fraction_uq10)<<" W\t\t|\n";

	  cout<<"\n| Effective frequency per AID: \t\t\t\t\t\t|";
          cout<<"\n-------------------------------------------------------------------------";
          cout<<"\n| AID | SOCCLK \t\t| VCLK \t\t| DCLK \t\t| LCLK \t\t|";
          cout<<"\n-------------------------------------------------------------------------";
          for(uint32_t j = 0; j < 4 ; j++){
              cout<<fixed<<setprecision(3)<<"\n| ["<<j<<"] | "
                <<(mtbl.socclk_frequency[j] * fraction_uq10)<<"MHz\t| "
                <<(mtbl.vclk_frequency[j] * fraction_uq10)<<"MHz\t| "
                <<(mtbl.dclk_frequency[j] * fraction_uq10)<<"MHz\t| "
                <<(mtbl.lclk_frequency[j] * fraction_uq10)<<"MHz\t| ";
          }
          cout<<"\n-------------------------------------------------------------------------\n";
          cout<<"\n-------------------------------------------------------------------------\n";
      }
    }
  }
  // Clean up resources allocated at amdsmi_init
  ret = amdsmi_shut_down();
  CHK_AMDSMI_RET(ret)

  return 0;
}
