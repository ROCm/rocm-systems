/*
 * Copyright (c) Broadcom Inc All Rights Reserved.
 *
 *  Developed by:
 *            Broadcom Inc
 *
 *            www.broadcom.com
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

#include <fstream>
#include <memory>
#include <regex>
#include <iomanip>

#include "amd_smi/impl/nic/amd_smi_lspci_commands.h"
#include "amd_smi/impl/amd_smi_utils.h"

amdsmi_status_t get_lspci_device_data(std::string bdfStr, std::string search_key, std::string& version) {
    std::string lspci_data;
    std::string command = "lspci -s " + bdfStr + " -vv | grep -i '" + search_key + "'";

    if (smi_brcm_execute_cmd_get_data(command, &lspci_data) != AMDSMI_STATUS_SUCCESS){
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to execute command: lspci -s " << bdfStr << " -vv | grep -i " << search_key << ".";
        LOG_ERROR(ss);

        return AMDSMI_STATUS_NOT_SUPPORTED;
    }

    int pos = lspci_data.find(search_key);
    if (pos != std::string::npos) {
        version = lspci_data.erase(0, lspci_data.find(search_key) + search_key.length());
        if (!version.empty() && version[version.length() - 1] == '\n') {
            version.erase(version.length() - 1);
        }
    }
    else
        version = "N/A";

    return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_lspci_root_switch(amdsmi_bdf_t devicehBdf, amdsmi_bdf_t *switchBdf) {

    amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
    std::string lspci_data;

    status = smi_brcm_execute_cmd_get_data("lspci -tvv", &lspci_data);

    if (status != AMDSMI_STATUS_SUCCESS) {
        std::ostringstream ss;
        ss << __PRETTY_FUNCTION__ << " | "
           << "Failed to execute command: lspci -tvv.";
        LOG_ERROR(ss);
        return status;
    }
    
    std::istringstream lines(lspci_data);

    std::string line;
    uint64_t bus_pos, dev_pos, fun_pos;

    std::vector<amdsmi_bdf_t> switch_list;
    amdsmi_bdf_t temp;


    // Loop through and get the switch list
    while (std::getline(lines, line)) {

        if(line.find("LSI PCIe Switch management endpoint") != std::string::npos){
            //get Bus
            bus_pos = line.rfind(']----');
            if (bus_pos == std::string::npos){
                // Check if the Bus position is not found, then continue to the next line
                continue;
            }
            
            //Get device
            dev_pos = line.rfind('.');
            if (dev_pos == std::string::npos){
                // Check if the device position is not found, then continue to the next line
                continue;
            }

            //Get function
            fun_pos = dev_pos + 1;

            std::ostringstream ss;
            ss << __PRETTY_FUNCTION__ << " | "
               << "Found switch at " << line.substr(bus_pos - 6, 2) << ":"
               << line.substr(dev_pos - 2, 2) << ":"
               << line.substr(fun_pos - 2, 1);
            LOG_DEBUG(ss);

            try
            {
                // Parse the BDF
                temp.bus_number =  std::stoi(line.substr(bus_pos - 6, 2), NULL, 16);
                temp.device_number =  std::stoi(line.substr(dev_pos - 2, 2), NULL, 16);
                temp.function_number =  std::stoi(line.substr(fun_pos - 2, 1), NULL, 16);
            } 
            catch (const std::invalid_argument& e) {
                std::ostringstream ss;
                ss << __PRETTY_FUNCTION__ << " | " << "Invalid input: Not a valid hexadecimal string.";
                LOG_ERROR(ss);
            }
            catch (const std::out_of_range& e) {
                std::ostringstream ss;
                ss << __PRETTY_FUNCTION__ << " | " << "Invalid input: Number out of range.";
                LOG_ERROR(ss);
            }
            
            switch_list.push_back(temp);
        }
    }

    //Reset Stream
    lines.clear();
    lines.seekg(0, std::ios::beg);


    for (const auto& d : switch_list){
        uint64_t switch_bus_start, switch_bus_end = 0x0 ;
        std::stringstream ss;
        ss << std::hex << std::setw(2) << std::setfill('0') << d.bus_number;

        while (std::getline(lines, line)) {

            if ((line.rfind('-' + ss.str() + ']') != std::string::npos)) {
                switch_bus_end = d.bus_number;
                
                bus_pos = line.rfind('-' + ss.str() + ']');
                //std::cout << line.substr(bus_pos - 2, 2) << std::endl;
                
                try
                {
                    switch_bus_start = std::stoi(line.substr(bus_pos - 2, 2), NULL, 16);
                } 
                catch (const std::invalid_argument& e) {
                    std::ostringstream ss;
                    ss << __PRETTY_FUNCTION__ << " | " << "Invalid input: Not a valid hexadecimal string.";
                    LOG_ERROR(ss);
                }
                catch (const std::out_of_range& e) {
                    std::ostringstream ss;
                    ss << __PRETTY_FUNCTION__ << " | " << "Invalid input: Number out of range.";
                    LOG_ERROR(ss);

                }

                std::ostringstream sst;
                sst << __PRETTY_FUNCTION__ << " | " << "Switch bus range: " << switch_bus_start << "-" << switch_bus_end;
                LOG_DEBUG(sst);

                break;
            }
            
        }

        if (devicehBdf.bus_number >= switch_bus_start  && devicehBdf.bus_number <= switch_bus_end){
            switchBdf->bus_number = d.bus_number;
            switchBdf->device_number = d.device_number;
            switchBdf->function_number = d.function_number;
            std::ostringstream ss;
            ss << __PRETTY_FUNCTION__ << " | " << "Found switch at BDF " << d.bus_number << ":" << d.device_number << ":" << d.function_number;
            LOG_DEBUG(ss);
                        
            break;
        }
    }

      return status;

}
