/*************************************************************************
 * Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#include "CollectiveArgs.hpp"
#include "StandaloneUtils.hpp"
#include <iostream>
#include <regex>


namespace RcclUnitTesting
{

std::string executeCommand(const char* cmd) {
    std::string result;
    FILE* pipe = popen(cmd, "r");

    if (!pipe) {
        std::cerr << "Error executing command: " << cmd << std::endl;
        return result;
    }

    char buffer[128];
    while (!feof(pipe)) {
        if (fgets(buffer, 128, pipe) != NULL) {
            result += buffer;
        }
    }

    pclose(pipe);
    return result;
}

std::vector<std::string> splitString(const std::string& str, char delimiter) {
    std::vector<std::string> result;
    std::istringstream iss(str);

    std::string line;
    while(std::getline(iss, line, delimiter)) {
        result.push_back(line);
    }

    return result;
}


ArchInfo parseMetadata(const std::vector<std::string>& list) {
    ArchInfo archInfo;
    KernelInfo currKernelInfo;
    
    std::regex amdhsaTargetRegex("amdhsa.target:\\s+(?:'?)amdgcn-amd-amdhsa--(\\w+)(?:'?)");
    std::regex kernelNameRegex("\\.name:\\s+(\\w+)");
    std::regex privateSegmentSizeRegex("\\.private_segment_fixed_size:\\s+(\\d+)");
    
    for (const auto& line : list) {
        std::smatch match;

        if (std::regex_search(line, match, amdhsaTargetRegex)) {
            archInfo.archName = match[1];
        } else if (std::regex_search(line, match, kernelNameRegex)) {
            currKernelInfo.name = match[1];
        } else if (std::regex_search(line, match, privateSegmentSizeRegex)) {
            currKernelInfo.privateSegmentFixedSize = std::stoi(match[1]);
        }
        
        if (!currKernelInfo.name.empty() && currKernelInfo.privateSegmentFixedSize != 0) {
            archInfo.kernels.push_back(currKernelInfo);
            currKernelInfo = {}; // Empty kernelInfo
        }
    }
    
    return archInfo;
}

}
