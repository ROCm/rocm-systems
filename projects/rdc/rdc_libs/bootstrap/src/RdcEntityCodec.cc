/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <rdc/rdc.h>
#include <rdc_lib/RdcEntityCodec.h>

#include <algorithm>
#include <iostream>
#include <string>

#include "common/rdc_utils.h"

rdc_entity_info_t rdc_get_info_from_entity_index(uint32_t entity_index) {
  rdc_entity_info_t info;
  info.device_type =
      (rdc_device_type_t)((entity_index >> RDC_ENTITY_TYPE_SHIFT) & RDC_ENTITY_TYPE_MASK);
  info.entity_role =
      (rdc_device_role_t)((entity_index >> RDC_ENTITY_ROLE_SHIFT) & RDC_ENTITY_ROLE_MASK);
  info.instance_index = (entity_index >> RDC_ENTITY_INSTANCE_SHIFT) & RDC_ENTITY_INSTANCE_MASK;
  info.device_index = (entity_index >> RDC_ENTITY_DEVICE_SHIFT) & RDC_ENTITY_DEVICE_MASK;
  return info;
}

uint32_t rdc_get_entity_index_from_info(rdc_entity_info_t info) {
  uint32_t entity_index = 0;
  entity_index |= ((info.device_type & RDC_ENTITY_TYPE_MASK) << RDC_ENTITY_TYPE_SHIFT);
  entity_index |= ((info.entity_role & RDC_ENTITY_ROLE_MASK) << RDC_ENTITY_ROLE_SHIFT);
  entity_index |= ((info.instance_index & RDC_ENTITY_INSTANCE_MASK) << RDC_ENTITY_INSTANCE_SHIFT);
  entity_index |= ((info.device_index & RDC_ENTITY_DEVICE_MASK) << RDC_ENTITY_DEVICE_SHIFT);
  return entity_index;
}

bool rdc_is_partition_string(const char* s) {
  if (!s || s[0] == '\0') {
    return false;
  }

  if (s[0] != 'g') {
    return false;
  }

  std::string str(s);
  size_t dotPos = str.find('.');
  if (dotPos == std::string::npos) return false;

  if (dotPos <= 1 || dotPos >= str.size() - 1) return false;

  std::string gpuPart = str.substr(1, dotPos - 1);
  std::string partitionPart = str.substr(dotPos + 1);

  if (!std::all_of(gpuPart.begin(), gpuPart.end(), ::isdigit) ||
      !std::all_of(partitionPart.begin(), partitionPart.end(), ::isdigit))
    return false;

  int gpuIndex = std::stoi(gpuPart);
  int partitionIndex = std::stoi(partitionPart);

  if (gpuIndex < 0 || gpuIndex >= RDC_MAX_NUM_DEVICES) return false;
  if (partitionIndex < 0 || partitionIndex >= RDC_MAX_NUM_PARTITIONS) return false;

  return true;
}

bool rdc_parse_partition_string(const char* s, uint32_t* physicalGpu, uint32_t* partition) {
  if (!s) {
    return false;
  }

  if (!rdc_is_partition_string(s)) {
    return false;
  }

  std::string str(s);

  std::string rest = str.substr(1);
  size_t pos = rest.find('.');

  if (pos == std::string::npos) return false;

  std::string gpuStr = rest.substr(0, pos);
  std::string partStr = rest.substr(pos + 1);

  // Ensure both parts are a number
  if (!(!gpuStr.empty() && std::all_of(gpuStr.begin(), gpuStr.end(), ::isdigit)) ||
      !(!partStr.empty() && std::all_of(partStr.begin(), partStr.end(), ::isdigit))) {
    return false;
  }

  *physicalGpu = std::stoi(gpuStr);
  *partition = std::stoi(partStr);
  return true;
}
