/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _OCLSYSINFO_H_
#define _OCLSYSINFO_H_
#include <string>

int oclSysInfo(std::string& info_string, bool useCPU, unsigned dev_id,
               unsigned int platformIndex = 0);

#endif  //_OCLSYSINFO_H_
