/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <map>

const std::map<std::string, std::string> mapKernelToFileName{
  {"Set", "Set.cpp"},
  {"HipTest::vectorADD", "vectorADD.inl"},
};