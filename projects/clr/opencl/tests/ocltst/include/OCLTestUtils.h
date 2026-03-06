/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef OCLTESTUTILS_H_
#define OCLTESTUTILS_H_
#include <string>

// @param FN Name of the file to be loaded
// @param S String to store the loaded file
// @brief Load file to a string
// @return true on success
bool loadFile(const char* FN, std::string& S);

#endif /* OCLTESTUTILS_H_ */
