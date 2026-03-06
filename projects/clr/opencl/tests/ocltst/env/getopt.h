/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

extern char* optarg;
extern int optind;

extern "C" int getopt(int argc, char* const argv[], const char* optstring);
