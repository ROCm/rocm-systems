// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Internal engine header — the single place inside the argparse layer that
// pulls in the timemory parser implementation. Other argparse TUs that need
// the complete tim::argparse::argument_parser / tim::vsettings types include
// THIS header rather than reaching for the timemory headers directly. Public
// argparse.hpp forward-declares these types and exposes them only behind
// references / pointers, so its consumers never pay the timemory include
// cost.
//
// When the engine is swapped (e.g. to CLI11), this is one of two files that
// must change — the other being parsed_values.cpp's specializations.

#pragma once

#include <timemory/settings/vsettings.hpp>
#include <timemory/utility/argparse.hpp>
