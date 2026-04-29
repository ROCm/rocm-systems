// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <string>

namespace rocm_compute
{
class pc_samples_writer_t
{
public:
    virtual             ~pc_samples_writer_t() = default;
    virtual void        write() = 0;
	virtual std::string get_result() = 0;
};
}
