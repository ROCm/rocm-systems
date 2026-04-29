// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "code_object_translator.h"

#include <string>

namespace rocm_compute
{
class pc_samples_writer_t
{
public:
    virtual             ~pc_samples_writer_t() = default;
    virtual void        start_code_obj_desc(const obj_symbol_t& desc) = 0;
    virtual void        end_code_obj_desc()                           = 0;
	virtual std::string get_result() = 0;
};
}
