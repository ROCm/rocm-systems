// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "backends/amd_smi/backend.hpp"
#include "backends/amd_smi/wrapper.hpp"

#include <memory>

namespace rocprofsys::avail
{

class smi_session
{
public:
    using backend_t = backends::amd_smi::backend<backends::amd_smi::wrapper>;

    smi_session()
    : m_backend{ std::make_shared<backend_t>() }
    {
        m_backend->initialize();
    }

    ~smi_session() { m_backend->shutdown(); }

    smi_session(const smi_session&)            = delete;
    smi_session& operator=(const smi_session&) = delete;
    smi_session(smi_session&&)                 = delete;
    smi_session& operator=(smi_session&&)      = delete;

    [[nodiscard]] const std::shared_ptr<backend_t>& backend() const noexcept
    {
        return m_backend;
    }

private:
    std::shared_ptr<backend_t> m_backend;
};

}  // namespace rocprofsys::avail
