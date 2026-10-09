// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "ph_future.hpp"

#include <algorithm>
#include <utility>

ph_future::ph_future(ph_progress_fn on_progress, ph_finished_fn on_finished) noexcept
: m_on_progress{ on_progress }
, m_on_finished{ on_finished }
{}

std::shared_ptr<ph_future>
ph_future::create(ph_progress_fn on_progress, ph_finished_fn on_finished)
{
    auto future              = std::make_shared<ph_future>(on_progress, on_finished);
    future->m_user_reference = future;
    return future;
}

bool
ph_future::try_attach()
{
    const std::scoped_lock lock{ m_mutex };
    return !std::exchange(m_attached, true);
}

std::stop_token
ph_future::stop_token() const noexcept
{
    return m_stop.get_token();
}

void
ph_future::report_progress(double value)
{
    if(m_on_progress == nullptr) return;

    {
        const std::scoped_lock lock{ m_mutex };
        if(m_finishing) return;
    }
    m_on_progress(this, std::clamp(value, 0.0, 1.0));
}

void
ph_future::finish(ph_future_status_t status, ph_result_t result)
{
    {
        const std::scoped_lock lock{ m_mutex };
        if(std::exchange(m_finishing, true)) return;
        m_result = result;
    }

    if(m_on_finished != nullptr) m_on_finished(this, status, result);

    {
        const std::scoped_lock lock{ m_mutex };
        m_done = true;
    }
    m_done_cv.notify_all();
}

ph_result_t
ph_future::wait()
{
    std::unique_lock lock{ m_mutex };
    if(!m_attached) return PH_RESULT_INVALID_ARGUMENT;

    m_done_cv.wait(lock, [this] { return m_done; });
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_future::cancel()
{
    {
        const std::scoped_lock lock{ m_mutex };
        if(!m_attached) return PH_RESULT_INVALID_ARGUMENT;
    }
    m_stop.request_stop();
    return PH_RESULT_SUCCESS;
}

ph_result_t
ph_future::result(ph_result_t& out)
{
    const std::scoped_lock lock{ m_mutex };
    if(!m_done) return PH_RESULT_INVALID_ARGUMENT;

    out = m_result;
    return PH_RESULT_SUCCESS;
}

bool
ph_future::release_user_reference()
{
    std::shared_ptr<ph_future> reference;
    {
        const std::scoped_lock lock{ m_mutex };
        reference = std::move(m_user_reference);
    }
    return reference != nullptr;
}
