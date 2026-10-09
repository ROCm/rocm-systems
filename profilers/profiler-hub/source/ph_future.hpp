// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <stop_token>

/**
 * Definition of the opaque ph_future handle. It is shared between the user, who holds
 * one reference until ph_future_free(), and the operation it was passed to, which holds
 * one until its on_finished callback has returned.
 */
struct ph_future : std::enable_shared_from_this<ph_future>
{
    ph_future(ph_progress_fn on_progress, ph_finished_fn on_finished) noexcept;

    /** Creates a future that the user reference keeps alive until
     * release_user_reference(). */
    [[nodiscard]] static std::shared_ptr<ph_future> create(ph_progress_fn on_progress,
                                                           ph_finished_fn on_finished);

    /** @return false if the future already serves an operation. */
    [[nodiscard]] bool try_attach();

    [[nodiscard]] std::stop_token stop_token() const noexcept;

    /** Calls on_progress with @p value clamped to [0, 1] and @p description;
     *  ignored once finished. */
    void report_progress(double value, ph_progress_description_t description);

    /** Ends the operation; only the first call has an effect. Calls on_finished. */
    void finish(ph_future_status_t status, ph_result_t result);

    [[nodiscard]] ph_result_t wait();
    [[nodiscard]] ph_result_t cancel();
    [[nodiscard]] ph_result_t result(ph_result_t& out);

    /** @return false if the user reference was already released. */
    [[nodiscard]] bool release_user_reference();

private:
    const ph_progress_fn m_on_progress;
    const ph_finished_fn m_on_finished;
    std::stop_source     m_stop;

    std::mutex                 m_mutex;
    std::condition_variable    m_done_cv;
    std::shared_ptr<ph_future> m_user_reference;
    bool                       m_attached{ false };
    bool                       m_finishing{ false };
    bool                       m_done{ false };
    ph_result_t                m_result{ PH_RESULT_SUCCESS };
};
