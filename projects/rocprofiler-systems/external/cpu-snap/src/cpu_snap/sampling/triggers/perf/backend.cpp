// triggers/perf/backend.cpp: out-of-line bodies for perf::backend.

#include "cpu_snap/sampling/triggers/perf/backend.hpp"

#include <cstdint>
#include <span>
#include <variant>

#include "cpu_snap/sampling/streaming_pipeline.hpp"

namespace cpu_snap::sampling::triggers::perf
{

backend::backend(kernel_drain::config cfg)
: m_drain{ cfg }
{}

void backend::attach_tid(pid_t tid) { m_drain.open_tid(tid); }

void backend::detach_tid(pid_t tid) noexcept { m_drain.close_tid(tid); }

int backend::wait(int timeout_ms) noexcept { return m_drain.wait(timeout_ms); }

void backend::drain_into(internal::streaming_pipeline& pipeline,
                         unwind::unwinder&           unw) noexcept
{
    std::visit(
        [this, &pipeline](auto& concrete_unwinder) {
            m_drain.drain_all(
                concrete_unwinder,
                [&pipeline](pid_t tid, std::uint64_t ts_ns,
                             std::span<const std::uint64_t> frames) noexcept {
                    pipeline.on_sample(tid, ts_ns, frames);
                },
                [&pipeline](std::uint64_t lost) noexcept { pipeline.on_lost(lost); });
        },
        unw);
}

void backend::disable_all() noexcept { m_drain.disable_all(); }

void backend::enable_all() noexcept { m_drain.enable_all(); }

void backend::close_all() noexcept { m_drain.close_all(); }

std::size_t backend::tracked_tids() const noexcept
{
    return m_drain.tracked_tids();
}

bool backend::is_tracked(pid_t tid) const noexcept
{
    return m_drain.is_tracked(tid);
}

void backend::collect_tids(std::vector<pid_t>& out) const
{
    out.clear();
    m_drain.for_each_tid([&out](pid_t tid) { out.push_back(tid); });
}

}  // namespace cpu_snap::sampling::triggers::perf
