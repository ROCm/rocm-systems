#include "cpu_snap/sampling/fork_registry.hpp"

#include <mutex>
#include <unordered_map>

#include <pthread.h>

namespace cpu_snap::sampling::internal::fork_registry
{

namespace
{

std::mutex&
registry_mutex() noexcept
{
    static std::mutex mu;
    return mu;
}

std::unordered_map<void*, child_callback>&
registry() noexcept
{
    static std::unordered_map<void*, child_callback> map;
    return map;
}

void
child_handler() noexcept
{
    // pthread_atfork(child) runs single-threaded in the forked child.
    // Do NOT acquire registry_mutex() here: if the parent held it at
    // fork time, the mutex is locked in the child too and lock_guard
    // would deadlock or throw. Iterate the registry directly; the
    // body is noexcept by construction (callbacks are noexcept,
    // unordered_map::clear/begin/end don't throw).
    for(const auto& [object, on_child] : registry())
    {
        on_child(object);
    }
    registry().clear();
}

void
ensure_atfork_registered() noexcept
{
    static const int once = []() {
        ::pthread_atfork(nullptr, nullptr, &child_handler);
        return 0;
    }();
    (void) once;
}

}  // namespace

void
add(void* object, child_callback on_child)
{
    ensure_atfork_registered();
    const std::lock_guard<std::mutex> lk(registry_mutex());
    registry().insert_or_assign(object, on_child);
}

void
remove(void* object) noexcept
{
    // noexcept: if mutex::lock throws system_error here, the runtime
    // calls std::terminate. That is the correct behavior - if the
    // registry mutex is poisoned, process state is already lost; do
    // not silently swallow.
    const std::lock_guard<std::mutex> lk(registry_mutex());
    registry().erase(object);
}

}  // namespace cpu_snap::sampling::internal::fork_registry
