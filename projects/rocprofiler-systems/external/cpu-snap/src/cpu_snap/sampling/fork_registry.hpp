#pragma once

/**
 * @file fork_registry.hpp
 * @brief Process-wide list of objects that must tear down their child-side
 *        state after fork().
 */

namespace cpu_snap::sampling::internal::fork_registry
{

/** @brief Callback run in the forked child for each registered object. */
using child_callback = void (*)(void* object) noexcept;

/**
 * @brief Register @p object; @p on_child runs in the forked child.
 *
 * The first call installs the pthread_atfork(child) handler, so the handler
 * is queued after every atfork handler registered earlier (for example the
 * signal backend's, which must run first).
 */
void
add(void* object, child_callback on_child);

/** @brief Unregister @p object; a no-op when it is not registered. */
void
remove(void* object) noexcept;

}  // namespace cpu_snap::sampling::internal::fork_registry
