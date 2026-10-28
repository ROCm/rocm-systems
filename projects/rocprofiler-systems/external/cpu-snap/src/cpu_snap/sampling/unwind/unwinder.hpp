#pragma once

/**
 * @file unwinder.hpp
 * @brief The closed set of unwind strategies, as a std::variant.
 *
 * The sampler owns exactly one `unwinder`, chosen once at construction from
 * the requested unwind_mode and the selected trigger's capabilities, and
 * `std::visit`s it once per drain batch - never per sample - so each
 * backend's per-sample loop runs against a single concrete alternative
 * (concrete, inlinable, no virtual call, no std::function).
 *
 * Header-only. C++20.
 */

#include <variant>

#include "cpu_snap/sampling/unwind/dwarf_unwinder.hpp"
#include "cpu_snap/sampling/unwind/fp_copy_unwinder.hpp"
#include "cpu_snap/sampling/unwind/kernel_callchain_unwinder.hpp"
#include "cpu_snap/sampling/unwind/machine_snapshot.hpp"

namespace cpu_snap::sampling::unwind
{

/** @brief Closed set; std::visit once per drain batch, never per sample. */
using unwinder = std::variant<kernel_callchain_unwinder, fp_copy_unwinder, dwarf_unwinder>;

static_assert(Unwinder<kernel_callchain_unwinder>);
static_assert(Unwinder<fp_copy_unwinder>);
static_assert(Unwinder<dwarf_unwinder>);

}  // namespace cpu_snap::sampling::unwind
