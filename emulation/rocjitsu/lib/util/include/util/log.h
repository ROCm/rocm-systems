// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UTIL_LOG_H_
#define UTIL_LOG_H_

#include <concepts>
#include <format>
#include <iostream>
#include <mutex>
#include <ostream>
#include <sstream>
#include <string_view>

namespace util {

/// @brief Unified logging and debug output for rocjitsu.
///
/// @details Log groups are independently enableable via a compile-time
/// bitmask (e.g., cmake -DRJ_LOG_GROUPS=VM). Debug output is controlled by
/// NDEBUG. All output is mutex-protected to prevent interleaved lines
/// from concurrent threads.
///
/// Two print overloads per group:
///
///   // Variadic fold, args evaluated eagerly:
///   Logger::vm("addr=0x", std::hex, addr, std::dec, " size=", size);
///
///   // Lambda, body only evaluated when group is enabled:
///   Logger::vm([&](auto &os) {
///     os << std::format("addr={:#x} size={}", addr, size);
///   });
///
/// The lambda form is preferred when arguments are expensive to compute
/// (e.g., hex dumps, register snapshots) since the lambda body compiles
/// to nothing when the group is disabled.
class Logger {
public:
  /// @brief Log group IDs, independently enableable via RJ_LOG_GROUPS.
  enum Group : unsigned {
    GROUP_VM = 0, ///< Kernel dispatch, instruction execution, memory access, etc.
    // Future groups added here
  };

  /// @brief Human-readable group name for log prefixes.
  static constexpr std::string_view group_name(unsigned group_id) {
    switch (group_id) {
    case GROUP_VM:
      return "VM";
    default:
      return " ";
    }
  }

  /// @brief Bitmask of enabled log groups (compile-time, via cmake).
#ifdef RJ_LOG_GROUPS
  static constexpr unsigned groups = RJ_LOG_GROUPS;
#else
  static constexpr unsigned groups = 0;
#endif

#ifndef NDEBUG
  static constexpr bool debug_enabled = true;
#else
  static constexpr bool debug_enabled = false;
#endif

  /// @brief Check if a specific group is enabled at compile time.
  static constexpr bool group_enabled(unsigned group_id) {
    return (groups & (1u << group_id)) != 0;
  }

  /// @brief Returns true when any logging is enabled.
  static constexpr bool enabled() { return groups != 0; }

  /// @brief Emit debug output. Compiled to nothing in Release builds.
  template <typename... Args> static void debug_print(Args &&...args) {
    if constexpr (debug_enabled) {
      std::ostringstream buf;
      buf << "[rj debug] ";
      (buf << ... << args);
      buf << '\n';
      emit(buf.str());
    }
  }

  /// @brief Lambda overload, body only evaluated in Debug builds.
  template <typename Fn>
    requires std::invocable<Fn, std::ostringstream &>
  static void debug_print(Fn &&fn) {
    if constexpr (debug_enabled) {
      std::ostringstream buf;
      buf << "[rj debug] ";
      fn(buf);
      buf << '\n';
      emit(buf.str());
    }
  }

  /// @brief Emit log output if the specified group is enabled.
  /// @details Compiles to nothing for disabled groups. Thread-safe via
  /// shared mutex.
  template <unsigned GroupId, typename... Args> static void print(Args &&...args) {
    if constexpr ((groups & (1u << GroupId)) != 0) {
      std::ostringstream buf;
      buf << std::format("[rj log {}] ", group_name(GroupId));
      (buf << ... << args);
      buf << '\n';
      emit(buf.str());
    }
  }

  /// @brief Lambda overload, body only evaluated when group is enabled.
  /// @details Preferred when arguments are expensive to prepare.
  template <unsigned GroupId, typename Fn>
    requires std::invocable<Fn, std::ostringstream &>
  static void print(Fn &&fn) {
    if constexpr ((groups & (1u << GroupId)) != 0) {
      std::ostringstream buf;
      auto pos_before = buf.tellp();
      fn(buf);
      if (buf.tellp() == pos_before)
        return;
      std::ostringstream out;
      out << std::format("[rj log {}] ", group_name(GroupId));
      out << buf.str() << '\n';
      emit(out.str());
    }
  }

  /// @brief Emit a warning unconditionally (always enabled, not group-gated).
  template <typename... Args> static void warn(Args &&...args) {
    std::ostringstream buf;
    buf << "[rj warn] ";
    (buf << ... << args);
    buf << '\n';
    emit(buf.str());
  }

  /// @brief Convenience for logging in GROUP_VM, variadic.
  template <typename... Args> static void vm(Args &&...args) {
    print<GROUP_VM>(std::forward<Args>(args)...);
  }

  /// @brief Convenience for logging in GROUP_VM, lambda.
  template <typename Fn>
    requires std::invocable<Fn, std::ostringstream &>
  static void vm(Fn &&fn) {
    print<GROUP_VM>(std::forward<Fn>(fn));
  }

private:
  static std::mutex &mutex() {
    static std::mutex m;
    return m;
  }

  static void emit(const std::string &s) {
    std::lock_guard<std::mutex> lock(mutex());
    std::cerr << s;
  }
};

} // namespace util

#endif // UTIL_LOG_H_
