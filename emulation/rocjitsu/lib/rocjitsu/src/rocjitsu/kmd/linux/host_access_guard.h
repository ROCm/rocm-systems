// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_HOST_ACCESS_GUARD_H_
#define ROCJITSU_KMD_LINUX_HOST_ACCESS_GUARD_H_

/// @file host_access_guard.h
/// @brief Turns a host fault taken inside an emulated access into a refusal.

#include <csetjmp>
#include <csignal>
#include <mutex>

#include <pthread.h>

namespace rocjitsu {

/// @brief Where a guarded access resumes when the host memory under it faults.
struct HostAccessGuardState {
  sigjmp_buf landing;
  /// @brief What si_addr reported, so the refusal can be classified afterwards.
  const void *fault_address = nullptr;
  bool armed = false;
};

/// @brief Per-thread guard state.
/// @details Per-thread because the access, the fault and the landing site are
/// all on one thread; a process-wide slot would let one thread's fault resume
/// another thread's stack frame.
inline HostAccessGuardState &host_access_guard_state() {
  static thread_local HostAccessGuardState state;
  return state;
}

/// @brief Divert a host fault back to the guarded access that caused it.
///
/// @details Called from the process's SIGSEGV/SIGBUS handler before anything
/// else looks at the fault. Returns normally when no guarded access is in
/// flight, which is every fault this does not own -- those must keep reaching
/// whatever the handler would otherwise do, or a genuine crash would be
/// swallowed and the process would carry on corrupt.
///
/// @param[in] signal The signal being handled.
/// @param[in] fault_address si_addr, kept for classifying the refusal.
inline void divert_host_fault_to_guard(int signal, const void *fault_address) {
  HostAccessGuardState &state = host_access_guard_state();
  if (!state.armed)
    return;
  state.armed = false;
  state.fault_address = fault_address;
  // The handler runs with this signal blocked, and siglongjmp here does not
  // restore the mask -- the guard deliberately does not pay for saving it on
  // every access. Unblock it explicitly, or the next fault on this thread is
  // undeliverable and the process dies on the second bad address instead of
  // reporting it.
  sigset_t unblock;
  sigemptyset(&unblock);
  sigaddset(&unblock, signal);
  pthread_sigmask(SIG_UNBLOCK, &unblock, nullptr);
  siglongjmp(state.landing, 1);
}

/// @brief The disposition that was in place before the guard took over.
/// @details Kept so every fault the guard does not own reaches whatever would
/// have handled it. Swallowing those would turn a genuine crash into a process
/// that carries on with corrupt state.
inline struct sigaction &previous_host_fault_action(int signal_number) {
  static struct sigaction segv {};
  static struct sigaction bus {};
  return signal_number == SIGBUS ? bus : segv;
}

/// @brief Absorb a fault belonging to a guarded access; pass on every other.
inline void host_fault_handler(int signal_number, siginfo_t *info, void *context) {
  // Does not return when this thread is inside a guarded access.
  divert_host_fault_to_guard(signal_number, info != nullptr ? info->si_addr : nullptr);

  const struct sigaction &previous = previous_host_fault_action(signal_number);
  if ((previous.sa_flags & SA_SIGINFO) != 0) {
    if (previous.sa_sigaction != nullptr) {
      previous.sa_sigaction(signal_number, info, context);
      return;
    }
  } else if (previous.sa_handler == SIG_IGN) {
    return;
  } else if (previous.sa_handler != SIG_DFL && previous.sa_handler != nullptr) {
    previous.sa_handler(signal_number);
    return;
  }
  // Nothing else wanted it: die the way an unhandled fault does, rather than
  // returning to the faulting instruction and looping on it forever.
  signal(signal_number, SIG_DFL);
  raise(signal_number);
}

/// @brief Make guarded accesses able to absorb host faults. Idempotent.
///
/// @details Installed by whoever performs emulated accesses rather than by the
/// interposer, because the two do not always ship together: the memory model is
/// linked by tools and tests that never preload the interposer, and a guard
/// that only worked when it happened to be present would fail exactly where a
/// bad address is most likely -- in a test.
inline void install_host_access_guard() {
  static std::once_flag once;
  std::call_once(once, [] {
    struct sigaction action {};
    action.sa_sigaction = host_fault_handler;
    // SA_ONSTACK so a fault taken on an exhausted stack still reaches this.
    action.sa_flags = SA_SIGINFO | SA_ONSTACK;
    sigemptyset(&action.sa_mask);
    sigaction(SIGSEGV, &action, &previous_host_fault_action(SIGSEGV));
    // A memfd shrunk under a live mapping reports SIGBUS, not SIGSEGV, and
    // reaches an emulated access identically.
    sigaction(SIGBUS, &action, &previous_host_fault_action(SIGBUS));
  });
}

/// @brief Run @p access, reporting whether the host memory under it faulted.
///
/// @details The alternative to this is asking the kernel, before every access,
/// whether the pages are still there and still permit it. That answer costs a
/// syscall -- a scan of /proc/self/maps for writability -- which is affordable
/// once per mapping but not once per access: a workload of many small accesses
/// measured 1.75x slower for it. It also cannot be cached, because the only
/// invalidation signal available covers mapping changes the interposer sees,
/// and one it missed would leave the cache authorising a store through a page
/// that is no longer writable.
///
/// So the question is not asked. The access is attempted, and the hardware
/// answers it: if the page is gone or refuses the access, the fault lands here
/// and becomes a refusal the caller reports as a GPU memory violation. Nothing
/// is paid on the path where the memory is fine, which is all of them but the
/// ones this exists to catch.
///
/// @warning A write that faults part-way has already stored the bytes before
/// the faulting one. That is the same partial transfer real hardware performs
/// when it walks into an unmapped page, and it is why callers report the whole
/// access as faulted rather than counting what landed.
///
/// @param[in] access Invoked once; must not itself arm a guard.
/// @returns true when @p access completed, false when it faulted.
template <typename F> [[nodiscard]] inline bool with_host_access_guard(F &&access) {
  // Nothing of this frame is held across the jump. A local live over sigsetjmp
  // has an indeterminate value after siglongjmp unless it is volatile -- the
  // compiler is free to leave it in a register the jump restores -- so the
  // state is re-fetched on the far side rather than carried there. GCC rejects
  // the carried form outright under -Werror=clobbered.
  //
  // savemask 0 on purpose: saving the signal mask costs a sigprocmask syscall
  // on every armed access, which is the cost this whole mechanism exists to
  // avoid. The handler restores what needs restoring instead.
  if (sigsetjmp(host_access_guard_state().landing, /*savemask=*/0) != 0)
    return false;
  host_access_guard_state().armed = true;
  access();
  host_access_guard_state().armed = false;
  return true;
}

/// @brief Address of the last fault this thread's guard absorbed.
inline const void *last_guarded_fault_address() { return host_access_guard_state().fault_address; }

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_HOST_ACCESS_GUARD_H_
