#!/usr/bin/env python3
"""Make an instrumented copy of Kurt's hipfileread-cpu-visibility.cpp (ab0761b4c3).

Changes (only active when the PROBE_* env vars are set; otherwise behaviour is Kurt's):
  PROBE_PRE / PROBE_POST  comma-separated interventions from interventions.h, applied right
                          before / right after hipFileRead (before Succeeded is stored)
  PROBE_COUNT=1           do not fail fast on a stale watcher: spin until the new value appears
                          (<=1 s), record lateness relative to the Succeeded store, keep going;
                          print a summary line at the end
"""
import sys

src = open(sys.argv[1]).read()


def sub(old, new, count=1):
    global src
    assert src.count(old) == count, (old, src.count(old))
    src = src.replace(old, new)


sub('#include "test-options.h"\n',
    '#include "test-options.h"\n#include "interventions.h"\n')

# globals for stats
sub('/// @param[out] passed_iterations Set to the number of iterations that passed',
    '''namespace probe {
inline const bool count_mode = std::getenv("PROBE_COUNT") != nullptr;
inline std::atomic<uint64_t> succeeded_ns{0};
inline std::mutex mtx;
inline std::vector<long> lateness;            // ns, per stale watcher event
inline std::vector<long> spreads;             // ns, per stale iteration with >=2 stale watchers
inline std::atomic<long> stale_events{0}, never_events{0};
inline long stale_iterations{0};
} // namespace probe

/// @param[out] passed_iterations Set to the number of iterations that passed''')

# per-watcher visible timestamps for the current iteration
sub('    std::barrier<> iteration_boundary{static_cast<std::ptrdiff_t>(watcher_count + 1)};\n',
    '    std::barrier<> iteration_boundary{static_cast<std::ptrdiff_t>(watcher_count + 1)};\n'
    '    std::vector<std::atomic<uint64_t>> probe_vis(watcher_count);\n'
    '    a1::Ctx probe_ctx;\n'
    '    a1::init(probe_ctx, std::getenv("PROBE_VERBOSE") != nullptr);\n'
    '    const auto probe_pre{a1::split(std::getenv("PROBE_PRE"))};\n'
    '    const auto probe_post{a1::split(std::getenv("PROBE_POST"))};\n')

sub('''                        if (state == ReadState::Succeeded) {
                            failures.add("Iteration ", iteration, ", watcher ", w, " (cache line ", line,
                                         "): observed a stale value after hipFileRead returned");
                            break;
                        }''',
    '''                        if (state == ReadState::Succeeded) {
                            if (probe::count_mode) {
                                uint32_t seen{observed};
                                while (seen != new_value &&
                                       a1::now_ns() - probe::succeeded_ns.load() < 1000000000ull) {
                                    std::atomic_thread_fence(std::memory_order_seq_cst);
                                    seen = *value;
                                }
                                const uint64_t t1{a1::now_ns()};
                                probe_vis[w].store(t1);
                                probe::stale_events++;
                                if (seen != new_value) probe::never_events++;
                                std::lock_guard g{probe::mtx};
                                probe::lateness.push_back(long(t1 - probe::succeeded_ns.load()));
                                break;
                            }
                            failures.add("Iteration ", iteration, ", watcher ", w, " (cache line ", line,
                                         "): observed a stale value after hipFileRead returned");
                            break;
                        }''')

sub('''            const auto    file_offset{static_cast<hoff_t>(((iteration + 1) % 2) * buffer_size)};
            const ssize_t nbytes{hipFileRead(handle, device_buffer, buffer_size, file_offset, 0)};''',
    '''            const auto    file_offset{static_cast<hoff_t>(((iteration + 1) % 2) * buffer_size)};
            for (auto &v : probe_vis) v.store(0);
            a1::apply_all(probe_ctx, probe_pre);
            const ssize_t nbytes{hipFileRead(handle, device_buffer, buffer_size, file_offset, 0)};''')

sub('''            const int     read_errno{errno};
''', '''            const int     read_errno{errno};
            (void)hipGetLastError(); // hipFileRead leaves a sticky invalid-argument error
            a1::apply_all(probe_ctx, probe_post);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            probe::succeeded_ns.store(a1::now_ns());
''')

sub('''            iteration_boundary.arrive_and_wait();
            if (!failures.any()) {
                passed_iterations++;
            }''',
    '''            iteration_boundary.arrive_and_wait();
            {
                uint64_t mn{UINT64_MAX}, mx{0};
                int      nstale{0};
                for (auto &v : probe_vis) {
                    const uint64_t t{v.load()};
                    if (t) { nstale++; mn = std::min(mn, t); mx = std::max(mx, t); }
                }
                if (nstale) probe::stale_iterations++;
                if (nstale >= 2) probe::spreads.push_back(long(mx - mn));
            }
            if (!failures.any()) {
                passed_iterations++;
            }''')

sub('''        done.store(true, std::memory_order_relaxed);
        iteration_boundary.arrive_and_wait();
    }
''',
    '''        done.store(true, std::memory_order_relaxed);
        iteration_boundary.arrive_and_wait();
    }
    if (probe::count_mode) {
        auto pct = [](std::vector<long> v, double p) {
            if (v.empty()) return 0L;
            std::sort(v.begin(), v.end());
            return v[size_t(p * double(v.size() - 1))] / 1000;
        };
        std::cout << "PROBE pre=" << (std::getenv("PROBE_PRE") ? std::getenv("PROBE_PRE") : "-")
                  << " post=" << (std::getenv("PROBE_POST") ? std::getenv("PROBE_POST") : "-")
                  << " buf=" << options.buffer_size << " cpus=" << options.cpu_count
                  << ": stale iterations " << probe::stale_iterations << "/" << passed_iterations
                  << ", stale watcher-events " << probe::stale_events.load() << ", never " << probe::never_events.load()
                  << ", lateness p50 " << pct(probe::lateness, 0.5) << " us, p90 " << pct(probe::lateness, 0.9)
                  << " us, worst " << pct(probe::lateness, 1.0) << " us; spread across stale watchers p50 "
                  << pct(probe::spreads, 0.5) << " us, worst " << pct(probe::spreads, 1.0) << " us\\n";
    }
''')

open(sys.argv[2], 'w').write(src)
