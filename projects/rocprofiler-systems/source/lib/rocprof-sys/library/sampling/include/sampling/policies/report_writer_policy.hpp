// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// ============================================================================
// ReportWriterPolicy — named requirement
// ============================================================================
// Required member functions:
//   void write_timer_samples(int64_t tid,
//                            std::vector<TimerSample> const& samples);
//       Accumulate timer samples for tid into the report. Called once per
//       thread per post_process() invocation.
//   void write_overflow_samples(int64_t tid,
//                               std::vector<OverflowSample> const& samples);
//       Accumulate overflow samples for tid into the report.
//   void flush();
//       Finalize and write all output files. Called once at the end of
//       post_process(). After flush(), the writer is in an undefined state.
//
// Production: rocprofsys::sampling::NativeReportWriter
//             - emits sampling_wall_clock.{txt,json}, sampling_cpu_clock.{txt,json},
//               sampling_percent.{txt,json}, hw_counters.{txt,json} (PAPI only)
//             - trip_count column appended per row
//             - same field names, column order, units, precision as timemory output
//             - emits L43 with "native report" (the single permitted log change)
// Test double: rocprofsys::sampling::test::NoopReportWriter
//             - counts per-tid write calls; no file I/O

namespace rocprofsys::sampling
{
class native_report_writer;
}
namespace rocprofsys::sampling::test
{
struct noop_report_writer;
}
