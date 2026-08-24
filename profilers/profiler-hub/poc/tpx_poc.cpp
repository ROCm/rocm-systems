// Perfetto trace_processor POC
// Demonstrates the CreateInstance -> ReadTrace -> ExecuteQuery -> Iterator
// call sequence against a Chrome JSON trace file.
// Build only when PROFILER_HUB_ENABLE_TPX is set.

#include <cstdio>
#include <cstdlib>
#include <string>

#include "perfetto/trace_processor/read_trace.h"
#include "perfetto/trace_processor/trace_processor.h"

int
main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr, "usage: tpx_poc <trace_file>\n");
        return 1;
    }
    const char* trace_path = argv[1];

    perfetto::trace_processor::Config cfg;
    auto tp = perfetto::trace_processor::TraceProcessor::CreateInstance(cfg);
    if(!tp)
    {
        fprintf(stderr, "TraceProcessor::CreateInstance failed\n");
        return 1;
    }

    auto status = perfetto::trace_processor::ReadTrace(tp.get(), trace_path);
    if(!status.ok())
    {
        fprintf(stderr, "ReadTrace(%s): %s\n", trace_path, status.message().c_str());
        return 1;
    }

    // Query the slice table — the primary interval table in trace_processor.
    auto it = tp->ExecuteQuery("SELECT ts, dur, name FROM slice ORDER BY ts LIMIT 10");
    int  row_count = 0;
    while(it.Next())
    {
        int64_t     ts       = it.Get(0).AsLong();
        int64_t     dur      = it.Get(1).AsLong();
        auto        name_val = it.Get(2);
        std::string name     = name_val.is_null() ? "<null>" : name_val.AsString();
        printf("slice row %d: ts=%lld dur=%lld name=%s\n",
               row_count,
               (long long) ts,
               (long long) dur,
               name.c_str());
        ++row_count;
    }
    if(!it.Status().ok())
    {
        fprintf(stderr, "ExecuteQuery error: %s\n", it.Status().message().c_str());
        return 1;
    }

    if(row_count == 0)
    {
        fprintf(stderr, "ERROR: no rows returned from slice table\n");
        return 1;
    }
    printf("POC PASS: %d slice row(s) read from %s\n", row_count, trace_path);
    return 0;
}
