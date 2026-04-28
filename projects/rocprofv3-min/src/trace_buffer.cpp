#include "trace_buffer.h"

#include <filesystem>

namespace rocprofv3_min {

TraceBuffers& TraceBuffers::instance() {
    static TraceBuffers inst;
    return inst;
}

void TraceBuffers::push_hip(HipApiRow&& row) {
    std::lock_guard<std::mutex> lk(hip_mu_);
    hip_rows_.emplace_back(std::move(row));
}
void TraceBuffers::push_kernel(KernelRow&& row) {
    std::lock_guard<std::mutex> lk(kern_mu_);
    kern_rows_.emplace_back(std::move(row));
}
void TraceBuffers::push_copy(CopyRow&& row) {
    std::lock_guard<std::mutex> lk(copy_mu_);
    copy_rows_.emplace_back(std::move(row));
}

static void open_csv(std::ofstream& out, const std::string& path) {
    out.open(path, std::ios::out | std::ios::trunc);
}

void TraceBuffers::flush(const std::string& out_dir, uint32_t pid) {
    namespace fs = std::filesystem;
    if (!out_dir.empty()) {
        std::error_code ec;
        fs::create_directories(out_dir, ec);
    }
    auto p = [&](const char* leaf) {
        std::string base = out_dir.empty() ? std::string(".") : out_dir;
        return base + "/rocprofv3_" + std::to_string(pid) + "_" + leaf;
    };

    {
        std::lock_guard<std::mutex> lk(hip_mu_);
        std::ofstream out;
        open_csv(out, p("hip_api_trace.csv"));
        if (out.is_open()) {
            out << "Domain,Function,Process_Id,Thread_Id,Correlation_Id,Start_Timestamp,End_Timestamp\n";
            for (auto& r : hip_rows_) {
                out << r.domain << "," << r.function_name << "," << r.process_id << ","
                    << r.thread_id << "," << r.correlation_id << "," << r.start_ns << ","
                    << r.end_ns << "\n";
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(kern_mu_);
        std::ofstream out;
        open_csv(out, p("kernel_trace.csv"));
        if (out.is_open()) {
            out << "Kind,Agent_Id,Queue_Id,Thread_Id,Dispatch_Id,Kernel_Id,Kernel_Name,"
                << "Correlation_Id,Start_Timestamp,End_Timestamp,Private_Segment_Size,"
                << "Group_Segment_Size,Workgroup_Size_X,Workgroup_Size_Y,Workgroup_Size_Z,"
                << "Grid_Size_X,Grid_Size_Y,Grid_Size_Z\n";
            for (auto& r : kern_rows_) {
                out << r.kind << "," << r.agent_id << "," << r.queue_id << "," << r.thread_id
                    << "," << r.dispatch_id << "," << r.kernel_id << ","
                    << r.kernel_name << "," << r.correlation_id << "," << r.start_ns << ","
                    << r.end_ns << "," << r.private_segment_size << "," << r.group_segment_size
                    << "," << r.workgroup_size_x << "," << r.workgroup_size_y << ","
                    << r.workgroup_size_z << "," << r.grid_size_x << "," << r.grid_size_y << ","
                    << r.grid_size_z << "\n";
            }
        }
    }
    {
        std::lock_guard<std::mutex> lk(copy_mu_);
        std::ofstream out;
        open_csv(out, p("memory_copy_trace.csv"));
        if (out.is_open()) {
            out << "Kind,Direction,Source_Agent_Id,Destination_Agent_Id,Correlation_Id,"
                << "Start_Timestamp,End_Timestamp,Bytes\n";
            for (auto& r : copy_rows_) {
                out << r.kind << "," << r.direction << "," << r.source_agent_id << ","
                    << r.destination_agent_id << "," << r.correlation_id << "," << r.start_ns
                    << "," << r.end_ns << "," << r.bytes << "\n";
            }
        }
    }
}

} // namespace rocprofv3_min
