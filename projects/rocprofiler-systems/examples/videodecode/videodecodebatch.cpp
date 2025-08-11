/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <iomanip>
#include <iostream>
#include <libgen.h>
#include <mutex>
#include <queue>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>
#if __cplusplus >= 201703L && __has_include(<filesystem>)
#    include <filesystem>
#else
#    include <experimental/filesystem>
#endif
#include "common.h"
#include "roc_video_dec.h"
#include "video_demuxer.h"

class ThreadPool
{
public:
    ThreadPool(int nthreads)
    : shutdown_(false)
    {
        // Create the specified number of threads
        threads_.reserve(nthreads);
        for(int i = 0; i < nthreads; ++i)
            threads_.emplace_back(std::bind(&ThreadPool::ThreadEntry, this, i));
    }

    ~ThreadPool() {}

    void JoinThreads()
    {
        {
            // Unblock any threads and tell them to stop
            std::unique_lock<std::mutex> lock(mutex_);
            shutdown_ = true;
            cond_var_.notify_all();
        }

        // Wait for all threads to stop
        for(auto& thread : threads_)
            thread.join();
    }

    void ExecuteJob(std::function<void()> func)
    {
        // Place a job on the queue and unblock a thread
        std::unique_lock<std::mutex> lock(mutex_);
        decode_jobs_queue_.emplace(std::move(func));
        cond_var_.notify_one();
    }

protected:
    void ThreadEntry(int i)
    {
        std::function<void()> execute_decode_job;

        while(true)
        {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cond_var_.wait(lock,
                               [&] { return shutdown_ || !decode_jobs_queue_.empty(); });
                if(decode_jobs_queue_.empty())
                {
                    // No jobs to do; shutting down
                    return;
                }

                execute_decode_job = std::move(decode_jobs_queue_.front());
                decode_jobs_queue_.pop();
            }

            // Execute the decode job without holding any locks
            execute_decode_job();
        }
    }

    std::mutex                        mutex_;
    std::condition_variable           cond_var_;
    bool                              shutdown_;
    std::queue<std::function<void()>> decode_jobs_queue_;
    std::vector<std::thread>          threads_;
};

struct DecoderInfo
{
    int                              dec_device_id;
    std::unique_ptr<RocVideoDecoder> viddec;
    std::uint32_t                    bit_depth;
    rocDecVideoCodec                 rocdec_codec_id;
    std::atomic_bool                 decoding_complete;

    DecoderInfo()
    : dec_device_id(0)
    , viddec(nullptr)
    , bit_depth(8)
    , decoding_complete(false)
    {}
};

void
DecProc(RocVideoDecoder* p_dec, VideoDemuxer* demuxer, int* pn_frame, double* pn_fps,
        std::atomic_bool& decoding_complete, bool& b_dump_output_frames,
        std::string& output_file_name, OutputSurfaceMemoryType mem_type)
{
    int                n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
    uint8_t *          p_video = nullptr, *p_frame = nullptr;
    int64_t            pts            = 0;
    double             total_dec_time = 0.0;
    OutputSurfaceInfo* surf_info;
    auto               start_time = std::chrono::high_resolution_clock::now();
    do
    {
        demuxer->Demux(&p_video, &n_video_bytes, &pts);
        n_frame_returned = p_dec->DecodeFrame(p_video, n_video_bytes, 0, pts);
        n_frame += n_frame_returned;
        if(b_dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED)
        {
            if(n_frame_returned)
            {
                if(!p_dec->GetOutputSurfaceInfo(&surf_info))
                {
                    std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                    break;
                }
            }
            for(int i = 0; i < n_frame_returned; i++)
            {
                p_frame = p_dec->GetFrame(&pts);
                p_dec->SaveFrameToFile(output_file_name, p_frame, surf_info);
                // release frame
                p_dec->ReleaseFrame(pts);
            }
        }
    } while(n_video_bytes);
    n_frame += p_dec->GetNumOfFlushedFrames();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_per_decode =
        std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Calculate average decoding time
    total_dec_time               = time_per_decode;
    double average_decoding_time = total_dec_time / n_frame;
    double n_fps                 = 1000 / average_decoding_time;
    *pn_fps                      = n_fps;
    *pn_frame                    = n_frame;
    p_dec->ResetSaveFrameToFile();
    decoding_complete = true;
}

void
ShowHelpAndExit(const char* option = NULL)
{
    std::cout
        << "Options:" << std::endl
        << "-i <directory containing input video files [required]> " << std::endl
        << "-t Number of threads ( 1 >= n_thread <= 64) - optional; default: 4"
        << std::endl
        << "-d Device ID (>= 0)  - optional; default: 0" << std::endl
        << "-o Directory for output YUV files - optional" << std::endl
        << "-m output_surface_memory_type - decoded surface memory; optional; default - 3"
        << " [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : "
           "OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]"
        << std::endl
        << "-disp_delay -specify the number of frames to be delayed for display; "
           "optional; default: 1"
        << std::endl;
    exit(0);
}

void
ParseCommandLine(std::string& input_folder_path, std::string& output_folder_path,
                 int& device_id, int& n_thread, bool& b_dump_output_frames,
                 OutputSurfaceMemoryType& mem_type, int& disp_delay, int argc,
                 char* argv[])
{
    // Parse command-line arguments
    if(argc <= 1)
    {
        ShowHelpAndExit();
    }
    for(int i = 1; i < argc; i++)
    {
        if(!strcmp(argv[i], "-h"))
        {
            ShowHelpAndExit();
        }
        if(!strcmp(argv[i], "-i"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-i");
            }
            input_folder_path = argv[i];
            continue;
        }
        if(!strcmp(argv[i], "-t"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-t");
            }
            n_thread = atoi(argv[i]);
            if(n_thread <= 0 || n_thread > 64)
            {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        if(!strcmp(argv[i], "-d"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            if(device_id < 0)
            {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        if(!strcmp(argv[i], "-o"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-o");
            }
            output_folder_path = argv[i];
#if __cplusplus >= 201703L && __has_include(<filesystem>)
            if(std::filesystem::is_directory(output_folder_path))
            {
                std::filesystem::remove_all(output_folder_path);
            }
            std::filesystem::create_directory(output_folder_path);
#else
            if(std::experimental::filesystem::is_directory(output_folder_path))
            {
                std::experimental::filesystem::remove_all(output_folder_path);
            }
            std::experimental::filesystem::create_directory(output_folder_path);
#endif
            b_dump_output_frames = true;
            continue;
        }
        if(!strcmp(argv[i], "-m"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-m");
            }
            mem_type = static_cast<OutputSurfaceMemoryType>(atoi(argv[i]));
            continue;
        }
        if(!strcmp(argv[i], "-disp_delay"))
        {
            if(++i == argc)
            {
                ShowHelpAndExit("-disp_delay");
            }
            disp_delay = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}

int
main(int argc, char** argv)
{
    std::string             input_folder_path, output_folder_path;
    int                     device_id = 0, num_files = 0;
    int                     n_thread               = 4;
    int                     disp_delay             = 1;
    Rect*                   p_crop_rect            = nullptr;
    bool                    b_extract_sei_messages = false;
    OutputSurfaceMemoryType mem_type =
        OUT_SURFACE_MEM_DEV_INTERNAL;  // set to decode only for performance
    bool                     b_force_zero_latency = false, b_dump_output_frames = false;
    std::vector<std::string> input_file_names;
    ParseCommandLine(input_folder_path, output_folder_path, device_id, n_thread,
                     b_dump_output_frames, mem_type, disp_delay, argc, argv);

    try
    {
#if __cplusplus >= 201703L && __has_include(<filesystem>)
        for(const auto& entry : std::filesystem::directory_iterator(input_folder_path))
        {
#else
        for(const auto& entry :
            std::experimental::filesystem::directory_iterator(input_folder_path))
        {
#endif
            input_file_names.push_back(entry.path());
            num_files++;
        }

        std::vector<std::string> output_file_names(num_files);
        n_thread                    = ((n_thread > num_files) ? num_files : n_thread);
        int             num_devices = 0, sd = 0;
        hipError_t      hip_status = hipSuccess;
        hipDeviceProp_t hip_dev_prop;
        std::string     gcn_arch_name;
        hip_status = hipGetDeviceCount(&num_devices);
        if(hip_status != hipSuccess)
        {
            std::cout << "ERROR: hipGetDeviceCount failed! (" << hip_status << ")"
                      << std::endl;
            return -1;
        }
        if(num_devices < 1)
        {
            ERR("ERROR: didn't find any GPU!");
            return -1;
        }

        hip_status = hipGetDeviceProperties(&hip_dev_prop, device_id);
        if(hip_status != hipSuccess)
        {
            ERR("ERROR: hipGetDeviceProperties for device (" + TOSTR(device_id) +
                " ) failed! (" + hipGetErrorName(hip_status) + ")");
            return -1;
        }

        gcn_arch_name   = hip_dev_prop.gcnArchName;
        std::size_t pos = gcn_arch_name.find_first_of(":");
        std::string gcn_arch_name_base =
            (pos != std::string::npos) ? gcn_arch_name.substr(0, pos) : gcn_arch_name;

        // gfx90a has two GCDs as two separate devices
        if(!gcn_arch_name_base.compare("gfx90a") && num_devices > 1)
        {
            sd = 1;
        }

        std::string         device_name;
        int                 pci_bus_id, pci_domain_id, pci_device_id;
        double              total_fps = 0;
        int                 n_total   = 0;
        std::vector<double> v_fps;
        std::vector<int>    v_frame;
        v_fps.resize(num_files, 0);
        v_frame.resize(num_files, 0);
        int hip_vis_dev_count = 0;
        GetEnvVar("HIP_VISIBLE_DEVICES", hip_vis_dev_count);

        std::cout << "info: Number of threads: " << n_thread << std::endl;

        std::vector<std::unique_ptr<VideoDemuxer>> v_demuxer(num_files);
        std::unique_ptr<RocVideoDecoder> dec_8bit_avc(nullptr), dec_8bit_hevc(nullptr),
            dec_10bit_hevc(nullptr), dec_8bit_av1(nullptr), dec_10bit_av1(nullptr),
            dec_8bit_vp9(nullptr), dec_10bit_vp9(nullptr);
        std::vector<std::unique_ptr<DecoderInfo>> v_dec_info;
        ThreadPool                                thread_pool(n_thread);

        // reconfig parameters
        ReconfigParams         reconfig_params      = { 0 };
        ReconfigDumpFileStruct reconfig_user_struct = { 0 };
        reconfig_params.p_fn_reconfigure_flush      = ReconfigureFlushCallback;
        if(!b_dump_output_frames)
        {
            reconfig_user_struct.b_dump_frames_to_file = false;
            reconfig_params.reconfig_flush_mode        = RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
        }
        else
        {
            reconfig_user_struct.b_dump_frames_to_file = true;
            reconfig_params.reconfig_flush_mode        = RECONFIG_FLUSH_MODE_NONE;
        }
        reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;

        for(int i = 0; i < num_files; i++)
        {
            std::unique_ptr<VideoDemuxer> demuxer(
                new VideoDemuxer(input_file_names[i].c_str()));
            v_demuxer[i]           = std::move(demuxer);
            std::size_t found_file = input_file_names[i].find_last_of('/');
            input_file_names[i]    = input_file_names[i].substr(found_file + 1);
            if(b_dump_output_frames)
            {
                std::size_t found_ext = input_file_names[i].find_last_of('.');
                std::string path      = output_folder_path + "/output_" +
                                   input_file_names[i].substr(0, found_ext) + ".yuv";
                output_file_names[i] = path;
            }
        }

        for(int i = 0; i < n_thread; i++)
        {
            v_dec_info.emplace_back(std::make_unique<DecoderInfo>());
            if(!hip_vis_dev_count)
            {
                if(device_id % 2 == 0)
                {
                    v_dec_info[i]->dec_device_id =
                        (i % 2 == 0) ? device_id : device_id + sd;
                }
                else
                    v_dec_info[i]->dec_device_id =
                        (i % 2 == 0) ? device_id - sd : device_id;
            }
            else
            {
                v_dec_info[i]->dec_device_id = i % hip_vis_dev_count;
            }

            v_dec_info[i]->rocdec_codec_id =
                AVCodec2RocDecVideoCodec(v_demuxer[i]->GetCodecID());
            v_dec_info[i]->bit_depth = v_demuxer[i]->GetBitDepth();
            if(v_dec_info[i]->bit_depth == 8)
            {
                if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_AVC)
                {
                    std::unique_ptr<RocVideoDecoder> dec_8bit_avc(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_8bit_avc);
                }
                else if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_HEVC)
                {
                    std::unique_ptr<RocVideoDecoder> dec_8bit_hevc(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_8bit_hevc);
                }
                else if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_AV1)
                {
                    std::unique_ptr<RocVideoDecoder> dec_8bit_av1(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_8bit_av1);
                }
                else if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_VP9)
                {
                    std::unique_ptr<RocVideoDecoder> dec_8bit_vp9(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_8bit_vp9);
                }
                else
                {
                    ERR("ERROR: codec type is not supported!");
                    return -1;
                }
            }
            else
            {  // bit depth = 10bit
                if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_HEVC)
                {
                    std::unique_ptr<RocVideoDecoder> dec_10bit_hevc(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_10bit_hevc);
                }
                else if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_AV1)
                {
                    std::unique_ptr<RocVideoDecoder> dec_10bit_av1(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_10bit_av1);
                }
                else if(v_dec_info[i]->rocdec_codec_id == rocDecVideoCodec_VP9)
                {
                    std::unique_ptr<RocVideoDecoder> dec_10bit_vp9(new RocVideoDecoder(
                        v_dec_info[i]->dec_device_id, mem_type,
                        v_dec_info[i]->rocdec_codec_id, b_force_zero_latency, p_crop_rect,
                        b_extract_sei_messages, disp_delay));
                    v_dec_info[i]->viddec = std::move(dec_10bit_vp9);
                }
                else
                {
                    ERR("ERROR: codec type is not supported!");
                    return -1;
                }
            }

            v_dec_info[i]->viddec->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id,
                                                 pci_domain_id, pci_device_id);
            std::cout << "info: decoding " << input_file_names[i] << " using GPU device "
                      << v_dec_info[i]->dec_device_id << " - " << device_name << "["
                      << gcn_arch_name << "] on PCI bus " << std::setfill('0')
                      << std::setw(2) << std::right << std::hex << pci_bus_id << ":"
                      << std::setfill('0') << std::setw(2) << std::right << std::hex
                      << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
        }

        std::mutex mutex;

        for(int j = 0; j < num_files; j++)
        {
            int thread_idx = j % n_thread;
            if(j >= n_thread)
            {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    while(!v_dec_info[thread_idx]->decoding_complete)
                        ;
                    v_dec_info[thread_idx]->decoding_complete = false;
                }
                uint32_t         bit_depth = v_demuxer[j]->GetBitDepth();
                rocDecVideoCodec codec_id =
                    AVCodec2RocDecVideoCodec(v_demuxer[j]->GetCodecID());
                if(v_dec_info[thread_idx]->bit_depth != bit_depth ||
                   v_dec_info[thread_idx]->rocdec_codec_id != codec_id)
                {
                    if(bit_depth == 8)
                    {  // can be HEVC or H.264 or AV1
                        if(dec_8bit_avc == nullptr && codec_id == rocDecVideoCodec_AVC)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_8bit_avc(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_8bit_avc);
                        }
                        else if(dec_8bit_hevc == nullptr &&
                                codec_id == rocDecVideoCodec_HEVC)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_8bit_hevc(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_8bit_hevc);
                        }
                        else if(dec_8bit_av1 == nullptr &&
                                codec_id == rocDecVideoCodec_AV1)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_8bit_av1(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_8bit_av1);
                        }
                        else if(dec_8bit_av1 == nullptr &&
                                codec_id == rocDecVideoCodec_VP9)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_8bit_vp9(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_8bit_vp9);
                        }
                        else
                        {
                            if(codec_id == rocDecVideoCodec_AVC)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_8bit_avc);
                            }
                            else if(codec_id == rocDecVideoCodec_HEVC)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_8bit_hevc);
                            }
                            else if(codec_id == rocDecVideoCodec_AV1)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_8bit_av1);
                            }
                            else if(codec_id == rocDecVideoCodec_VP9)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_8bit_vp9);
                            }
                            else
                            {
                                ERR("ERROR: codec type is not supported!");
                                return -1;
                            }
                        }
                        v_dec_info[thread_idx]->bit_depth       = bit_depth;
                        v_dec_info[thread_idx]->rocdec_codec_id = codec_id;
                    }
                    else
                    {  // bit_depth = 10bit; HEVC or AV1
                        if(dec_10bit_hevc == nullptr && codec_id == rocDecVideoCodec_HEVC)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_10bit_hevc(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_10bit_hevc);
                        }
                        else if(dec_10bit_av1 == nullptr &&
                                codec_id == rocDecVideoCodec_AV1)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_10bit_av1(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_10bit_av1);
                        }
                        else if(dec_10bit_vp9 == nullptr &&
                                codec_id == rocDecVideoCodec_VP9)
                        {
                            std::unique_ptr<RocVideoDecoder> dec_10bit_vp9(
                                new RocVideoDecoder(v_dec_info[thread_idx]->dec_device_id,
                                                    mem_type, codec_id,
                                                    b_force_zero_latency, p_crop_rect,
                                                    b_extract_sei_messages, disp_delay));
                            v_dec_info[thread_idx]->viddec = std::move(dec_10bit_vp9);
                        }
                        else
                        {
                            if(codec_id == rocDecVideoCodec_HEVC)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_10bit_hevc);
                            }
                            else if(codec_id == rocDecVideoCodec_AV1)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_10bit_av1);
                            }
                            else if(codec_id == rocDecVideoCodec_VP9)
                            {
                                v_dec_info[thread_idx]->viddec.swap(dec_10bit_vp9);
                            }
                            else
                            {
                                ERR("ERROR: codec type is not supported!");
                                return -1;
                            }
                        }
                        v_dec_info[thread_idx]->bit_depth       = bit_depth;
                        v_dec_info[thread_idx]->rocdec_codec_id = codec_id;
                    }
                }
                v_dec_info[thread_idx]->viddec->GetDeviceinfo(
                    device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
                std::cout << "info: decoding " << input_file_names[j]
                          << " using GPU device " << v_dec_info[thread_idx]->dec_device_id
                          << " - " << device_name << "[" << gcn_arch_name
                          << "] on PCI bus " << std::setfill('0') << std::setw(2)
                          << std::right << std::hex << pci_bus_id << ":"
                          << std::setfill('0') << std::setw(2) << std::right << std::hex
                          << pci_domain_id << "." << pci_device_id << std::dec
                          << std::endl;
            }
            if(!v_dec_info[thread_idx]->viddec->CodecSupported(
                   v_dec_info[thread_idx]->dec_device_id,
                   v_dec_info[thread_idx]->rocdec_codec_id,
                   v_dec_info[thread_idx]->bit_depth))
            {
                std::cerr << "Codec not supported on GPU, skipping this file!"
                          << std::endl;
                v_dec_info[thread_idx]->decoding_complete = true;
                continue;
            }
            thread_pool.ExecuteJob(
                std::bind(DecProc, v_dec_info[thread_idx]->viddec.get(),
                          v_demuxer[j].get(), &v_frame[j], &v_fps[j],
                          std::ref(v_dec_info[thread_idx]->decoding_complete),
                          b_dump_output_frames, output_file_names[j], mem_type));
        }

        thread_pool.JoinThreads();
        for(int i = 0; i < num_files; i++)
        {
            total_fps +=
                v_fps[i] * static_cast<double>(n_thread) / static_cast<double>(num_files);
            n_total += v_frame[i];
        }
        if(!b_dump_output_frames)
        {
            std::cout << "info: Total frame decoded: " << n_total << std::endl;
            std::cout << "info: avg decoding time per frame: " << 1000 / total_fps
                      << " ms" << std::endl;
            std::cout << "info: avg FPS: " << total_fps << std::endl;
        }
        else
        {
            if(mem_type == OUT_SURFACE_MEM_NOT_MAPPED)
            {
                std::cout << "info: saving frames with -m 3 option is not supported!"
                          << std::endl;
            }
            else
            {
                for(int i = 0; i < num_files; i++)
                    std::cout << "info: saved frames into " << output_file_names[i]
                              << std::endl;
            }
        }

    } catch(const std::exception& ex)
    {
        std::cout << ex.what() << std::endl;
    }

    return 0;
}
