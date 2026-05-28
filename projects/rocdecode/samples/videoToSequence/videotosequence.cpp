/*
Copyright (c) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <iomanip>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <thread>
#include <functional>
#include <sys/stat.h>
#include <libgen.h>

#if __cplusplus >= 201703L && __has_include(<filesystem>)
    #include <filesystem>
#else
    #include <experimental/filesystem>
#endif
#include "video_demuxer.h"
#include "roc_video_dec.h"
#include "common.h"

class ThreadPool {
    public:
        ThreadPool(int nthreads) : shutdown_(false) {
            // Create the specified number of threads
            threads_.reserve(nthreads);
            for (int i = 0; i < nthreads; ++i)
                threads_.emplace_back(std::bind(&ThreadPool::ThreadEntry, this, i));
        }

        ~ThreadPool() {}

        void JoinThreads() {
            {
                // Unblock any threads and tell them to stop
                std::unique_lock<std::mutex> lock(mutex_);
                shutdown_ = true;
                cond_var_.notify_all();
            }

            // Wait for all threads to stop
            for (auto& thread : threads_)
                thread.join();
        }

        void ExecuteJob(std::function<void()> func) {
            // Place a job on the queue and unblock a thread
            std::unique_lock<std::mutex> lock(mutex_);
            decode_jobs_queue_.emplace(std::move(func));
            cond_var_.notify_one();
        }

    protected:
        void ThreadEntry(int i) {
            std::function<void()> execute_decode_job;

            while (true) {
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cond_var_.wait(lock, [&] {return shutdown_ || !decode_jobs_queue_.empty();});
                    if (decode_jobs_queue_.empty()) {
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

        std::mutex mutex_;
        std::condition_variable cond_var_;
        bool shutdown_;
        std::queue<std::function<void()>> decode_jobs_queue_;
        std::vector<std::thread> threads_;
};

struct DecoderInfo {
    int dec_device_id;
    std::unique_ptr<RocVideoDecoder> viddec;
    std::uint32_t bit_depth;
    rocDecVideoCodec rocdec_codec_id;
    std::atomic_bool decoding_complete;

    DecoderInfo() : dec_device_id(0), viddec(nullptr), bit_depth(8) , decoding_complete(false) {}
};

struct SeqInfo {
    int batch_size;     // seq_info.batch_size: #of sequences in output
    int seq_length;     // length of each sequence: #frames per seq
    int step;           // step in number of frames to skip from one sequence to next
    int stride;         // stride in muber of frames to skip between consecutive frames in a seq
};

void DecProc(RocVideoDecoder *p_dec, VideoDemuxer *demuxer, int *pn_frame, double *pn_fps, std::atomic_bool &decoding_complete, int &seek_mode, bool &b_dump_output_frames, SeqInfo &seq_info, std::string *p_output_file_name, OutputSurfaceMemoryType mem_type) {
    
    int n_video_bytes = 0, n_frame_returned = 0;
    int64_t n_frame = 0;
    uint8_t *p_video = nullptr, *p_frame = nullptr;
    int64_t pts = 0;
    double total_dec_time = 0.0;
    int seq_id = 0;
    OutputSurfaceInfo *surf_info;
    VideoSeekContext video_seek_ctx;
    int seq_frame_start[seq_info.batch_size];
    seq_frame_start[0] = 0;
    for (int i = 1; i < seq_info.batch_size; i++) {
        seq_frame_start[i] = seq_frame_start[i-1] +  (seq_info.seq_length - 1) * seq_info.stride + seq_info.step;
        //std::cout << "seq: " << i << " seq_start: " << seq_frame_start[i] << std::endl;
    }
    auto start_time = std::chrono::high_resolution_clock::now();
    int n_frames_skipped = 0, n_frame_seq = 0, num_seq = 0;
    int next_frame_num = 0;
    bool seq_start = true;
    std::string seq_output_file_name = p_output_file_name[num_seq];
    //set reconfig before decode start
    ReconfigParams reconfig_params = { 0 };
    ReconfigDumpFileStruct reconfig_user_struct = { 0 };
    reconfig_params.p_fn_reconfigure_flush = ReconfigureFlushCallback;
    reconfig_user_struct.b_dump_frames_to_file = false;
    reconfig_params.reconfig_flush_mode = RECONFIG_FLUSH_MODE_NONE;
    reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;
    p_dec->SetReconfigParams(&reconfig_params, true); // force reconfig flush mode

    do {
        if (seek_mode && seq_start) {
            // todo:: reconfigure before seeking
            video_seek_ctx.seek_frame_ = seq_frame_start[num_seq];
            video_seek_ctx.seek_crit_ = SEEK_CRITERIA_FRAME_NUM;
            video_seek_ctx.seek_mode_ = SEEK_MODE_PREV_KEY_FRAME;
            demuxer->Seek(video_seek_ctx, &p_video, &n_video_bytes);
            pts = video_seek_ctx.out_frame_pts_;
            n_frame = static_cast<int64_t> (pts * demuxer->GetFrameRate());     // start frame number
            seq_start = false;
            p_dec->FlushAndReconfigure();
        } else {
            demuxer->Demux(&p_video, &n_video_bytes, &pts);
        }
        n_frame_returned = p_dec->DecodeFrame(p_video, n_video_bytes, 0, pts);
        if (b_dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED) {
            if (!n_frame && !p_dec->GetOutputSurfaceInfo(&surf_info)) {
                std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                break;
            }
            for (int i = 0; i < n_frame_returned; i++) {
                if ((n_frame + i) == next_frame_num) {
                    p_frame = p_dec->GetFrame(&pts);
                    if (n_frame_seq < seq_info.seq_length) {
                        p_dec->SaveFrameToFile(seq_output_file_name, p_frame, surf_info);
                        //std::cout << "saving " << next_frame_num << " to " << seq_output_file_name << std::endl;
                        n_frame_seq ++;
                    }
                    p_dec->ReleaseFrame(pts);
                    next_frame_num += seq_info.stride;
                }else {
                    p_frame = p_dec->GetFrame(&pts);
                    p_dec->ReleaseFrame(pts);
                }
            }
        } 
        n_frame += n_frame_returned;
        if (n_frame_seq >= seq_info.seq_length) {
            n_frame_seq = 0; //reset for next sequence
            seq_start = true;
            num_seq ++;
            if (num_seq < seq_info.batch_size) {
                next_frame_num = seq_frame_start[num_seq];
                seq_output_file_name = p_output_file_name[num_seq];
            }
            p_dec->ResetSaveFrameToFile();
            // needed to flush last sequence frames before decoding the next sequence by passing EOS to parser
            n_frame_returned = p_dec->DecodeFrame(nullptr, 0, ROCDEC_PKT_ENDOFSTREAM, -1);
        }
    } while (n_video_bytes && num_seq < seq_info.batch_size);
    
    //n_frame += p_dec->GetNumOfFlushedFrames();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_per_decode = std::chrono::duration<double, std::milli>(end_time - start_time).count();

    // Calculate average decoding time
    total_dec_time = time_per_decode;
    double average_decoding_time = total_dec_time / n_frame;
    double n_fps = 1000 / average_decoding_time;
    *pn_fps = n_fps;
    *pn_frame = n_frame;
    p_dec->ResetSaveFrameToFile();
    decoding_complete = true;
}


void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input File / Folder Path - required" << std::endl
    << "-o Output folder to dump sequences - dumps output if requested; optional" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
    << "-b seq_info.batch_size - specify the number of sequences to be decoded; (default: all sequences till eof)" << std::endl
    << "-step - frame interval between each sequence; (default: sequence length)" << std::endl
    << "-stride - distance between consective frames in a sequence; (default: 1)" << std::endl
    << "-l - Number of frames in each sequence; (default: 3)" << std::endl
    << "-crop crop rectangle for output (not used when using interopped decoded frame); optional; default: 0" << std::endl
    << "-seek_mode option for seeking (0: no seek 1: seek to prev key frame); optional; default: 0" << std::endl
    << "-m output_surface_memory_type - decoded surface memory; optional; default - 0"
    << " [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]" << std::endl
    << "-disp_delay -specify the number of frames to be delayed for display; optional; default: 1" << std::endl;
    exit(0);
}
// input_folder_path, output_folder_path, device_id, n_threads, seq_info, seek_mode, mem_type, argc, argv
void ParseCommandLine(std::string &input_folder_path, std::string &output_folder_path, int &device_id, int &n_thread, SeqInfo &seq_info, int &seek_mode, 
                bool &b_dump_output_frames, OutputSurfaceMemoryType &mem_type, int &disp_delay, int argc, char *argv[]) {
    // Parse command-line arguments
    if(argc <= 1) {
        ShowHelpAndExit();
    }
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!strcmp(argv[i], "-i")) {
            if (++i == argc) {
                ShowHelpAndExit("-i");
            }
            input_folder_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-t")) {
            if (++i == argc) {
                ShowHelpAndExit("-t");
            }
            n_thread = atoi(argv[i]);
            if (n_thread <= 0 || n_thread > 64) {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            if (device_id < 0) {
                ShowHelpAndExit(argv[i]);
            }
            continue;
        }
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_folder_path = argv[i];
            if (!output_folder_path.empty()) {
#if __cplusplus >= 201703L && __has_include(<filesystem>)
                if (std::filesystem::is_directory(output_folder_path)) {
                    std::filesystem::remove_all(output_folder_path);
                }
                std::filesystem::create_directory(output_folder_path);
#else
                if (std::experimental::filesystem::is_directory(output_folder_path)) {
                    std::experimental::filesystem::remove_all(output_folder_path);
                }
                std::experimental::filesystem::create_directory(output_folder_path);
#endif
                b_dump_output_frames = true;
            }
            continue;
        }
        if (!strcmp(argv[i], "-m")) {
            if (++i == argc) {
                ShowHelpAndExit("-m");
            }
            mem_type = static_cast<OutputSurfaceMemoryType>(atoi(argv[i]));
            continue;
        }
        if (!strcmp(argv[i], "-b")) {
            if (++i == argc) {
                ShowHelpAndExit("-b");
            }
            seq_info.batch_size = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-l")) {
            if (++i == argc) {
                ShowHelpAndExit("-l");
            }
            seq_info.seq_length = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-step")) {
            if (++i == argc) {
                ShowHelpAndExit("-step");
            }
            seq_info.step = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-stride")) {
            if (++i == argc) {
                ShowHelpAndExit("-stride");
            }
            seq_info.stride = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-seek_mode")) {
            if (++i == argc) {
                ShowHelpAndExit("-seek_mode");
            }
            seek_mode = atoi(argv[i]);
            if (seek_mode != 0 && seek_mode != 1)
                ShowHelpAndExit("-seek_mode");
            continue;
        }
        if (!strcmp(argv[i], "-disp_delay")) {
            if (++i == argc) {
                ShowHelpAndExit("-disp_delay");
            }
            disp_delay = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
}


int main(int argc, char **argv) {

    std::string input_folder_path, output_folder_path;
    int dump_output_frames = 0;
    int device_id = 0, num_files = 0, seek_mode = 0;
    SeqInfo seq_info = {4, 1, 1, 4};        //default values
    int n_threads = 1;
    int disp_delay = 1;
    bool b_extract_sei_messages = false;
    bool b_flush_frames_during_reconfig = true, b_dump_output_frames = false;
    Rect *p_crop_rect = nullptr;            // specify crop_rect if output cropping is needed
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;      // set to internal

    uint32_t num_decoded_frames = 0;  // default value is 0, meaning decode the entire stream
    std::vector<std::string> input_file_names;

    ParseCommandLine(input_folder_path, output_folder_path, device_id, n_threads, seq_info, seek_mode, b_dump_output_frames, mem_type, disp_delay, argc, argv);

    try {

#if __cplusplus >= 201703L && __has_include(<filesystem>)
        for (const auto& entry : std::filesystem::directory_iterator(input_folder_path)) {
#else
        for (const auto& entry : std::experimental::filesystem::directory_iterator(input_folder_path)) {
#endif
            input_file_names.push_back(entry.path());
            num_files++;
        }
        n_threads = ((n_threads > num_files) ? num_files : n_threads);
        std::vector<std::string> output_seq_file_names;
        output_seq_file_names.resize(seq_info.batch_size * num_files);
        int num_devices = 0, sd = 0;
        hipError_t hip_status = hipSuccess;
        hipDeviceProp_t hip_dev_prop;
        std::string gcn_arch_name;
        if (hipGetDeviceCount(&num_devices) != hipSuccess) {
            std::cout << "ERROR: hipGetDeviceCount failed! (" << hip_status << ")" << std::endl;
            return -1;
        }
        if (num_devices < 1) {
            ROCDEC_ERR("ERROR: didn't find any GPU!");
            return -1;
        }

        if (hipSuccess != hipGetDeviceProperties(&hip_dev_prop, device_id)) {
            ROCDEC_ERR("ERROR: hipGetDeviceProperties for device (" +TOSTR(device_id) + " ) failed! (" + hipGetErrorName(hip_status) + ")" );
            return -1;
        }

        gcn_arch_name = hip_dev_prop.gcnArchName;
        std::size_t pos = gcn_arch_name.find_first_of(":");
        std::string gcn_arch_name_base = (pos != std::string::npos) ? gcn_arch_name.substr(0, pos) : gcn_arch_name;

        // gfx90a has two GCDs as two separate devices 
        if (!gcn_arch_name_base.compare("gfx90a") && num_devices > 1) {
            sd = 1;
        }

        std::string device_name;
        int pci_bus_id, pci_domain_id, pci_device_id;
        double total_fps = 0;
        int n_total = 0;
        std::vector<double> v_fps;
        std::vector<int> v_frame;
        v_fps.resize(num_files, 0);
        v_frame.resize(num_files, 0);
        int hip_vis_dev_count = 0;
        GetEnvVar("HIP_VISIBLE_DEVICES", hip_vis_dev_count);

        std::vector<std::unique_ptr<VideoDemuxer>> v_demuxer;
        std::vector<std::unique_ptr<DecoderInfo>> v_dec_info;
        ThreadPool thread_pool(n_threads);
        std::mutex mutex;

        for (int i = 0; i < num_files; i++) {
            v_demuxer.push_back(std::make_unique<VideoDemuxer>(input_file_names[i].c_str()));
            std::size_t found_file = input_file_names[i].find_last_of('/');
            input_file_names[i] = input_file_names[i].substr(found_file + 1);
            if (b_dump_output_frames) {
                std::size_t found_ext = input_file_names[i].find_last_of('.');
                std::string path = output_folder_path + "/output_" + input_file_names[i].substr(0, found_ext);
                for (int n = 0; n < seq_info.batch_size; n++) {
                    output_seq_file_names[i * seq_info.batch_size + n] = path + "_seq_" + std::to_string(n) + ".yuv";
                }
            }
        }

        for (int i = 0; i < n_threads; i++) {
            v_dec_info.emplace_back(std::make_unique<DecoderInfo>());
            if (!hip_vis_dev_count) {
                if (device_id % 2 == 0) {
                    v_dec_info[i]->dec_device_id = (i % 2 == 0) ? device_id : device_id + sd;
                } else
                    v_dec_info[i]->dec_device_id = (i % 2 == 0) ? device_id - sd : device_id;
            } else {
                v_dec_info[i]->dec_device_id = i % hip_vis_dev_count;
            }
            v_dec_info[i]->rocdec_codec_id = AVCodec2RocDecVideoCodec(v_demuxer[i]->GetCodecID());
            v_dec_info[i]->bit_depth = v_demuxer[i]->GetBitDepth();
            v_dec_info[i]->viddec = std::make_unique<RocVideoDecoder>(v_dec_info[i]->dec_device_id, mem_type, v_dec_info[i]->rocdec_codec_id, false, p_crop_rect, b_extract_sei_messages, disp_delay);
            v_dec_info[i]->viddec->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
            std::cout << "info: decoding " << input_file_names[i] << " using GPU device " << v_dec_info[i]->dec_device_id << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
            std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
            std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
        }

        for (int j = 0; j < num_files; j++) {
            int thread_idx = j % n_threads;
            if (j >= n_threads) {
                {
                    std::unique_lock<std::mutex> lock(mutex);
                    while (!v_dec_info[thread_idx]->decoding_complete)
                        sleep(1);
                    v_dec_info[thread_idx]->decoding_complete = false;
                }
                uint32_t bit_depth = v_demuxer[j]->GetBitDepth();
                rocDecVideoCodec codec_id = AVCodec2RocDecVideoCodec(v_demuxer[j]->GetCodecID());
                // If the codec_type or bit_depth has changed, recreate the decoder
                //if (v_dec_info[thread_idx]->bit_depth != bit_depth || v_dec_info[thread_idx]->rocdec_codec_id != codec_id) {
                    (v_dec_info[thread_idx]->viddec).release();
                    v_dec_info[thread_idx]->viddec = std::make_unique<RocVideoDecoder>(v_dec_info[thread_idx]->dec_device_id, mem_type, codec_id, false, p_crop_rect, b_extract_sei_messages, disp_delay);
                //}
                v_dec_info[thread_idx]->viddec->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
                std::cout << "info: decoding " << input_file_names[j] << " using GPU device " << v_dec_info[thread_idx]->dec_device_id << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
                std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
                std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
            }
            if (!v_dec_info[thread_idx]->viddec->CodecSupported(v_dec_info[thread_idx]->dec_device_id, v_dec_info[thread_idx]->rocdec_codec_id, v_dec_info[thread_idx]->bit_depth)) {
                std::cerr << "Codec not supported on GPU, skipping this file!" << std::endl;
                continue;
            }
            thread_pool.ExecuteJob(std::bind(DecProc, v_dec_info[thread_idx]->viddec.get(), v_demuxer[j].get(), &v_frame[j], &v_fps[j], std::ref(v_dec_info[thread_idx]->decoding_complete), 
                                    seek_mode, b_dump_output_frames, seq_info, &output_seq_file_names[j*seq_info.batch_size], mem_type));
        }

        thread_pool.JoinThreads();
        for (int i = 0; i < num_files; i++) {
            total_fps += v_fps[i] * static_cast<double>(n_threads) / static_cast<double>(num_files);
            n_total += v_frame[i];
        }
        if (!b_dump_output_frames) {
            std::cout << "info: Total frame decoded: " << n_total  << std::endl;
            std::cout << "info: avg decoding time per frame: " << 1000 / total_fps << " ms" << std::endl;
            std::cout << "info: avg FPS: " << total_fps  << std::endl;
        }
        else {
            if (mem_type == OUT_SURFACE_MEM_NOT_MAPPED) {
                std::cout << "info: saving frames with -m 3 option is not supported!" << std::endl;
            } else {
                for (int i = 0; i < num_files; i++){
                    for (int n = 0; n < seq_info.batch_size; n++) {
                        std::cout << "info: saved frames into " << output_seq_file_names[i * seq_info.batch_size + n] << std::endl;
                    }
                }
            }
        }

    } catch (const std::exception &ex) {
      std::cout << ex.what() << std::endl;
      exit(1);
    }


    return 0;
}
