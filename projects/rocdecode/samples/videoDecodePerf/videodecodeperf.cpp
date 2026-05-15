/*
Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <iomanip>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
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
#if ENABLE_HOST_DECODE
    #include "ffmpeg_video_dec.h"
#endif

void DecProc(RocVideoDecoder *p_dec, VideoDemuxer *demuxer, int *pn_frame, int *pn_pic_dec, double *pn_fps, double *pn_fps_dec, int max_num_frames, OutputSurfaceMemoryType mem_type) {
    int n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
    int n_pic_decoded = 0, decoded_pics = 0;
    uint8_t *p_video = nullptr;
    int64_t pts = 0;
    double total_dec_time = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    do {
        demuxer->Demux(&p_video, &n_video_bytes, &pts);
        n_frame_returned = p_dec->DecodeFrame(p_video, n_video_bytes, 0, pts, &decoded_pics);
        n_frame += n_frame_returned;
        n_pic_decoded += decoded_pics;
        if (max_num_frames && max_num_frames <= n_frame) {
            break;
        }
    } while (n_video_bytes);
    if (mem_type == OUT_SURFACE_MEM_NOT_MAPPED) {
        p_dec->WaitForDecodeCompletion();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto time_per_decode = std::chrono::duration<double, std::milli>(end_time - start_time).count();
    auto session_overhead = p_dec->GetDecoderSessionOverHead(std::this_thread::get_id());
    // Calculate average decoding time
    total_dec_time = time_per_decode - session_overhead;
    double average_output_time = total_dec_time / n_frame;
    double average_decoding_time = total_dec_time / n_pic_decoded;
    double n_fps = 1000 / average_output_time;
    double n_fps_dec = 1000 / average_decoding_time;
    *pn_fps = n_fps;
    *pn_fps_dec = n_fps_dec;
    *pn_frame = n_frame;
    *pn_pic_dec = n_pic_decoded;
}

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input File Path - required" << std::endl
    << "-t Number of threads (>= 1) - optional; default: 1" << std::endl
    << "-d Device ID (>= 0) - optional; default: 0" << std::endl
    << "-z Force zero latency (decoded frames will be flushed out for display immediately) - optional" << std::endl
    << "-disp_delay -specify the number of frames to be delayed for display; optional; default: 1" << std::endl
    << "-m Memory type (integer values between 0 to 3: specifies where to store the decoded output:" << std::endl
    << "                                               0 = decoded output will be in internal interopped memory," << std::endl
    << "                                               1 = decoded output will be copied to a separate device memory," << std::endl
    << "                                               2 = decoded output will be copied to a separate host memory," << std::endl
    << "                                               3 = decoded output will not be available (decode only)) - optional; default: 3" << std::endl;
    exit(0);
}

int main(int argc, char **argv) {

    std::string input_file_path;
    int device_id = 0;
    int n_thread = 1;
    Rect *p_crop_rect = nullptr;
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_NOT_MAPPED;        // set to decode only for performance
    bool b_force_zero_latency = false;
    uint32_t max_num_frames = 0;  // max number of frames to be decoded. default value is 0, meaning decode the entire stream
    int disp_delay = 1;
    int backend = 0;

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
            input_file_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-t")) {
            if (++i == argc) {
                ShowHelpAndExit("-t");
            }
            n_thread = atoi(argv[i]);
            if (n_thread <= 0) {
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
        if (!strcmp(argv[i], "-disp_delay")) {
            if (++i == argc) {
                ShowHelpAndExit("-disp_delay");
            }
            disp_delay = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-f")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            max_num_frames = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-z")) {
            if (i == argc) {
                ShowHelpAndExit("-z");
            }
            b_force_zero_latency = true;
            continue;
        }
        if (!strcmp(argv[i], "-m")) {
            if (++i == argc) {
                ShowHelpAndExit("-m");
            }
            mem_type = static_cast<OutputSurfaceMemoryType>(atoi(argv[i]));
            continue;
        }
        if (!strcmp(argv[i], "-backend")) {
            if (++i == argc) {
                ShowHelpAndExit("-backend");
            }
            backend = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
    
    try {
        // TODO: Change this block to use VCN query API 
        int num_devices = 0, sd = 0;
        hipError_t hip_status = hipSuccess;
        hipDeviceProp_t hip_dev_prop;
        std::string gcn_arch_name;
        hip_status = hipGetDeviceCount(&num_devices);
        if (hip_status != hipSuccess) {
            std::cout << "ERROR: hipGetDeviceCount failed! (" << hip_status << ")" << std::endl;
            return -1;
        }

        if (num_devices < 1) {
            ROCDEC_ERR("ERROR: didn't find any GPU!");
            return -1;
        }

        hip_status = hipGetDeviceProperties(&hip_dev_prop, device_id);
        if (hip_status != hipSuccess) {
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

        std::vector<std::unique_ptr<VideoDemuxer>> v_demuxer;
        std::vector<std::unique_ptr<RocVideoDecoder>> v_viddec;
        std::vector<int> v_device_id(n_thread);

        int hip_vis_dev_count = 0;
        GetEnvVar("HIP_VISIBLE_DEVICES", hip_vis_dev_count);

        std::size_t found_file = input_file_path.find_last_of('/');
        std::cout << "info: Input file: " << input_file_path.substr(found_file + 1) << std::endl;
        std::cout << "info: Number of threads: " << n_thread << std::endl;

        for (int i = 0; i < n_thread; i++) {
            std::unique_ptr<VideoDemuxer> demuxer(new VideoDemuxer(input_file_path.c_str()));
            rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer->GetCodecID());
            if (!hip_vis_dev_count) {
                if (device_id % 2 == 0)
                    v_device_id[i] = (i % 2 == 0) ? device_id : device_id + sd;
                else
                    v_device_id[i] = (i % 2 == 0) ? device_id - sd : device_id;
            } else {
                v_device_id[i] = i % hip_vis_dev_count;
            }
            std::unique_ptr<RocVideoDecoder> dec;
            if (!backend) { // gpu backend
                dec = std::make_unique<RocVideoDecoder>(v_device_id[i], mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, false, disp_delay);
            } else {
            #if ENABLE_HOST_DECODE
                std::cout << "info: RocDecode is using CPU backend!" << std::endl;
                uint32_t max_width = demuxer->GetWidth();
                uint32_t max_height =demuxer->GetHeight();
                mem_type = OUT_SURFACE_MEM_HOST_COPIED;
                dec = std::make_unique<FFMpegVideoDecoder>(v_device_id[i], mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, false, disp_delay, max_width, max_height);
            #else
                std::cout << "Error: RocDecode HOST library is not found and backend is not supported!" << std::endl;
                return 0;
            #endif
            }

            if (!dec->CodecSupported(v_device_id[i], rocdec_codec_id, demuxer->GetBitDepth())) {
                std::cerr << "Codec not supported on GPU, skipping this file!" << std::endl;
                continue;
            }
            v_demuxer.push_back(std::move(demuxer));
            v_viddec.push_back(std::move(dec));
        }

        float total_fps = 0;
        float total_fps_dec = 0;
        std::vector<std::thread> v_thread;
        std::vector<double> v_fps, v_fps_dec;
        std::vector<int> v_frame, v_frame_dec;
        v_fps.resize(n_thread, 0);
        v_fps_dec.resize(n_thread, 0);
        v_frame.resize(n_thread, 0);
        v_frame_dec.resize(n_thread, 0);
        int n_total = 0;
        int n_total_dec = 0;
        OutputSurfaceInfo *p_surf_info;

        std::string device_name;
        int pci_bus_id, pci_domain_id, pci_device_id;

        for (int i = 0; i < n_thread; i++) {
            v_viddec[i]->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
            if (!backend) std::cout << "info: stream " << i << " using GPU device " << v_device_id[i] << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
            std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
            std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
            std::cout << "info: decoding started for thread " << i << " ,please wait!" << std::endl;
        }

        for (int i = 0; i < n_thread; i++) {
            v_thread.push_back(std::thread(DecProc, v_viddec[i].get(), v_demuxer[i].get(), &v_frame[i], &v_frame_dec[i], &v_fps[i], &v_fps_dec[i], max_num_frames, mem_type));
        }

        for (int i = 0; i < n_thread; i++) {
            v_thread[i].join();
            total_fps += v_fps[i];
            total_fps_dec += v_fps_dec[i];
            n_total += v_frame[i];
            n_total_dec += v_frame_dec[i];
        }

        std::cout << "info: Total pictures decoded: " << n_total_dec  << std::endl;
        std::cout << "info: Total frames output/displayed: " << n_total  << std::endl;
        std::cout << "info: avg decoding time per picture: " << 1000 / total_fps_dec << " ms" << std::endl;
        std::cout << "info: avg decode FPS: " << total_fps_dec  << std::endl;
        std::cout << "info: avg output/display time per frame: " << 1000 / total_fps << " ms" << std::endl;
        std::cout << "info: avg output/display FPS: " << total_fps  << std::endl;
    } catch (const std::exception &ex) {
      std::cout << ex.what() << std::endl;
      exit(1);
    }

    return 0;
}
