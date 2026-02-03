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
#include <fstream>
#include <cstring>
#include <string>
#include <iomanip>
#include <unistd.h>
#include <vector>
#include <string>
#include <chrono>
#include <sys/stat.h>
#include <libgen.h>
#include "video_demuxer.h"
#include "rocdecode/roc_bitstream_reader.h"
#include "roc_video_dec.h"
#include "common.h"
#if ENABLE_HOST_DECODE
    #include "ffmpeg_video_dec.h"
#endif

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input picture files - required" << std::endl
    << "-codec Codec type (0: HEVC, 1: AVC; 2: AV1; 3: VP9) - required" << std::endl
    << "-l Number of iterations - optional; default: 1" << std::endl
    << "-o Output File Path - dumps output if requested; optional" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
    << "-backend backend (0 for GPU, 1 CPU-FFMpeg, 2 CPU-FFMpeg No threading); optional; default: 0" << std::endl
    << "-f Number of decoded frames - specify the number of pictures to be decoded; optional" << std::endl
    << "-z force_zero_latency (force_zero_latency, Decoded frames will be flushed out for display immediately); optional;" << std::endl
    << "-disp_delay -specify the number of frames to be delayed for display; optional; default: 1" << std::endl
    << "-md5 generate MD5 message digest on the decoded YUV image sequence; optional;" << std::endl
    << "-md5_check MD5 File Path - generate MD5 message digest on the decoded YUV image sequence and compare to the reference MD5 string in a file; optional;" << std::endl
    << "-crop crop rectangle for output (not used when using interopped decoded frame); optional; default: 0" << std::endl
    << "-m output_surface_memory_type - decoded surface memory; optional; default - 0"
    << " [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]" << std::endl;
    exit(0);
}

int main(int argc, char **argv) {

    std::vector<const char*> file_names;
    std::string output_file_path, md5_file_path;
    std::fstream ref_md5_file;
    int codec_type = 0;
    int num_iterations = 1;
    int dump_output_frames = 0;
    int device_id = 0;
    int disp_delay = 1;
    int backend = 0;
    bool b_force_zero_latency = false;     // false by default: enabling this option might affect decoding performance
    bool b_extract_sei_messages = false;
    bool b_generate_md5 = false;
    bool b_md5_check = false;
    bool b_flush_frames_during_reconfig = true;
    Rect crop_rect = {};
    Rect *p_crop_rect = nullptr;
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;      // set to internal
    ReconfigParams reconfig_params = { 0 };
    ReconfigDumpFileStruct reconfig_user_struct = { 0 };
    uint32_t num_decoded_frames = 0;  // default value is 0, meaning decode the entire stream

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
            for (; i < argc; i++) {
                file_names.push_back(argv[i]);
                if (i + 1 < argc) {
                    if (argv[i + 1][0] == '-') {
                        break;
                    }
                }
            }
            continue;
        }
        if (!strcmp(argv[i], "-codec")) {
            if (++i == argc) {
                ShowHelpAndExit("-codec");
            }
            codec_type = atoi(argv[i]);
            if (codec_type < 0 || codec_type > 3) {
                ShowHelpAndExit("-codec");
            }
            continue;
        }
        if (!strcmp(argv[i], "-l")) {
            if (++i == argc) {
                ShowHelpAndExit("-l");
            }
            num_iterations = atoi(argv[i]);
            if (num_iterations < 1) {
                ShowHelpAndExit("-l");
            }
            continue;
        }
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_file_path = argv[i];
            dump_output_frames = 1;
            continue;
        }
        if (!strcmp(argv[i], "-backend")) {
            if (++i == argc) {
                ShowHelpAndExit("-backend");
            }
            backend = atoi(argv[i]);
            continue;
        }

        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
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
                ShowHelpAndExit("-f");
            }
            num_decoded_frames = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-z")) {
            if (i == argc) {
                ShowHelpAndExit("-z");
            }
            b_force_zero_latency = true;
            continue;
        }
        if (!strcmp(argv[i], "-md5")) {
            if (i == argc) {
                ShowHelpAndExit("-md5");
            }
            b_generate_md5 = true;
            continue;
        }
        if (!strcmp(argv[i], "-md5_check")) {
            if (++i == argc) {
                ShowHelpAndExit("-md5_check");
            }
            b_generate_md5 = true;
            b_md5_check = true;
            md5_file_path = argv[i];
            continue;
        }
        if (!strcmp(argv[i], "-crop")) {
            if (++i == argc || 4 != sscanf(argv[i], "%d,%d,%d,%d", &crop_rect.left, &crop_rect.top, &crop_rect.right, &crop_rect.bottom)) {
                ShowHelpAndExit("-crop");
            }
            if ((crop_rect.right - crop_rect.left) % 2 == 1 || (crop_rect.bottom - crop_rect.top) % 2 == 1) {
                std::cout << "output crop rectangle must have width and height of even numbers" << std::endl;
                exit(1);
            }
            p_crop_rect = &crop_rect;
            continue;
        }
        if (!strcmp(argv[i], "-m")) {
            if (++i == argc) {
                ShowHelpAndExit("-m");
            }
            mem_type = static_cast<OutputSurfaceMemoryType>(atoi(argv[i]));
            continue;
        }
        if (!strcmp(argv[i], "flush")) {
            b_flush_frames_during_reconfig = atoi(argv[i]) ? true : false;
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }

    try {
        std::cout << "Total frame number = " << file_names.size() << std::endl;
        rocDecVideoCodec rocdec_codec_id;
        switch (codec_type) {
            case 0:
                rocdec_codec_id = rocDecVideoCodec_HEVC;
                break;
            case 1:
                rocdec_codec_id = rocDecVideoCodec_AVC;
                break;
            case 2:
                rocdec_codec_id = rocDecVideoCodec_AV1;
                break;
            case 3:
                rocdec_codec_id = rocDecVideoCodec_VP9;
                break;
            default:
                std::cerr << "Unsupported stream codec type." << std::endl;
                return 1;
        }

        RocVideoDecoder *viddec;
        if (!backend)   // gpu backend
            viddec = new RocVideoDecoder(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);
        else {
            std::cout << "info: RocDecode is using CPU backend!" << std::endl;
            bool use_threading = false;
            if (mem_type == OUT_SURFACE_MEM_DEV_INTERNAL) mem_type = OUT_SURFACE_MEM_DEV_COPIED;    // mem_type internal is not supported in this mode
            if (backend == 1) {
                viddec = new FFMpegVideoDecoder(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);
            } else
                viddec = new FFMpegVideoDecoder(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay, true);
        }

        std::string device_name, gcn_arch_name;
        int pci_bus_id, pci_domain_id, pci_device_id;

        viddec->GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
        std::cout << "info: Using GPU device " << device_id << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
        std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
        std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
        std::cout << "info: decoding started, please wait!" << std::endl;

        int n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
        int n_pic_decoded = 0, decoded_pics = 0;
        std::vector<uint8_t> bitstream(5 * 1024 * 1024);
        int pkg_flags = 0;
        uint8_t *pframe = nullptr;
        int64_t pts = 0;
        OutputSurfaceInfo *surf_info;
        uint32_t width, height;
        double total_dec_time = 0;
        bool first_frame = true;
        MD5Generator *md5_generator = nullptr;

        // initialize reconfigure params: the following is configured to dump to output which is relevant for this sample
        reconfig_params.p_fn_reconfigure_flush = ReconfigureFlushCallback;
        reconfig_user_struct.b_dump_frames_to_file = dump_output_frames;
        reconfig_user_struct.output_file_name = output_file_path;
        reconfig_params.reconfig_flush_mode = RECONFIG_FLUSH_MODE_NONE;
        if (dump_output_frames) {
            reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_DUMP_TO_FILE;
        }
        if (b_generate_md5) {
            reconfig_params.reconfig_flush_mode |= RECONFIG_FLUSH_MODE_CALCULATE_MD5;
        }
        reconfig_params.p_reconfig_user_struct = &reconfig_user_struct;

        if (b_generate_md5) {
            md5_generator = new MD5Generator();
            md5_generator->InitMd5();
            reconfig_user_struct.md5_generator_handle = static_cast<void*>(md5_generator);
        }
        viddec->SetReconfigParams(&reconfig_params);

        for (int i = 0; i < num_iterations; i++) {
            int num_frames_decoded_in_loop = 0;
            pkg_flags = 0;
            for ( const char* file_name : file_names) {
                std::ifstream in_file(file_name, std::ios::binary);
                if (!in_file) {
                    std::cerr << "Error: Failed to open " << file_name << " for reading." << std::endl;
                    exit(1);
                }
                in_file.seekg(0, std::ios::end);
                n_video_bytes = in_file.tellg();
                if (n_video_bytes > bitstream.size()) {
                    bitstream.resize(n_video_bytes);
                }
                in_file.seekg(0, std::ios::beg);
                if (!in_file.read(reinterpret_cast<char*>(bitstream.data()), n_video_bytes)) {
                    std::cerr << "Error: Failed to read " << file_name << "." << std::endl;
                    exit(1);
                }
                // Close the file
                in_file.close();

                auto start_time = std::chrono::high_resolution_clock::now();
                if (num_frames_decoded_in_loop + 1 == file_names.size()) {
                    pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
                }
                n_frame_returned = viddec->DecodeFrame(bitstream.data(), n_video_bytes, pkg_flags, pts, &decoded_pics);
                num_frames_decoded_in_loop++;

                if (!n_frame && !viddec->GetOutputSurfaceInfo(&surf_info)) {
                    std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                    break;
                }
                for (int i = 0; i < n_frame_returned; i++) {
                    pframe = viddec->GetFrame(&pts);
                    if (b_generate_md5 && pframe) {
                        md5_generator->UpdateMd5ForFrame(pframe, surf_info);
                    }
                    if (dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED) {
                        viddec->SaveFrameToFile(output_file_path, pframe, surf_info);
                    }
                    // release frame
                    viddec->ReleaseFrame(pts);
                }
                auto end_time = std::chrono::high_resolution_clock::now();
                auto time_per_decode = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                total_dec_time += time_per_decode;
                n_frame += n_frame_returned;
                n_pic_decoded += decoded_pics;
                if (num_decoded_frames && num_decoded_frames <= n_frame) {
                    break;
                }
            }
        }
        n_frame += viddec->GetNumOfFlushedFrames();

        std::cout << "info: Total pictures decoded: " << n_pic_decoded << std::endl;
        std::cout << "info: Total frames output/displayed: " << n_frame << std::endl;
        if (!dump_output_frames) {
            std::cout << "info: avg decoding time per picture: " << total_dec_time / n_pic_decoded << " ms" <<std::endl;
            std::cout << "info: avg decode FPS: " << (n_pic_decoded / total_dec_time) * 1000 << std::endl;
            std::cout << "info: avg output/display time per frame: " << total_dec_time / n_frame << " ms" <<std::endl;
            std::cout << "info: avg output/display FPS: " << (n_frame / total_dec_time) * 1000 << std::endl;
        } else {
            if (mem_type == OUT_SURFACE_MEM_NOT_MAPPED) {
                std::cout << "info: saving frames with -m 3 option is not supported!" << std::endl;
            } else {
                std::cout << "info: saved frames into " << output_file_path << std::endl;
            }
        }
        if (b_generate_md5) {
            uint8_t *digest;
            md5_generator->FinalizeMd5(&digest);
            std::cout << "MD5 message digest: ";
            for (int i = 0; i < 16; i++) {
                std::cout << std::setfill('0') << std::setw(2) << std::hex << static_cast<int>(digest[i]);
            }
            std::cout << std::endl;
            if (b_md5_check) {
                std::string ref_md5_string(33, 0);
                uint8_t ref_md5[16];
                ref_md5_file.open(md5_file_path.c_str(), std::ios::in);
                if ((ref_md5_file.rdstate() & std::ifstream::failbit) != 0) {
                    std::cerr << "Failed to open MD5 file." << std::endl;
                    return 1;
                }
                ref_md5_file.getline(ref_md5_string.data(), ref_md5_string.length());
                if ((ref_md5_file.rdstate() & std::ifstream::badbit) != 0) {
                    std::cerr << "Failed to read MD5 digest string." << std::endl;
                    return 1;
                }
                for (int i = 0; i < 16; i++) {
                    std::string part = ref_md5_string.substr(i * 2, 2);
                    ref_md5[i] = std::stoi(part, nullptr, 16);
                }
                if (memcmp(digest, ref_md5, 16) == 0) {
                    std::cout << "MD5 digest matches the reference MD5 digest: ";
                } else {
                    std::cout << "MD5 digest does not match the reference MD5 digest: ";
                }
                std::cout << ref_md5_string.c_str() << std::endl;
                ref_md5_file.close();
            }
            delete md5_generator;
        }
    } catch (const std::exception &ex) {
        std::cout << ex.what() << std::endl;
        exit(1);
    }

    return 0;
}
