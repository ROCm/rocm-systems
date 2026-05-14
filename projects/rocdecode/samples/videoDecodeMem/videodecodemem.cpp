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
#include <iomanip>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
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
#include "md5.h"

class FileStreamProvider : public VideoDemuxer::StreamProvider {
public:
    FileStreamProvider(const char *input_file_path) {
        fp_in_.open(input_file_path, std::ifstream::in | std::ifstream::binary);
        if (!fp_in_) {
            std::cerr << "Unable to open input file: " << input_file_path << std::endl;
            exit(-1);
        }
        fp_in_.seekg (0, fp_in_.end);
        int length = fp_in_.tellg();
        fp_in_.seekg (0, fp_in_.beg);
        io_buffer_size_ = length;
    }
    ~FileStreamProvider() {
        fp_in_.close();
    }
    // Fill in the buffer owned by the demuxer
    int GetData(uint8_t *p_buf, int n_buf) {
        // We read a file for this example. You may get your data from network or somewhere else
        return static_cast<int>(fp_in_.read(reinterpret_cast<char*>(p_buf), n_buf).gcount());
    }
    size_t GetBufferSize() { return io_buffer_size_; };

private:
    std::ifstream fp_in_;
    size_t  io_buffer_size_;
};

void ShowHelpAndExit(const char *option = NULL) {
    std::cout << "Options:" << std::endl
    << "-i Input File Path - required" << std::endl
    << "-o Output File Path - dumps output if requested; optional" << std::endl
    << "-d GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
    << "-z force_zero_latency (force_zero_latency, Decoded frames will be flushed out for display immediately); optional;" << std::endl
    << "-sei extract SEI messages; optional;" << std::endl
    << "-md5 generate MD5 message digest on the decoded YUV image sequence; optional;" << std::endl
    << "-md5_check MD5 File Path - generate MD5 message digest on the decoded YUV image sequence and compare to the reference MD5 string in a file; optional;" << std::endl
    << "-crop crop rectangle for output (not used when using interopped decoded frame); optional; default: 0" << std::endl
    << "-m output_surface_memory_type - decoded surface memory; optional; default - 0"
    << " [0 : OUT_SURFACE_MEM_DEV_INTERNAL/ 1 : OUT_SURFACE_MEM_DEV_COPIED/ 2 : OUT_SURFACE_MEM_HOST_COPIED/ 3 : OUT_SURFACE_MEM_NOT_MAPPED]" << std::endl
    << "-disp_delay -specify the number of frames to be delayed for display; optional; default: 1" << std::endl;
    exit(0);
}

int main(int argc, char **argv) {

    std::string input_file_path, output_file_path, md5_file_path;
    std::fstream ref_md5_file;
    int dump_output_frames = 0;
    int device_id = 0;
    bool b_force_zero_latency = false;     // false by default: enabling this option might affect decoding performance
    bool b_extract_sei_messages = false;
    bool b_generate_md5 = false;
    bool b_md5_check = false;
    int disp_delay = 1;
    Rect crop_rect = {};
    Rect *p_crop_rect = nullptr;
    OutputSurfaceMemoryType mem_type = OUT_SURFACE_MEM_DEV_INTERNAL;        // set to internal
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
        if (!strcmp(argv[i], "-o")) {
            if (++i == argc) {
                ShowHelpAndExit("-o");
            }
            output_file_path = argv[i];
            dump_output_frames = 1;
            continue;
        }
        if (!strcmp(argv[i], "-d")) {
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            device_id = atoi(argv[i]);
            continue;
        }
        if (!strcmp(argv[i], "-z")) {
            if (i == argc) {
                ShowHelpAndExit("-z");
            }
            b_force_zero_latency = true;
            continue;
        }
        if (!strcmp(argv[i], "-sei")) {
            if (i == argc) {
                ShowHelpAndExit("-sei");
            }
            b_extract_sei_messages = true;
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
        if (!strcmp(argv[i], "-disp_delay")) {
            if (++i == argc) {
                ShowHelpAndExit("-disp_delay");
            }
            disp_delay = atoi(argv[i]);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }
    try {
        FileStreamProvider stream_provider(input_file_path.c_str());
        VideoDemuxer demuxer(&stream_provider);
        rocDecVideoCodec rocdec_codec_id = AVCodec2RocDecVideoCodec(demuxer.GetCodecID());
        RocVideoDecoder viddec(device_id, mem_type, rocdec_codec_id, b_force_zero_latency, p_crop_rect, b_extract_sei_messages, disp_delay);
        if(!viddec.CodecSupported(device_id, rocdec_codec_id, demuxer.GetBitDepth())) {
            std::cerr << "GPU doesn't support codec!" << std::endl;
            return 0;
        }

        std::string device_name, gcn_arch_name;
        int pci_bus_id, pci_domain_id, pci_device_id;

        viddec.GetDeviceinfo(device_name, gcn_arch_name, pci_bus_id, pci_domain_id, pci_device_id);
        std::cout << "info: Using GPU device " << device_id << " - " << device_name << "[" << gcn_arch_name << "] on PCI bus " <<
        std::setfill('0') << std::setw(2) << std::right << std::hex << pci_bus_id << ":" << std::setfill('0') << std::setw(2) <<
        std::right << std::hex << pci_domain_id << "." << pci_device_id << std::dec << std::endl;
        std::cout << "info: decoding started, please wait!" << std::endl;

        int n_video_bytes = 0, n_frame_returned = 0, n_frame = 0;
        uint8_t *pvideo = nullptr;
        int pkg_flags = 0;
        uint8_t *pframe = nullptr;
        int64_t pts = 0;
        OutputSurfaceInfo *surf_info;
        uint32_t width, height;
        double total_dec_time = 0;
        MD5Generator *md5_generator = nullptr;

        if (b_generate_md5) {
            md5_generator = new MD5Generator();
            md5_generator->InitMd5();
        }

        do {
            auto start_time = std::chrono::high_resolution_clock::now();
            demuxer.Demux(&pvideo, &n_video_bytes, &pts);
            // Treat 0 bitstream size as end of stream indicator
            if (n_video_bytes == 0) {
                pkg_flags |= ROCDEC_PKT_ENDOFSTREAM;
            }
            n_frame_returned = viddec.DecodeFrame(pvideo, n_video_bytes, pkg_flags, pts);
            auto end_time = std::chrono::high_resolution_clock::now();
            auto time_per_frame = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            total_dec_time += time_per_frame;
            if (!n_frame && !viddec.GetOutputSurfaceInfo(&surf_info)) {
                std::cerr << "Error: Failed to get Output Surface Info!" << std::endl;
                break;
            }
            for (int i = 0; i < n_frame_returned; i++) {
                pframe = viddec.GetFrame(&pts);
                if (b_generate_md5 && pframe) {
                    md5_generator->UpdateMd5ForFrame(pframe, surf_info);
                }
                if (dump_output_frames && mem_type != OUT_SURFACE_MEM_NOT_MAPPED) {
                    viddec.SaveFrameToFile(output_file_path, pframe, surf_info);
                }
                // release frame
                viddec.ReleaseFrame(pts);
            }
            n_frame += n_frame_returned;
        } while (n_video_bytes);

        std::cout << "info: Total frame decoded: " << n_frame << std::endl;
        if (!dump_output_frames) {
            std::cout << "info: avg decoding time per frame (ms): " << total_dec_time / n_frame << std::endl;
            std::cout << "info: avg FPS: " << (n_frame / total_dec_time) * 1000 << std::endl;
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
                std::cout << ref_md5_string << std::endl;
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
