/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

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
#include <cstring>
#include <cstdlib>
#include <climits>
#include <string>
#include <sstream>
#include "hip/hip_runtime.h"
#include "rocjpeg/rocjpeg.h"

struct SubsamplingEntry {
    RocJpegChromaSubsampling subsampling;
    const char *name;
};

struct OutputFormatEntry {
    RocJpegOutputFormat format;
    const char *name;
};

struct CodingProcessEntry {
    RocJpegCodingProcess process;
    const char *name;
};

// Pixel format of the decoded image written to the RocJpegImage channels: plane i is written to channel[i].
struct ImageLayout {
    const char *name;
    int num_planes;
    const char *description;
};

static const SubsamplingEntry kSubsamplings[] = {
    {ROCJPEG_CSS_444, "4:4:4"},
    {ROCJPEG_CSS_440, "4:4:0"},
    {ROCJPEG_CSS_422, "4:2:2"},
    {ROCJPEG_CSS_420, "4:2:0"},
    {ROCJPEG_CSS_411, "4:1:1"},
    {ROCJPEG_CSS_400, "4:0:0"},
};

static const OutputFormatEntry kOutputFormats[] = {
    {ROCJPEG_OUTPUT_NATIVE,     "ROCJPEG_OUTPUT_NATIVE"},
    {ROCJPEG_OUTPUT_YUV_PLANAR, "ROCJPEG_OUTPUT_YUV_PLANAR"},
    {ROCJPEG_OUTPUT_Y,          "ROCJPEG_OUTPUT_Y"},
    {ROCJPEG_OUTPUT_RGB,        "ROCJPEG_OUTPUT_RGB"},
    {ROCJPEG_OUTPUT_RGB_PLANAR, "ROCJPEG_OUTPUT_RGB_PLANAR"},
};

static const CodingProcessEntry kCodingProcesses[] = {
    {ROCJPEG_CODING_PROCESS_BASELINE,    "Baseline (sequential, Huffman)"},
    {ROCJPEG_CODING_PROCESS_EXTENDED,    "Extended (sequential)"},
    {ROCJPEG_CODING_PROCESS_PROGRESSIVE, "Progressive"},
    {ROCJPEG_CODING_PROCESS_LOSSLESS,    "Lossless"},
};

static const ImageLayout kYuv444p = {"YUV444P", 3, "Y, U, V planes; U and V at full resolution"};
static const ImageLayout kYuv440p = {"YUV440P", 3, "Y, U, V planes; U and V at half height"};
static const ImageLayout kYuv422p = {"YUV422P", 3, "Y, U, V planes; U and V at half width"};
static const ImageLayout kYuv420p = {"YUV420P", 3, "Y, U, V planes; U and V at half width and half height"};
static const ImageLayout kNv12    = {"NV12",    2, "Y plane, then interleaved UV plane at half width and half height"};
static const ImageLayout kYuyv    = {"YUYV",    1, "packed Y0 U Y1 V, 2 bytes per pixel"};
static const ImageLayout kY8      = {"Y8",      1, "Y (luma) plane only"};
static const ImageLayout kRgb24   = {"RGB24",   1, "packed R, G, B, 3 bytes per pixel"};
static const ImageLayout kRgbp    = {"RGBP",    3, "R, G, B planes"};

// Centers text in a field of the given width.
std::string Center(const std::string &text, int width) {
    int left = (width - static_cast<int>(text.size())) / 2;
    if (left <= 0) {
        return text;
    }
    return std::string(left, ' ') + text + std::string(width - left - text.size(), ' ');
}

std::string TrimRight(std::string s) {
    s.erase(s.find_last_not_of(' ') + 1);
    return s;
}

// The NATIVE layout is reported by rocJpegGetDecoderCaps; the other output formats are derived from it
// as described for RocJpegOutputFormat in rocjpeg.h. Returns nullptr if there is no layout.
const ImageLayout *GetImageLayout(RocJpegOutputFormat output_format, uint32_t native_surface_format) {
    switch (output_format) {
        case ROCJPEG_OUTPUT_NATIVE:
            switch (native_surface_format) {
                case ROCJPEG_SURFACE_FORMAT_YUV444P: return &kYuv444p;
                case ROCJPEG_SURFACE_FORMAT_YUV440P: return &kYuv440p;
                case ROCJPEG_SURFACE_FORMAT_YUYV:    return &kYuyv;
                case ROCJPEG_SURFACE_FORMAT_NV12:    return &kNv12;
                case ROCJPEG_SURFACE_FORMAT_Y8:      return &kY8;
                default:                             return nullptr;
            }
        case ROCJPEG_OUTPUT_YUV_PLANAR:
            switch (native_surface_format) {
                case ROCJPEG_SURFACE_FORMAT_YUV444P: return &kYuv444p;
                case ROCJPEG_SURFACE_FORMAT_YUV440P: return &kYuv440p;
                case ROCJPEG_SURFACE_FORMAT_YUYV:    return &kYuv422p;
                case ROCJPEG_SURFACE_FORMAT_NV12:    return &kYuv420p;
                case ROCJPEG_SURFACE_FORMAT_Y8:      return &kY8;
                default:                             return nullptr;
            }
        case ROCJPEG_OUTPUT_Y:          return native_surface_format != ROCJPEG_SURFACE_FORMAT_NONE ? &kY8 : nullptr;
        case ROCJPEG_OUTPUT_RGB:        return native_surface_format != ROCJPEG_SURFACE_FORMAT_NONE ? &kRgb24 : nullptr;
        case ROCJPEG_OUTPUT_RGB_PLANAR: return native_surface_format != ROCJPEG_SURFACE_FORMAT_NONE ? &kRgbp : nullptr;
        default:                        return nullptr;
    }
}

void ShowHelpAndExit(const char *option = nullptr) {
    if (option) {
        std::cout << "Invalid option: " << option << std::endl;
    }
    std::cout << "Options:" << std::endl
    << "-d  GPU device ID (0 for the first device, 1 for the second, etc.); optional; default: 0" << std::endl
    << "-be rocJPEG backend (0 for hardware-accelerated JPEG decoding using VCN," << std::endl
    << "    1 for hybrid JPEG decoding using CPU and GPU HIP kernels (currently not supported)); optional; default: 0" << std::endl;
    exit(option ? 1 : 0);
}

bool ParseInt(const char *str, long min_value, long max_value, long *value) {
    char *end = nullptr;
    long parsed = strtol(str, &end, 10);
    if (end == str || *end != '\0' || parsed < min_value || parsed > max_value) {
        return false;
    }
    *value = parsed;
    return true;
}

std::string SubsamplingsString(uint32_t subsampling_mask) {
    std::string subsamplings;
    for (const auto &css : kSubsamplings) {
        if (subsampling_mask & (1u << css.subsampling)) {
            if (!subsamplings.empty()) {
                subsamplings += ", ";
            }
            subsamplings += css.name;
        }
    }
    return subsamplings.empty() ? "none" : subsamplings;
}

int main(int argc, char **argv) {
    int device_id = 0;
    RocJpegBackend backend = ROCJPEG_BACKEND_HARDWARE;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-h")) {
            ShowHelpAndExit();
        }
        if (!strcmp(argv[i], "-d")) {
            long parsed = 0;
            if (++i == argc) {
                ShowHelpAndExit("-d");
            }
            if (!ParseInt(argv[i], 0, INT_MAX, &parsed)) {
                std::cerr << "Error: invalid device ID '" << argv[i] << "'; expected a non-negative integer." << std::endl;
                return 1;
            }
            device_id = static_cast<int>(parsed);
            continue;
        }
        if (!strcmp(argv[i], "-be")) {
            long parsed = 0;
            if (++i == argc) {
                ShowHelpAndExit("-be");
            }
            if (!ParseInt(argv[i], ROCJPEG_BACKEND_HARDWARE, ROCJPEG_BACKEND_HYBRID, &parsed)) {
                std::cerr << "Error: invalid backend '" << argv[i] << "'; expected 0 or 1." << std::endl;
                return 1;
            }
            backend = static_cast<RocJpegBackend>(parsed);
            continue;
        }
        ShowHelpAndExit(argv[i]);
    }

    int num_devices = 0;
    if (hipGetDeviceCount(&num_devices) != hipSuccess || num_devices < 1) {
        std::cerr << "Error: No GPU device found!" << std::endl;
        return 1;
    }
    if (device_id >= num_devices) {
        std::cerr << "Error: device ID " << device_id << " is out of range; " << num_devices
                  << " device(s) available." << std::endl;
        return 1;
    }

    hipDeviceProp_t hip_dev_prop;
    if (hipSetDevice(device_id) != hipSuccess ||
        hipGetDeviceProperties(&hip_dev_prop, device_id) != hipSuccess) {
        std::cerr << "Error: Failed to query device properties for device " << device_id << std::endl;
        return 1;
    }

    char pci_bus_id[64] = {0};
    if (hipDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), device_id) != hipSuccess) {
        std::cerr << "Error: Failed to query PCI bus ID for device " << device_id << std::endl;
        return 1;
    }

    const char *backend_name = (backend == ROCJPEG_BACKEND_HARDWARE) ? "HARDWARE" : "HYBRID";
    std::cout << "JPEG decoder capabilities for GPU device " << device_id << " - " << hip_dev_prop.name
              << " [" << hip_dev_prop.gcnArchName << "] on PCI bus " << pci_bus_id << std::endl << std::endl;

    // rocJpegCreate may return a handle even when it fails; that handle must still be destroyed.
    RocJpegHandle rocjpeg_handle = nullptr;
    RocJpegStatus status = rocJpegCreate(backend, device_id, &rocjpeg_handle);
    if (status != ROCJPEG_STATUS_SUCCESS) {
        std::cout << "info: JPEG decoding is not available on GPU device " << device_id << " with the "
                  << backend_name << " backend (" << rocJpegGetErrorName(status) << ")." << std::endl;
        if (rocjpeg_handle != nullptr) {
            rocJpegDestroy(rocjpeg_handle);
        }
        return 0;
    }

    RocJpegDecodeCaps decode_caps = {};
    status = rocJpegGetDecoderCaps(rocjpeg_handle, &decode_caps);
    rocJpegDestroy(rocjpeg_handle);
    if (status != ROCJPEG_STATUS_SUCCESS) {
        std::cerr << "Error: rocJpegGetDecoderCaps failed: " << rocJpegGetErrorName(status) << std::endl;
        return 1;
    }

    if (!decode_caps.is_supported) {
        std::cout << "info: the rocJPEG decoder on GPU device " << device_id
                  << " reports no supported chroma subsampling." << std::endl;
        return 0;
    }

    const int label_width = 27;
    std::string coding_processes;
    for (const auto &process : kCodingProcesses) {
        if (decode_caps.coding_process_mask & (1u << process.process)) {
            coding_processes += (coding_processes.empty() ? "" : ", ") + std::string(process.name);
        }
    }
    std::string rgb_conversion;
    if (decode_caps.hw_rgb_subsampling_mask != 0) {
        rgb_conversion = "hardware (" + SubsamplingsString(decode_caps.hw_rgb_subsampling_mask) + ")";
    }
    uint32_t sw_rgb_subsampling_mask = decode_caps.subsampling_mask & ~decode_caps.hw_rgb_subsampling_mask;
    if (sw_rgb_subsampling_mask != 0) {
        rgb_conversion += (rgb_conversion.empty() ? "" : ", ");
        rgb_conversion += "software (" + SubsamplingsString(sw_rgb_subsampling_mask) + ")";
    }
    std::cout << std::left
              << std::setw(label_width) << "Backend" << ": " << backend_name << std::endl
              << std::setw(label_width) << "JPEG coding process" << ": " << (coding_processes.empty() ? "none" : coding_processes) << std::endl
              << std::setw(label_width) << "JPEG cores" << ": " << decode_caps.num_jpeg_cores << std::endl
              << std::setw(label_width) << "Min resolution (WxH)" << ": "
              << decode_caps.min_width << "x" << decode_caps.min_height << std::endl
              << std::setw(label_width) << "Max resolution (WxH)" << ": "
              << decode_caps.max_width << "x" << decode_caps.max_height << std::endl
              << std::setw(label_width) << "Max components" << ": " << static_cast<int>(decode_caps.max_components) << std::endl
              << std::setw(label_width) << "ROI (crop) decode" << ": "
              << (decode_caps.roi_decode ? (decode_caps.hw_roi_decode ? "yes (hardware)" : "yes (software)") : "no") << std::endl
              << std::setw(label_width) << "RGB color conversion" << ": " << rgb_conversion << std::endl
              << std::endl;

    // One row per output format; two columns (pixel format, number of planes) per supported chroma subsampling,
    // grouped under a centered chroma subsampling heading.
    const int format_width = 27;
    const int layout_width = 9;
    const int planes_width = 6;
    const int group_width = layout_width + planes_width;
    const std::string group_gap(3, ' ');
    int num_supported = 0;
    for (const auto &css : kSubsamplings) {
        num_supported += (decode_caps.subsampling_mask & (1u << css.subsampling)) ? 1 : 0;
    }
    std::cout << "Output pixel format and number of planes per output format (RocJpegOutputFormat) and chroma subsampling:"
              << std::endl << std::endl;
    std::string heading_row(format_width, ' '), separator_row(format_width, ' '), column_row = "Output format";
    column_row.resize(format_width, ' ');
    for (const auto &css : kSubsamplings) {
        if (!(decode_caps.subsampling_mask & (1u << css.subsampling))) {
            continue;
        }
        heading_row += Center(css.name, group_width) + group_gap;
        separator_row += std::string(group_width, '-') + group_gap;
        std::ostringstream columns;
        columns << std::left << std::setw(layout_width) << "Format" << std::setw(planes_width) << "Planes";
        column_row += columns.str() + group_gap;
    }
    std::cout << TrimRight(heading_row) << std::endl << TrimRight(separator_row) << std::endl << TrimRight(column_row) << std::endl
              << std::string(format_width + num_supported * (group_width + group_gap.size()), '-') << std::endl;

    const ImageLayout *used_layouts[] = {&kYuv444p, &kYuv440p, &kYuv422p, &kYuv420p, &kNv12, &kYuyv, &kY8, &kRgb24, &kRgbp};
    bool layout_used[sizeof(used_layouts) / sizeof(used_layouts[0])] = {};
    for (const auto &fmt : kOutputFormats) {
        if (!(decode_caps.output_format_mask & (1u << fmt.format))) {
            continue;
        }
        std::ostringstream row;
        row << std::left << std::setw(format_width) << fmt.name;
        for (const auto &css : kSubsamplings) {
            if (!(decode_caps.subsampling_mask & (1u << css.subsampling))) {
                continue;
            }
            const ImageLayout *layout = GetImageLayout(fmt.format, decode_caps.native_surface_format[css.subsampling]);
            if (layout == nullptr) {
                row << std::setw(layout_width) << "-" << Center("-", planes_width) << group_gap;
                continue;
            }
            row << std::setw(layout_width) << layout->name << Center(std::to_string(layout->num_planes), planes_width) << group_gap;
            for (size_t i = 0; i < sizeof(used_layouts) / sizeof(used_layouts[0]); i++) {
                layout_used[i] = layout_used[i] || (used_layouts[i] == layout);
            }
        }
        std::cout << TrimRight(row.str()) << std::endl;
    }

    std::cout << std::endl << "Pixel formats (plane i is written to RocJpegImage::channel[i]):" << std::endl;
    for (size_t i = 0; i < sizeof(used_layouts) / sizeof(used_layouts[0]); i++) {
        if (layout_used[i]) {
            std::cout << "  " << std::setw(layout_width) << used_layouts[i]->name << used_layouts[i]->description << std::endl;
        }
    }

    std::cout << std::endl << "info: " << num_supported << " supported chroma subsampling format(s)." << std::endl;
    return 0;
}
