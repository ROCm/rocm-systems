# ROCm-API-Tracer

A comprehensive toolkit for tracing ROCm/HIP API calls using three different approaches: **eBPF**, **Frida**, and **ltrace**. Each method offers unique advantages for different use cases, from high-performance production monitoring to simple development debugging.

## 🚀 Quick Start

Choose the method that best fits your needs:

- **Need maximum performance?** → [eBPF](./eBPF/) (minimal overhead, production-ready, call stack tracking)
- **Need flexibility and dynamic analysis?** → [Frida](./frida/) (JavaScript-based, highly customizable)
- **Need simplicity and quick debugging?** → [ltrace](./ltrace/) (easy to use, minimal setup)

## 📁 Project Structure

```
ROCm-API-Tracer/
├── eBPF/                          # High-performance unified tracing (HIP APIs + GPU kernels)
│   ├── hip_kernel_unified.bpf.c  # Unified eBPF program (kernel-space)
│   ├── hip_kernel_unified_tracer.c # Unified tracer (user-space)
│   ├── hsa_hybrid_shim.cpp       # HSA kernel dispatch shim (LD_PRELOAD)
│   ├── hip_trace.bpf.c           # Legacy HIP-only eBPF program
│   ├── hip_trace.c               # Legacy HIP-only tracer
│   ├── CMakeLists.txt            # CMake build system
│   ├── chrome_trace_writer.c     # Chrome Trace JSON writer
│   ├── README.md                 # eBPF-specific documentation
│   ├── design-docs/              # Technical documentation
│   │   ├── eBPF_HIP_Tracing_Technical_Guide.md
│   │   └── Unified_Tracer_Architecture.md
│   └── (build outputs and generated files)
├── frida/                         # Dynamic instrumentation with JavaScript
│   ├── frida-hip-tracer.js       # JavaScript tracing script
│   ├── frida-hip-trace.sh        # Shell wrapper script
│   ├── README.md                 # Frida-specific documentation
│   ├── FRIDA_INJECTION_MECHANISM.md  # Technical implementation details
│   ├── hip-frida-trace.log       # Sample log output
│   └── hip-frida-trace.csv       # Sample CSV output
├── ltrace/                        # Simple library call tracing
│   ├── ltrace-hip-final.sh       # Main tracing script
│   ├── analyze-final-trace.sh    # Analysis and CSV generation
│   ├── README.md                 # ltrace-specific documentation
│   ├── LTRACE_TECHNICAL_DETAILS.md  # Technical implementation
│   ├── hip-trace.log             # Sample trace output
│   ├── hip-trace_readable.log    # Human-readable trace
│   └── trace_analysis.csv        # Sample analysis output
├── TRACING_METHODS_COMPARISON.md  # Comprehensive comparison of all methods
├── README.md                      # This file
└── LICENSE                        # Project license
```

## 🎯 Method Selection Guide

### eBPF - The Performance Champion

**Recent Updates**: **Unified Tracer** combining HIP API tracing (eBPF uprobes) with GPU kernel dispatch tracing (HSA shim). Generates Chrome Trace JSON with complete GPU workload visibility.

**Best for**: Production monitoring, high-performance applications, complete GPU workload analysis

**Key Features**:
- ✅ **Unified Tracing**: HIP APIs + GPU kernel dispatches in one tool
- ✅ **Minimal overhead** (< 1% CPU impact)
- ✅ **GPU-accurate timestamps** from AMD profiling API
- ✅ **Chrome Trace JSON output** (Perfetto compatible)
- ✅ **Event correlation** linking HIP calls to kernel launches
- ✅ **Comprehensive metadata** (grid sizes, memory usage, etc.)
- ✅ **No rocprofiler-sdk dependency** for HIP tracing
- ✅ **Production-ready** with libbpf 1.7.0 optimizations

**Quick Start**:
```bash
cd eBPF
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# Terminal 1: Start tracer
sudo ./hip_kernel_unified_tracer -o trace.json

# Terminal 2: Run HIP app with shim
LD_PRELOAD=./libhsa_hybrid_shim.so /path/to/your/hip_app
```

**Requirements**: Linux kernel 5.4+, root privileges, libbpf-dev, ROCm 5.0+

### Frida - The Flexibility Leader
**Best for**: Dynamic analysis, reverse engineering, debugging, prototyping

**Key Features**:
- ✅ JavaScript-based customization
- ✅ Dynamic function discovery
- ✅ Rich argument type information
- ✅ No root privileges required
- ✅ Highly flexible and extensible

**Quick Start**:
```bash
cd frida
pip install frida-tools frida
./frida-hip-trace.sh /path/to/your/hip-app
```

**Requirements**: Python, frida-tools, target application

### ltrace - The Simplicity Winner
**Best for**: Quick debugging, simple analysis, development workflow

**Key Features**:
- ✅ Easiest to use and understand
- ✅ Low overhead (< 5% CPU impact)
- ✅ No root privileges required
- ✅ Automatic function discovery
- ✅ Minimal setup and dependencies

**Quick Start**:
```bash
cd ltrace
sudo apt-get install ltrace
./ltrace-hip-final.sh /path/to/your/hip-app
```

**Requirements**: ltrace utility, target application

## 📊 Comparison Summary

| Aspect | eBPF (Unified) | Frida | ltrace |
|--------|------|-------|--------|
| **Performance** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **GPU Kernel Tracing** | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| **Chrome Trace Output** | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| **Event Correlation** | ⭐⭐⭐⭐⭐ | ❌ | ❌ |
| **Ease of Use** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Setup Complexity** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Production Ready** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **HIP API Coverage** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

## 🔍 Detailed Comparison

For a comprehensive analysis of all three methods, including technical details, performance benchmarks, and use case recommendations, see [TRACING_METHODS_COMPARISON.md](./TRACING_METHODS_COMPARISON.md).

## 🛠️ Common Use Cases

### 1. Production Performance Monitoring
**Recommended**: eBPF (Unified Tracer)
- Complete GPU workload visibility (HIP APIs + kernel execution)
- Chrome Trace JSON output for Perfetto visualization
- GPU-accurate timestamps and correlation
- Minimal performance impact

### 2. Development and Debugging
**Recommended**: Frida or ltrace
- Debug HIP API usage during development
- Analyze function call patterns
- Identify performance bottlenecks

### 3. Reverse Engineering
**Recommended**: Frida
- Understand unknown HIP applications
- Dynamic analysis and modification
- Custom JavaScript-based logic

### 4. Quick Analysis
**Recommended**: ltrace
- Fast setup and execution
- Simple function call tracing
- Educational purposes

## 📈 Output Formats

All methods generate structured output for analysis:

### CSV Format
```csv
timestamp,thread_id,function_name,event,arguments,return_value,duration_ms
2025-01-01 12:00:00.000,12345,hipMalloc,ENTER,"void**=0x7fff12345678; size_t=4194304",,0
2025-01-01 12:00:00.001,12345,hipMalloc,LEAVE,"void**=0x7fff12345678; size_t=4194304",0x7fa000000000,1
```

### Log Format
```
[2025-01-01 12:00:00.000] [TID:12345] ENTER hipMalloc() [void**=0x7fff12345678, size_t=4194304]
[2025-01-01 12:00:00.001] [TID:12345] LEAVE hipMalloc() [ret=0x7fa000000000] [duration=1ms]
```

## 🔧 Analysis Tools

Each method includes analysis scripts:

- **eBPF**: `demo_csv_analysis.sh` - Comprehensive CSV analysis
- **Frida**: Built-in JavaScript analysis capabilities
- **ltrace**: `analyze-final-trace.sh` - Trace analysis and CSV generation

## 📚 Documentation

- **[eBPF Technical Guide](./eBPF/eBPF_HIP_Tracing_Technical_Guide.md)** - Deep dive into eBPF implementation
- **[Frida Injection Mechanism](./frida/FRIDA_INJECTION_MECHANISM.md)** - How Frida works internally
- **[ltrace Technical Details](./ltrace/LTRACE_TECHNICAL_DETAILS.md)** - ltrace implementation details
- **[Methods Comparison](./TRACING_METHODS_COMPARISON.md)** - Comprehensive comparison

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with all three methods
5. Submit a pull request

## 📄 License

This project is licensed under the GPL-2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [libbpf](https://github.com/libbpf/libbpf) for eBPF support
- Uses [Frida](https://github.com/frida/frida) for dynamic instrumentation
- Leverages [ltrace](https://man7.org/linux/man-pages/man1/ltrace.1.html) for library tracing
- Inspired by modern observability and profiling tools

## 🆘 Support

For issues and questions:

1. Check the method-specific README files
2. Review the technical documentation
3. Consult the comparison document
4. Open an issue on the repository

## 🚀 Getting Started

1. **Choose your method** based on your requirements
2. **Navigate to the appropriate directory** (eBPF, frida, or ltrace)
3. **Follow the Quick Start guide** in that directory
4. **Run the analysis tools** to understand your HIP application
5. **Refer to the documentation** for advanced usage

Happy tracing! 🎉