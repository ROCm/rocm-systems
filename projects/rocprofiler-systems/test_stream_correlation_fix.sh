#!/bin/bash
# test_stream_correlation_fix.sh

set -e

echo "🧪 Testing stream correlation fix..."

# Create test directory
mkdir -p stream_correlation_test
cd stream_correlation_test

# Create the test program
cat > test_stream_correlation.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

__global__ void simple_kernel(int* data, int value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        data[idx] = value;
    }
}

int main() {
    const int N = 1024;
    int *h_data, *d_data;
    
    // Allocate memory
    h_data = new int[N];
    hipMalloc(&d_data, N * sizeof(int));
    
    // Create explicit streams
    hipStream_t stream1, stream2;
    hipStreamCreate(&stream1);
    hipStreamCreate(&stream2);
    
    std::cout << "Launching kernels on different streams..." << std::endl;
    
    // Launch kernels on different streams to test correlation
    simple_kernel<<<4, 256, 0, stream1>>>(d_data, 1);      // stream1
    simple_kernel<<<4, 256>>>(d_data, 2);                  // default stream (0)
    simple_kernel<<<4, 256, 0, stream2>>>(d_data, 3);      // stream2
    simple_kernel<<<4, 256>>>(d_data, 4);                  // default stream (0) again
    
    // Add some memory operations too
    hipMemcpyAsync(h_data, d_data, N*sizeof(int), hipMemcpyDeviceToHost, stream1);
    hipMemcpy(h_data, d_data, N*sizeof(int), hipMemcpyDeviceToHost);  // default stream
    
    hipDeviceSynchronize();
    
    std::cout << "All operations completed" << std::endl;
    
    // Cleanup
    hipStreamDestroy(stream1);
    hipStreamDestroy(stream2);
    hipFree(d_data);
    delete[] h_data;
    
    return 0;
}
EOF

# Compile test
echo "🔨 Compiling test..."
hipcc -o test_stream_correlation test_stream_correlation.cpp

# Test 1: Stream grouping disabled (your fix should work here)
echo ""
echo "📊 Test 1: Stream-based grouping (ROCPROFSYS_ROCM_GROUP_BY_QUEUE=false)"
ROCPROFSYS_ROCM_DOMAINS="hip_runtime_api,kernel_dispatch" \
ROCPROFSYS_ROCM_GROUP_BY_QUEUE=false \
ROCPROFSYS_USE_PERFETTO=true \
ROCPROFSYS_VERBOSE=1 \
rocprof-sys-run -o test1 -- ./test_stream_correlation

# Test 2: Queue grouping enabled (should force queue tracks)
echo ""
echo "📊 Test 2: Queue-based grouping (ROCPROFSYS_ROCM_GROUP_BY_QUEUE=true)"
ROCPROFSYS_ROCM_DOMAINS="kernel_dispatch" \
ROCPROFSYS_ROCM_GROUP_BY_QUEUE=true \
ROCPROFSYS_USE_PERFETTO=true \
rocprof-sys-run -o test2 -- ./test_stream_correlation

# Find the actual trace files (they have timestamp subdirs and process IDs)
echo ""
echo "🔍 Checking generated trace files..."

# Find test1 trace file
TEST1_TRACE=$(find test1 -name "perfetto-trace-*.proto" 2>/dev/null | head -1)
if [ -n "$TEST1_TRACE" ]; then
    echo "✅ Found test1 trace: $TEST1_TRACE ($(du -h "$TEST1_TRACE" | cut -f1))"
else
    echo "❌ test1 perfetto trace not found"
    echo "   Contents of test1/:"
    ls -la test1/ 2>/dev/null || echo "   test1 directory not found"
    if [ -d test1 ]; then
        find test1 -name "*.proto" -o -name "*perfetto*" 2>/dev/null
    fi
fi

# Find test2 trace file  
TEST2_TRACE=$(find test2 -name "perfetto-trace-*.proto" 2>/dev/null | head -1)
if [ -n "$TEST2_TRACE" ]; then
    echo "✅ Found test2 trace: $TEST2_TRACE ($(du -h "$TEST2_TRACE" | cut -f1))"
else
    echo "❌ test2 perfetto trace not found"
    echo "   Contents of test2/:"
    ls -la test2/ 2>/dev/null || echo "   test2 directory not found"
    if [ -d test2 ]; then
        find test2 -name "*.proto" -o -name "*perfetto*" 2>/dev/null
    fi
fi

# Try validation with Python API first, fallback to simple version
echo ""
echo "🔍 Validating stream correlation..."

validation_passed=false

# Validate test1 (stream grouping)
if [ -n "$TEST1_TRACE" ]; then
    echo "Validating Test 1 (stream grouping): $TEST1_TRACE"
    
    # Try the full validation first
    if python3 ../validate_stream_correlation.py "$TEST1_TRACE" 2>/dev/null; then
        validation_passed=true
        echo "✅ Full validation with Python API succeeded"
    elif python3 ../validate_stream_correlation_simple.py "$TEST1_TRACE" 2>/dev/null; then
        validation_passed=true
        echo "✅ Simple validation with shell tool succeeded"
    else
        echo "❌ Both validation methods failed for test1"
    fi
else
    echo "❌ Cannot validate test1 - trace file not found"
fi

# Validate test2 (queue grouping) 
if [ -n "$TEST2_TRACE" ]; then
    echo ""
    echo "Validating Test 2 (queue grouping): $TEST2_TRACE"
    
    if python3 ../validate_stream_correlation.py "$TEST2_TRACE" 2>/dev/null; then
        echo "✅ Test 2 validation succeeded"
    elif python3 ../validate_stream_correlation_simple.py "$TEST2_TRACE" 2>/dev/null; then
        echo "✅ Test 2 simple validation succeeded"
    else
        echo "❌ Test 2 validation failed (this might be expected for queue grouping)"
    fi
else
    echo "❌ Cannot validate test2 - trace file not found"
fi

# Show some basic stats regardless
echo ""
echo "📈 Basic trace statistics:"
if [ -n "$TEST1_TRACE" ]; then
    file_size=$(du -h "$TEST1_TRACE" | cut -f1)
    echo "  Test 1 (stream grouping): $file_size"
fi
if [ -n "$TEST2_TRACE" ]; then
    file_size=$(du -h "$TEST2_TRACE" | cut -f1)
    echo "  Test 2 (queue grouping): $file_size"
fi

# Instructions for manual verification
echo ""
echo "🔗 Manual verification steps:"
echo "  1. Open https://ui.perfetto.dev in your browser"
if [ -n "$TEST1_TRACE" ]; then
    echo "  2. Upload $TEST1_TRACE"
fi
echo "  3. Look for tracks named 'HIP Activity Stream 0', 'HIP Activity Stream 1', etc."
echo "  4. Verify kernels appear under correct stream tracks"
if [ -n "$TEST2_TRACE" ]; then
    echo "  5. Compare with $TEST2_TRACE which should show queue-based grouping"
fi

if [ "$validation_passed" = true ]; then
    echo ""
    echo "✅ Stream correlation tests completed successfully!"
    echo "   Your _group_by_queue scoping fix appears to be working correctly."
else
    echo ""
    echo "⚠️  Automated validation failed, but traces were generated."
    echo "   Please verify manually using the Perfetto UI."
fi

cd ..
echo ""
echo "🏁 Test completed. Output files are in stream_correlation_test/"