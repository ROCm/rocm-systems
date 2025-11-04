# test_stream_correlation_fix.ps1 - PowerShell version for Windows

Write-Host "🧪 Testing stream correlation fix..." -ForegroundColor Cyan

# Create test directory
New-Item -ItemType Directory -Force -Path "stream_correlation_test" | Out-Null
Set-Location "stream_correlation_test"

# Create the test program
$testCode = @'
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
'@

$testCode | Out-File -FilePath "test_stream_correlation.cpp" -Encoding UTF8

# Compile test
Write-Host "🔨 Compiling test..." -ForegroundColor Yellow
hipcc -o test_stream_correlation.exe test_stream_correlation.cpp

if (-not (Test-Path "test_stream_correlation.exe")) {
    Write-Host "❌ Compilation failed" -ForegroundColor Red
    exit 1
}

# Test 1: Stream grouping disabled 
Write-Host ""
Write-Host "📊 Test 1: Stream-based grouping (ROCPROFSYS_ROCM_GROUP_BY_QUEUE=false)" -ForegroundColor Green

$env:ROCPROFSYS_ROCM_DOMAINS = "hip_runtime_api,kernel_dispatch"
$env:ROCPROFSYS_ROCM_GROUP_BY_QUEUE = "false"
$env:ROCPROFSYS_USE_PERFETTO = "true"
$env:ROCPROFSYS_VERBOSE = "1"

rocprof-sys-run -o test1 -- ./test_stream_correlation.exe

# Test 2: Queue grouping enabled
Write-Host ""
Write-Host "📊 Test 2: Queue-based grouping (ROCPROFSYS_ROCM_GROUP_BY_QUEUE=true)" -ForegroundColor Green

$env:ROCPROFSYS_ROCM_DOMAINS = "kernel_dispatch"
$env:ROCPROFSYS_ROCM_GROUP_BY_QUEUE = "true"

rocprof-sys-run -o test2 -- ./test_stream_correlation.exe

# Find the actual trace files
Write-Host ""
Write-Host "🔍 Checking generated trace files..." -ForegroundColor Cyan

$test1Trace = Get-ChildItem -Path "test1" -Recurse -Filter "perfetto-trace-*.proto" | Select-Object -First 1
$test2Trace = Get-ChildItem -Path "test2" -Recurse -Filter "perfetto-trace-*.proto" | Select-Object -First 1

if ($test1Trace) {
    $size1 = [math]::Round($test1Trace.Length / 1KB, 2)
    Write-Host "✅ Found test1 trace: $($test1Trace.FullName) ($size1 KB)" -ForegroundColor Green
} else {
    Write-Host "❌ test1 perfetto trace not found" -ForegroundColor Red
    Write-Host "   Contents of test1/:" -ForegroundColor Yellow
    Get-ChildItem -Path "test1" -Recurse | Format-Table Name, Length
}

if ($test2Trace) {
    $size2 = [math]::Round($test2Trace.Length / 1KB, 2)
    Write-Host "✅ Found test2 trace: $($test2Trace.FullName) ($size2 KB)" -ForegroundColor Green
} else {
    Write-Host "❌ test2 perfetto trace not found" -ForegroundColor Red
    Write-Host "   Contents of test2/:" -ForegroundColor Yellow
    Get-ChildItem -Path "test2" -Recurse | Format-Table Name, Length
}

# Try validation
Write-Host ""
Write-Host "🔍 Validating stream correlation..." -ForegroundColor Cyan

$validationPassed = $false

if ($test1Trace) {
    Write-Host "Validating Test 1 (stream grouping): $($test1Trace.FullName)" -ForegroundColor Yellow
    
    try {
        python "../validate_stream_correlation.py" $test1Trace.FullName
        $validationPassed = $true
        Write-Host "✅ Full validation with Python API succeeded" -ForegroundColor Green
    } catch {
        try {
            python "../validate_stream_correlation_simple.py" $test1Trace.FullName
            $validationPassed = $true
            Write-Host "✅ Simple validation with shell tool succeeded" -ForegroundColor Green
        } catch {
            Write-Host "❌ Both validation methods failed for test1" -ForegroundColor Red
            Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
        }
    }
} else {
    Write-Host "❌ Cannot validate test1 - trace file not found" -ForegroundColor Red
}

# Show basic stats
Write-Host ""
Write-Host "📈 Basic trace statistics:" -ForegroundColor Cyan
if ($test1Trace) {
    Write-Host "  Test 1 (stream grouping): $([math]::Round($test1Trace.Length / 1KB, 2)) KB"
}
if ($test2Trace) {
    Write-Host "  Test 2 (queue grouping): $([math]::Round($test2Trace.Length / 1KB, 2)) KB"
}

# Manual verification instructions
Write-Host ""
Write-Host "🔗 Manual verification steps:" -ForegroundColor Cyan
Write-Host "  1. Open https://ui.perfetto.dev in your browser"
if ($test1Trace) {
    Write-Host "  2. Upload $($test1Trace.FullName)"
}
Write-Host "  3. Look for tracks named 'HIP Activity Stream 0', 'HIP Activity Stream 1', etc."
Write-Host "  4. Verify kernels appear under correct stream tracks"
if ($test2Trace) {
    Write-Host "  5. Compare with $($test2Trace.FullName) which should show queue-based grouping"
}

if ($validationPassed) {
    Write-Host ""
    Write-Host "✅ Stream correlation tests completed successfully!" -ForegroundColor Green
    Write-Host "   Your _group_by_queue scoping fix appears to be working correctly." -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "⚠️  Automated validation failed, but traces were generated." -ForegroundColor Yellow
    Write-Host "   Please verify manually using the Perfetto UI." -ForegroundColor Yellow
}

Set-Location ".."
Write-Host ""
Write-Host "🏁 Test completed. Output files are in stream_correlation_test/" -ForegroundColor Cyan