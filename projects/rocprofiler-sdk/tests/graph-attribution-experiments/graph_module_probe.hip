#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CHECK(call)                                                                                \
    do                                                                                             \
    {                                                                                              \
        auto e = (call);                                                                           \
        if(e != hipSuccess)                                                                        \
        {                                                                                          \
            fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e));                              \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)
__global__ void
static_add(int* x, int v)
{
    if(threadIdx.x == 0) *x += v;
}
int
main(int argc, char** argv)
{
    if(argc != 2) return 2;
    hipModule_t   module;
    hipFunction_t add, mul;
    hipStream_t   stream;
    CHECK(hipModuleLoad(&module, argv[1]));
    CHECK(hipModuleGetFunction(&add, module, "module_add"));
    CHECK(hipModuleGetFunction(&mul, module, "module_mul"));
    CHECK(hipStreamCreate(&stream));
    int* x;
    CHECK(hipMalloc(&x, sizeof(int)));
    CHECK(hipMemset(x, 0, sizeof(int)));
    int            two = 2, three = 3, four = 4;
    void*          a[] = {&x, &two};
    void*          b[] = {&x, &three};
    void*          c[] = {&x, &four};
    hipGraph_t     graph;
    hipGraphExec_t exec;
    CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    hipLaunchKernelGGL(static_add, dim3(1), dim3(1), 0, stream, x, 1);
    CHECK(hipModuleLaunchKernel(add, 1, 1, 1, 1, 1, 1, 0, stream, a, nullptr));
    CHECK(hipModuleLaunchKernel(mul, 1, 1, 1, 1, 1, 1, 0, stream, a, nullptr));
    CHECK(hipModuleLaunchKernel(add, 1, 1, 1, 1, 1, 1, 0, stream, b, nullptr));
    CHECK(hipExtModuleLaunchKernel(
        add, 1, 1, 1, 1, 1, 1, 0, stream, c, nullptr, nullptr, nullptr, 0));
    hipLaunchKernelGGL(static_add, dim3(1), dim3(1), 0, stream, x, 5);
    CHECK(hipStreamEndCapture(stream, &graph));
    CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for(int i = 0; i < 3; ++i)
        CHECK(hipGraphLaunch(exec, stream));
    CHECK(hipStreamSynchronize(stream));
    int result = 0;
    CHECK(hipMemcpy(&result, x, sizeof(int), hipMemcpyDeviceToHost));
    if(result != 126)
    {
        fprintf(stderr, "wrong GPU result %d\n", result);
        return 3;
    }
    printf("module graph GPU oracle passed: %d\n", result);
    CHECK(hipGraphExecDestroy(exec));
    CHECK(hipGraphDestroy(graph));
    CHECK(hipFree(x));
    CHECK(hipStreamDestroy(stream));
    CHECK(hipModuleUnload(module));
}
