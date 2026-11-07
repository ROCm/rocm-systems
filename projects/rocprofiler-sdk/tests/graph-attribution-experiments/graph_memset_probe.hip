#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x)                                                                                   \
    do                                                                                             \
    {                                                                                              \
        auto check_status = (x);                                                                   \
        if(check_status != hipSuccess)                                                             \
        {                                                                                          \
            fprintf(stderr, "%s: %s\n", #x, hipGetErrorString(check_status));                      \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)
struct Fill
{
    size_t   offset, count;
    unsigned value, element;
};
int
main()
{
    unsigned char* p;
    hipStream_t    s;
    CHECK(hipMalloc(&p, 16384));
    CHECK(hipStreamCreate(&s));
    auto issue = [&](Fill f) {
        if(f.element == 1)
            CHECK(hipMemsetD8Async((hipDeviceptr_t) (p + f.offset), f.value, f.count, s));
        else if(f.element == 2)
            CHECK(hipMemsetD16Async((hipDeviceptr_t) (p + f.offset), f.value, f.count, s));
        else
            CHECK(hipMemsetD32Async((hipDeviceptr_t) (p + f.offset), f.value, f.count, s));
    };
    auto run = [&](hipGraphExec_t ex, const std::vector<Fill>& fills) {
        for(int rep = 0; rep < 3; ++rep)
        {
            CHECK(hipMemsetAsync(p, 0, 16384, s));
            CHECK(hipGraphLaunch(ex, s));
            CHECK(hipStreamSynchronize(s));
            std::vector<unsigned char> got(16384), want(16384, 0);
            CHECK(hipMemcpy(got.data(), p, 16384, hipMemcpyDeviceToHost));
            for(auto f : fills)
                for(size_t i = 0; i < f.count * f.element; ++i)
                    want[f.offset + i] = (f.value >> (8 * (i % f.element))) & 255;
            if(got != want)
            {
                fprintf(stderr, "GPU byte oracle FAILED\n");
                exit(2);
            }
        }
    };
    auto capture = [&](const std::vector<Fill>& fs) {
        hipGraph_t g;
        CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
        for(auto f : fs)
            issue(f);
        CHECK(hipStreamEndCapture(s, &g));
        return g;
    };
    auto instantiate = [&](hipGraph_t g) {
        hipGraphExec_t e;
        CHECK(hipGraphInstantiate(&e, g, nullptr, nullptr, 0));
        return e;
    };
    std::vector<Fill> fs{{3, 33, 0xa5, 1}, {130, 17, 0xbeef, 2}, {260, 9, 0xa1b2c3d4, 4}};
    auto              g  = capture(fs);
    auto              ex = instantiate(g);
    run(ex, fs);
    size_t n = 0;
    CHECK(hipGraphGetNodes(g, nullptr, &n));
    std::vector<hipGraphNode_t> ns(n);
    CHECK(hipGraphGetNodes(g, ns.data(), &n));
    hipGraphNode_t  target = nullptr;
    hipMemsetParams params{};
    for(auto node : ns)
    {
        hipMemsetParams q{};
        CHECK(hipGraphMemsetNodeGetParams(node, &q));
        if(q.dst == p + 3)
        {
            target = node;
            params = q;
        }
    }
    if(!target) return 3;
    if(hipGraphExecMemsetNodeSetParams(ex, nullptr, &params) == hipSuccess) return 4;
    run(ex, fs);  // failed mutation must preserve attribution
    params.dst   = p + 512;
    params.value = 0x5a;
    CHECK(hipGraphExecMemsetNodeSetParams(ex, target, &params));
    auto changed      = fs;
    changed[0].offset = 512;
    changed[0].value  = 0x5a;
    run(ex, changed);
    auto fresh = instantiate(g);
    run(fresh, fs);
    CHECK(hipGraphExecDestroy(fresh));
    CHECK(hipGraphExecDestroy(ex));
    CHECK(hipGraphMemsetNodeSetParams(target, &params));
    fresh = instantiate(g);
    run(fresh, changed);
    CHECK(hipGraphExecDestroy(fresh));
    CHECK(hipGraphDestroy(g));
    // Same destination/length/value is ambiguous; different values are not.
    for(bool same : {true, false})
    {
        std::vector<Fill> two{{3, 33, 0xa5, 1}, {3, 33, same ? 0xa5u : 0x5au, 1}};
        g  = capture(two);
        ex = instantiate(g);
        run(ex, two);
        CHECK(hipGraphExecDestroy(ex));
        CHECK(hipGraphDestroy(g));
    }
    // A manual duplicate cannot borrow the capture API ID.
    g = capture({fs[0]});
    n = 1;
    hipGraphNode_t dep, added;
    CHECK(hipGraphGetNodes(g, &dep, &n));
    params             = {};
    params.dst         = p + 3;
    params.value       = 0xa5;
    params.elementSize = 1;
    params.width       = 33;
    params.height      = 1;
    CHECK(hipGraphAddMemsetNode(&added, g, &dep, 1, &params));
    ex = instantiate(g);
    run(ex, {fs[0]});
    CHECK(hipGraphExecDestroy(ex));
    CHECK(hipGraphDestroy(g));
    // Unsupported 2D fill must not let a modeled 1D fill borrow a source ID.
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    issue(fs[0]);
    CHECK(hipMemset2DAsync(p + 1024, 64, 0x7e, 33, 2, s));
    CHECK(hipStreamEndCapture(s, &g));
    ex = instantiate(g);
    run(ex, {fs[0], {1024, 33, 0x7e, 1}, {1088, 33, 0x7e, 1}});
    CHECK(hipGraphExecDestroy(ex));
    CHECK(hipGraphDestroy(g));
    // Small/unaligned/large and typed periodic patterns.
    std::vector<Fill> edge{{1, 1, 0xff, 1},
                           {33, 14, 0, 1},
                           {81, 33, 0x83, 1},
                           {160, 64, 0x6b, 1},
                           {257, 8193, 0xa7, 1},
                           {9002, 17, 0x1234, 2},
                           {9252, 9, 0x12345678, 4}};
    g  = capture(edge);
    ex = instantiate(g);
    run(ex, edge);
    CHECK(hipGraphExecDestroy(ex));
    CHECK(hipGraphDestroy(g));
    CHECK(hipStreamDestroy(s));
    CHECK(hipFree(p));
    puts("PASS: GPU byte oracle; unique/mutation/duplicate/manual/typed/edge memset graphs");
}
