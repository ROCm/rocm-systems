// Fixed harness for gather_chain.hip: random inputs, a CPU reference check of every
// output, and timing of gather_launch (see sa_common.h).
#include <cmath>

#include "kernel_api.h"
#include "sa_common.h"

int
main(int argc, char** argv)
{
    sa::Args           args = sa::parse(argc, argv);
    sa::Rng            rng(args.seed);
    std::vector<int>   index(size_t(THREADS) * STEPS);
    std::vector<float> data(TABLE);
    for(auto& v : index)
        v = rng.below(TABLE);
    for(auto& v : data)
        v = rng.uniform(-1.0f, 1.0f);

    int*   dindex;
    float *ddata, *dout;
    HIP_CHECK(hipMalloc(&dindex, index.size() * sizeof(int)));
    HIP_CHECK(hipMalloc(&ddata, data.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, THREADS * sizeof(float)));
    HIP_CHECK(hipMemcpy(dindex, index.data(), index.size() * sizeof(int), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(ddata, data.data(), data.size() * sizeof(float), hipMemcpyHostToDevice));
    auto launch = [&] { gather_launch(dindex, ddata, dout); };
    if(args.once)
    {
        launch();
        sa::sync_all();
        return 0;
    }

    auto correct = [&] {
        std::vector<float> out(THREADS);
        HIP_CHECK(hipMemcpy(out.data(), dout, THREADS * sizeof(float), hipMemcpyDeviceToHost));
        for(int t = 0; t < THREADS; ++t)
        {
            double expect = 0, mag = 0;
            for(int j = 0; j < STEPS; ++j)
            {
                float v = data[index[size_t(j) * THREADS + t]];
                expect += v;
                mag += std::fabs(v);
            }
            if(!(std::fabs(out[t] - expect) <= 1e-5 * mag + 1e-5))
            {
                std::printf("output %d: %f, expected %f\n", t, out[t], expect);
                return false;
            }
        }
        return true;
    };
    sa::poison(dout, THREADS * sizeof(float));
    launch();
    sa::sync_all();
    if(!correct()) return sa::fail("first call");

    auto before = [&](int) {
        size_t p       = size_t(rng.next() % index.size());
        index[p]       = rng.below(TABLE);
        data[index[p]] = rng.uniform(-1.0f, 1.0f);
        HIP_CHECK(hipMemcpy(dindex + p, &index[p], sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(
            hipMemcpy(ddata + index[p], &data[index[p]], sizeof(float), hipMemcpyHostToDevice));
        sa::poison(dout, THREADS * sizeof(float));
    };
    double ms = sa::median_ms(args.iters, before, launch);
    if(!correct()) return sa::fail("result after the timed calls");
    std::printf("PASS  %.4f ms\n", ms);
    return 0;
}
