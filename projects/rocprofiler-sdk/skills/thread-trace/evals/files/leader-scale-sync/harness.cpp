// Fixed harness for leader_scale.hip: random inputs, a new decay on every call, a CPU
// reference check of every output, and timing of leader_scale_launch (see sa_common.h).
#include <cmath>

#include "kernel_api.h"
#include "sa_common.h"

int
main(int argc, char** argv)
{
    sa::Args           args = sa::parse(argc, argv);
    sa::Rng            rng(args.seed);
    const size_t       n = size_t(BLOCKS) * ROUNDS * 256;
    std::vector<float> in(n);
    for(auto& v : in)
        v = rng.uniform(-50.0f, 50.0f);
    float decay = 0.999f;

    float *din, *dout;
    HIP_CHECK(hipMalloc(&din, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dout, n * sizeof(float)));
    HIP_CHECK(hipMemcpy(din, in.data(), n * sizeof(float), hipMemcpyHostToDevice));
    auto launch = [&] { leader_scale_launch(din, dout, decay); };
    if(args.once)
    {
        launch();
        sa::sync_all();
        return 0;
    }

    auto correct = [&] {
        std::vector<float> out(n);
        HIP_CHECK(hipMemcpy(out.data(), dout, n * sizeof(float), hipMemcpyDeviceToHost));
        for(int r = 0; r < ROUNDS; ++r)
        {
            double s = 0, term = 1;
            for(int k = 1; k <= SERIES; ++k)
                term *= double(decay), s += term / (k + r);
            double scale = 1 + 1e-3 * s;
            for(int b = 0; b < BLOCKS; ++b)
            {
                for(int x = 0; x < 256; ++x)
                {
                    size_t i = (size_t(b) * ROUNDS + r) * 256 + x;
                    if(!(std::fabs(out[i] - in[i] * scale) <= 1e-4 * (1 + std::fabs(in[i]))))
                    {
                        std::printf("output %zu: %f, expected %f\n", i, out[i], in[i] * scale);
                        return false;
                    }
                }
            }
        }
        return true;
    };
    sa::poison(dout, n * sizeof(float));
    launch();
    sa::sync_all();
    if(!correct()) return sa::fail("first call");

    auto before = [&](int) {
        decay = 0.998f + 0.0015f * rng.uniform(0.0f, 1.0f);
        sa::poison(dout, n * sizeof(float));
    };
    double ms = sa::median_ms(args.iters, before, launch);
    if(!correct()) return sa::fail("result after the timed calls");
    std::printf("PASS  %.4f ms\n", ms);
    return 0;
}
