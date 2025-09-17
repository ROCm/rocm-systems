#include <Kokkos_Core.hpp>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << "Kokkos configuration:" << std::endl;
    #ifdef KOKKOS_ENABLE_HIP
        std::cout << "  HIP enabled" << std::endl;
    #endif
    #ifdef KOKKOS_ENABLE_CUDA
        std::cout << "  CUDA enabled" << std::endl;
    #endif
    #ifdef KOKKOS_ENABLE_OPENMP
        std::cout << "  OpenMP enabled" << std::endl;
    #endif
    #ifdef KOKKOS_ENABLE_SERIAL
        std::cout << "  Serial enabled" << std::endl;
    #endif

    Kokkos::initialize(argc, argv);
    std::cout << "Default execution space: " << Kokkos::DefaultExecutionSpace::name() << std::endl;

    {
        int N = 1024;
        Kokkos::View<int*> data("data", N);

        Kokkos::parallel_for("FillData", N, KOKKOS_LAMBDA(const int i) {
            data(i) = i * i;
        });

        int sum = 0;
        Kokkos::parallel_reduce("SumData", N, KOKKOS_LAMBDA(const int i, int& lsum) {
            lsum += data(i);
        }, sum);

        std::cout << "Sum of squares from 0 to " << N-1 << " is " << sum << std::endl;
    } // <-- All Kokkos Views destroyed here

    Kokkos::finalize();
    return 0;
}