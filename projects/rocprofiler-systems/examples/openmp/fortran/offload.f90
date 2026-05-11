! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! OpenMP offload tracing workload. Launches NUM_KERNELS GPU kernels

program target_burner
    use omp_lib
    implicit none

    integer, parameter :: NUM_KERNELS     = 10
    integer, parameter :: WORK_PER_KERNEL = 65536

    real(8) :: total, partial, xx
    integer :: kernel, i

    total = 0.0d0

    do kernel = 1, NUM_KERNELS
        partial = 0.0d0

        !$omp target teams distribute parallel do                    &
        !$omp&   private(xx) reduction(+: partial)
        do i = 1, WORK_PER_KERNEL
            xx = real(i, kind=8) * 1.0d-4 + real(kernel, kind=8)
            partial = partial + sin(xx) * cos(xx)
        end do
        !$omp end target teams distribute parallel do

        total = total + partial
    end do

    print *, "kernels launched =", NUM_KERNELS
    print *, "result (sink)    =", total
end program target_burner
