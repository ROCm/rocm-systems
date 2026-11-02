! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! OpenMP host example that runs a round-robin parallel loop with two threads.

program host_combined
    use omp_lib
    implicit none

    call run_ordered_phase()

contains
    subroutine run_ordered_phase()
        integer, parameter :: N = 20
        integer :: i
        integer :: values(0:N)

        do i = 0, N
            values(i) = i
        end do

        !$omp parallel do num_threads(2) shared(values) schedule(static, 1)
        do i = 0, N
            values(i) = values(i) + omp_get_thread_num()
        end do
        !$omp end parallel do

        print *, "[phase 1] final values:"
        do i = 0, N
            print *, "values(", i, ")=", values(i)
        end do
    end subroutine run_ordered_phase

end program host_combined
