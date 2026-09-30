! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! OpenMP offload tracing workload. Performs the transpose of a matrix on the GPU.

program target_transpose
    use omp_lib
    implicit none

    integer, parameter :: ROWS = 128
    integer, parameter :: COLS = 256

    integer :: input(ROWS, COLS), output(COLS, ROWS)
    integer :: i, j, errors

    do j = 1, COLS
        do i = 1, ROWS
            input(i, j) = i + (j * 10)
        end do
    end do
    output = 0

    ! Distribute the transpose operation across GPU threads
    !$omp target teams distribute parallel do collapse(2) map(to: input) map(from: output)
    do j = 1, COLS
        do i = 1, ROWS
            output(j, i) = input(i, j)
        end do
    end do
    !$omp end target teams distribute parallel do

    ! Validate that the transpose was performed correctly
    errors = 0
    do j = 1, COLS
        do i = 1, ROWS
            if (output(j, i) /= input(i, j)) errors = errors + 1
        end do
    end do

    print *, "matrix size       =", ROWS, "x", COLS
    print *, "validation errors =", errors
end program target_transpose
