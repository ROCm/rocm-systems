! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! Combined OpenMP host example that runs two distinct phases in sequence:
!   Phase 1: ordered parallel loop
!   Phase 2: parallel task with detach (Currently Disabled)


program host_combined
    use omp_lib
    implicit none

    call run_ordered_phase()
    print *, "----"
    ! DISABLED until underlying SDK issue is fixed
    ! (ROCP fatal occurs when early_fulfill arrives after task_complete, which should not be the case)
    ! call run_task_detach_phase()

contains

    ! ------------------------------------------------------------------ !
    ! Phase 1: ordered parallel loop                                      !
    ! ------------------------------------------------------------------ !
    subroutine run_ordered_phase()
        integer, parameter :: N = 16
        integer :: i, k
        real(8) :: input(N), output(N)
        real(8) :: running_sum, local

        do i = 1, N
            input(i) = real(i, kind=8)
        end do
        output = 0.0d0
        running_sum = 0.0d0

        print *, "[phase 1] ordered parallel loop"

        !$omp parallel do ordered num_threads(2)              &
        !$omp&     shared(input, output, running_sum)         &
        !$omp&     private(k, local)                          &
        !$omp&     schedule(static, 1)
        do i = 1, N
            local = 0.0d0
            do k = 1, 1000
                local = local + sin(real(i * k, kind=8))
            end do
            output(i) = 2.0d0 * input(i) + 1.0d-12 * local

            !$omp ordered
                running_sum = running_sum + output(i)
                print *, "i=", i, "thread=", omp_get_thread_num(), &
                         "output=", output(i), "running_sum=", running_sum
            !$omp end ordered
        end do
        !$omp end parallel do

        print *, "[phase 1] final running_sum =", running_sum
    end subroutine run_ordered_phase

    ! ------------------------------------------------------------------ !
    ! Phase 2: parallel task with detach                                  !
    ! ------------------------------------------------------------------ !
    subroutine run_task_detach_phase()
        integer, parameter :: N = 1000
        integer(kind=omp_event_handle_kind) :: event
        real(8) :: data(N)
        real(8) :: result
        integer :: i, j
        real(8) :: s

        do i = 1, N
            data(i) = real(i, kind=8)
        end do
        result = 0.0d0

        print *, "[phase 2] parallel task with detach"

        !$omp parallel num_threads(2) shared(data, result, event)
        !$omp single
            !$omp task detach(event) shared(data, result) private(j, s)
                s = 0.0d0
                do j = 1, N
                    s = s + data(j) * data(j)
                end do
                result = s
            !$omp end task

            !$omp task shared(event)
                call simulate_async_work()
                call omp_fulfill_event(event)
            !$omp end task

            !$omp taskwait
        !$omp end single
        !$omp end parallel

        print *, "[phase 2] result =", result
    end subroutine run_task_detach_phase

    ! ------------------------------------------------------------------ !
    ! Helper: simulate some host-side work before fulfilling the event    !
    ! ------------------------------------------------------------------ !
    subroutine simulate_async_work()
        integer :: k
        real(8) :: tmp
        tmp = 0.0d0
        do k = 1, 100000
            tmp = tmp + sin(real(k, kind=8))
        end do
        ! Use tmp so the loop is not eliminated by the optimizer
        if (tmp > huge(tmp)) print *, tmp
    end subroutine simulate_async_work

end program host_combined
