! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! Combined OpenMP host example that runs two distinct phases in sequence:
!   Phase 1: round-robin parallel loop
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
    ! Phase 1: round-robin parallel loop                                  !
    ! ------------------------------------------------------------------ !
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
