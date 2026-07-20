! Copyright (c) Advanced Micro Devices, Inc.
! SPDX-License-Identifier: MIT
!
! OpenMP task-detach example.
!
! A task that carries a detach(event) clause is NOT considered complete until
! BOTH of the following have happened:
!   (1) its structured block has finished executing, AND
!   (2) its completion event has been fulfilled via omp_fulfill_event(event).
!
! Because those two conditions can occur in any order and the event can be
! fulfilled by any thread, there are several distinct "detach routes". This
! program is deliberately small but walks through every one of them:
!
!   Phase 1 : late fulfill by a sibling task    - the detached body finishes
!             first, then a different, concurrently running task releases it.
!   Phase 2 : early fulfill from inside the task - the event is fulfilled at
!             the top of the detached body, BEFORE the body finishes.
!   Phase 3 : fulfill by the generating thread   - the thread that created the
!             task releases the event itself before it synchronizes.
!   Phase 4 : fan-out, fulfilled out of order    - many detached tasks with a
!             mix of early and late fulfillment across several events.

program task_detach_routes
    use omp_lib
    implicit none

    call run_late_sibling_fulfill()
    print *, "----"
    call run_early_self_fulfill()
    print *, "----"
    call run_generator_fulfill()
    print *, "----"
    call run_fanout_fulfill()

contains

    ! ------------------------------------------------------------------ !
    ! Phase 1: late fulfill by a sibling task                            !
    !                                                                    !
    ! The classic asynchronous pattern: a detached "consumer" runs to    !
    ! the end of its body and then parks, waiting for its event. A       !
    ! separate "producer" task simulates asynchronous latency and then   !
    ! releases the consumer by fulfilling the event.                     !
    ! ------------------------------------------------------------------ !
    subroutine run_late_sibling_fulfill()
        integer, parameter :: N = 1000
        integer(kind=omp_event_handle_kind) :: event
        real(8) :: result
        result = 0.0d0

        print *, "[phase 1] late fulfill by a sibling task"

        !$omp parallel num_threads(2) shared(result, event)
        !$omp single
            !$omp task detach(event) shared(result)
                result = reduce_squares(N)
            !$omp end task

            !$omp task shared(event)
                call busy_work(200000)
                call omp_fulfill_event(event)
            !$omp end task

            !$omp taskwait
        !$omp end single
        !$omp end parallel

        print *, "[phase 1] result =", result
    end subroutine run_late_sibling_fulfill

    ! ------------------------------------------------------------------ !
    ! Phase 2: early fulfill from inside the detached task               !
    !                                                                    !
    ! The event is fulfilled at the very top of the detached body, so    !
    ! the event is already satisfied while the body is still running.    !
    ! The task still only completes once the body returns.               !
    ! ------------------------------------------------------------------ !
    subroutine run_early_self_fulfill()
        integer, parameter :: N = 1000
        integer(kind=omp_event_handle_kind) :: event
        real(8) :: result
        result = 0.0d0

        print *, "[phase 2] early fulfill from inside the detached task"

        !$omp parallel num_threads(2) shared(result, event)
        !$omp single
            !$omp task detach(event) shared(result)
                call omp_fulfill_event(event)
                call busy_work(200000)
                result = reduce_squares(N)
            !$omp end task

            !$omp taskwait
        !$omp end single
        !$omp end parallel

        print *, "[phase 2] result =", result
    end subroutine run_early_self_fulfill

    ! ------------------------------------------------------------------ !
    ! Phase 3: fulfill by the generating thread                          !
    !                                                                    !
    ! The thread that generates the detached task does some unrelated    !
    ! work and then fulfills the event itself, before reaching the       !
    ! taskwait synchronization point.                                    !
    ! ------------------------------------------------------------------ !
    subroutine run_generator_fulfill()
        integer, parameter :: N = 1000
        integer(kind=omp_event_handle_kind) :: event
        real(8) :: result
        result = 0.0d0

        print *, "[phase 3] fulfill by the generating thread"

        !$omp parallel num_threads(2) shared(result, event)
        !$omp single
            !$omp task detach(event) shared(result)
                result = reduce_squares(N)
            !$omp end task

            call busy_work(200000)
            call omp_fulfill_event(event)

            !$omp taskwait
        !$omp end single
        !$omp end parallel

        print *, "[phase 3] result =", result
    end subroutine run_generator_fulfill

    ! ------------------------------------------------------------------ !
    ! Phase 4: fan-out, fulfilled out of order                           !
    !                                                                    !
    ! Three detached tasks are all outstanding at once, mixing the       !
    ! routes above: task 1 is released late by a sibling, task 2         !
    ! fulfills itself early, and task 3 is released by the generating    !
    ! thread. The events are fulfilled in the opposite order to the one  !
    ! the tasks were created in, which stresses tracking of several      !
    ! simultaneously outstanding completion events.                      !
    ! ------------------------------------------------------------------ !
    subroutine run_fanout_fulfill()
        integer(kind=omp_event_handle_kind) :: ev1, ev2, ev3
        integer :: r1, r2, r3
        r1 = 0
        r2 = 0
        r3 = 0

        print *, "[phase 4] many detached tasks, fulfilled out of order"

        !$omp parallel num_threads(4) shared(ev1, ev2, ev3, r1, r2, r3)
        !$omp single
            ! Three detached tasks are created up front...
            !$omp task detach(ev1) shared(r1)
                call busy_work(150000)
                r1 = 1
            !$omp end task

            ! ev2 is implicitly firstprivate via the detach clause, so it must
            ! not appear in a shared clause on this same task construct.
            !$omp task detach(ev2) shared(r2)
                call omp_fulfill_event(ev2)        ! task 2 releases itself early
                call busy_work(100000)
                r2 = 2
            !$omp end task

            !$omp task detach(ev3) shared(r3)
                call busy_work(50000)
                r3 = 3
            !$omp end task

            ! ...and released in reverse order: the generating thread frees
            ! task 3 itself, while a sibling task frees task 1 later.
            call omp_fulfill_event(ev3)

            !$omp task shared(ev1)
                call busy_work(80000)
                call omp_fulfill_event(ev1)
            !$omp end task

            !$omp taskwait
        !$omp end single
        !$omp end parallel

        print *, "[phase 4] results =", r1, r2, r3
    end subroutine run_fanout_fulfill

    ! ------------------------------------------------------------------ !
    ! Helpers                                                            !
    ! ------------------------------------------------------------------ !

    ! Reduce the sum of squares of 1..n. Gives each detached body a small
    ! amount of real, order-independent work to perform.
    function reduce_squares(n) result(s)
        integer, intent(in) :: n
        real(8) :: s
        integer :: j
        s = 0.0d0
        do j = 1, n
            s = s + real(j, kind=8) * real(j, kind=8)
        end do
    end function reduce_squares

    ! Spin on a trivial floating-point loop to simulate latency. The guard on
    ! tmp keeps the optimizer from eliminating the loop.
    subroutine busy_work(iters)
        integer, intent(in) :: iters
        integer :: k
        real(8) :: tmp
        tmp = 0.0d0
        do k = 1, iters
            tmp = tmp + sin(real(k, kind=8))
        end do
        if (tmp > huge(tmp)) print *, tmp
    end subroutine busy_work

end program task_detach_routes
