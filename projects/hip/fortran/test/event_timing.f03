!!!!!!!!!!!!!!
! HIP runtime event timing and cross-stream ordering (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipEventCreateWithFlags, hipEventQuery, hipEventElapsedTime and
! hipStreamWaitEvent. The ordering check poisons a buffer and keeps the
! producing stream busy, so a missing wait reads poison rather than the data.
!!!!!!!!!!!!!!
!
program event_timing
  use iso_c_binding
  use hip

  implicit none

  ! dptr: 64 MB for timing + stream_a load   dptr2: small buffer for ordering test
  integer(c_size_t), parameter :: nbig   = 16_c_size_t * 1024 * 1024
  integer(c_size_t), parameter :: nsmall = 1024_c_size_t
  integer(c_size_t), parameter :: bytes_big   = nbig   * 4_c_size_t
  integer(c_size_t), parameter :: bytes_small = nsmall * 4_c_size_t

  type(c_ptr) :: estart = c_null_ptr, estop = c_null_ptr, emark = c_null_ptr
  type(c_ptr) :: stream_a = c_null_ptr, stream_b = c_null_ptr
  type(c_ptr) :: dptr = c_null_ptr, dptr2 = c_null_ptr
  real(c_float), target :: hsrc(nsmall), hdst(nsmall)
  real(c_float)         :: ms
  integer(kind(hipSuccess)) :: istat
  integer :: i

  write(*,"(a)",advance="no") &
    "-- Running test 'hip event_timing' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream_a))
  call hipCheck(hipStreamCreate(stream_b))
  call hipCheck(hipMalloc(dptr,  bytes_big))
  call hipCheck(hipMalloc(dptr2, bytes_small))

  ! estart: default flags; estop: blocking-sync so hipEventSynchronize is efficient
  call hipCheck(hipEventCreateWithFlags(estart, hipEventDefault))
  call hipCheck(hipEventCreateWithFlags(estop,  hipEventBlockingSync))
  call hipCheck(hipEventCreateWithFlags(emark,  hipEventDefault))

  ! ---- timing section: bracket four large async memsets on stream_a ----
  call hipCheck(hipEventRecord(estart, stream_a))
  call hipCheck(hipMemsetAsync(dptr, 0, bytes_big, stream_a))
  call hipCheck(hipMemsetAsync(dptr, 1, bytes_big, stream_a))
  call hipCheck(hipMemsetAsync(dptr, 2, bytes_big, stream_a))
  call hipCheck(hipMemsetAsync(dptr, 3, bytes_big, stream_a))
  call hipCheck(hipEventRecord(estop, stream_a))

  ! before sync: hipErrorNotReady or hipSuccess are both legitimate
  istat = hipEventQuery(estop)
  if (istat /= hipSuccess .and. istat /= hipErrorNotReady) then
    write(*,*) "FAILED! hipEventQuery (pre-sync) unexpected istat = ", istat
    call exit(1)
  end if

  call hipCheck(hipEventSynchronize(estop))

  ! after sync: must be hipSuccess
  istat = hipEventQuery(estop)
  if (istat /= hipSuccess) then
    write(*,*) "FAILED! hipEventQuery (post-sync) = ", istat, " expected hipSuccess=0"
    call exit(1)
  end if

  call hipCheck(hipEventElapsedTime(ms, estart, estop))
  if (ms <= 0.0 .or. ms >= 10000.0) then
    write(*,*) "FAILED! elapsed time = ", ms, " ms (expected 0 < ms < 10000)"
    call exit(1)
  end if
  write(*,"(a,f9.4,a)",advance="no") "(elapsed ", ms, " ms) "

  ! Cross-stream ordering: poison the buffer, then keep stream_a busy so a
  ! missing wait lets stream_b read poison instead of the pattern.
  call hipCheck(hipMemset(dptr2, 255, bytes_small))
  call hipCheck(hipDeviceSynchronize())

  do i = 1, int(nsmall)
    hsrc(i) = real(i, c_float)
  end do
  hdst = 0.0_c_float

  do i = 1, 12
    call hipCheck(hipMemsetAsync(dptr, i, bytes_big, stream_a))
  end do
  call hipCheck(hipMemcpyAsync(dptr2, c_loc(hsrc(1)), bytes_small, &
                               hipMemcpyHostToDevice, stream_a))
  call hipCheck(hipEventRecord(emark, stream_a))

  ! stream_b must wait for emark before reading dptr2
  call hipCheck(hipStreamWaitEvent(stream_b, emark, 0))
  call hipCheck(hipMemcpyWithStream(c_loc(hdst(1)), dptr2, bytes_small, &
                                    hipMemcpyDeviceToHost, stream_b))

  call hipCheck(hipStreamSynchronize(stream_b))

  do i = 1, int(nsmall)
    if (hdst(i) /= hsrc(i)) then
      write(*,*) "FAILED! cross-stream hdst(", i, ") = ", hdst(i), &
                 " expected ", hsrc(i)
      call exit(1)
    end if
  end do

  call hipCheck(hipEventDestroy(estart))
  call hipCheck(hipEventDestroy(estop))
  call hipCheck(hipEventDestroy(emark))
  call hipCheck(hipStreamDestroy(stream_a))
  call hipCheck(hipStreamDestroy(stream_b))
  call hipCheck(hipFree(dptr))
  call hipCheck(hipFree(dptr2))

  write(*,*) "PASSED!"

end program event_timing
