!!!!!!!!!!!!!!
! HIP runtime event timing
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipEventCreate, hipEventRecord, hipEventSynchronize,
! hipEventElapsedTime and hipEventDestroy by timing a device memset on the
! default stream.
!!!!!!!!!!!!!!
!
program event
  use iso_c_binding
  use hip

  implicit none

  type(c_ptr) :: estart = c_null_ptr, estop = c_null_ptr, dptr = c_null_ptr
  real(c_float) :: ms
  integer(c_size_t), parameter :: nbytes = 4 * 1024 * 1024

  write(*,"(a)",advance="no") "-- Running test 'hip event' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipMalloc(dptr, nbytes))

  call hipCheck(hipEventCreate(estart))
  call hipCheck(hipEventCreate(estop))

  ! Record around some device work on the default stream (stream = c_null_ptr).
  call hipCheck(hipEventRecord(estart, c_null_ptr))
  call hipCheck(hipMemset(dptr, 0, nbytes))
  call hipCheck(hipEventRecord(estop, c_null_ptr))

  call hipCheck(hipEventSynchronize(estop))
  call hipCheck(hipEventElapsedTime(ms, estart, estop))

  if (ms < 0.0) then
     write(*,*) "FAILED! elapsed time = ", ms, " ms (expected >= 0)"
     call exit(1)
  end if

  call hipCheck(hipEventDestroy(estart))
  call hipCheck(hipEventDestroy(estop))
  call hipCheck(hipFree(dptr))

  write(*,*) "PASSED!"

end program event
