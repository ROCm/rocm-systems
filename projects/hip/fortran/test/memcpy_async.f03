!!!!!!!!!!!!!!
! HIP runtime asynchronous copies
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Copies host -> device with hipMemcpyAsync and device -> host with
! hipMemcpyWithStream on a user stream, then verifies the round trip.
!!!!!!!!!!!!!!
!
program memcpy_async
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 1024
  real(c_double), target :: hsrc(n), hdst(n)
  type(c_ptr) :: dptr = c_null_ptr, stream = c_null_ptr
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip memcpy_async' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8

  do i = 1, n
     hsrc(i) = real(i, c_double)
  end do
  hdst = 0.0d0

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipStreamCreate(stream))
  call hipCheck(hipMalloc(dptr, nbytes))

  ! Host -> device (async on the stream)
  call hipCheck(hipMemcpyAsync(dptr, c_loc(hsrc(1)), nbytes, hipMemcpyHostToDevice, stream))
  ! Device -> host (with the same stream)
  call hipCheck(hipMemcpyWithStream(c_loc(hdst(1)), dptr, nbytes, hipMemcpyDeviceToHost, stream))

  call hipCheck(hipStreamSynchronize(stream))

  do i = 1, n
     if (hdst(i) /= hsrc(i)) then
        write(*,*) "FAILED! hdst(", i, ") = ", hdst(i), " (expected ", hsrc(i), ")"
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(dptr))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program memcpy_async
