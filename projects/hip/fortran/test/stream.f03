!!!!!!!!!!!!!!
! HIP runtime stream management
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Exercises hipStreamCreate, hipStreamGetDevice, hipStreamSynchronize and
! hipStreamDestroy.
!!!!!!!!!!!!!!
!
program test_stream
  use iso_c_binding
  use hip

  implicit none

  type(c_ptr)    :: stream = c_null_ptr
  integer(c_int) :: dev, sdev

  write(*,"(a)",advance="no") "-- Running test 'hip stream' (Fortran 2003 interfaces) - "

  call hipCheck(hipSetDevice(0))
  call hipCheck(hipGetDevice(dev))

  call hipCheck(hipStreamCreate(stream))
  if (.not. c_associated(stream)) then
     write(*,*) "FAILED! stream is null after hipStreamCreate"
     call exit(1)
  end if

  call hipCheck(hipStreamGetDevice(stream, sdev))
  if (sdev /= dev) then
     write(*,*) "FAILED! stream device = ", sdev, " (expected ", dev, ")"
     call exit(1)
  end if

  call hipCheck(hipStreamSynchronize(stream))
  call hipCheck(hipStreamDestroy(stream))

  write(*,*) "PASSED!"

end program test_stream
