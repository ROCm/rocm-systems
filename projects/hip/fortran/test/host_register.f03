!!!!!!!!!!!!!!
! HIP runtime hipHostRegister / hipHostGetDevicePointer (Fortran 2003 interfaces)
! see: https:!rocm.docs.amd.com/projects/HIP/en/latest/
!
! Page-locks an existing host array, obtains its device pointer, copies through
! it on the device and verifies the data, then unregisters the array.
!!!!!!!!!!!!!!
!
program host_register
  use iso_c_binding
  use hip

  implicit none

  integer(c_int), parameter :: n = 256
  real(c_double), target :: harr(n), hcheck(n)
  type(c_ptr) :: ddev = c_null_ptr, dregdev = c_null_ptr
  integer(c_size_t) :: nbytes
  integer :: i

  write(*,"(a)",advance="no") "-- Running test 'hip host_register' (Fortran 2003 interfaces) - "

  nbytes = int(n, c_size_t) * 8
  do i = 1, n
     harr(i) = real(i, c_double)
  end do

  call hipCheck(hipSetDevice(0))

  ! Page-lock the host array (mapped so it has a device address) and fetch it.
  call hipCheck(hipHostRegister(c_loc(harr(1)), nbytes, hipHostRegisterMapped))
  call hipCheck(hipHostGetDevicePointer(dregdev, c_loc(harr(1)), 0))

  ! Copy registered host memory (via its device pointer) into a device buffer,
  ! then back to a plain host array.
  call hipCheck(hipMalloc(ddev, nbytes))
  call hipCheck(hipMemcpy(ddev, dregdev, nbytes, hipMemcpyDeviceToDevice))

  hcheck = 0.0d0
  call hipCheck(hipMemcpy(c_loc(hcheck(1)), ddev, nbytes, hipMemcpyDeviceToHost))

  do i = 1, n
     if (hcheck(i) /= harr(i)) then
        write(*,*) "FAILED! hcheck(", i, ") = ", hcheck(i), " expected ", harr(i)
        call exit(1)
     end if
  end do

  call hipCheck(hipFree(ddev))
  call hipCheck(hipHostUnregister(c_loc(harr(1))))

  write(*,*) "PASSED!"

end program host_register
